"""Duration prediction for the long pipeline stages (QC, dataset build, train,
export, eval, and external mWW training) from a small JSON-lines history ledger.

Ledger: one JSON object per line at `config.DATA_DIR / "timings.jsonl"`
(override with `KWS_TIMINGS`) -- `record()` appends, `predict()` reads. Rate
(seconds per unit of `size`) rather than raw seconds is the modelled quantity,
so a prediction scales with whatever `size` a caller passes (epochs, wav
count, ...) instead of assuming every run is the same size.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from kws_de import config

_HISTORY = 10  # per-stage/host records a prediction is drawn from
_POLL_SECONDS = 30


def _ledger_path() -> Path:
    override = os.environ.get("KWS_TIMINGS")
    return Path(override) if override else config.DATA_DIR / "timings.jsonl"


def host_tag() -> str:
    """8 hex chars, never the raw hostname (KWS_HOST_TAG overrides for a
    stable tag across a machine's re-imaging/renaming)."""
    raw = os.environ.get("KWS_HOST_TAG") or platform.node()
    return sha256(raw.encode()).hexdigest()[:8]


def record(stage: str, size: float, seconds: float, note: str = "") -> None:
    """Append one finished-stage measurement to the ledger."""
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "stage": stage,
        "size": size,
        "seconds": seconds,
        "host": host_tag(),
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows(stage: str) -> Iterator[dict]:
    path = _ledger_path()
    if not path.exists():
        return
    host = host_tag()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("stage") != stage or row.get("host") != host or row.get("note") == "failed":
                continue
            yield row


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method) --
    exact for the hand-checked numbers in tests/test_eta.py."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


@dataclass(frozen=True)
class Prediction:
    seconds: float
    low: float
    high: float
    n: int


def predict(stage: str, size: float) -> Prediction | None:
    """Median of per-unit rates (seconds/size) from up to the last 10 records
    of `stage` on this host, scaled by `size`; low/high from the 20th/80th
    percentile rate. `None` with no usable history."""
    rates = [r["seconds"] / r["size"] for r in _rows(stage) if r["size"] > 0][-_HISTORY:]
    if not rates:
        return None
    ordered = sorted(rates)
    median_rate = statistics.median(rates)
    return Prediction(
        seconds=median_rate * size,
        low=_percentile(ordered, 20) * size,
        high=_percentile(ordered, 80) * size,
        n=len(rates),
    )


def format_eta(pred: Prediction | None, size: float) -> str:
    del size  # symmetric with predict()'s signature; not needed to render the text
    if pred is None:
        return "ETA unknown (first run of this stage; will be recorded)"
    runs = "run" if pred.n == 1 else "runs"
    return (
        f"ETA ~{pred.seconds / 60:.1f} min "
        f"(range {pred.low / 60:.1f}–{pred.high / 60:.1f}, from {pred.n} {runs})"
    )


class Timed:
    """`with Timed("train", size=20000, note="v3"): ...` -- prints the ETA on
    entry, records the measured wall time on exit (including on an exception,
    where `note` becomes "failed" so it's excluded from future predictions)."""

    def __init__(self, stage: str, size: float, note: str = ""):
        self.stage = stage
        self.size = size
        self.note = note
        self._t0 = 0.0

    def __enter__(self) -> "Timed":
        print(format_eta(predict(self.stage, self.size), self.size))
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        seconds = time.monotonic() - self._t0
        note = "failed" if exc_type is not None else self.note
        record(self.stage, self.size, seconds, note=note)


def _cmd_predict(args: argparse.Namespace) -> int:
    print(format_eta(predict(args.stage, args.size), args.size))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    record(args.stage, args.size, args.seconds, note=args.note)
    print(f"recorded: {args.stage} size={args.size} seconds={args.seconds:.1f}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    print(format_eta(predict(args.stage, args.size), args.size))
    t0 = time.monotonic()
    proc = subprocess.run(args.command)  # noqa: S603 - caller-supplied command, run as given
    seconds = time.monotonic() - t0
    record(args.stage, args.size, seconds, note="" if proc.returncode == 0 else "failed")
    return proc.returncode


def _latest_step(logfile: Path, pattern: re.Pattern) -> int | None:
    text = logfile.read_text(errors="replace")
    matches = list(pattern.finditer(text))
    return int(matches[-1].group(1)) if matches else None


def _cmd_watch(args: argparse.Namespace) -> int:
    logfile = Path(args.logfile)
    pattern = re.compile(args.pattern)
    st = logfile.stat()
    birth = getattr(st, "st_birthtime", st.st_ctime)
    while True:
        step = _latest_step(logfile, pattern)
        if step is not None:
            elapsed_min = max(time.time() - birth, 1e-9) / 60
            rate = step / elapsed_min
            eta_min = (args.total - step) / rate if rate > 0 else float("inf")
            print(f"step {step}/{args.total}, {rate:.0f} steps/min, ETA {eta_min:.1f} min")
            if args.stage:
                pred = predict(args.stage, args.total)
                if pred:
                    print(f"  predicted (history): {format_eta(pred, args.total)}")
            if step >= args.total:
                return 0
        if args.once:
            return 0
        time.sleep(_POLL_SECONDS)


def main() -> None:  # pragma: no cover - CLI wrapper
    ap = argparse.ArgumentParser(prog="kws-eta", description="stage-duration prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="print the ETA for a stage/size")
    p.add_argument("stage")
    p.add_argument("size", type=float)
    p.set_defaults(func=_cmd_predict)

    r = sub.add_parser("record", help="append a finished-stage measurement")
    r.add_argument("stage")
    r.add_argument("size", type=float)
    r.add_argument("seconds", type=float)
    r.add_argument("--note", default="")
    r.set_defaults(func=_cmd_record)

    run = sub.add_parser("run", help="run a command, print ETA first, record wall time after")
    run.add_argument("stage")
    run.add_argument("size", type=float)
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- <command...>")
    run.set_defaults(func=_cmd_run)

    w = sub.add_parser("watch", help="tail a training log, print step rate + ETA")
    w.add_argument("logfile")
    w.add_argument("--total", type=int, required=True)
    w.add_argument("--pattern", required=True, help="regex with one capturing group: the step")
    w.add_argument("--stage", default=None, help="also print the ledger's prediction for it")
    w.add_argument("--once", action="store_true", help="check once and exit (no 30s poll loop)")
    w.set_defaults(func=_cmd_watch)

    args = ap.parse_args()
    if args.cmd == "run" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.cmd == "run" and not args.command:
        ap.error("run: no command given (usage: kws-eta run <stage> <size> -- <command...>)")
    raise SystemExit(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    main()
