"""E70: grammar-constrained decoding (review S1) measured offline, host-only.

Runs each model ONCE over every approved phrase and negative clip, caches the
per-step posteriors, then replays `kws_de.window_intent.align_scores` +
`decide` over a grid of (smooth_win, floor, tau, delta) and both window-close
orders -- align first with the device's fire path as fallback, or fires first
with align as fallback -- against today's device path (`decode_window`).
Read-only: touches neither the models nor the firmware.

Tuning set = spk10's phrases; report set = every other speaker's phrases; the
negatives are a gate, never a tuning set (a setting whose false accepts exceed
today's on the same model is never selected). See
docs/superpowers/specs/2026-10-10-grammar-constrained-decoding-design.md §3.

Usage:
  CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/sweep-align.py \
      [--approved DIR] [--cache-dir DIR] [--out CSV] [--model NAME=PATH ...] [--dry-run]
"""

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.codegen import model_bytes
from kws_de.eval import _stream_posteriors, make_command_predict_fn, prompt_intent
from kws_de.grammar import Intent
from kws_de.window_intent import align_scores, decide, decode_window

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP_MS = 100
TUNE_SPEAKER = "spk10"
DEFAULT_MODELS = {
    "deployed": REPO_ROOT / "firmware" / "main" / "gen" / "model_data.h",
    **{
        f"gain2_s{s}": config.MODELS_DIR / "e64" / f"gain2_s{s}" / "command_v3_gain2_w48_qat.tflite"
        for s in range(3)
    },
}
SMOOTH = (1, 3)
FLOORS = (0.10, 0.25)
TAUS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)
DELTAS = (0.0, 0.02, 0.05, 0.10)
ORDERS = ("align-first", "fires-first")


def clips(approved: Path) -> list[dict]:
    """Every phrase (target Intent) and negative (target None), with speaker."""
    rows = []
    for kind in ("phrases", "negatives"):
        with (approved / kind / "index.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                target = prompt_intent(r["prompt"]) if kind == "phrases" else None
                if kind == "phrases" and not isinstance(target, Intent):
                    print(f"skip {r['file']}: prompt does not parse ({target})", file=sys.stderr)
                    continue
                rows.append({"file": r["file"], "speaker": r["speaker"], "target": target})
    return rows


def cached_posteriors(name: str, path: Path, rows: list[dict], approved: Path, cache: Path):
    """{file: posteriors (T x labels)} for `name`, computed once and cached as npz."""
    npz = cache / f"{name}.npz"
    if npz.exists():
        with np.load(npz) as z:
            got = {k: z[k] for k in z.files}
        if all(r["file"] in got for r in rows):
            return got
    predict_fn = make_command_predict_fn(model_bytes(path))
    step = int(config.SAMPLE_RATE * STEP_MS / 1000)
    out = {}
    for r in rows:
        sig, _ = sf.read(approved / r["file"], dtype="float32", always_2d=True)
        out[r["file"]] = np.stack(_stream_posteriors(predict_fn, sig[:, 0], step)).astype(
            np.float32
        )
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **out)
    return out


def _zone_flags(target: Intent, got) -> tuple[int, int]:
    """(zone dropped, zone invented) for a wrong-but-same-device/action answer."""
    if not isinstance(got, Intent) or got == target:
        return 0, 0
    same = got.device == target.device and got.action == target.action
    return int(same and target.zone and not got.zone), int(same and got.zone and not target.zone)


def evaluate(rows, decoded, labels) -> dict:
    """Score one decoder (file -> Intent|Rejection) over phrases (tune / report /
    all, by length, zone errors) and negatives (false accepts)."""
    m = defaultdict(int)
    for r in rows:
        got = decoded[r["file"]]
        t = r["target"]
        if t is None:
            m["neg"] += 1
            m["fa"] += isinstance(got, Intent)
            continue
        ok = got == t
        split = "tune" if r["speaker"] == TUNE_SPEAKER else "report"
        m[f"n_{split}"] += 1
        m[f"ok_{split}"] += ok
        L = 3 if t.zone else 2
        m[f"n_{L}w"] += 1
        m[f"ok_{L}w"] += ok
        d, i = _zone_flags(t, got)
        m["zone_dropped"] += d
        m["zone_invented"] += i
    m["ok_all"] = m["ok_tune"] + m["ok_report"]
    m["n_all"] = m["n_tune"] + m["n_report"]
    return dict(m)


def sweep_model(name: str, posts: dict, rows: list[dict], labels) -> list[dict]:
    today = {
        f: decode_window(p, labels, STEP_MS, first_ms=config.CLIP_MS) for f, p in posts.items()
    }
    base = evaluate(rows, today, labels)
    results = [{"model": name, "order": "today", **base}]
    for smooth, floor in itertools.product(SMOOTH, FLOORS):
        scored = {
            f: align_scores(
                p, labels, floor=floor, step_ms=STEP_MS, first_ms=config.CLIP_MS, smooth_win=smooth
            )
            for f, p in posts.items()
        }
        oracle = sum(
            bool(scored[r["file"]]) and scored[r["file"]][0][2] == r["target"]
            for r in rows
            if r["target"] is not None
        )
        for tau, delta in itertools.product(TAUS, DELTAS):
            aligned = {f: decide(s, tau, delta)[0] for f, s in scored.items()}
            for order in ORDERS:
                first, second = (aligned, today) if order == "align-first" else (today, aligned)
                decoded = {
                    f: first[f] if isinstance(first[f], Intent) else second[f] for f in posts
                }
                results.append(
                    {
                        "model": name,
                        "order": order,
                        "smooth": smooth,
                        "floor": floor,
                        "tau": tau,
                        "delta": delta,
                        "oracle": oracle,
                        **evaluate(rows, decoded, labels),
                    }
                )
    return results


def pick(results: list[dict], model: str, order: str) -> dict | None:
    """Best tune-set exact intent with false accepts <= today's, ties to the
    more conservative (higher tau, then delta) setting."""
    base = next(r for r in results if r["model"] == model and r["order"] == "today")
    ok = [
        r for r in results if r["model"] == model and r["order"] == order and r["fa"] <= base["fa"]
    ]
    return max(ok, key=lambda r: (r["ok_tune"], r["tau"], r["delta"])) if ok else None


def pick_shared(results: list[dict], models: list[str], order: str) -> tuple | None:
    """One (smooth, floor, tau, delta) for every model: max summed tune exact
    subject to false accepts <= today's on each model."""
    base = {
        m: next(r for r in results if r["model"] == m and r["order"] == "today")["fa"]
        for m in models
    }
    by_key = defaultdict(dict)
    for r in results:
        if r["order"] == order:
            by_key[(r["smooth"], r["floor"], r["tau"], r["delta"])][r["model"]] = r
    best = None
    for key, per in by_key.items():
        if len(per) < len(models) or any(per[m]["fa"] > base[m] for m in models):
            continue
        tot = sum(per[m]["ok_tune"] for m in models)
        cand = (tot, key[2], key[3], key, per)
        if best is None or cand[:3] > best[:3]:
            best = cand
    return best


def fmt(r: dict) -> str:
    return (
        f"{r['ok_tune']}/{r['n_tune']} | {r['ok_report']}/{r['n_report']}"
        f" | {r['ok_all']}/{r['n_all']} | {r['fa']}/{r['neg']}"
        f" | {r['ok_2w']}/{r['n_2w']} | {r['ok_3w']}/{r['n_3w']}"
        f" | {r['zone_dropped']} / {r['zone_invented']}"
    )


HEAD = (
    "| model | setting | tune spk10 | report | all | FA | 2-word | 3-word"
    " | zone dropped / invented |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--approved", type=Path, default=config.DATA_DIR / "recordings" / "approved")
    ap.add_argument("--cache-dir", type=Path, default=config.DATA_DIR.parent / "cache" / "e70")
    ap.add_argument("--out", type=Path, default=None, help="CSV of every grid row")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="model to sweep (default: deployed header + E68 gain2 s0/s1/s2)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    models = dict(m.split("=", 1) for m in a.model) if a.model else DEFAULT_MODELS
    models = {k: Path(v) for k, v in models.items()}
    rows = clips(a.approved)
    n_ph = sum(r["target"] is not None for r in rows)
    print(f"{n_ph} phrases, {len(rows) - n_ph} negatives; models: {', '.join(models)}")
    if a.dry_run:
        return
    labels = config.COMMAND_LABELS
    results = []
    for name, path in models.items():
        posts = cached_posteriors(name, path, rows, a.approved, a.cache_dir)
        results += sweep_model(name, posts, rows, labels)
        print(f"{name}: swept", file=sys.stderr)

    if a.out:
        with a.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in results for k in r}))
            w.writeheader()
            w.writerows(results)

    print("\n## Today (device path) and per-model best, false accepts capped at today's\n")
    print(HEAD)
    for name in models:
        base = next(r for r in results if r["model"] == name and r["order"] == "today")
        print(f"| {name} | today | {fmt(base)} |")
        for order in ORDERS:
            r = pick(results, name, order)
            if r:
                print(
                    f"| {name} | {order} s{r['smooth']} f{r['floor']} t{r['tau']} d{r['delta']}"
                    f" (oracle {r['oracle']}) | {fmt(r)} |"
                )
    print("\n## One shared setting for every model\n")
    print(HEAD)
    for order in ORDERS:
        got = pick_shared(results, list(models), order)
        if not got:
            print(f"| all | {order}: no setting keeps false accepts at today's on every model |")
            continue
        _, _, _, key, per = got
        for name in models:
            r = per[name]
            print(f"| {name} | {order} s{key[0]} f{key[1]} t{key[2]} d{key[3]} | {fmt(r)} |")


if __name__ == "__main__":
    main()
