"""Sweep the streaming decoder's fire threshold and hangover over every field take's
command span, against the DEPLOYED command model, and report where each setting lands
relative to the firmware's current constants. Read-only: never touches firmware or the
model, just prints a table.

Why: `firmware/main/stream.h`'s smoothing/debounce (KWS_THRESHOLD, KWS_MIN_CONSECUTIVE,
KWS_GAP_STEPS) was picked once and never swept against real field speech. Field takes
(`qc/*/qc.csv` rows with `set=field`) already carry a Whisper-derived ground truth in
their `prompt` column (empty = no command heard = negative material) — see
`kws_de.eval.prompt_intent` and `docs/sphinx/pipeline.rst` "The two evaluation figures".

Model: the bytes the firmware actually embeds (`kws_de.codegen.model_bytes` on
`firmware/main/gen/model_data.h`), not whatever `.tflite` happens to sit in `models/` —
same reasoning as `kws_de.eval.make_command_predict_fn`'s docstring.

Command span: default is the WHOLE take (`--span-start-s 0`). A `kws_de.qc.WAKE_MAX_S`
(2.5s) skip-the-preroll heuristic was tried first — field takes here run ~3.5s, and it
cut agreement to 0/18 across every setting, while streaming the whole take instead
recovers real agreement (up to 4/18 at the settings tried). The preroll's wake phrase
ends at a variable point inside it, not reliably at its edge, so a fixed skip risks
cutting into the actual command; the model doesn't recognise the wake phrase as any
command word anyway, so streaming over it costs a few harmless extra windows, nothing
more. `--span-start-s` stays available for experimentation.

"Hangover" here means extra confirmation frames beyond the first: hangover N means
`min_consecutive = N + 1` (hangover 0 fires on a single qualifying frame; hangover 1 is
`KWS_MIN_CONSECUTIVE=2`, the deployed value). Smoothing window and gap steps are left at
the deployed constants (`KWS_SMOOTH_WIN`, `KWS_GAP_STEPS`) — only threshold and hangover
are swept, per the task brief.

The model runs ONCE per take (posteriors cached); every threshold/hangover combination
then replays the cached posteriors through a fresh `kws_de.stream.KeywordStream` — cheap,
no repeated inference.

Usage:
  uv run --no-sync python scripts/sweep-decoder.py [--qc-root DIR] [--model PATH]
      [--span-start-s 0] [--dry-run]
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.codegen import model_bytes
from kws_de.eval import make_command_predict_fn, prompt_intent
from kws_de.grammar import Intent
from kws_de.stream import KeywordStream

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "firmware" / "main" / "gen" / "model_data.h"
FEATURES_CONFIG = REPO_ROOT / "firmware" / "main" / "gen" / "features_config.h"

THRESHOLDS = [round(0.3 + 0.1 * i, 1) for i in range(7)]  # 0.3 .. 0.9
HANGOVERS = (0, 1, 2)
STEP_MS = 100


def firmware_constants(path: Path = FEATURES_CONFIG) -> dict:
    """The deployed streaming constants, read straight out of the generated header —
    never hardcoded, so this always compares against whatever is actually shipped."""
    text = path.read_text()
    out = {}
    for name in ("KWS_THRESHOLD", "KWS_MIN_CONSECUTIVE", "KWS_GAP_STEPS", "KWS_SMOOTH_WIN"):
        m = re.search(rf"#define\s+{name}\s+([\d.]+)f?", text)
        if not m:
            raise ValueError(f"{path}: no #define {name}")
        out[name] = float(m.group(1))
    return out


def field_rows(qc_root: Path) -> list[dict]:
    """Every `set=field` row across `qc_root`'s session stamps, `file` resolved and
    checked to exist (a row surviving in qc.csv after its incoming/ tree was pruned is
    skipped, not fatal)."""
    rows = []
    for stamp in sorted(Path(qc_root).iterdir()):
        csv_path = stamp / "qc.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("set") == "field" and Path(r["file"]).exists():
                    rows.append(r)
    return rows


def command_span(path: Path, span_start_s: float) -> np.ndarray:
    """The take's audio from `span_start_s` onward (mono float32) — the part after
    where the wake phrase is expected to have ended. Shorter than that: empty."""
    sig, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = sig[:, 0]
    start = int(span_start_s * sr)
    return mono[start:]


def posteriors(predict_fn, audio: np.ndarray) -> list[np.ndarray]:
    """Run the model once over `audio`, trailing-window every STEP_MS — the same
    windowing `kws_de.eval._stream_events` uses, without the decode baked in, so the
    decode can be replayed cheaply per threshold/hangover setting below."""
    step_samples = int(config.SAMPLE_RATE * STEP_MS / 1000)
    n = len(audio)
    pos = config.CLIP_SAMPLES
    if n < pos:
        audio = np.pad(audio, (pos - n, 0))
        n = pos
    out = []
    while pos <= n:
        out.append(predict_fn(audio[pos - config.CLIP_SAMPLES : pos]))
        pos += step_samples
    return out


def decode(posts: list[np.ndarray], threshold: float, min_consecutive: int, fw: dict):
    ks = KeywordStream(
        predict_fn=None,
        labels=config.COMMAND_LABELS,
        smooth_win=int(fw["KWS_SMOOTH_WIN"]),
        threshold=threshold,
        min_consecutive=min_consecutive,
        gap_steps=int(fw["KWS_GAP_STEPS"]),
    )
    events = []
    for p in posts:
        events += ks.push(p)
    return events


def sweep(cached: list[tuple[list, object]], fw: dict) -> list[dict]:
    """`cached`: [(posteriors, target_or_None)] — one per field take, target is the
    Whisper-derived Intent or None for negative material. Returns one row per
    (threshold, hangover) with agreement over targets and false-fire rate over
    negatives."""
    from kws_de.grammar import parse

    results = []
    for threshold in THRESHOLDS:
        for hangover in HANGOVERS:
            min_consecutive = hangover + 1
            compared = agree = neg_total = false_fires = 0
            for posts, target in cached:
                pred = parse(decode(posts, threshold, min_consecutive, fw))
                if target is None:
                    neg_total += 1
                    false_fires += isinstance(pred, Intent)
                else:
                    compared += 1
                    agree += pred == target
            results.append(
                {
                    "threshold": threshold,
                    "hangover": hangover,
                    "min_consecutive": min_consecutive,
                    "agree": agree,
                    "compared": compared,
                    "false_fires": false_fires,
                    "negatives": neg_total,
                }
            )
    return results


def render(results: list[dict], fw: dict) -> str:
    out = ["| threshold | hangover | agreement | false fires |", "|---|---|---|---|"]
    for r in results:
        agree_rate = f"{r['agree']}/{r['compared']}" if r["compared"] else "n/a"
        fire_rate = f"{r['false_fires']}/{r['negatives']}" if r["negatives"] else "n/a"
        out.append(f"| {r['threshold']:.1f} | {r['hangover']} | {agree_rate} | {fire_rate} |")

    def rate(r):
        return r["agree"] / r["compared"] if r["compared"] else -1.0

    zero_fire = [r for r in results if r["false_fires"] == 0]
    best = max(zero_fire or results, key=rate)
    fw_hangover = int(fw["KWS_MIN_CONSECUTIVE"]) - 1
    current = next(
        (
            r
            for r in results
            if r["hangover"] == fw_hangover and r["threshold"] == fw["KWS_THRESHOLD"]
        ),
        None,
    )
    out.append("\nBest (false fires == 0 preferred), vs the firmware's current constants:")
    out.append(
        f"  best:    threshold={best['threshold']:.1f} hangover={best['hangover']} -> "
        f"agreement {best['agree']}/{best['compared']}, "
        f"false fires {best['false_fires']}/{best['negatives']}"
    )
    if current:
        out.append(
            f"  current: threshold={fw['KWS_THRESHOLD']:.1f} hangover={fw_hangover} "
            f"(KWS_THRESHOLD/KWS_MIN_CONSECUTIVE) -> "
            f"agreement {current['agree']}/{current['compared']}, "
            f"false fires {current['false_fires']}/{current['negatives']}"
        )
    else:
        out.append(
            f"  current: threshold={fw['KWS_THRESHOLD']:.1f} hangover={fw_hangover} "
            "is outside the swept grid — not directly comparable above."
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qc-root", default=str(config.DATA_DIR / "recordings" / "qc"), help="qc/<stamp>/ root"
    )
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="firmware model header/.tflite")
    ap.add_argument(
        "--span-start-s",
        type=float,
        default=0.0,
        help="seconds into each take where the command span starts (default: 0, whole take)",
    )
    ap.add_argument("--dry-run", action="store_true", help="list field-row counts, run nothing")
    args = ap.parse_args()

    span_start_s = args.span_start_s
    rows = field_rows(Path(args.qc_root))
    n_pos = sum(1 for r in rows if r.get("prompt"))
    print(f"{len(rows)} field takes ({n_pos} with a parsed command, {len(rows) - n_pos} negative)")
    if args.dry_run or not rows:
        return

    predict_fn = make_command_predict_fn(model_bytes(Path(args.model)))
    cached = []
    for r in rows:
        audio = command_span(Path(r["file"]), span_start_s)
        target = prompt_intent(r["prompt"]) if r.get("prompt") else None
        cached.append((posteriors(predict_fn, audio), target))

    fw = firmware_constants()
    print(render(sweep(cached, fw), fw))


if __name__ == "__main__":
    main()
