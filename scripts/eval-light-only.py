"""#133: the light-only decoder (`LIGHT_ONLY`) against the full grammar, host-only.

Replays the per-step posteriors that scripts/sweep-align.py caches (same clips,
same models, same cache) through `decode_window` twice -- full grammar, and
restricted to Licht an / Licht aus -- and prints per model:

  hits   Licht an / aus phrases decoded to their own intent
  wrong  Licht an / aus phrases decoded to any other intent
  other  other phrases decoded to Licht an / aus (a light action nobody asked for)
  fa     negatives decoded to any intent (light-only: to Licht an / aus)

Usage:
  CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/eval-light-only.py [--model NAME=PATH ...]
"""

import argparse
import importlib.util
from pathlib import Path

from kws_de import config
from kws_de.grammar import Intent
from kws_de.window_intent import LIGHT_ONLY, decode_window

_spec = importlib.util.spec_from_file_location(
    "sweep_align", Path(__file__).parent / "sweep-align.py"
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def score(posts, rows, labels, allowed):
    n = {"light": 0, "hits": 0, "wrong": 0, "other": 0, "fa": 0, "neg": 0}
    for r in rows:
        got = decode_window(
            posts[r["file"]], labels, sweep.STEP_MS, first_ms=config.CLIP_MS, allowed=allowed
        )
        got = got if isinstance(got, Intent) else None
        t = r["target"]
        if t is None:
            n["neg"] += 1
            n["fa"] += got is not None
        elif t in LIGHT_ONLY:
            n["light"] += 1
            n["hits"] += got == t
            n["wrong"] += got is not None and got != t
        else:
            n["other"] += got in LIGHT_ONLY
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--approved", type=Path, default=config.DATA_DIR / "recordings" / "approved")
    ap.add_argument("--cache-dir", type=Path, default=config.DATA_DIR.parent / "cache" / "e70")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.model) if a.model else sweep.DEFAULT_MODELS
    rows = sweep.clips(a.approved)
    labels = config.COMMAND_LABELS
    print(f"{'model':10} {'decoder':10} hits/light wrong other(Licht an/aus) fa/neg")
    for name, path in models.items():
        posts = sweep.cached_posteriors(name, Path(path), rows, a.approved, a.cache_dir)
        for dec, allowed in (("full", None), ("light", LIGHT_ONLY)):
            n = score(posts, rows, labels, allowed)
            print(
                f"{name:10} {dec:10} {n['hits']}/{n['light']} {n['wrong']:5} {n['other']:5}"
                f"             {n['fa']}/{n['neg']}"
            )


if __name__ == "__main__":
    main()
