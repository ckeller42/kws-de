"""Thin build layer over `kws_de.data`: speaker-disjoint train/val/test feature
tensors + a verifiable manifest, all reproducible from one seed. See
docs/superpowers/plans/2026-09-01-kws-dataset-phase0.md.
"""

import argparse

import numpy as np

from kws_de import config
from kws_de.data import _origin_flags, build_dataset


def assemble(clips_ws, noises, rng, labels, commands):
    """Raw (clip, speaker) dict for one split -> (X, y, is_tts). Wraps build_dataset
    (features + labels) and _origin_flags (per-row real/TTS origin, same iteration
    order) -- neither is duplicated here, just composed."""
    clips = {lbl: [c for c, _ in items] for lbl, items in clips_ws.items()}
    X, y = build_dataset(clips, noises, rng, labels=labels, commands=commands)
    is_tts = _origin_flags(clips_ws, snrs=(20, 10, 0), words=commands)
    return X, y, np.asarray(is_tts, bool)


def load_split(name: str):
    """Load `data/features_{name}.npz` -> (X, y, is_tts)."""
    d = np.load(config.DATA_DIR / f"features_{name}.npz")
    return d["X"], d["y"], d["is_tts"]


def main() -> None:  # pragma: no cover - I/O wrapper (needs the raw-clip cache)
    """`kws-dataset build [--seed N]`."""
    ap = argparse.ArgumentParser(prog="kws-dataset")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()
    # 1. load cached raw (clip, speaker) dict (real MSWC + TTS-filled), see kws_de.data
    # 2. rng = np.random.default_rng(args.seed); split_three_way(..., keep_speaker=True)
    # 3. assemble(...) each split; np.savez data/features_{train,val,test}.npz
    # 4. manifest = build_manifest({...}, seed=args.seed, labels=config.COMMAND_LABELS)
    #    (config.DATA_DIR/"manifest.json").write_text(json.dumps(manifest, indent=2))
    raise NotImplementedError("wire the cached raw clips -> split_three_way -> assemble; see plan")
