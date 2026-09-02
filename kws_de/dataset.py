"""Thin build layer over `kws_de.data`: speaker-disjoint train/val/test feature
tensors + a verifiable manifest, all reproducible from one seed. See
docs/superpowers/plans/2026-09-01-kws-dataset-phase0.md.
"""

import argparse
import json
import pickle

import numpy as np

from kws_de import config
from kws_de.data import (
    _fill_with_tts,
    _origin_flags,
    build_dataset,
    command_words,
    merge_recordings,
    split_three_way,
)
from kws_de.manifest import build_manifest


def assemble(clips_ws, noises, rng, labels, commands):
    """Raw (clip, speaker) dict for one split -> (X, y, is_tts). Wraps build_dataset
    (features + labels) and _origin_flags (per-row real/TTS origin, same iteration
    order) -- neither is duplicated here, just composed. TTS clips get one perturbed
    copy (build_dataset `synthetic`), mirrored in the flags."""
    clips = {lbl: [c for c, _ in items] for lbl, items in clips_ws.items()}
    synthetic = {lbl: [s.startswith("tts:") for _, s in items] for lbl, items in clips_ws.items()}
    X, y = build_dataset(clips, noises, rng, labels=labels, commands=commands, synthetic=synthetic)
    is_tts = _origin_flags(clips_ws, snrs=(20, 10, 0), words=commands, perturb_tts=True)
    return X, y, np.asarray(is_tts, bool)


def load_split(name: str, prefix: str = "features"):
    """Load `data/{prefix}_{name}.npz` -> (X, y, is_tts). prefix "features" is the
    frozen v2 dataset, "features_v3" the real-speech rebuild."""
    d = np.load(config.DATA_DIR / f"{prefix}_{name}.npz")
    return d["X"], d["y"], d["is_tts"]


def force_rec_to_train(train: dict, *others: dict) -> int:
    """Move every device-recording (`rec:`) clip out of `others` into `train`.

    The product is a personalised device: a speaker records their own voice so
    the model learns it, so by default their clips belong in the training split
    (`kws-dataset build --recordings-split train`). With only one or two device
    speakers the global speaker-disjoint draw can otherwise put all of them in
    val/test, training on none of them. `--recordings-split auto` skips this and
    leaves them to the draw. Returns the number of clips moved."""
    moved = 0
    for other in others:
        for label, items in other.items():
            keep = [(c, s) for c, s in items if not s.startswith("rec:")]
            rec = [(c, s) for c, s in items if s.startswith("rec:")]
            if rec:
                train.setdefault(label, []).extend(rec)
                other[label] = keep
                moved += len(rec)
    return moved


def build(  # pragma: no cover - I/O
    seed: int = 0,
    cache_name: str = "raw_clips_merged.pkl",
    out_prefix: str = "features",
    recordings_split: str = "train",
):
    """Dataset build, deterministic from one seed given the cached raw clips (Piper TTS
    synthesis itself is stochastic per call, so newly-filled clips are persisted back to
    the cache — see `_fill_with_tts` call below — making the cache the reproducibility
    anchor): cached raw clips -> speaker-disjoint train/val/test features + manifest,
    written under config.DATA_DIR. Clean per-word features only
    (no transition-window augmentation — that is a training-time choice, kept out of the
    reusable dataset). Writes `data/manifest<suffix>.json` (suffix `_v3` for prefix
    `features_v3`). QC-approved device recordings are merged in on EVERY build
    (`kws_de.data.merge_recordings`), not baked into the cache, and with
    `recordings_split="train"` (the default) their speakers all go to the train split —
    see `force_rec_to_train`. Returns the manifest dict."""
    words = command_words()
    labels = config.COMMAND_LABELS
    # pickle: our own gitignored local cache written by kws_de.data, never untrusted input.
    with open(config.DATA_DIR / cache_name, "rb") as fh:
        cached = pickle.load(fh)  # noqa: S301
    clips_ws = cached["clips"]
    with open(config.DATA_DIR / "noise.pkl", "rb") as fh:
        noises = pickle.load(fh)  # noqa: S301

    # ensure every command word has clips; persist any TTS fill back to the cache so a
    # rebuild reuses these exact clips instead of resynthesizing (Piper is stochastic).
    tts_added = _fill_with_tts(clips_ws, words=words)
    if tts_added:
        with open(config.DATA_DIR / cache_name, "wb") as fh:
            pickle.dump(cached, fh)
        print(f"[tts] added: {tts_added}")
    # After the cache is persisted: QC-approved device recordings are re-read on every
    # build (QC output changes between builds, the cached MSWC/TTS clips do not).
    merged = merge_recordings(clips_ws)
    if merged:
        print(f"[recordings] merged: {merged}")
    # Split assignment is deterministic in `seed`; augmentation uses a derived stream
    # so the split (the reproducibility contract) is independent of augmentation draws.
    tr_ws, va_ws, te_ws = split_three_way(clips_ws, np.random.default_rng(seed), keep_speaker=True)
    if recordings_split == "train":
        moved = force_rec_to_train(tr_ws, va_ws, te_ws)
        if moved:
            print(f"[recordings] moved {moved} device clips into train (--recordings-split train)")
    splits = {}
    speakers = {}
    for i, (name, ws) in enumerate((("train", tr_ws), ("val", va_ws), ("test", te_ws))):
        X, y, is_tts = assemble(ws, noises, np.random.default_rng(seed + 1 + i), labels, words)
        np.savez(config.DATA_DIR / f"{out_prefix}_{name}.npz", X=X, y=y, is_tts=is_tts)
        splits[name] = (X, y, is_tts)
        speakers[name] = [s for items in ws.values() for _, s in items]

    manifest = build_manifest(splits, seed=seed, labels=labels, speakers=speakers)
    suffix = out_prefix.removeprefix("features")
    (config.DATA_DIR / f"manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(
        f"[dataset] built seed={seed}: " + ", ".join(f"{n}={splits[n][0].shape[0]}" for n in splits)
    )
    return manifest


def main() -> None:  # pragma: no cover - CLI wrapper
    """`kws-dataset build [--seed N] [--cache raw_clips_v3.pkl] [--prefix features_v3]
    [--recordings-split train|auto]`."""
    ap = argparse.ArgumentParser(prog="kws-dataset")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="raw_clips_merged.pkl", help="raw clip cache under data/")
    ap.add_argument("--prefix", default="features", help="output npz prefix (features_v3 ...)")
    ap.add_argument(
        "--recordings-split",
        choices=["train", "auto"],
        default="train",
        help="where QC-approved device recordings go: 'train' (default) forces every "
        "rec: speaker into the train split (personalised, in-training model); 'auto' "
        "leaves them to the global speaker-disjoint draw",
    )
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    build(
        seed=args.seed,
        cache_name=args.cache,
        out_prefix=args.prefix,
        recordings_split=args.recordings_split,
    )
