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
    SYNTHETIC_PREFIXES,
    _fill_with_tts,
    _origin_flags,
    build_dataset,
    command_words,
    merge_recordings,
    split_three_way,
)
from kws_de.manifest import build_manifest


def assemble(clips_ws, noises, rng, labels, commands, shift_ms=200, context_mix=0):
    """Raw (clip, speaker) dict for one split -> (X, y, is_tts). Wraps build_dataset
    (features + labels) and _origin_flags (per-row real/TTS origin, same iteration
    order) -- neither is duplicated here, just composed. TTS clips get one perturbed
    copy (build_dataset `synthetic`), mirrored in the flags; `context_mix` rows
    (build_dataset) likewise."""
    clips = {lbl: [c for c, _ in items] for lbl, items in clips_ws.items()}
    # `clone:` rows are synthetic too: `--real-weight` must not triple them, and they get
    # the same acoustic-variety augmentation TTS voices need.
    synthetic = {
        lbl: [s.startswith(SYNTHETIC_PREFIXES) for _, s in items] for lbl, items in clips_ws.items()
    }
    speakers = {lbl: [s for _, s in items] for lbl, items in clips_ws.items()}
    X, y = build_dataset(
        clips,
        noises,
        rng,
        labels=labels,
        commands=commands,
        synthetic=synthetic,
        shift_ms=shift_ms,
        context_mix=context_mix,
        speakers=speakers,
    )
    is_tts = _origin_flags(
        clips_ws, snrs=(20, 10, 0), words=commands, perturb_tts=True, context_mix=context_mix
    )
    return X, y, np.asarray(is_tts, bool)


def load_split(name: str, prefix: str = "features"):
    """Load `data/{prefix}_{name}.npz` -> (X, y, is_tts). prefix "features" is the
    frozen v2 dataset, "features_v3" the real-speech rebuild."""
    d = np.load(config.DATA_DIR / f"{prefix}_{name}.npz")
    return d["X"], d["y"], d["is_tts"]


def force_rec_to_train(train: dict, *others: dict) -> int:
    """Move every device-recording (`rec:`/`ctx:`) clip out of `others` into `train`.

    The product is a personalised device: a speaker records their own voice so
    the model learns it, so by default their clips belong in the training split
    (`kws-dataset build --recordings-split train`). With only one or two device
    speakers the global speaker-disjoint draw can otherwise put all of them in
    val/test, training on none of them. Context clips (`ctx:`, E48) are cut from
    the same device speakers' takes and get the same treatment, as do their
    voice-cloned clips (`clone:`): a clone of a training speaker must not become
    a val/test "speaker". `--recordings-split auto` skips this and leaves them
    to the draw. Returns the number of clips moved."""
    device = ("rec:", "ctx:", "clone:")
    moved = 0
    for other in others:
        for label, items in other.items():
            keep = [(c, s) for c, s in items if not s.startswith(device)]
            rec = [(c, s) for c, s in items if s.startswith(device)]
            if rec:
                train.setdefault(label, []).extend(rec)
                other[label] = keep
                moved += len(rec)
    return moved


def split_for_build(clips_ws: dict, seed: int, recordings_split: str = "train"):
    """The build's split: speaker-disjoint train/val/test drawn from `seed`, device
    recordings forced into train (`force_rec_to_train`), and the `clone:` clips set
    aside WITHOUT taking part in the draw. `split_three_way` permutes the sorted union of
    speaker ids, so three extra `clone:spkNN` "speakers" would re-deal every MSWC/TTS
    speaker: a build with clones would differ from the one without by a whole re-split,
    not by the clones (E64). Returns (train, val, test, clones) -- the three splits are
    what a build without the clone tree gives; `clones` is `{label: [(clip, speaker)]}`,
    train material (`assemble_clones`)."""
    base = {
        lbl: [(c, s) for c, s in items if not s.startswith("clone:")]
        for lbl, items in clips_ws.items()
    }
    clones = {
        lbl: [(c, s) for c, s in items if s.startswith("clone:")] for lbl, items in clips_ws.items()
    }
    tr_ws, va_ws, te_ws = split_three_way(base, np.random.default_rng(seed), keep_speaker=True)
    if recordings_split == "train":
        moved = force_rec_to_train(tr_ws, va_ws, te_ws)
        if moved:
            print(f"[recordings] moved {moved} device clips into train (--recordings-split train)")
    return tr_ws, va_ws, te_ws, {lbl: items for lbl, items in clones.items() if items}


def assemble_clones(clones_ws, noises, rng, labels, commands, shift_ms=200):
    """Feature rows of the `clone:` clips alone -> (X, y, is_tts), to be appended to the
    train split. Same treatment as a TTS clip in `assemble` (clean + per-snr rows, and a
    pitch/tempo-perturbed copy of each), but from their own `rng`: drawn inside the train
    split's stream they would shift every later augmentation draw, and the rows the two
    arms of a with/without-clones comparison share would stop being identical (E64).
    No context-mix rows for clones. build_dataset always emits `_silence_` rows; the
    train split already has its own, so they are dropped here."""
    clips = {lbl: [c for c, _ in items] for lbl, items in clones_ws.items()}
    synthetic = {lbl: [True] * len(items) for lbl, items in clones_ws.items()}
    X, y = build_dataset(
        clips, noises, rng, labels=labels, commands=commands, synthetic=synthetic, shift_ms=shift_ms
    )
    keep = y != list(labels).index("_silence_")
    return X[keep], y[keep], np.ones(int(keep.sum()), bool)


def build(  # pragma: no cover - I/O
    seed: int = 0,
    cache_name: str = "raw_clips_merged.pkl",
    out_prefix: str = "features",
    recordings_split: str = "train",
    shift_ms: int = 200,
    context_mix: int = 0,
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
    see `force_rec_to_train`. `context_mix` (multi-word context rows, see
    `kws_de.data.build_dataset`) applies to the train split only — val/test stay
    single-word rows. Returns the manifest dict."""
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
    tr_ws, va_ws, te_ws, clones = split_for_build(clips_ws, seed, recordings_split)
    splits = {}
    speakers = {}
    for i, (name, ws) in enumerate((("train", tr_ws), ("val", va_ws), ("test", te_ws))):
        X, y, is_tts = assemble(
            ws,
            noises,
            np.random.default_rng(seed + 1 + i),
            labels,
            words,
            shift_ms=shift_ms,
            context_mix=context_mix if name == "train" else 0,
        )
        speakers[name] = [s for items in ws.values() for _, s in items]
        if name == "train" and clones:
            # Appended after the split's own rows, from a stream of their own: every row
            # above is byte-identical to the build without the clone tree.
            Xc, yc, tc = assemble_clones(
                clones, noises, np.random.default_rng([seed, 64]), labels, words, shift_ms
            )
            X, y, is_tts = (
                np.concatenate([X, Xc]),
                np.concatenate([y, yc]),
                np.concatenate([is_tts, tc]),
            )
            clone_ids = [s for items in clones.values() for _, s in items]
            speakers[name] += clone_ids
            print(f"[recordings] appended {len(clone_ids)} clone clips ({len(yc)} rows) to train")
        np.savez(config.DATA_DIR / f"{out_prefix}_{name}.npz", X=X, y=y, is_tts=is_tts)
        splits[name] = (X, y, is_tts)

    manifest = build_manifest(
        splits,
        seed=seed,
        labels=labels,
        speakers=speakers,
        shift_ms=shift_ms,
        context_mix=context_mix,
    )
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
    [--recordings-split train|auto] [--shift-ms N] [--context-mix K]`."""
    ap = argparse.ArgumentParser(prog="kws-dataset")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--shift-ms",
        type=int,
        default=200,
        dest="shift_ms",
        help="max random time-shift of each word row in ms, either direction (default 200)",
    )
    ap.add_argument(
        "--context-mix",
        type=int,
        default=0,
        dest="context_mix",
        help="K multi-word context rows per train word clip (+1 gap-centred _unknown_ "
        "row each; default 0 = off, see kws_de.data.build_dataset)",
    )
    ap.add_argument("--cache", default="raw_clips_merged.pkl", help="raw clip cache under data/")
    ap.add_argument("--prefix", default="features", help="output npz prefix (features_v3 ...)")
    ap.add_argument(
        "--recordings-split",
        choices=["train", "auto"],
        default="train",
        help="where QC-approved device recordings go: 'train' (default) forces every "
        "rec:/ctx: speaker into the train split (personalised, in-training model); "
        "'auto' leaves them to the global speaker-disjoint draw",
    )
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    build(
        seed=args.seed,
        cache_name=args.cache,
        out_prefix=args.prefix,
        recordings_split=args.recordings_split,
        shift_ms=args.shift_ms,
        context_mix=args.context_mix,
    )
