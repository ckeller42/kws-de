# NOTE: pickle is used only for this script's own local, gitignored data/ cache
# (raw_clips.pkl / noise.pkl) — written and read by this same code, never untrusted
# input — so the arbitrary-code-execution risk on unpickling doesn't apply here.
import argparse
import io
import logging
import os
import pickle
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from kws_de import config, tts
from kws_de.augment import mix_at_snr, perturb, van_augment
from kws_de.features import mfcc


def recordings_root() -> Path:
    """QC-approved word clips if the pipeline has run, else the legacy hand-dropped layout."""
    approved = config.DATA_DIR / "recordings" / "approved" / "words"
    return approved if approved.is_dir() else config.DATA_DIR / "recordings"


def merge_recordings(clips_ws: dict, root: Path | None = None) -> dict[str, int]:
    """Fold the QC-approved device recordings into a cached clip dict — on EVERY
    build, not just the one that created the cache. QC output changes between
    builds (a new session is ingested and approved); the cached MSWC/TTS clips do
    not, so the recordings are re-read here instead of being baked into the
    pickle. Previous ``rec:``/``ctx:`` entries are dropped first, so a re-build
    replaces them rather than duplicating them.

    `root` is the recordings directory (default ``config.DATA_DIR/"recordings"``).
    Two word trees are merged, kept distinguishable by speaker-id prefix so a
    consumer can tell them apart (or filter to one) without a schema change:
    ``approved/words/<label>/`` (guided single-word takes) becomes speaker
    ``rec:<spk>``; ``approved/context/<label>/`` (E48: word clips cut from a
    sentence/field/elicit take — real usage but with another vocabulary word
    often right next to the label in the 1 s window) becomes ``ctx:<spk>``.
    Approved negatives become ``_unknown_`` material via `negative_windows`.
    No-op returning ``{}`` when neither tree exists, which keeps v1/v2 builds
    byte-identical. Returns {label: n merged} (words + context combined)."""
    from kws_de.recordings import load_recordings

    root = Path(root) if root is not None else config.DATA_DIR / "recordings"
    word_trees = {"rec:": root / "approved" / "words", "ctx:": root / "approved" / "context"}
    if not any(d.is_dir() for d in word_trees.values()):
        return {}
    for lbl, items in clips_ws.items():
        clips_ws[lbl] = [(c, s) for c, s in items if not s.startswith(("rec:", "ctx:"))]
    merged: dict[str, int] = {}
    for prefix, d in word_trees.items():
        if not d.is_dir():
            continue
        labels = sorted(x.name for x in d.iterdir() if x.is_dir())
        for w, items in load_recordings(d, labels, prefix=prefix).items():
            if items:
                clips_ws.setdefault(w, []).extend(items)
                merged[w] = merged.get(w, 0) + len(items)
    neg_root = root / "approved" / "negatives"
    if neg_root.is_dir():
        wins = negative_windows(neg_root)
        if wins:
            clips_ws.setdefault("_unknown_", []).extend(wins)
            merged["_unknown_"] = len(wins)
    return merged


def negative_windows(root: Path) -> list[tuple[np.ndarray, str]]:
    """1 s windows at 1 s hops from every approved negative phrase -> `_unknown_` material,
    speaker id `rec:<spk>` so speaker-disjoint splitting treats them like other real clips.
    A file at the wrong sample rate, or shorter than one window, is skipped with a warning.
    Short files are NOT rare: QC's floor is 300 ms, so a negative take the recorder's VAD
    cut short (~0.85 s — 5 of the 10 approved negatives in the first real session) passes
    QC and is dropped here for want of a full 1 s window. Emitting a zero-padded window
    instead would teach the model silence, so the drop is deliberate; the fix is on the
    recording side."""
    import soundfile as sf

    out = []
    n = config.CLIP_SAMPLES
    for f in sorted(Path(root).glob("*/*.wav")):
        sig, sr = sf.read(f, dtype="float32", always_2d=True)
        sig = sig[:, 0]
        if sr != config.SAMPLE_RATE:
            logging.warning(
                "negative_windows: %s sample rate %d != %d, skipping", f, sr, config.SAMPLE_RATE
            )
            continue
        if len(sig) < n:
            logging.warning(
                "negative_windows: %s shorter than one window (%d < %d samples), skipping",
                f,
                len(sig),
                n,
            )
            continue
        for start in range(0, len(sig) - n + 1, n):
            out.append((sig[start : start + n].astype(np.float32), f"rec:{f.parent.name}"))
    return out


_ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

# macOS `say` German voices actually installed on this machine (checked via
# `say -v '?' | grep de_DE`), used to TTS-fill command words MSWC has too few
# real clips for. Varying voice+rate+punctuation gives combinatorial diversity;
# a synthesized clip's "speaker id" is its (voice, rate) pair so a speaker-disjoint
# split holds out entire unseen voice/rate combos, not just individual utterances.
_TTS_VOICES = ["Anna", "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"]
_TTS_RATES = [120, 140, 160, 180, 200, 220, 240, 260, 280]
_TTS_PHRASINGS = ["{w}", "{w}.", "{w}!", "{w}?"]


def split_by_speaker(clips_with_speakers: dict, rng, test_frac: float = 0.2, *, keep_speaker=False):
    """Split each label's (clip, speaker_id) list into train/test by speaker.

    No speaker appears in both splits -- across ALL labels, not just within one: the
    speaker->split assignment is drawn once over the union of every label's speaker ids,
    then applied to filter each label (a speaker recorded under two different labels, e.g.
    a device speaker's word clips and their `_unknown_` negatives sharing one `rec:<spk>`
    id, still lands in exactly one split). Returns (train_clips, test_clips). By default
    each is a ``dict[label] -> list[np.ndarray]`` (speaker id dropped) suitable for
    ``build_dataset``; with ``keep_speaker=True`` the speaker id is kept (``dict[label] ->
    list[(np.ndarray, speaker_id)]``), e.g. to later tell real MSWC clips from
    TTS-synthesized ones (speaker id prefixed ``"tts:"``).
    """
    all_speakers = sorted({spk for items in clips_with_speakers.values() for _, spk in items})
    order = rng.permutation(len(all_speakers))
    n_test = max(1, round(len(all_speakers) * test_frac)) if all_speakers else 0
    test_speakers = {all_speakers[i] for i in order[:n_test]}
    train_clips: dict = {}
    test_clips: dict = {}
    for label, items in clips_with_speakers.items():
        if keep_speaker:
            train_clips[label] = [(c, spk) for c, spk in items if spk not in test_speakers]
            test_clips[label] = [(c, spk) for c, spk in items if spk in test_speakers]
        else:
            train_clips[label] = [c for c, spk in items if spk not in test_speakers]
            test_clips[label] = [c for c, spk in items if spk in test_speakers]
    return train_clips, test_clips


def split_three_way(
    clips_with_speakers: dict,
    rng,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    *,
    keep_speaker=False,
):
    """Speaker-disjoint train/val/test split. No speaker appears in more than one
    split -- across ALL labels, not just within one: the speaker->split assignment is
    drawn once over the union of every label's speaker ids, then applied to filter each
    label (a speaker recorded under two different labels, e.g. a device speaker's word
    clips and their `_unknown_` negatives sharing one `rec:<spk>` id, still lands in
    exactly one split). Fractions are of the speaker set (val_frac, test_frac; the rest
    is train). Val exists alongside `split_by_speaker`'s train/test so model selection
    never touches test. See `split_by_speaker` for the two-way version and the
    `keep_speaker` semantics this mirrors."""
    all_speakers = sorted({spk for items in clips_with_speakers.values() for _, spk in items})
    order = rng.permutation(len(all_speakers))
    n = len(all_speakers)
    n_val = round(n * val_frac) if n else 0
    n_test = round(n * test_frac) if n else 0
    # guarantee non-empty val/test when there are enough speakers
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    val_s = {all_speakers[i] for i in order[:n_val]}
    test_s = {all_speakers[i] for i in order[n_val : n_val + n_test]}
    train_s = set(all_speakers) - val_s - test_s

    train, val, test = {}, {}, {}
    for label, items in clips_with_speakers.items():

        def pick(keep, items=items):
            return [(c, s) if keep_speaker else c for c, s in items if s in keep]

        val[label] = pick(val_s)
        test[label] = pick(test_s)
        train[label] = pick(train_s)
    return train, val, test


def _random_shift(clip, rng, max_shift_ms: int = 200):
    """Random time-shift by up to +/-max_shift_ms, zero-filled (not circular).
    Pads/truncates to config.CLIP_SAMPLES first so the shift is well-defined.
    Words must be recognisable at any window offset, not just clip-start."""
    sig = np.asarray(clip, dtype=np.float32).ravel()
    n = config.CLIP_SAMPLES
    if sig.shape[0] < n:
        sig = np.pad(sig, (0, n - sig.shape[0]))
    sig = sig[:n]
    max_shift = int(config.SAMPLE_RATE * max_shift_ms / 1000)
    if max_shift <= 0:
        return sig
    shift = int(rng.integers(-max_shift, max_shift + 1))
    out = np.zeros_like(sig)
    if shift > 0:
        out[shift:] = sig[: n - shift]
    elif shift < 0:
        out[: n + shift] = sig[-shift:]
    else:
        out[:] = sig
    return out


VAN_SNRS = (0, 5, 10)  # dB — cabin-ish, noisier than the general snrs= default


def _trimmed(clip):
    """Leading/trailing silence off (same call as `kws_de.recordings.load_recordings`)."""
    import librosa

    sig = np.asarray(clip, np.float32).ravel()
    out, _ = librosa.effects.trim(sig, top_db=30)
    return out if len(out) else sig


def _context_window(target, left, right, rng):
    """`[left] + gap + target + gap + [right]` (each gap 50–200 ms of silence, either
    neighbour optional but not both) -> the 1 s window centred on `target` and the 1 s
    window centred on the gap after it (before it when there is no right neighbour).
    Zero-padded where the sequence runs out. Clips are silence-trimmed already."""

    def gap():
        return np.zeros(
            int(rng.integers(config.SAMPLE_RATE // 20, config.SAMPLE_RATE // 5 + 1)), np.float32
        )

    head = [left, gap()] if left is not None else []
    tail = [gap(), right] if right is not None else []
    parts = head + [target] + tail
    t0 = sum(len(p) for p in head)
    centre = t0 + len(target) // 2
    gap_centre = t0 + len(target) + len(tail[0]) // 2 if tail else len(head[0]) + len(head[1]) // 2
    half = config.CLIP_SAMPLES // 2
    padded = np.pad(np.concatenate(parts), (half, half))
    return padded[centre : centre + config.CLIP_SAMPLES], padded[
        gap_centre : gap_centre + config.CLIP_SAMPLES
    ]


def van_augmentation_enabled() -> bool:
    """Van-cabin augmentation for REAL clips (rec:/MSWC) is opt-in: set both
    KWS_NOISE_DIR (recommended: `<mww-train>/data/fma_16k`, or `negative_datasets`)
    and KWS_RIR_DIR (recommended: `<mww-train>/data/mit_rirs`) to directories of 16kHz
    wav files. No default — this repo commits no path to that external training data."""
    return bool(os.environ.get("KWS_NOISE_DIR")) and bool(os.environ.get("KWS_RIR_DIR"))


_VAN_FILES: dict[tuple[str, str], tuple[list[Path], list[Path]]] = {}


def _van_noise_and_rir(rng):
    """One random noise clip (KWS_NOISE_DIR) and one random room IR (KWS_RIR_DIR), drawn
    fresh from `rng` on every call — `build_dataset` calls it once per REAL clip, so each
    clip gets its own cabin instead of one pair shared across the whole build. The two
    wav lists are globbed once per directory pair and kept in `_VAN_FILES`."""
    import soundfile as sf

    dirs = os.environ["KWS_NOISE_DIR"], os.environ["KWS_RIR_DIR"]
    if dirs not in _VAN_FILES:
        noises, rirs = (sorted(Path(d).glob("*.wav")) for d in dirs)
        if not noises or not rirs:
            raise FileNotFoundError(f"no .wav files under {dirs[0]} or {dirs[1]}")
        _VAN_FILES[dirs] = (noises, rirs)
    noises, rirs = _VAN_FILES[dirs]
    noise, _ = sf.read(noises[int(rng.integers(0, len(noises)))], dtype="float32")
    rir, _ = sf.read(rirs[int(rng.integers(0, len(rirs)))], dtype="float32")
    return noise, rir


def build_dataset(
    clips,
    noises,
    rng,
    snrs=(20, 10, 0),
    labels=None,
    commands=None,
    synthetic=None,
    shift_ms: int = 200,
    context_mix: int = 0,
    speakers=None,
):
    """Build (X, y) from raw clips. `labels`/`commands` default to the v1 vocab
    (`config.LABELS`/`config.COMMANDS`) so existing v1 callers are unaffected;
    pass `labels=config.COMMAND_LABELS, commands=command_words()` for v2.

    `synthetic` (optional `{label: [bool]}` aligned to `clips`) marks TTS clips; each one
    also contributes a pitch/tempo-perturbed copy (±2 semitones, 0.85–1.15× tempo, drawn
    from `rng`) through the same clean + per-snr pipeline, so rows per TTS clip double.
    A handful of synthetic voices need the acoustic variety real speakers bring for free;
    real clips are left alone so the real:TTS row ratio does not get worse.

    Every word class (commands AND `_unknown_`) sees the SAME audio domains: one
    clean (time-shifted by up to ±`shift_ms`, see `_random_shift`) copy plus a
    noise-mixed (time-shifted) copy at each snr
    in `snrs` -- per-clip count = 1 + len(snrs). This matters: if commands were
    noise-only and `_unknown_` clean-only, the model learns "clean audio implies
    _unknown_" instead of the actual words. `_silence_` stays noise-only (that IS
    its definition) plus a few pure-zero clean samples so clean input alone
    doesn't uniquely signal any one class.

    Van-cabin augmentation (`van_augmentation_enabled()`, opt-in via
    KWS_NOISE_DIR/KWS_RIR_DIR) adds `len(VAN_SNRS)` extra rows per REAL (non-`perturbed`)
    clip: a room IR drawn per clip convolved in, mixed with a noise clip drawn per clip
    at 0/5/10 dB — see `_van_noise_and_rir` and `kws_de.augment.van_augment`. Off by
    default, so `_origin_flags` must be told the same way (`van_augmentation_enabled()`)
    to keep its row count in sync.

    `context_mix=K` (`kws-dataset build --context-mix`) appends, AFTER all the rows above
    (so they stay byte-identical to a K=0 build), K multi-word rows per command clip:
    the silence-trimmed clip between 1–2 other trimmed clips of this split (same speaker
    when that speaker has others, any label — `speakers` is `{label: [speaker_id]}`
    aligned to `clips` like `synthetic`), 50–200 ms gaps, the 1 s window centred on the
    target (`_context_window`), labelled with the target word, shifted and mixed at ONE
    draw from {clean} ∪ `snrs` rather than the whole ladder (row count). The first
    sequence of each clip also yields the window centred on the gap next to the target,
    labelled `_unknown_`, so a word-boundary window has a class. Per command clip: K + 1
    extra rows, all flagged with the target clip's origin in `_origin_flags`.
    """
    labels = list(labels) if labels is not None else config.LABELS
    commands = list(commands) if commands is not None else config.COMMANDS
    van = van_augmentation_enabled()
    X, y = [], []

    def add(sig, label):
        X.append(mfcc(sig))
        y.append(labels.index(label))

    def add_word_clip(clip, label, perturbed=False):
        add(_random_shift(clip, rng, shift_ms), label)
        for snr in snrs:
            noise = noises[int(rng.integers(0, len(noises)))]
            add(mix_at_snr(_random_shift(clip, rng, shift_ms), noise, snr, rng), label)
        if perturbed:
            # A pitch/tempo-perturbed copy gets the SAME clean+per-snr treatment above,
            # not a recursive add_word_clip call — that would also fall into the van
            # branch below (a TTS-perturbed copy is not a REAL clip either).
            n_steps = float(rng.uniform(-2.0, 2.0))
            rate = float(rng.uniform(0.85, 1.15))
            pclip = perturb(clip, n_steps, rate, config.SAMPLE_RATE)
            add(_random_shift(pclip, rng, shift_ms), label)
            for snr in snrs:
                noise = noises[int(rng.integers(0, len(noises)))]
                add(mix_at_snr(_random_shift(pclip, rng, shift_ms), noise, snr, rng), label)
        elif van:  # REAL clip (rec:/MSWC): van-cabin variety, its own (noise, RIR) pair
            van_noise, van_rir = _van_noise_and_rir(rng)
            for snr in VAN_SNRS:
                shifted = _random_shift(clip, rng, shift_ms)
                add(van_augment(shifted, van_noise, van_rir, snr, rng), label)

    def flags_for(label):
        return (synthetic or {}).get(label) or []

    for cmd in commands:
        flags = flags_for(cmd)
        for i, clip in enumerate(clips.get(cmd, [])):
            add_word_clip(clip, cmd, perturbed=bool(flags[i]) if i < len(flags) else False)
    flags = flags_for("_unknown_")
    for i, clip in enumerate(clips.get("_unknown_", [])):
        add_word_clip(clip, "_unknown_", perturbed=bool(flags[i]) if i < len(flags) else False)
    n_sil = max(1, len(clips.get("_unknown_", [])))
    for _ in range(n_sil):
        noise = noises[int(rng.integers(0, len(noises)))]
        sil = mix_at_snr(np.zeros(config.CLIP_SAMPLES, np.float32), noise, 0.0, rng)
        add(sil, "_silence_")
    n_clean_sil = max(1, n_sil // 10)
    for _ in range(n_clean_sil):
        add(np.zeros(config.CLIP_SAMPLES, np.float32), "_silence_")
    if context_mix:
        pool = [
            (lbl, i) for lbl in [*commands, "_unknown_"] for i in range(len(clips.get(lbl, [])))
        ]
        trimmed = {key: _trimmed(clips[key[0]][key[1]]) for key in pool}

        def spk(key):
            ids = (speakers or {}).get(key[0]) or []
            return ids[key[1]] if key[1] < len(ids) else None

        by_spk: dict = {}
        for key in pool:
            by_spk.setdefault(spk(key), []).append(key)

        def add_context(win, label):
            snr = ((None,) + tuple(snrs))[int(rng.integers(0, len(snrs) + 1))]
            sig = _random_shift(win, rng, shift_ms)
            if snr is not None:
                sig = mix_at_snr(sig, noises[int(rng.integers(0, len(noises)))], snr, rng)
            add(sig, label)

        for cmd in commands:
            for i in range(len(clips.get(cmd, []))):
                me = (cmd, i)
                same = [k for k in by_spk[spk(me)] if k != me]
                others = same if len(same) >= 2 else [k for k in pool if k != me]
                for j in range(context_mix):
                    n = min(int(rng.integers(1, 3)), len(others))
                    pick = [
                        trimmed[others[int(x)]] for x in rng.choice(len(others), n, replace=False)
                    ]
                    if n == 1:
                        pick = [pick[0], None] if rng.random() < 0.5 else [None, pick[0]]
                    win, gap_win = _context_window(trimmed[me], pick[0], pick[1], rng)
                    add_context(win, cmd)
                    if j == 0:
                        add_context(gap_win, "_unknown_")
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def command_words() -> list[str]:
    """Slot-command words that need clips (devices + zones + actions)."""
    return config.DEVICES + config.ZONES + config.ACTIONS


def main() -> None:  # pragma: no cover - thin I/O wrapper (manual/integration)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="stream MSWC-de + ESC-50, cache raw clips")
    ap.add_argument(
        "--safety-cap",
        type=int,
        default=300_000,
        help="max MSWC examples to scan (stream is alphabetical by keyword; German "
        "'a'/'b' words alone can exceed 300k, so a real run may need to raise this)",
    )
    ap.add_argument(
        "--v2", action="store_true", help="use the v2 slot-command vocab instead of v1 COMMANDS"
    )
    ap.add_argument(
        "--v3",
        action="store_true",
        help="v2 vocab, real speech mined from an extracted MSWC-de tarball "
        "(--mswc-root) plus data/recordings/, TTS only as backstop",
    )
    ap.add_argument(
        "--mswc-root",
        default=str(config.DATA_DIR / "mswc" / "de"),
        help="extracted MSWC-de tarball root (contains clips/ and de_splits.csv)",
    )
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    v2 = args.v2 or args.v3
    words = command_words() if v2 else None
    cache_name = "raw_clips_v3.pkl" if args.v3 else ("raw_clips_v2.pkl" if v2 else "raw_clips.pkl")
    if args.fetch:
        _fetch_and_cache(
            safety_cap=args.safety_cap,
            words=words,
            cache_name=cache_name,
            mswc_root=Path(args.mswc_root) if args.v3 else None,
            n_unknown=2000 if args.v3 else 600,
        )


def _fetch_and_cache(
    n_per_word=300,
    n_unknown=600,
    safety_cap=300_000,
    words=None,
    cache_name="raw_clips.pkl",
    mswc_root: Path | None = None,
) -> None:  # pragma: no cover
    """Stream MSWC-de (MLCommons/ml_spoken_words, config 'de_wav') + download ESC-50
    noise, caching raw clips under config.DATA_DIR so re-runs don't re-download.
    `words` defaults to `config.COMMANDS` (v1); pass `command_words()` for v2 (cache
    under a different `cache_name` so the v1 cache is untouched). With `mswc_root`,
    mine the extracted tarball (`kws_de.mswc.mine`) and merge `data/recordings/`
    instead of streaming; `_unknown_` gets `n_unknown` clips either way.
    """
    words = list(words) if words is not None else config.COMMANDS
    clips_path = config.DATA_DIR / cache_name
    if clips_path.exists():
        print(f"[mswc] cache hit: {clips_path}")
    else:
        if mswc_root is not None:
            from kws_de.mswc import mine
            from kws_de.recordings import load_recordings

            clips = mine(mswc_root, words, n_per_word=n_per_word, n_unknown=n_unknown)
            # legacy hand-dropped layout only; the QC-approved tree is merged on every
            # build by `merge_recordings`, never baked into the cache.
            if not (config.DATA_DIR / "recordings" / "approved" / "words").is_dir():
                for w, items in load_recordings(recordings_root(), words).items():
                    clips[w].extend(items)
            scanned = "mswc-tarball"
        else:
            clips, scanned = _fetch_mswc(words, n_per_word, n_unknown, safety_cap)
        counts = {c: len(clips[c]) for c in words}
        print(f"[mswc] done: scanned={scanned} counts={counts} unknown={len(clips['_unknown_'])}")
        with open(clips_path, "wb") as fh:
            pickle.dump({"clips": clips, "scanned": scanned}, fh)

    noise_path = config.DATA_DIR / "noise.pkl"
    if noise_path.exists():
        print(f"[noise] cache hit: {noise_path}")
    else:
        audio_dir = _download_esc50(config.DATA_DIR)
        noises = _load_noises(audio_dir)
        print(f"[noise] loaded {len(noises)} ESC-50 clips")
        with open(noise_path, "wb") as fh:
            pickle.dump(noises, fh)


def _fetch_mswc(  # pragma: no cover - network I/O (manual/integration)
    words: list[str],
    n_per_word: int,
    n_unknown: int,
    safety_cap: int,
    unknown_per_word_cap: int = 5,
):
    """Stream MSWC-de, early-stopping each target word at n_per_word valid clips
    and collecting a diverse ~n_unknown pool of other words for `_unknown_`.
    Returns (clips: dict[label] -> list[(np.ndarray, speaker_id)], scanned: int).
    """
    from datasets import load_dataset

    target_words = {w.lower(): w for w in words}
    clips: dict = {w: [] for w in words}
    clips["_unknown_"] = []
    unknown_word_counts: dict = {}

    ds = load_dataset(
        "MLCommons/ml_spoken_words",
        "de_wav",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    scanned = 0
    for ex in ds:
        scanned += 1
        if ex.get("is_valid"):
            kw = (ex["keyword"] or "").lower()
            if kw in target_words:
                cmd = target_words[kw]
                if len(clips[cmd]) < n_per_word:
                    audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
                    clips[cmd].append((audio, ex["speaker_id"]))
            elif len(clips["_unknown_"]) < n_unknown:
                seen = unknown_word_counts.get(kw, 0)
                if seen < unknown_per_word_cap:
                    audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
                    clips["_unknown_"].append((audio, ex["speaker_id"]))
                    unknown_word_counts[kw] = seen + 1

        if scanned % 20_000 == 0:
            counts = {c: len(clips[c]) for c in words}
            print(
                f"[mswc] scanned={scanned} unknown={len(clips['_unknown_'])} "
                f"last_keyword={ex['keyword']!r} counts={counts}",
                flush=True,
            )

        done_targets = all(len(clips[c]) >= n_per_word for c in words)
        done_unknown = len(clips["_unknown_"]) >= n_unknown
        if (done_targets and done_unknown) or scanned >= safety_cap:
            break
    return clips, scanned


def _download_esc50(dest_dir: Path) -> Path:  # pragma: no cover - network I/O
    audio_dir = dest_dir / "ESC-50-master" / "audio"
    if audio_dir.exists() and any(audio_dir.iterdir()):
        return audio_dir
    print(f"[noise] downloading ESC-50 from {_ESC50_URL} ...")
    with urlopen(_ESC50_URL, timeout=300) as resp:  # noqa: S310 - fixed, known-good URL
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    return audio_dir


def _load_noises(audio_dir: Path) -> list:  # pragma: no cover - file I/O (manual/integration)
    import librosa
    import soundfile as sf

    noises = []
    for f in sorted(audio_dir.glob("*.wav")):
        y, sr = sf.read(str(f))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr != config.SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=config.SAMPLE_RATE, res_type="fft")
        noises.append(y.astype(np.float32))
    return noises


def _say_one(word: str, voice: str, rate: int, phrasing: str, wav_path: Path):  # pragma: no cover
    import soundfile as sf

    # `say` can deadlock under heavy parallelism — bound each call and skip on failure
    # (returns None) rather than hanging the whole thread pool.
    try:
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(rate),
                "--data-format=LEI16@16000",
                "-o",
                str(wav_path),
                phrasing.format(w=word),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        wav_path.unlink(missing_ok=True)
        return None
    y, _sr = sf.read(str(wav_path))
    # The wav stays on disk: `kws_de.tts.synthesize` manifests it and the TTS gate has to
    # be able to transcribe it. Whoever asked for the synthesis deletes it.
    return y.astype(np.float32), f"tts:{voice}:{rate}"


def tts_engines() -> list[str]:
    """Which TTS engines `_tts_fill_word` draws from. Multi-engine (voice diversity across
    say+piper+...) by default; set env var ``KWS_TTS_ENGINES=say`` to reproduce the
    say-only baseline (docs/eval-report-v2.md). Pure aside from the env read."""
    override = os.environ.get("KWS_TTS_ENGINES")
    if override:
        return [e.strip() for e in override.split(",") if e.strip()]
    return tts.available_engines()


def _tts_combo_plan(
    word: str, n: int, engines: list[str], voices_by_engine: dict[str, list[str]] | None = None
) -> list[tuple[str, str, int]]:
    """Pure selection logic: `n` (engine, voice, rate) combos to synthesize `word` with,
    drawn round-robin across `engines` (kws_de.tts.voice_combos) so multiple engines
    contribute EQUALLY — plain round-robin alone stops being balanced once the smallest
    engine's voice/rate pool runs out and the larger one keeps filling the rest alone.
    Once the balanced pool itself is exhausted (n exceeds every engine's distinct
    voice/rate combos), CYCLES back through it to reach n — engines whose backend is
    stochastic per call (e.g. Piper's noise_scale) still produce fresh distinct audio on
    a repeat; deterministic ones (macOS `say`) produce an exact repeat, no worse than
    the redundancy any oversampling scheme adds. No backends touched.

    `voices_by_engine`, when given, overrides `kws_de.tts.engine_voices` per engine —
    the voice-gated build passes only gate-passing voices; an engine with none left is
    dropped rather than skewing the round-robin with an empty pool."""

    def voices(e: str) -> list[str]:
        return voices_by_engine[e] if voices_by_engine is not None else tts.engine_voices(e)

    engines = [e for e in engines if voices(e)]
    if not engines:
        return []
    pool = min(len(voices(e)) * len(tts.RATES) for e in engines)
    base = tts.voice_combos(len(engines) * pool, engines, voices_by_engine)
    if not base:
        return []
    reps = -(-max(n, 0) // len(base))  # ceil division
    return (base * reps)[: max(n, 0)]


def tts_gate_transcriber():  # pragma: no cover - loads Whisper
    """The transcriber the synthetic-clip gate needs: Whisper with language DETECTION on
    (not forced to German), since catching a clip that came out English is the point.
    Returns None — gate disabled, every clip kept — when ``KWS_TTS_GATE=0`` or when
    mlx-whisper is not installed (Linux CI has no MLX; it also builds no TTS clips)."""
    if os.environ.get("KWS_TTS_GATE") == "0":
        return None
    try:
        from kws_de.qc import whisper_transcriber

        return whisper_transcriber(language=None)
    except Exception as e:  # noqa: BLE001 - missing/unloadable model must not fail a build
        print(f"[tts] gate disabled — no Whisper ({type(e).__name__}: {e})")
        return None


VOICE_GATE_CACHE_NAME = "tts_voice_gate.json"


def voice_gate_cache_path() -> Path:
    return config.DATA_DIR / VOICE_GATE_CACHE_NAME


def passing_voices(engine: str, transcriber, voices: list[str] | None = None) -> list[str]:
    """Voices for `engine` that pass `kws_de.qc.voice_gate`, cached by `f"{engine}:{voice}"`
    in `voice_gate_cache_path()` (default `$KWS_DATA_ROOT/data/tts_voice_gate.json`) so a
    voice is gated once, ever — its audio doesn't change, so a cached verdict never
    expires. `voices` overrides `kws_de.tts.engine_voices(engine)` (mainly for tests);
    a multi-speaker Piper voice's speakers (`de_DE-mls-medium#N`) are already separate
    entries there, so each is gated and cached independently."""
    import json
    from datetime import date

    from kws_de.qc import voice_gate

    path = voice_gate_cache_path()
    try:
        cache = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    dirty = False
    out = []
    for v in voices if voices is not None else tts.engine_voices(engine):
        key = f"{engine}:{v}"
        entry = cache.get(key)
        if entry is None:
            entry = voice_gate(engine, v, transcriber)
            entry["date"] = date.today().isoformat()
            cache[key] = entry
            dirty = True
        if entry.get("ok"):
            out.append(v)
    if dirty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False))
    return out


def _tts_fill_word(
    word: str,
    n: int,
    tmp_dir: Path,
    max_workers: int = 4,
    voices_by_engine: dict[str, list[str]] | None = None,
) -> list:
    # pragma: no cover - shells out / loads models
    """Synthesize up to n clips of `word` across all engines from `tts_engines()`
    (parallelized — each synthesis call is independent, so this is I/O/compute-bound and
    speeds up with a thread pool), varied by engine/voice/rate. Returns
    [(np.ndarray, speaker_id)] with speaker_id="tts:{engine}:{voice}" — rate is augmentation,
    not identity, so the speaker-disjoint split holds out whole voices.

    `voices_by_engine` (from `passing_voices`, one Whisper pass per VOICE, not per clip)
    restricts synthesis to voices that already passed `kws_de.qc.voice_gate`; every clip
    then only needs `kws_de.qc.tts_cheap_gate` (duration, not silent — no model) before
    being kept, since the voice-level gate already answered "is this German and does
    this voice say what it's told". `voices_by_engine=None` synthesizes from every known
    voice, ungated — the caller decides."""
    from concurrent.futures import ThreadPoolExecutor

    from kws_de.qc import tts_cheap_gate

    combos = _tts_combo_plan(word, n, tts_engines(), voices_by_engine)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _job(args):
        i, (engine, voice, rate) = args
        wav = tmp_dir / f"{word}_{i}.wav"
        audio = tts.synthesize(word, engine, voice, rate, wav)
        wav.unlink(missing_ok=True)
        return None if audio is None else (audio, f"tts:{engine}:{voice}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = [r for r in ex.map(_job, enumerate(combos)) if r is not None]
    kept, dropped = [], {}
    for audio, speaker in results:
        ok, reason = tts_cheap_gate(audio, config.SAMPLE_RATE)
        if ok:
            kept.append((audio, speaker))
        else:
            dropped[f"{speaker} {reason}"] = dropped.get(f"{speaker} {reason}", 0) + 1
    if dropped:
        print(f"[tts] {word}: gate dropped {sum(dropped.values())}/{len(results)}: {dropped}")
    return kept


def tts_text_for(word: str) -> str:
    """The text a TTS engine should actually SAY to produce a clip labelled
    `word`. For an ordinary command label the label IS the spoken word ("Licht").
    A config.SCENE_TRIGGERS token, though, is a space-free join ("GuteNacht") that
    no engine pronounces correctly — its natural spelling lives in
    config.SCENE_TRIGGER_PROMPTS ("Gute Nacht"), which is what gets synthesized.
    This is the wiring the future retrain relies on: once a scene trigger is
    promoted into COMMAND_LABELS, `_fill_with_tts` bootstraps it from the natural
    spelling, not the token, with no further change (see docs/paper-notes E55)."""
    return config.SCENE_TRIGGER_PROMPTS.get(word, word)


def _fill_with_tts(clips: dict, target: int = 300, words=None) -> dict:  # pragma: no cover
    """Top up any word (from `words`, default `config.COMMANDS`) under `target`
    real clips with TTS clips from gate-passing voices only. Returns
    {word: n_tts_kept}. A config.SCENE_TRIGGERS token is synthesized from its
    natural spelling (`tts_text_for`), not the space-free token."""
    import shutil

    words = list(words) if words is not None else config.COMMANDS
    tmp_dir = config.DATA_DIR / "tts_tmp"
    transcriber = tts_gate_transcriber()
    # One voice-gate pass per (engine, voice), cached, up front — not per word: the
    # voice's German-ness doesn't depend on which command it is about to say.
    voices_by_engine = (
        {e: passing_voices(e, transcriber) for e in tts_engines()} if transcriber else None
    )
    added = {}
    for cmd in words:
        have = len(clips.get(cmd, []))
        if have >= target:
            continue
        need = target - have
        print(f"[tts] {cmd}: {have} real clips, synthesizing {need} more")
        new = _tts_fill_word(
            tts_text_for(cmd).lower(), need, tmp_dir, voices_by_engine=voices_by_engine
        )
        clips.setdefault(cmd, []).extend(new)
        added[cmd] = len(new)
    shutil.rmtree(tmp_dir, ignore_errors=True)  # clips, and the manifest beside them
    return added


def _origin_flags(
    clips_ws: dict, snrs, words=None, perturb_tts: bool = False, context_mix: int = 0
) -> np.ndarray:
    """Boolean array flagging TTS-synthesized origin (speaker id prefix "tts:"),
    aligned row-for-row to build_dataset's output for the same clips/snrs — must
    mirror build_dataset's iteration order (commands then unknown, each clip's
    clean copy + one row per snr, then silence, then clean silence) exactly.
    With `perturb_tts`, TTS clips count twice (build_dataset's perturbed copy). A REAL
    clip also picks up `len(VAN_SNRS)` extra (False) rows when van augmentation is
    enabled (`van_augmentation_enabled()`) — must track build_dataset's own check.
    `context_mix=K` appends build_dataset's K + 1 context rows per command clip (after
    the silence rows), flagged with that clip's origin."""
    words = list(words) if words is not None else config.COMMANDS
    per_clip = 1 + len(snrs)
    van_rows = len(VAN_SNRS) if van_augmentation_enabled() else 0
    flags = []

    def rows(spk):
        is_tts = spk.startswith("tts:")
        base = [is_tts] * (per_clip * (2 if perturb_tts and is_tts else 1))
        return base if is_tts else base + [False] * van_rows

    for cmd in words:
        for _clip, spk in clips_ws.get(cmd, []):
            flags.extend(rows(spk))
    for _clip, spk in clips_ws.get("_unknown_", []):
        flags.extend(rows(spk))
    n_sil = max(1, len(clips_ws.get("_unknown_", [])))
    flags.extend([False] * n_sil)
    n_clean_sil = max(1, n_sil // 10)
    flags.extend([False] * n_clean_sil)
    if context_mix:
        for cmd in words:
            for _clip, spk in clips_ws.get(cmd, []):
                flags.extend([spk.startswith("tts:")] * (context_mix + 1))
    return np.asarray(flags, dtype=bool)
