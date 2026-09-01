# NOTE: pickle is used only for this script's own local, gitignored data/ cache
# (raw_clips.pkl / noise.pkl) — written and read by this same code, never untrusted
# input — so the arbitrary-code-execution risk on unpickling doesn't apply here.
import argparse
import io
import os
import pickle
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from kws_de import config, tts
from kws_de.augment import mix_at_snr
from kws_de.features import mfcc

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

    No speaker appears in both splits. Returns (train_clips, test_clips). By
    default each is a ``dict[label] -> list[np.ndarray]`` (speaker id dropped)
    suitable for ``build_dataset``; with ``keep_speaker=True`` the speaker id is
    kept (``dict[label] -> list[(np.ndarray, speaker_id)]``), e.g. to later tell
    real MSWC clips from TTS-synthesized ones (speaker id prefixed ``"tts:"``).
    """
    train_clips: dict = {}
    test_clips: dict = {}
    for label, items in clips_with_speakers.items():
        speakers = sorted({spk for _, spk in items})
        order = rng.permutation(len(speakers))
        n_test = max(1, round(len(speakers) * test_frac)) if speakers else 0
        test_speakers = {speakers[i] for i in order[:n_test]}
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
    split. Fractions are of the speaker set (val_frac, test_frac; the rest is
    train). Val exists alongside `split_by_speaker`'s train/test so model
    selection never touches test. See `split_by_speaker` for the two-way version
    and the `keep_speaker` semantics this mirrors."""
    train, val, test = {}, {}, {}
    for label, items in clips_with_speakers.items():
        speakers = sorted({spk for _, spk in items})
        order = rng.permutation(len(speakers))
        n = len(speakers)
        n_val = round(n * val_frac) if n else 0
        n_test = round(n * test_frac) if n else 0
        # guarantee non-empty val/test when there are enough speakers
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        val_s = {speakers[i] for i in order[:n_val]}
        test_s = {speakers[i] for i in order[n_val : n_val + n_test]}

        def pick(keep, items=items):
            return [(c, s) if keep_speaker else c for c, s in items if s in keep]

        val[label] = pick(val_s)
        test[label] = pick(test_s)
        train[label] = pick(set(speakers) - val_s - test_s)
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


def make_transition_windows(clips_by_word, rng, n_pairs, gap_ms=250):
    """Build transition-aware training windows from TRAIN-split word clips only
    (never test -- see `_build_and_split`, which is the only caller and enforces
    this by only ever passing it the train-split clips dict).

    Concatenates two random command words with `gap_ms` of silence between them
    -- the SAME gap `eval._intent_audio` uses to build catalog phrases -- then
    cuts CLIP_SAMPLES windows two ways:
      - straddling the boundary (tail-of-A + gap + head-of-B, neither word's
        center inside the window) -> these are the boundary-transition ghosts
        the streaming decoder actually sees; labeled "_unknown_" so the model
        learns "not any single word" instead of guessing one.
      - centered on one word with the neighbor's audio bleeding in at an edge
        -> "in-context positive", labeled with the centered word -- matches
        what streaming actually feeds the model (a word rarely arrives alone
        in its 1s window).

    Returns (unknown_windows, context_positives):
      unknown_windows: list[np.ndarray] of CLIP_SAMPLES float32 arrays.
      context_positives: list[(np.ndarray, str)] of (window, word_label).
    """
    words = [w for w, clips in clips_by_word.items() if clips]
    if not words:
        return [], []
    n = config.CLIP_SAMPLES
    gap = np.zeros(int(config.SAMPLE_RATE * gap_ms / 1000), np.float32)

    def cut(seq, center):
        """CLIP_SAMPLES window of `seq` centered at sample `center`, zero-padded
        past either end -- works whether `seq` is longer or shorter than
        CLIP_SAMPLES, so short clips don't need special-casing."""
        start = int(round(center)) - n // 2
        out = np.zeros(n, np.float32)
        src_start, src_end = max(start, 0), min(start + n, len(seq))
        if src_end > src_start:
            dst = src_start - start
            out[dst : dst + (src_end - src_start)] = seq[src_start:src_end]
        return out

    unknown_windows = []
    context_positives = []
    for _ in range(n_pairs):
        # distinct words when possible -- an unrelated-word ghost is the
        # failure mode this targets; only fall back to a repeat if there's
        # just one command word to draw from.
        wa, wb = rng.choice(words, size=2, replace=len(words) < 2)
        clip_a = clips_by_word[wa][int(rng.integers(0, len(clips_by_word[wa])))]
        clip_b = clips_by_word[wb][int(rng.integers(0, len(clips_by_word[wb])))]
        a = np.asarray(clip_a, np.float32).ravel()
        b = np.asarray(clip_b, np.float32).ravel()
        seq = np.concatenate([a, gap, b])
        len_a, len_gap = len(a), len(gap)

        jitter = int(rng.integers(-len_gap // 2, len_gap // 2 + 1)) if len_gap else 0
        unknown_windows.append(cut(seq, len_a + len_gap / 2 + jitter))
        context_positives.append((cut(seq, len_a / 2), wa))
        context_positives.append((cut(seq, len_a + len_gap + len(b) / 2), wb))

    return unknown_windows, context_positives


def build_dataset(
    clips,
    noises,
    rng,
    snrs=(20, 10, 0),
    labels=None,
    commands=None,
    transition_unknown=None,
    transition_positives=None,
):
    """Build (X, y) from raw clips. `labels`/`commands` default to the v1 vocab
    (`config.LABELS`/`config.COMMANDS`) so existing v1 callers are unaffected;
    pass `labels=config.COMMAND_LABELS, commands=command_words()` for v2.

    Every word class (commands AND `_unknown_`) sees the SAME audio domains: one
    clean (time-shifted) copy plus a noise-mixed (time-shifted) copy at each snr
    in `snrs` -- per-clip count = 1 + len(snrs). This matters: if commands were
    noise-only and `_unknown_` clean-only, the model learns "clean audio implies
    _unknown_" instead of the actual words. `_silence_` stays noise-only (that IS
    its definition) plus a few pure-zero clean samples so clean input alone
    doesn't uniquely signal any one class.

    `transition_unknown`/`transition_positives` (from `make_transition_windows`,
    train-split only) are already-cut CLIP_SAMPLES windows with deliberate
    boundary geometry, so they skip `_random_shift` (which would destroy that
    geometry) but still get the same clean+per-snr noise augmentation.
    """
    labels = list(labels) if labels is not None else config.LABELS
    commands = list(commands) if commands is not None else config.COMMANDS
    X, y = [], []

    def add(sig, label):
        X.append(mfcc(sig))
        y.append(labels.index(label))

    def add_word_clip(clip, label):
        add(_random_shift(clip, rng), label)
        for snr in snrs:
            noise = noises[int(rng.integers(0, len(noises)))]
            add(mix_at_snr(_random_shift(clip, rng), noise, snr, rng), label)

    def add_fixed_window(sig, label):
        add(sig, label)
        for snr in snrs:
            noise = noises[int(rng.integers(0, len(noises)))]
            add(mix_at_snr(sig, noise, snr, rng), label)

    for cmd in commands:
        for clip in clips.get(cmd, []):
            add_word_clip(clip, cmd)
    for clip in clips.get("_unknown_", []):
        add_word_clip(clip, "_unknown_")
    for win in transition_unknown or []:
        add_fixed_window(win, "_unknown_")
    for win, label in transition_positives or []:
        add_fixed_window(win, label)
    n_sil = max(1, len(clips.get("_unknown_", [])))
    for _ in range(n_sil):
        noise = noises[int(rng.integers(0, len(noises)))]
        sil = mix_at_snr(np.zeros(config.CLIP_SAMPLES, np.float32), noise, 0.0, rng)
        add(sil, "_silence_")
    n_clean_sil = max(1, n_sil // 10)
    for _ in range(n_clean_sil):
        add(np.zeros(config.CLIP_SAMPLES, np.float32), "_silence_")
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
    ap.add_argument("--build", action="store_true", help="speaker-split + augment cached clips")
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
    labels = config.COMMAND_LABELS if v2 else None
    out_prefix = "features_v3" if args.v3 else ("features_v2" if v2 else "features")
    if args.fetch:
        _fetch_and_cache(
            safety_cap=args.safety_cap,
            words=words,
            cache_name=cache_name,
            mswc_root=Path(args.mswc_root) if args.v3 else None,
            n_unknown=2000 if args.v3 else 600,
        )
    if args.build:
        _build_and_split(cache_name=cache_name, words=words, labels=labels, out_prefix=out_prefix)


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
            for w, items in load_recordings(config.DATA_DIR / "recordings", words).items():
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
    wav_path.unlink()
    return y.astype(np.float32), f"tts:{voice}:{rate}"


def tts_engines() -> list[str]:
    """Which TTS engines `_tts_fill_word` draws from. Multi-engine (voice diversity across
    say+piper+...) by default; set env var ``KWS_TTS_ENGINES=say`` to reproduce the
    say-only baseline (docs/eval-report-v2.md). Pure aside from the env read."""
    override = os.environ.get("KWS_TTS_ENGINES")
    if override:
        return [e.strip() for e in override.split(",") if e.strip()]
    return tts.available_engines()


def _tts_combo_plan(word: str, n: int, engines: list[str]) -> list[tuple[str, str, int]]:
    """Pure selection logic: `n` (engine, voice, rate) combos to synthesize `word` with,
    drawn round-robin across `engines` (kws_de.tts.voice_combos) so multiple engines
    contribute EQUALLY — plain round-robin alone stops being balanced once the smallest
    engine's voice/rate pool runs out and the larger one keeps filling the rest alone.
    Once the balanced pool itself is exhausted (n exceeds every engine's distinct
    voice/rate combos), CYCLES back through it to reach n — engines whose backend is
    stochastic per call (e.g. Piper's noise_scale) still produce fresh distinct audio on
    a repeat; deterministic ones (macOS `say`) produce an exact repeat, no worse than
    the redundancy any oversampling scheme adds. No backends touched."""
    if not engines:
        return []
    pool = min(len(tts.ENGINE_VOICES.get(e) or ["default"]) * len(tts.RATES) for e in engines)
    base = tts.voice_combos(len(engines) * pool, engines)
    if not base:
        return []
    reps = -(-max(n, 0) // len(base))  # ceil division
    return (base * reps)[: max(n, 0)]


def _tts_fill_word(word: str, n: int, tmp_dir: Path, max_workers: int = 4) -> list:
    # pragma: no cover - shells out / loads models
    """Synthesize up to n clips of `word` across all engines from `tts_engines()`
    (parallelized — each synthesis call is independent, so this is I/O/compute-bound and
    speeds up with a thread pool), varied by engine/voice/rate. Returns
    [(np.ndarray, speaker_id)] with speaker_id="tts:{engine}:{voice}:{rate}" so the
    speaker-disjoint split holds out whole voice combos, per engine."""
    from concurrent.futures import ThreadPoolExecutor

    combos = _tts_combo_plan(word, n, tts_engines())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _job(args):
        i, (engine, voice, rate) = args
        audio = tts.synthesize(word, engine, voice, rate, tmp_dir / f"{word}_{i}.wav")
        return None if audio is None else (audio, f"tts:{engine}:{voice}:{rate}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_job, enumerate(combos)))
    return [r for r in results if r is not None]


def _fill_with_tts(clips: dict, target: int = 300, words=None) -> dict:  # pragma: no cover
    """Top up any word (from `words`, default `config.COMMANDS`) under `target`
    real clips with macOS-`say` TTS clips. Returns {word: n_tts_added}."""
    words = list(words) if words is not None else config.COMMANDS
    tmp_dir = config.DATA_DIR / "tts_tmp"
    added = {}
    for cmd in words:
        have = len(clips.get(cmd, []))
        if have >= target:
            continue
        need = target - have
        print(f"[tts] {cmd}: {have} real clips, synthesizing {need} more")
        clips.setdefault(cmd, []).extend(_tts_fill_word(cmd.lower(), need, tmp_dir))
        added[cmd] = need
    if tmp_dir.exists():
        tmp_dir.rmdir()
    return added


def _origin_flags(clips_ws: dict, snrs, words=None) -> np.ndarray:
    """Boolean array flagging TTS-synthesized origin (speaker id prefix "tts:"),
    aligned row-for-row to build_dataset's output for the same clips/snrs — must
    mirror build_dataset's iteration order (commands then unknown, each clip's
    clean copy + one row per snr, then silence, then clean silence) exactly."""
    words = list(words) if words is not None else config.COMMANDS
    per_clip = 1 + len(snrs)
    flags = []
    for cmd in words:
        for _clip, spk in clips_ws.get(cmd, []):
            flags.extend([spk.startswith("tts:")] * per_clip)
    for _clip, spk in clips_ws.get("_unknown_", []):
        flags.extend([spk.startswith("tts:")] * per_clip)
    n_sil = max(1, len(clips_ws.get("_unknown_", [])))
    flags.extend([False] * n_sil)
    n_clean_sil = max(1, n_sil // 10)
    flags.extend([False] * n_clean_sil)
    return np.asarray(flags, dtype=bool)


def _build_and_split(
    test_frac: float = 0.2,
    seed: int = 0,
    cache_name: str = "raw_clips.pkl",
    words=None,
    labels=None,
    out_prefix: str = "features",
) -> None:  # pragma: no cover
    """TTS-fill thin words, speaker-disjoint split the cached raw clips, then
    augment + extract features, saving data/{out_prefix}_train.npz and
    data/{out_prefix}_test.npz (the latter also carries an ``is_tts`` row flag
    so eval can isolate real-speech-only accuracy). `words`/`labels` default to
    the v1 vocab; pass `command_words()`/`config.COMMAND_LABELS` for v2.
    """
    words = list(words) if words is not None else config.COMMANDS
    rng = np.random.default_rng(seed)
    with open(config.DATA_DIR / cache_name, "rb") as fh:
        cached = pickle.load(fh)
    with open(config.DATA_DIR / "noise.pkl", "rb") as fh:
        noises = pickle.load(fh)

    clips_with_speakers = cached["clips"]
    tts_added = _fill_with_tts(clips_with_speakers, words=words)
    if tts_added:
        with open(config.DATA_DIR / cache_name, "wb") as fh:
            pickle.dump(cached, fh)
        print(f"[tts] added: {tts_added}")

    snrs = (20, 10, 0)
    train_ws, test_ws = split_by_speaker(clips_with_speakers, rng, test_frac, keep_speaker=True)
    train_clips = {k: [c for c, _ in v] for k, v in train_ws.items()}
    test_clips = {k: [c for c, _ in v] for k, v in test_ws.items()}

    # Transition-aware training data (see make_transition_windows docstring):
    # built from TRAIN-split command-word clips ONLY, never test, to avoid
    # leakage. ~2000 boundary-straddling "_unknown_" negatives + ~4000
    # word-in-context positives (2 per pair) -- teaches the model what
    # inter-word transition audio looks like, which isolated-word clips never
    # show it.
    command_train_clips = {w: train_clips[w] for w in words if train_clips.get(w)}
    trans_unknown, trans_positive = make_transition_windows(command_train_clips, rng, n_pairs=600)

    X_train, y_train = build_dataset(
        train_clips,
        noises,
        rng,
        snrs=snrs,
        labels=labels,
        commands=words,
        transition_unknown=trans_unknown,
        transition_positives=trans_positive,
    )
    X_test, y_test = build_dataset(
        test_clips, noises, rng, snrs=snrs, labels=labels, commands=words
    )
    is_tts_test = _origin_flags(test_ws, snrs, words=words)
    np.savez(config.DATA_DIR / f"{out_prefix}_train.npz", X=X_train, y=y_train)
    np.savez(config.DATA_DIR / f"{out_prefix}_test.npz", X=X_test, y=y_test, is_tts=is_tts_test)
    print(f"[build] train X={X_train.shape} test X={X_test.shape}")
