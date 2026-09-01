# NOTE: pickle is used only for this script's own local, gitignored data/ cache
# (raw_clips.pkl / noise.pkl) — written and read by this same code, never untrusted
# input — so the arbitrary-code-execution risk on unpickling doesn't apply here.
import argparse
import io
import itertools
import pickle
import random
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from kws_de import config
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


def build_dataset(clips, noises, rng, snrs=(20, 10, 0), labels=None, commands=None):
    """Build (X, y) from raw clips. `labels`/`commands` default to the v1 vocab
    (`config.LABELS`/`config.COMMANDS`) so existing v1 callers are unaffected;
    pass `labels=config.COMMAND_LABELS, commands=command_words()` for v2."""
    labels = list(labels) if labels is not None else config.LABELS
    commands = list(commands) if commands is not None else config.COMMANDS
    X, y = [], []

    def add(sig, label):
        X.append(mfcc(sig))
        y.append(labels.index(label))

    for cmd in commands:
        for clip in clips.get(cmd, []):
            for snr in snrs:
                noise = noises[int(rng.integers(0, len(noises)))]
                add(mix_at_snr(clip, noise, snr, rng), cmd)
    for clip in clips.get("_unknown_", []):
        add(clip, "_unknown_")
    n_sil = max(1, len(clips.get("_unknown_", [])))
    for _ in range(n_sil):
        noise = noises[int(rng.integers(0, len(noises)))]
        sil = mix_at_snr(np.zeros(config.CLIP_SAMPLES, np.float32), noise, 0.0, rng)
        add(sil, "_silence_")
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
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        _fetch_and_cache(safety_cap=args.safety_cap)
    if args.build:
        _build_and_split()


def _fetch_and_cache(n_per_word=300, n_unknown=600, safety_cap=300_000) -> None:  # pragma: no cover
    """Stream MSWC-de (MLCommons/ml_spoken_words, config 'de_wav') + download ESC-50
    noise, caching raw clips under config.DATA_DIR so re-runs don't re-download.
    """
    clips_path = config.DATA_DIR / "raw_clips.pkl"
    if clips_path.exists():
        print(f"[mswc] cache hit: {clips_path}")
    else:
        clips, scanned = _fetch_mswc(n_per_word, n_unknown, safety_cap)
        counts = {c: len(clips[c]) for c in config.COMMANDS}
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
    n_per_word: int, n_unknown: int, safety_cap: int, unknown_per_word_cap: int = 5
):
    """Stream MSWC-de, early-stopping each command word at n_per_word valid clips
    and collecting a diverse ~n_unknown pool of other words for `_unknown_`.
    Returns (clips: dict[label] -> list[(np.ndarray, speaker_id)], scanned: int).
    """
    from datasets import load_dataset

    target_words = {c.lower(): c for c in config.COMMANDS}
    clips: dict = {c: [] for c in config.COMMANDS}
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
            counts = {c: len(clips[c]) for c in config.COMMANDS}
            print(
                f"[mswc] scanned={scanned} unknown={len(clips['_unknown_'])} "
                f"last_keyword={ex['keyword']!r} counts={counts}",
                flush=True,
            )

        done_targets = all(len(clips[c]) >= n_per_word for c in config.COMMANDS)
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


def _tts_fill_word(word: str, n: int, tmp_dir: Path) -> list:  # pragma: no cover - shells out
    """Synthesize up to n clips of `word` via macOS `say`, varied by voice/rate/
    punctuation. Returns [(np.ndarray, speaker_id)] with speaker_id="tts:{voice}:{rate}".
    """
    import soundfile as sf

    combos = list(itertools.product(_TTS_VOICES, _TTS_RATES, _TTS_PHRASINGS))
    random.Random(abs(hash(word)) % (2**32)).shuffle(combos)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (voice, rate, phrasing) in enumerate(combos):
        if len(out) >= n:
            break
        wav_path = tmp_dir / f"{word}_{i}.wav"
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
        )
        y, _sr = sf.read(str(wav_path))
        wav_path.unlink()
        out.append((y.astype(np.float32), f"tts:{voice}:{rate}"))
    return out


def _fill_with_tts(clips: dict, target: int = 300) -> dict:  # pragma: no cover - shells out
    """Top up any command word under `target` real clips with macOS-`say` TTS clips.
    Returns {word: n_tts_added} for words that needed filling."""
    tmp_dir = config.DATA_DIR / "tts_tmp"
    added = {}
    for cmd in config.COMMANDS:
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


def _origin_flags(clips_ws: dict, snrs) -> np.ndarray:
    """Boolean array flagging TTS-synthesized origin (speaker id prefix "tts:"),
    aligned row-for-row to build_dataset's output for the same clips/snrs — must
    mirror build_dataset's iteration order (commands x snrs, then unknown, then
    silence) exactly."""
    flags = []
    for cmd in config.COMMANDS:
        for _clip, spk in clips_ws.get(cmd, []):
            flags.extend([spk.startswith("tts:")] * len(snrs))
    for _clip, _spk in clips_ws.get("_unknown_", []):
        flags.append(False)
    n_sil = max(1, len(clips_ws.get("_unknown_", [])))
    flags.extend([False] * n_sil)
    return np.asarray(flags, dtype=bool)


def _build_and_split(test_frac: float = 0.2, seed: int = 0) -> None:  # pragma: no cover
    """TTS-fill thin command words, speaker-disjoint split the cached raw clips,
    then augment + extract features, saving data/features_train.npz and
    data/features_test.npz (the latter also carries an ``is_tts`` row flag so
    eval can isolate real-speech-only accuracy).
    """
    rng = np.random.default_rng(seed)
    with open(config.DATA_DIR / "raw_clips.pkl", "rb") as fh:
        cached = pickle.load(fh)
    with open(config.DATA_DIR / "noise.pkl", "rb") as fh:
        noises = pickle.load(fh)

    clips_with_speakers = cached["clips"]
    tts_added = _fill_with_tts(clips_with_speakers)
    if tts_added:
        with open(config.DATA_DIR / "raw_clips.pkl", "wb") as fh:
            pickle.dump(cached, fh)
        print(f"[tts] added: {tts_added}")

    snrs = (20, 10, 0)
    train_ws, test_ws = split_by_speaker(clips_with_speakers, rng, test_frac, keep_speaker=True)
    train_clips = {k: [c for c, _ in v] for k, v in train_ws.items()}
    test_clips = {k: [c for c, _ in v] for k, v in test_ws.items()}

    X_train, y_train = build_dataset(train_clips, noises, rng, snrs=snrs)
    X_test, y_test = build_dataset(test_clips, noises, rng, snrs=snrs)
    is_tts_test = _origin_flags(test_ws, snrs)
    np.savez(config.DATA_DIR / "features_train.npz", X=X_train, y=y_train)
    np.savez(config.DATA_DIR / "features_test.npz", X=X_test, y=y_test, is_tts=is_tts_test)
    print(f"[build] train X={X_train.shape} test X={X_test.shape}")
