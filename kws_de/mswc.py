"""Mine an extracted MSWC-de tarball (https://mlcommons.org/datasets/multilingual-spoken-words/,
CC-BY 4.0) by keyword folder, instead of streaming the HuggingFace mirror
alphabetically (`kws_de.data._fetch_mswc`), which never reached most of our
command words before its scan cap.

Expected layout under `root` (gitignored, e.g. `data/mswc/de/`):

    clips/<keyword>/<clip>.opus      1 s, 16 kHz mono
    de_splits.csv                    SET,LINK,WORD,VALID,SPEAKER,GENDER

Returns the same `{label: [(np.ndarray float32, speaker_id)]}` dict as
`_fetch_mswc`, so split/augment/manifest code is unchanged.
"""

import csv
import subprocess
from pathlib import Path

import numpy as np

from kws_de import config


def _folder_index(root: Path) -> dict[str, Path]:
    """Lower-cased keyword folder name -> folder path (case-insensitive lookup:
    config spells `Küche`, MSWC folders are lower-case)."""
    clips = Path(root) / "clips"
    return {p.name.lower(): p for p in sorted(clips.iterdir()) if p.is_dir()}


def _pick(items: list, n: int, rng: np.random.Generator) -> list:
    """Seeded shuffle, first `n`. Does not mutate `items`."""
    order = rng.permutation(len(items))
    return [items[i] for i in order[:n]]


def _valid_speakers(root: Path) -> dict[str, str]:
    """`<keyword>/<file>` -> speaker id for rows with VALID == TRUE."""
    out: dict[str, str] = {}
    with open(Path(root) / "de_splits.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["VALID"].strip().upper() == "TRUE":
                out[row["LINK"]] = row["SPEAKER"]
    return out


def _decode(path: Path) -> np.ndarray:  # pragma: no cover - audio I/O
    """float32 mono 16 kHz. soundfile handles wav and (libsndfile >= 1.0.29) opus;
    otherwise fall back to ffmpeg."""
    import soundfile as sf

    try:
        sig, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - libsndfile without opus support
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(config.SAMPLE_RATE),
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        return np.frombuffer(raw, np.float32).copy()
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa

        sig = librosa.resample(sig, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return np.asarray(sig, np.float32)


def mine(
    root: Path,
    words: list[str],
    *,
    n_per_word: int = 300,
    n_unknown: int = 2000,
    unknown_per_word_cap: int = 5,
    seed: int = 0,
) -> dict[str, list[tuple[np.ndarray, str]]]:
    """Per target word: up to `n_per_word` VALID clips (seeded pick). `_unknown_`:
    `n_unknown` VALID clips from keyword folders NOT in `words`, at most
    `unknown_per_word_cap` per keyword, folders visited in seeded-shuffled order."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    index = _folder_index(root)
    speakers = _valid_speakers(root)

    def valid_files(folder: Path) -> list[tuple[Path, str]]:
        out = []
        for p in sorted(folder.iterdir()):
            spk = speakers.get(f"{folder.name}/{p.name}")
            if spk is not None:
                out.append((p, spk))
        return out

    clips: dict = {}
    targets = set()
    for w in words:
        folder = index.get(w.lower())
        targets.add(w.lower())
        files = valid_files(folder) if folder else []
        clips[w] = [(_decode(p), spk) for p, spk in _pick(files, n_per_word, rng)]

    clips["_unknown_"] = []
    others = [k for k in index if k not in targets]
    for k in _pick(others, len(others), rng):
        if len(clips["_unknown_"]) >= n_unknown:
            break
        room = min(unknown_per_word_cap, n_unknown - len(clips["_unknown_"]))
        for p, spk in _pick(valid_files(index[k]), room, rng):
            clips["_unknown_"].append((_decode(p), spk))
    return clips
