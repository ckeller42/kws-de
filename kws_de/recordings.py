"""Drop-in self-recorded clips for words no public corpus has (e.g. a rare
camper-hardware compound). Layout, gitignored:

    data/recordings/<word>/<speaker>_<n>.wav|m4a|...

Any sample rate/channels; each file is one utterance. Returns the same
`{label: [(clip, speaker_id)]}` dict as `kws_de.mswc.mine`, with speaker id
`rec:<speaker>` so the speaker-disjoint split holds out whole people and
`_origin_flags` counts them as real (only `tts:` is synthetic).
"""

from pathlib import Path

import numpy as np

from kws_de import config


def centre(sig: np.ndarray, n: int = config.CLIP_SAMPLES) -> np.ndarray:
    """Zero-pad or crop `sig` symmetrically to exactly `n` samples."""
    sig = np.asarray(sig, np.float32).ravel()
    if sig.shape[0] >= n:
        start = (sig.shape[0] - n) // 2
        return sig[start : start + n]
    pad = n - sig.shape[0]
    return np.pad(sig, (pad // 2, pad - pad // 2))


def load_recordings(root: Path, words: list[str]) -> dict[str, list[tuple[np.ndarray, str]]]:
    """For each word in `words`, load `root/<word>/*` (skipped if absent),
    trim leading/trailing silence and centre in a CLIP_SAMPLES window."""
    import librosa  # pragma: no cover

    root = Path(root)
    clips: dict = {}
    for w in words:
        clips[w] = []
        folder = root / w
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.name.startswith(".") or not p.is_file():
                continue
            sig, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)  # pragma: no cover
            trimmed, _ = librosa.effects.trim(sig, top_db=30)
            speaker = p.stem.rsplit("_", 1)[0]
            clips[w].append((centre(trimmed), f"rec:{speaker}"))
    return clips
