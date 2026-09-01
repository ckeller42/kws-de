import librosa
import numpy as np

from kws_de import config


def _fit_length(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).ravel()
    if x.shape[0] < config.CLIP_SAMPLES:
        x = np.pad(x, (0, config.CLIP_SAMPLES - x.shape[0]))
    return x[: config.CLIP_SAMPLES]


def mfcc(samples: np.ndarray) -> np.ndarray:
    x = _fit_length(samples)
    m = librosa.feature.mfcc(
        y=x,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        n_fft=config.WIN_SAMPLES,
        hop_length=config.HOP_SAMPLES,
        n_mels=config.N_MELS,
        center=False,
    )  # shape (N_MFCC, frames)
    out = m.T.astype(np.float32)  # (frames, N_MFCC)
    return out[: config.N_FRAMES]


def mfcc_sequence(samples: np.ndarray) -> np.ndarray:
    """Same MFCC front-end as `mfcc`, but over the WHOLE signal -- no fixed 1 s
    window: no truncation, and only the minimal zero-pad needed to fill one
    analysis window. Frame count `T` grows with input duration; used for
    streaming/phrase audio (`kws_de.phrases`) instead of isolated-word clips."""
    x = np.asarray(samples, dtype=np.float32).ravel()
    if x.shape[0] < config.WIN_SAMPLES:
        x = np.pad(x, (0, config.WIN_SAMPLES - x.shape[0]))
    m = librosa.feature.mfcc(
        y=x,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        n_fft=config.WIN_SAMPLES,
        hop_length=config.HOP_SAMPLES,
        n_mels=config.N_MELS,
        center=False,
    )  # shape (N_MFCC, frames)
    return m.T.astype(np.float32)  # (frames, N_MFCC)
