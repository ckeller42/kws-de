"""E69 MFCC-DTW arm of the few-shot enrollment recogniser: match a signal against a
few enrolled takes per label by dynamic time warping over the device frontend's
MFCC frames (`kws_de.features.mfcc_sequence`), each utterance mean/variance
normalised per coefficient. Distance to a template = accumulated euclidean path
cost / path length; distance to a label = min over that label's templates.
Harness: scripts/e69_enroll_eval.py --arm dtw.

Self-check: uv run --no-sync python scripts/e69_dtw.py --selftest
"""

import sys
from collections.abc import Callable

import librosa
import numpy as np

from kws_de.features import mfcc_sequence


def features(sig: np.ndarray) -> np.ndarray:
    """(n_mfcc, T) per-utterance CMVN'd MFCC frames, librosa.sequence.dtw's layout."""
    m = mfcc_sequence(sig)
    return ((m - m.mean(0)) / (m.std(0) + 1e-6)).T


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    D, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def make_matcher(
    templates: dict[str, list[np.ndarray]],
) -> Callable[[np.ndarray], list[tuple[str, float]]]:
    feats = {lab: [features(s) for s in sigs] for lab, sigs in templates.items() if sigs}

    def match(sig: np.ndarray) -> list[tuple[str, float]]:
        q = features(sig)
        dists = [(lab, min(dtw_distance(q, t) for t in ts)) for lab, ts in feats.items()]
        return sorted(dists, key=lambda x: x[1])

    return match


def _selftest() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    chirp = np.sin(2 * np.pi * (300 + 1500 * t) * t).astype(np.float32)
    noise = rng.standard_normal(16000).astype(np.float32) * 0.3
    match = make_matcher({"chirp": [chirp], "noise": [noise]})
    (lab, d), (_, d2) = match(chirp)
    assert lab == "chirp" and d < 1e-6 < d2, (lab, d, d2)
    assert match(noise + 0.01 * rng.standard_normal(16000).astype(np.float32))[0][0] == "noise"
    print("e69_dtw selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
