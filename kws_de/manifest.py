"""Verifiable fingerprint for a built dataset: per-split counts + content hashes.

`build_manifest` is pure (no I/O) so it's trivially testable; `kws_de.dataset.main`
is the only caller that writes its output to `data/manifest.json`.
"""

import hashlib

import numpy as np

from kws_de import config


def build_manifest(splits: dict, *, seed: int, labels: list[str]) -> dict:
    """`splits` maps "train"/"val"/"test" -> (X, y, is_tts). Returns a
    JSON-serialisable dict: seed, labels, mfcc params, and per-split
    {n, real, tts, per_label_counts, hash} — hash is sha256 of the X bytes, so
    any rebuild can be verified byte-for-byte against a committed manifest."""
    out: dict = {
        "seed": seed,
        "labels": list(labels),
        "mfcc": {
            "n_mfcc": config.N_MFCC,
            "n_frames": config.N_FRAMES,
            "win": config.WIN_SAMPLES,
            "hop": config.HOP_SAMPLES,
            "n_mels": config.N_MELS,
            "sample_rate": config.SAMPLE_RATE,
        },
        "splits": {},
    }
    for name, (X, y, is_tts) in splits.items():
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        is_tts = np.asarray(is_tts, bool)
        counts = {labels[i]: int((y == i).sum()) for i in range(len(labels))}
        out["splits"][name] = {
            "n": int(len(y)),
            "real": int((~is_tts).sum()),
            "tts": int(is_tts.sum()),
            "per_label_counts": counts,
            "hash": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        }
    return out
