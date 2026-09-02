"""Verifiable fingerprint for a built dataset: per-split counts + content hashes.

`build_manifest` is pure (no I/O) so it's trivially testable; `kws_de.dataset.main`
is the only caller that writes its output to `data/manifest.json`.
"""

import hashlib
from datetime import UTC, datetime

import numpy as np

from kws_de import config


def build_manifest(
    splits: dict, *, seed: int, labels: list[str], speakers: dict[str, list[str]] | None = None
) -> dict:
    """`splits` maps "train"/"val"/"test" -> (X, y, is_tts). Returns a
    JSON-serialisable dict: seed, built_at (ISO-8601 UTC, when this manifest was
    built -- consumers like `kws_de.eval.eval_recordings` use it to flag how
    stale a "speaker was in training" match might be), labels, mfcc params, and
    per-split {n, real, tts, per_label_counts, hash} — hash is sha256 of the X
    bytes, so any rebuild can be verified byte-for-byte against a committed
    manifest. With `speakers` (per-split flat list of speaker ids, prefixed
    "tts:"/"rec:"/plain-mswc), each split additionally gets "sources" (counts by
    origin) and "speakers" (sorted numeric ids of device recordings only, "rec:"
    stripped) — provenance for QC-approved device recordings mixed into the
    build."""
    out: dict = {
        "seed": seed,
        "built_at": datetime.now(UTC).isoformat(),
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
        if speakers is not None:
            spk = speakers.get(name, [])
            recording = sum(1 for s in spk if s.startswith("rec:"))
            tts_n = sum(1 for s in spk if s.startswith("tts:"))
            mswc_n = len(spk) - recording - tts_n
            out["splits"][name]["sources"] = {"tts": tts_n, "recording": recording, "mswc": mswc_n}
            out["splits"][name]["speakers"] = sorted({s[4:] for s in spk if s.startswith("rec:")})
    return out
