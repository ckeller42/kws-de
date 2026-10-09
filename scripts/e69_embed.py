"""E69 CNN-embedding arm: few-shot enrollment matching on the deployed command model's
48-d penultimate (global-average-pool) output, read from the deployed INT8 tflite so the
embedding is exactly what runs on the ESP.

Each signal -> sequence of L2-normalised embeddings over 1 s windows (hop 100 ms, tail
zero-padded). Distance per label:
  proto -- best-energy window, cosine distance to the mean template embedding (clips <=1.2 s)
  dtw   -- cosine-distance DTW over the embedding sequences, length-normalised, min over takes
E69_EMBED_MODE=proto|dtw|auto (default auto: proto when the query and all templates are <=1.2 s).

Self-check: uv run --no-sync python scripts/e69_embed.py
"""

import hashlib
import os
from collections.abc import Callable

import numpy as np

from kws_de import config
from kws_de.features import mfcc

MODE = os.environ.get("E69_EMBED_MODE", "auto")
MODEL = config.MODELS_DIR / "command_v3_w48_qat.tflite"  # deployed 86b7105e
EMBED_TENSOR = "quant_global_average_pooling2d_1/Mean"
HOP = config.SAMPLE_RATE // 10
PROTO_MAX_S = 1.2

_itp = None
_cache: dict[bytes, np.ndarray] = {}


def _interpreter():
    global _itp
    if _itp is None:
        import tensorflow as tf

        _itp = tf.lite.Interpreter(model_path=str(MODEL), experimental_preserve_all_tensors=True)
        _itp.allocate_tensors()
    return _itp


def _windows(sig: np.ndarray) -> np.ndarray:
    n = max(len(sig), config.CLIP_SAMPLES)
    n = config.CLIP_SAMPLES + -(-(n - config.CLIP_SAMPLES) // HOP) * HOP
    x = np.pad(sig, (0, n - len(sig)))
    return np.lib.stride_tricks.sliding_window_view(x, config.CLIP_SAMPLES)[::HOP]


def embed(sig: np.ndarray) -> np.ndarray:
    """(n_windows, 48) L2-normalised embeddings, rows in time order. Cached by signal bytes."""
    sig = np.asarray(sig, dtype=np.float32).ravel()
    key = hashlib.sha1(sig.tobytes()).digest()
    if key not in _cache:
        itp = _interpreter()
        inp = itp.get_input_details()[0]
        emb = next(t for t in itp.get_tensor_details() if t["name"].endswith(EMBED_TENSOR))
        in_s, in_zp = inp["quantization"]
        e_s, e_zp = emb["quantization"]
        rows = []
        for w in _windows(sig):
            q = np.clip(np.round(mfcc(w) / in_s + in_zp), -128, 127).astype(np.int8)
            itp.set_tensor(inp["index"], q[None, ..., None])
            itp.invoke()
            rows.append((itp.get_tensor(emb["index"])[0].astype(np.float32) - e_zp) * e_s)
        e = np.stack(rows)
        _cache[key] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return _cache[key]


def _best_window(sig: np.ndarray) -> np.ndarray:
    w = _windows(np.asarray(sig, dtype=np.float32).ravel())
    return embed(sig)[int(np.argmax((w**2).sum(axis=1)))]


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-distance DTW (steps right/down/diagonal), normalised by len(a)+len(b)."""
    cost = 1.0 - a @ b.T
    acc = np.full((len(a) + 1, len(b) + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[-1, -1] / (len(a) + len(b)))


def make_matcher(
    templates: dict[str, list[np.ndarray]],
) -> Callable[[np.ndarray], list[tuple[str, float]]]:
    """templates: label -> list of float32 16 kHz mono signals (enrollment takes).
    Returns match(sig) -> [(label, distance), ...] sorted ascending (lower = closer)."""
    limit = PROTO_MAX_S * config.SAMPLE_RATE
    short = all(len(s) <= limit for takes in templates.values() for s in takes)
    protos = {}
    if MODE == "proto" or (MODE == "auto" and short):
        for lab, takes in templates.items():
            p = np.mean([_best_window(s) for s in takes], axis=0)
            protos[lab] = p / np.linalg.norm(p)
    seqs = {lab: [embed(s) for s in takes] for lab, takes in templates.items()}

    def match(sig: np.ndarray) -> list[tuple[str, float]]:
        if MODE == "proto" or (MODE == "auto" and protos and len(sig) <= limit):
            q = _best_window(sig)
            scores = [(lab, float(1.0 - q @ p)) for lab, p in protos.items()]
        else:
            q = embed(sig)
            scores = [(lab, min(dtw(q, t) for t in ts)) for lab, ts in seqs.items()]
        return sorted(scores, key=lambda x: x[1])

    return match


if __name__ == "__main__":
    import time

    import soundfile as sf

    words = config.DATA_DIR / "recordings" / "approved" / "words"
    a_path, b_path = sorted(words.glob("Licht/*.wav"))[0], sorted(words.glob("Heizung/*.wav"))[0]
    a, b = (sf.read(p, dtype="float32")[0] for p in (a_path, b_path))
    embed(b)  # warm-up: tf import + interpreter load
    t0 = time.perf_counter()
    n_win = len(embed(np.concatenate([a, b, a])))
    ms_win = (time.perf_counter() - t0) * 1e3 / n_win
    match = make_matcher({"Licht": [a], "Heizung": [b]})
    t0 = time.perf_counter()
    res = match(a)
    ms_match = (time.perf_counter() - t0) * 1e3
    d = dict(res)
    assert res[0][0] == "Licht" and d["Licht"] < 1e-5, res
    assert d["Heizung"] > d["Licht"], res
    assert dtw(embed(a), embed(a)) < 1e-5 and dtw(embed(a), embed(b)) > 1e-3
    print(f"ok mode={MODE} {res}  {ms_win:.2f} ms/window  {ms_match:.2f} ms/match (2 labels)")
