"""E69 arm (c): a small metric-learning encoder for few-shot enrollment matching.

DS-CNN backbone of the deployed size (`kws_de.model.build_dscnn`, width 48, head dropped)
-> Dense(64) -> L2 normalisation. Same (49, 10, 1) MFCC input as the command model.
Loss (`--loss`, default `amsm`): additive-margin softmax (CosFace) over the word classes
plus a fixed "reject" logit at cosine `t`: word rows must reach cos(centre) - m above t,
`_unknown_`/`_silence_` rows must stay below t - m for EVERY word centre, so non-command
speech is pushed away from all word clusters (the point is rejection). `--loss triplet` =
batch-hard triplet on cosine distance (Hermans et al. 2017; Rusci & Tuytelaars, IS 2023),
non-word rows only ever negatives; it collapsed here (loss pinned at the margin from step
250, every embedding the same point), so it is kept only for the record. Each batch =
K rows of every word class + U `_unknown_`/`_silence_` rows.

Training data = `raw_clips_v3.pkl` (MSWC + gated TTS; the 3 compound labels Küchenlicht /
Außenlicht / Leselicht become extra word classes) + `merge_recordings` (approved words +
context + negatives), augmented by `kws_de.data.build_dataset` as in `kws-dataset build`,
then filtered: all spk22 clips and all approved negatives are dropped (approved phrases
are never merged by the pipeline). `--no-approved` drops every approved recording.

  train:    uv run --no-sync python scripts/e69_triplet.py train --tag s0 --seed 0
  harness:  uv run --no-sync python scripts/e69_enroll_eval.py --arm triplet --pool both
  check:    uv run --no-sync python scripts/e69_triplet.py --selftest

Matcher (`make_matcher`, shared E69 interface): 1 s windows hop 100 ms (e69_embed), proto
for clips <= 1.2 s, cosine DTW otherwise; E69_TRIPLET_MODE=proto|dtw|auto,
E69_TRIPLET_MODEL=<.tflite> (default: newest non-int8 e69_triplet_*.tflite in models_dir).
"""

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from e69_embed import PROTO_MAX_S, _windows, dtw

from kws_de import config
from kws_de.features import mfcc

MODE = os.environ.get("E69_TRIPLET_MODE", "auto")
EXTRA_WORDS = ["Küchenlicht", "Außenlicht", "Leselicht"]
HELD_OUT_SPK = "spk22"


# ---------------------------------------------------------------- training


def build_features(seed: int, no_approved: bool):
    """(X, y, is_tts, labels): pipeline features with the E69 hold-outs removed.
    y indexes `labels`; the last two labels are `_unknown_`, `_silence_`."""
    import pickle

    from kws_de.data import command_words, merge_recordings
    from kws_de.dataset import assemble

    words = command_words() + EXTRA_WORDS
    labels = words + ["_unknown_", "_silence_"]
    with open(config.DATA_DIR / "raw_clips_v3.pkl", "rb") as fh:
        clips = pickle.load(fh)["clips"]  # noqa: S301 - own local cache
    with open(config.DATA_DIR / "noise.pkl", "rb") as fh:
        noises = pickle.load(fh)  # noqa: S301
    print("[recordings] merged:", merge_recordings(clips))

    def keep(lbl, spk):
        if no_approved and spk.startswith(("rec:", "ctx:")):
            return False
        if lbl == "_unknown_" and spk.startswith("rec:"):  # approved negatives
            return False
        return spk.split(":", 1)[-1] != HELD_OUT_SPK

    ws = {lbl: [(c, s) for c, s in clips.get(lbl, []) if keep(lbl, s)] for lbl in labels}
    dropped = sum(len(clips.get(lbl, [])) - len(v) for lbl, v in ws.items())
    assert all(ws[w] for w in words), [w for w in words if not ws[w]]
    print(f"[data] {sum(map(len, ws.values()))} clips kept, {dropped} dropped (hold-outs)")
    X, y, is_tts = assemble(ws, noises, np.random.default_rng(seed + 1), labels, words)
    return X, y, is_tts, labels


def batch_hard_loss(e, y, margin: float):
    """Batch-hard triplet loss on cosine distance. e: (n, d) unit rows; y: (n,) int,
    -1 = non-word row (never an anchor or a positive, always a negative).
    Returns (loss, fraction of anchors with a violated margin)."""
    import tensorflow as tf

    d = 1.0 - tf.matmul(e, e, transpose_b=True)
    same = tf.equal(y[:, None], y[None, :])
    word = y >= 0
    pos = same & ~tf.eye(tf.shape(y)[0], dtype=tf.bool) & word[:, None]
    hp = tf.reduce_max(tf.where(pos, d, -1.0), axis=1)
    hn = tf.reduce_min(tf.where(same, 9.0, d), axis=1)
    viol = tf.boolean_mask(tf.nn.relu(hp - hn + margin), word)
    return tf.reduce_mean(viol), tf.reduce_mean(tf.cast(viol > 0, tf.float32))


def am_reject_loss(e, y, centres, s: float, m: float, t: float):
    """AM-softmax with a constant reject logit. e: (n, d) unit rows; y: (n,) int, -1 =
    non-word row (target = reject); centres: (d, C). Returns (loss, batch accuracy)."""
    import tensorflow as tf

    cos = tf.matmul(e, tf.math.l2_normalize(centres, axis=0))
    tgt = tf.where(y >= 0, y, tf.shape(cos)[1])
    logits = tf.concat([cos, tf.fill([tf.shape(cos)[0], 1], t)], axis=1)
    logits -= m * tf.one_hot(tgt, tf.shape(logits)[1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(tgt, s * logits)
    acc = tf.cast(tf.equal(tf.argmax(logits, 1, output_type=tf.int32), tgt), tf.float32)
    return tf.reduce_mean(ce), tf.reduce_mean(acc)


def build_encoder(width: int = 48, dim: int = 64):
    import tensorflow as tf

    from kws_de.model import build_dscnn  # sets GPU memory growth

    base = build_dscnn(width=width)
    x = tf.keras.layers.Dense(dim, name="proj")(base.layers[-2].output)
    out = tf.keras.layers.UnitNormalization(name="l2")(x)
    return tf.keras.Model(base.input, out, name="e69_triplet")


def train(args) -> None:
    import tensorflow as tf

    from kws_de.export import balanced_calibration, to_int8_tflite

    t0 = time.time()
    cache = config.DATA_DIR / (
        f"e69_triplet_feats_s{args.seed}{'_noapproved' if args.no_approved else ''}.npz"
    )
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        X, y, is_tts, labels = d["X"], d["y"], d["is_tts"], list(d["labels"])
    else:
        X, y, is_tts, labels = build_features(args.seed, args.no_approved)
        np.savez(cache, X=X, y=y, is_tts=is_tts, labels=np.array(labels))
    print(f"[data] {len(y)} rows, {len(labels)} labels, {time.time() - t0:.0f} s")
    n_words = len(labels) - 2
    by_cls = [np.flatnonzero(y == c) for c in range(n_words)]
    other = np.flatnonzero(y >= n_words)
    w = np.where(is_tts, 1.0, float(args.real_weight))  # real speech over-represented

    def pick(idx, k):
        p = w[idx] / w[idx].sum()
        return rng.choice(idx, k, p=p)

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_encoder(args.width, args.dim)
    print(f"[model] {model.count_params()} params")
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, args.steps)
    opt = tf.keras.optimizers.Adam(lr)
    Xt = tf.constant(X[..., None])
    centres = tf.Variable(tf.random.normal([args.dim, n_words]), name="centres")
    params = model.trainable_variables + ([centres] if args.loss == "amsm" else [])

    @tf.function
    def step(idx, yb):
        with tf.GradientTape() as tape:
            e = model(tf.gather(Xt, idx), training=True)
            if args.loss == "amsm":
                loss, frac = am_reject_loss(e, yb, centres, args.scale, args.margin, args.t)
            else:
                loss, frac = batch_hard_loss(e, yb, args.margin)
        grads = tape.gradient(loss, params)
        opt.apply_gradients(zip(grads, params, strict=True))
        return loss, frac

    hist = []
    for s in range(args.steps):
        idx = np.concatenate(
            [pick(i, args.k) for i in by_cls] + [rng.choice(other, args.u, replace=False)]
        )
        yb = np.where(y[idx] < n_words, y[idx], -1).astype(np.int32)
        hist.append([float(v) for v in step(tf.constant(idx), tf.constant(yb))])
        if (s + 1) % 250 == 0:
            m = np.mean(hist[-250:], axis=0)
            print(f"step {s + 1}: loss {m[0]:.4f} acc/active {m[1]:.3f} ({time.time() - t0:.0f} s)")

    out = config.MODELS_DIR / f"e69_triplet_{args.tag}"
    model.save(out.with_suffix(".keras"))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    out.with_suffix(".tflite").write_bytes(conv.convert())
    rep = balanced_calibration(X, y, per_class=20, seed=args.seed)
    Path(f"{out}_int8.tflite").write_bytes(to_int8_tflite(model, rep))
    print(f"saved {out}.keras/.tflite/_int8.tflite, wall {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- matcher

_itp = None
_cache: dict[bytes, np.ndarray] = {}


def model_path() -> Path:
    env = os.environ.get("E69_TRIPLET_MODEL")
    if env:
        return Path(env)
    cands = [p for p in config.MODELS_DIR.glob("e69_triplet_*.tflite") if "_int8" not in p.name]
    return max(cands, key=lambda p: p.stat().st_mtime)


def _interpreter():
    global _itp
    if _itp is None:
        import tensorflow as tf

        _itp = tf.lite.Interpreter(model_path=str(model_path()))
        _itp.allocate_tensors()
    return _itp


def embed(sig: np.ndarray) -> np.ndarray:
    """(n_windows, dim) L2-normalised embeddings, rows in time order. Cached by bytes."""
    sig = np.asarray(sig, dtype=np.float32).ravel()
    key = hashlib.sha1(sig.tobytes()).digest()
    if key not in _cache:
        itp = _interpreter()
        inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
        in_s, in_zp = inp["quantization"]
        o_s, o_zp = out["quantization"]
        rows = []
        for win in _windows(sig):
            x = mfcc(win)[None, ..., None]
            if inp["dtype"] == np.int8:
                x = np.clip(np.round(x / in_s + in_zp), -128, 127).astype(np.int8)
            itp.set_tensor(inp["index"], x)
            itp.invoke()
            r = itp.get_tensor(out["index"])[0].astype(np.float32)
            rows.append((r - o_zp) * o_s if out["dtype"] == np.int8 else r)
        e = np.stack(rows)
        _cache[key] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return _cache[key]


def _best_window(sig: np.ndarray) -> np.ndarray:
    w = _windows(np.asarray(sig, dtype=np.float32).ravel())
    return embed(sig)[int(np.argmax((w**2).sum(axis=1)))]


def make_matcher(
    templates: dict[str, list[np.ndarray]],
) -> Callable[[np.ndarray], list[tuple[str, float]]]:
    """templates: label -> enrollment takes (float32 16 kHz). match(sig) -> [(label,
    cosine distance)] ascending. Same proto/DTW rule as scripts/e69_embed.py."""
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


def _selftest() -> None:
    import soundfile as sf
    import tensorflow as tf

    # loss: separated clusters -> 0; everything collapsed onto one point -> margin
    e = tf.math.l2_normalize(tf.constant([[1.0, 0], [1, 0.01], [0, 1], [0, 1.01], [-1, 0]]), 1)
    y = tf.constant([0, 0, 1, 1, -1])
    assert float(batch_hard_loss(e, y, 0.3)[0]) < 1e-6
    loss, frac = batch_hard_loss(tf.ones((5, 2)) / np.sqrt(2), y, 0.3)
    assert abs(float(loss) - 0.3) < 1e-5 and float(frac) == 1.0
    centres = tf.constant([[1.0, 0], [0, 1]])
    e = tf.constant([[1.0, 0], [0, 1], [-1, 0]])
    loss, acc = am_reject_loss(e, tf.constant([0, 1, -1]), centres, 30.0, 0.2, 0.4)
    assert float(loss) < 0.05 and float(acc) == 1.0, (loss, acc)
    loss, _ = am_reject_loss(
        tf.constant([[1.0, 0]] * 3), tf.constant([0, 1, -1]), centres, 30, 0.2, 0.4
    )
    assert float(loss) > 1.0, loss

    words = config.DATA_DIR / "recordings" / "approved" / "words"
    a_path, b_path = sorted(words.glob("Licht/*.wav"))[0], sorted(words.glob("Heizung/*.wav"))[0]
    a, b = (sf.read(p, dtype="float32")[0] for p in (a_path, b_path))
    ea = embed(np.concatenate([a, b]))
    assert np.allclose(np.linalg.norm(ea, axis=1), 1.0, atol=1e-4)
    t0 = time.perf_counter()
    res = make_matcher({"Licht": [a], "Heizung": [b]})(a)
    d = dict(res)
    assert res[0][0] == "Licht" and d["Licht"] < 1e-5 < d["Heizung"], res
    assert dtw(embed(a), embed(a)) < 1e-5
    ms = (time.perf_counter() - t0) * 1e3
    print(f"e69_triplet selftest ok model={model_path().name} mode={MODE} {res} {ms:.1f} ms")


def main() -> None:
    if "--selftest" in sys.argv:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["train"])
    ap.add_argument("--tag", required=True, help="-> models/e69_triplet_<tag>.keras")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--k", type=int, default=8, help="rows per word class per batch")
    ap.add_argument("--u", type=int, default=64, help="_unknown_/_silence_ rows per batch")
    ap.add_argument("--loss", choices=["amsm", "triplet"], default="amsm")
    ap.add_argument("--margin", type=float, default=0.2, help="amsm m / triplet margin")
    ap.add_argument("--scale", type=float, default=30.0, help="amsm logit scale s")
    ap.add_argument("--t", type=float, default=0.4, help="amsm reject cosine")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--real-weight", type=float, default=3.0)
    ap.add_argument("--no-approved", action="store_true", help="drop ALL approved recordings")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
