import os
import sys

# tfmot (quantisation-aware training, --qat below) only wraps tf.keras models
# built under Keras 2 ("legacy" mode, package `tf_keras`); TF 2.18's default
# tf.keras is Keras 3, which tfmot's QuantizeWrapperV2 cannot wrap. The env
# var that selects legacy Keras must be set before `tensorflow` is imported
# anywhere in the process, and this module is the one place `import
# tensorflow` happens for the CLI -- so when `--qat` is on the command line,
# re-exec once with the var set before any TF-touching import below runs.
# Harmless for every other invocation (tests, `kws_de.benchmark` importing
# `train()`): argv without "--qat" never triggers this branch.
if "--qat" in sys.argv and os.environ.get("TF_USE_LEGACY_KERAS") != "1":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "kws_de.train", *sys.argv[1:]])

import argparse  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from kws_de import config  # noqa: E402
from kws_de.eta import Timed  # noqa: E402
from kws_de.model import build_dscnn  # noqa: E402


def _class_weights(y) -> dict:
    """Inverse-frequency weights so an over-represented class (e.g. _unknown_
    swollen by transition-window negatives) can't dominate the loss and
    suppress word recall. Shared by `train()` and `train_qat()`."""
    n = len(y)
    counts = np.bincount(y)
    k = len(counts)
    return {i: (n / (k * c) if c else 0.0) for i, c in enumerate(counts)}


def upweight_real(X, y, is_tts, weight: int):
    """Repeat every real (``~is_tts`` -- device ``rec:`` recordings and MSWC
    clips; the feature-cache npz keeps only the ``is_tts`` flag, not the
    original ``rec:``/MSWC speaker id, so "real" here is that same population)
    training row `weight`-1 extra times, so real speech gets `weight`x
    representation relative to TTS synthetic clips before class weighting.
    `weight <= 1` is a no-op returning `X, y` unchanged (same arrays, not copies)."""
    X = np.asarray(X)
    y = np.asarray(y)
    if weight <= 1:
        return X, y
    is_tts = np.asarray(is_tts, bool)
    real_idx = np.flatnonzero(~is_tts)
    if real_idx.size == 0:
        return X, y
    extra = np.tile(real_idx, weight - 1)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


def train(
    X,
    y,
    epochs=30,
    seed=0,
    num_classes=None,
    class_weight=True,
    model=None,
    validation_data=None,
    callbacks=None,
    width=32,
    cosine=False,
):
    """Fit `model` (default: a fresh `build_dscnn`) on (X, y). `validation_data`/
    `callbacks` are passed straight through to `model.fit` (e.g. a `ModelCheckpoint`
    to select the best-val-accuracy epoch) -- kws_de.benchmark reuses this for the
    architecture zoo instead of duplicating the class-weight/fit logic. `width` is
    forwarded to `build_dscnn` when `model` is not given. `cosine` decays Adam's
    default 1e-3 learning rate to 0 over the whole run (CosineDecay) instead of
    holding it flat."""
    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(X, np.float32)[..., None]
    y = np.asarray(y)
    if model is None:
        model = build_dscnn(num_classes=num_classes, width=width)
    optimizer = "adam"
    if cosine:
        steps = epochs * math.ceil(len(y) / config.BATCH_SIZE)
        optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(1e-3, steps))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw = _class_weights(y) if class_weight else None
    if validation_data is not None:
        Xv, yv = validation_data
        validation_data = (np.asarray(Xv, np.float32)[..., None], np.asarray(yv))
    h = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        verbose=0,
        class_weight=cw,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    return model, h.history


def train_qat(
    model,
    X,
    y,
    epochs=10,
    seed=0,
    class_weight=True,
    learning_rate=1e-5,
    validation_data=None,
    callbacks=None,
):
    """Quantisation-aware fine-tune: wrap `model` with
    `tfmot.quantization.keras.quantize_model` (inserts per-tensor fake-quant
    ops on every activation and per-channel fake-quant on conv/dense
    kernels, mirroring the eventual INT8 TFLite graph) and fine-tune it for a
    few epochs at a low learning rate, so the INT8 export sees quantisation
    error during training instead of only at PTQ conversion time. Requires
    the process to already be running under `TF_USE_LEGACY_KERAS=1` (see the
    module-level re-exec above) -- `model` must be a Keras-2 (`tf_keras`)
    model, e.g. from `build_dscnn()` called in this same process.

    Returns the wrapped, fine-tuned model (still exposes every quantised
    layer's fake-quant min/max ranges) and its fit history."""
    import tensorflow_model_optimization as tfmot

    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(X, np.float32)[..., None]
    y = np.asarray(y)
    qmodel = tfmot.quantization.keras.quantize_model(model)
    qmodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cw = _class_weights(y) if class_weight else None
    if validation_data is not None:
        Xv, yv = validation_data
        validation_data = (np.asarray(Xv, np.float32)[..., None], np.asarray(yv))
    h = qmodel.fit(
        X,
        y,
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        verbose=0,
        class_weight=cw,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    return qmodel, h.history


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--v2", action="store_true", help="train on config.COMMAND_LABELS (23 classes)")
    ap.add_argument("--prefix", default=None, help="npz prefix (default: features[_v2])")
    ap.add_argument("--out", default=None, help="model filename (default: kws/command.keras)")
    ap.add_argument(
        "--qat",
        action="store_true",
        help="also produce a quantisation-aware-fine-tuned model (suffix _qat) for "
        "comparison against plain post-training INT8 quantisation",
    )
    ap.add_argument(
        "--qat-epochs", type=int, default=10, help="fine-tune epochs for --qat (default 10)"
    )
    ap.add_argument(
        "--width", type=int, default=32, help="conv/depthwise-separable channel count (default 32)"
    )
    ap.add_argument(
        "--real-weight",
        type=int,
        default=1,
        dest="real_weight",
        help="repeat every real (non-TTS, `~is_tts`) train row this many times before "
        "class weighting, so real speech is over-represented relative to synthetic TTS "
        "clips (default 1, no-op)",
    )
    ap.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    ap.add_argument(
        "--cosine",
        action="store_true",
        help="cosine-decay the float-phase learning rate from 1e-3 to 0 over --epochs "
        "(QAT fine-tune unchanged)",
    )
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or ("features_v2" if args.v2 else "features")
    num_classes = len(config.COMMAND_LABELS) if args.v2 else None
    out_name = args.out or ("command.keras" if args.v2 else "kws.keras")
    float_path = config.MODELS_DIR / out_name
    data = np.load(config.DATA_DIR / f"{prefix}_train.npz")
    # Older/toy feature caches (e.g. tests/test_train_qat.py's fixture) predate the
    # `is_tts` row flag; --real-weight is a no-op without it (real_idx empty below).
    is_tts = data["is_tts"] if "is_tts" in data.files else np.ones(data["X"].shape[0], dtype=bool)
    X, y = upweight_real(data["X"], data["y"], is_tts, args.real_weight)
    # Val-selected best epoch when the build wrote a val split (kws-dataset build does);
    # older/toy caches without one fall back to the last epoch.
    val_path = config.DATA_DIR / f"{prefix}_val.npz"
    val = np.load(val_path) if val_path.exists() else None
    size = args.epochs * X.shape[0]
    with tempfile.TemporaryDirectory() as td, Timed("train", size=size, note=prefix):
        ckpt = str(Path(td) / "best.weights.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            ckpt, save_best_only=True, save_weights_only=True, monitor="val_accuracy", mode="max"
        )
        progress = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, logs: print(
                f"epoch {e + 1}: val_accuracy {logs['val_accuracy']:.4f}"
            )
        )
        model, history = train(
            X,
            y,
            epochs=args.epochs,
            seed=args.seed,
            num_classes=num_classes,
            width=args.width,
            cosine=args.cosine,
            validation_data=None if val is None else (val["X"], val["y"]),
            callbacks=None if val is None else [checkpoint, progress],
        )
        if val is not None:
            va = history["val_accuracy"]
            best = int(np.argmax(va))
            model.load_weights(ckpt)  # best val_accuracy epoch, not necessarily the last
            print(f"best epoch {best + 1}/{args.epochs}: val accuracy {va[best]:.4f}")
    model.save(float_path)
    print(f"final train accuracy: {history['accuracy'][-1]:.4f}")

    if args.qat:
        qmodel, qhistory = train_qat(model, X, y, epochs=args.qat_epochs, seed=args.seed)
        # SavedModel dir, not `.keras`: tfmot's QuantizeWrapperV2 layers fail to
        # reload from the `.keras` zip format (variable-name mismatch on
        # reload, a known tfmot/Keras-3-format interaction) but round-trip
        # correctly through the SavedModel format `save_format="tf"` selects.
        qat_dir = config.MODELS_DIR / (out_name.removesuffix(".keras") + "_qat")
        qmodel.save(qat_dir, save_format="tf")
        print(f"QAT final train accuracy: {qhistory['accuracy'][-1]:.4f}")
        print(f"QAT model saved to {qat_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
