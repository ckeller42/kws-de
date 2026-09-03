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

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from kws_de import config  # noqa: E402
from kws_de.model import build_dscnn  # noqa: E402


def _class_weights(y) -> dict:
    """Inverse-frequency weights so an over-represented class (e.g. _unknown_
    swollen by transition-window negatives) can't dominate the loss and
    suppress word recall. Shared by `train()` and `train_qat()`."""
    n = len(y)
    counts = np.bincount(y)
    k = len(counts)
    return {i: (n / (k * c) if c else 0.0) for i, c in enumerate(counts)}


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
):
    """Fit `model` (default: a fresh `build_dscnn`) on (X, y). `validation_data`/
    `callbacks` are passed straight through to `model.fit` (e.g. a `ModelCheckpoint`
    to select the best-val-accuracy epoch) -- kws_de.benchmark reuses this for the
    architecture zoo instead of duplicating the class-weight/fit logic."""
    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(X, np.float32)[..., None]
    y = np.asarray(y)
    if model is None:
        model = build_dscnn(num_classes=num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or ("features_v2" if args.v2 else "features")
    num_classes = len(config.COMMAND_LABELS) if args.v2 else None
    out_name = args.out or ("command.keras" if args.v2 else "kws.keras")
    float_path = config.MODELS_DIR / out_name
    data = np.load(config.DATA_DIR / f"{prefix}_train.npz")

    if args.qat and float_path.exists():
        # Reuse the already-trained float model's weights (a same-architecture
        # `.keras` file, possibly saved under a different Keras major version --
        # `load_weights` round-trips fine across Keras 2/3 even when a full
        # `load_model` does not) rather than retraining, so the QAT fine-tune
        # starts from the exact model the PTQ path already exports.
        model = build_dscnn(num_classes=num_classes)
        model.load_weights(float_path)
        print(f"loaded existing float model weights from {out_name} for QAT fine-tune")
    else:
        model, history = train(data["X"], data["y"], epochs=args.epochs, num_classes=num_classes)
        model.save(float_path)
        print(f"final train accuracy: {history['accuracy'][-1]:.4f}")

    if args.qat:
        qmodel, qhistory = train_qat(model, data["X"], data["y"], epochs=args.qat_epochs)
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
