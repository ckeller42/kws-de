import argparse

import numpy as np
import tensorflow as tf

from kws_de import config
from kws_de.eta import Timed
from kws_de.model import build_dscnn


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
    cw = None
    if class_weight:
        # inverse-frequency weights so an over-represented class (e.g. _unknown_ swollen by
        # transition-window negatives) can't dominate the loss and suppress word recall.
        n = len(y)
        counts = np.bincount(y)
        k = len(counts)
        cw = {i: (n / (k * c) if c else 0.0) for i, c in enumerate(counts)}
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


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--v2", action="store_true", help="train on config.COMMAND_LABELS (23 classes)")
    ap.add_argument("--prefix", default=None, help="npz prefix (default: features[_v2])")
    ap.add_argument("--out", default=None, help="model filename (default: kws/command.keras)")
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or ("features_v2" if args.v2 else "features")
    num_classes = len(config.COMMAND_LABELS) if args.v2 else None
    out_name = args.out or ("command.keras" if args.v2 else "kws.keras")
    data = np.load(config.DATA_DIR / f"{prefix}_train.npz")
    size = args.epochs * data["X"].shape[0]
    with Timed("train", size=size, note=prefix):
        model, history = train(data["X"], data["y"], epochs=args.epochs, num_classes=num_classes)
    model.save(config.MODELS_DIR / out_name)
    print(f"final train accuracy: {history['accuracy'][-1]:.4f}")
