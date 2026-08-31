import argparse

import numpy as np
import tensorflow as tf

from kws_de import config
from kws_de.model import build_dscnn


def train(X, y, epochs=30, seed=0):
    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(X, np.float32)[..., None]
    model = build_dscnn()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(X, np.asarray(y), epochs=epochs, batch_size=32, verbose=0)
    return model, h.history


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(config.DATA_DIR / "features.npz")
    model, _ = train(X["X"], X["y"], epochs=args.epochs)
    model.save(config.MODELS_DIR / "kws.keras")
