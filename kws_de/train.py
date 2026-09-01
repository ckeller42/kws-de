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
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(config.DATA_DIR / "features_train.npz")
    model, history = train(data["X"], data["y"], epochs=args.epochs)
    model.save(config.MODELS_DIR / "kws.keras")
    print(f"final train accuracy: {history['accuracy'][-1]:.4f}")
