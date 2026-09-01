import argparse

import numpy as np
import tensorflow as tf

from kws_de import config
from kws_de.model import build_dscnn


def train(X, y, epochs=30, seed=0, num_classes=None):
    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(X, np.float32)[..., None]
    model = build_dscnn(num_classes=num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(X, np.asarray(y), epochs=epochs, batch_size=32, verbose=0)
    return model, h.history


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument(
        "--v2", action="store_true", help="train on config.COMMAND_LABELS (26 classes)"
    )
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "features_v2" if args.v2 else "features"
    num_classes = len(config.COMMAND_LABELS) if args.v2 else None
    out_name = "command.keras" if args.v2 else "kws.keras"
    data = np.load(config.DATA_DIR / f"{prefix}_train.npz")
    model, history = train(data["X"], data["y"], epochs=args.epochs, num_classes=num_classes)
    model.save(config.MODELS_DIR / out_name)
    print(f"final train accuracy: {history['accuracy'][-1]:.4f}")
