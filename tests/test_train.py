import numpy as np
from kws_de import config
from kws_de.train import train


def test_smoke_overfits_tiny_separable_data():
    rng = np.random.default_rng(0)
    # Two easily-separable clusters mapped to 2 classes -> accuracy must beat chance.
    n = 60
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % config.NUM_CLASSES).astype(np.int64)
    X += y[:, None, None]  # inject class-dependent shift so it's learnable
    model, hist = train(X, y, epochs=8, seed=0)
    assert hist["accuracy"][-1] > hist["accuracy"][0]  # learning happened
    assert hist["accuracy"][-1] > 1.5 / config.NUM_CLASSES
