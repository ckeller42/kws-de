import numpy as np

from kws_de import config
from kws_de.train import train, upweight_real


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


def test_upweight_real_is_noop_below_weight_2():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    is_tts = np.array([True, False, True, False, False])
    Xw, yw = upweight_real(X, y, is_tts, 1)
    assert np.array_equal(Xw, X)
    assert np.array_equal(yw, y)
    Xw, yw = upweight_real(X, y, is_tts, 0)
    assert np.array_equal(Xw, X)
    assert np.array_equal(yw, y)


def test_upweight_real_repeats_only_non_tts_rows():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])
    is_tts = np.array([True, False, True, False, False])  # real rows: 1, 3, 4
    Xw, yw = upweight_real(X, y, is_tts, 3)
    # original 5 rows plus 2 extra copies of each of the 3 real rows
    assert len(yw) == 5 + 2 * 3
    # every original row is still present exactly once
    for i in range(5):
        assert (yw == y[i]).sum() == (1 if is_tts[i] else 3)
        assert np.array_equal(Xw[yw == y[i]][0], X[i])


def test_upweight_real_noop_when_no_real_rows():
    X = np.arange(6).reshape(3, 2)
    y = np.array([0, 1, 2])
    is_tts = np.array([True, True, True])
    Xw, yw = upweight_real(X, y, is_tts, 5)
    assert np.array_equal(Xw, X)
    assert np.array_equal(yw, y)


def test_cosine_decays_learning_rate_to_near_zero():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % config.NUM_CLASSES).astype(np.int64)
    model, _ = train(X, y, epochs=3, seed=0, cosine=True)
    lr = model.optimizer.learning_rate
    lr = lr(model.optimizer.iterations) if callable(lr) else lr  # Keras 2 returns the schedule
    assert float(lr) < 1e-4
