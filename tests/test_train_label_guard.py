import numpy as np
import pytest

from kws_de import config
from kws_de.train import _check_labels, train


def _xy(n, n_classes):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % n_classes).astype(np.int64)
    return X, y


def test_check_labels_accepts_in_range_and_rejects_out_of_range():
    _check_labels(np.array([0, 5, 22]), 23, "train")  # fine
    with pytest.raises(ValueError, match="labels span 0..25 but the model has 23"):
        _check_labels(np.array([0, 25]), 23, "train")
    with pytest.raises(ValueError, match="labels span -1"):
        _check_labels(np.array([-1, 3]), 23, "train")


def test_train_refuses_stale_npz_labels_before_fitting():
    # The thinky failure mode (E61): a 26-class npz against the 23-class config
    # trained to chance silently on CUDA. Must raise before fit on every platform.
    X, y = _xy(52, 26)
    with pytest.raises(ValueError, match="model has 23 outputs"):
        train(X, y, epochs=1, seed=0, num_classes=23)


def test_train_checks_validation_labels_too():
    X, y = _xy(46, 23)
    Xv, yv = _xy(26, 26)  # validation split from the stale build
    with pytest.raises(ValueError, match="validation labels"):
        train(X, y, epochs=1, seed=0, num_classes=23, validation_data=(Xv, yv))
