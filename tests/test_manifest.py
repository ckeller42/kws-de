import numpy as np

from kws_de import config
from kws_de.manifest import build_manifest


def _split(rng, n):
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % config.NUM_CLASSES).astype(np.int64)
    is_tts = np.arange(n) % 2 == 0
    return X, y, is_tts


def test_manifest_shape_counts_and_stable_hash():
    rng = np.random.default_rng(0)
    splits = {"train": _split(rng, 30), "val": _split(rng, 6), "test": _split(rng, 6)}
    m = build_manifest(splits, seed=0, labels=config.LABELS)
    assert m["seed"] == 0 and m["labels"] == config.LABELS
    assert m["splits"]["train"]["n"] == 30
    assert m["splits"]["train"]["tts"] + m["splits"]["train"]["real"] == 30
    # hash is a deterministic function of the X bytes
    assert (
        m["splits"]["train"]["hash"]
        == build_manifest(splits, seed=0, labels=config.LABELS)["splits"]["train"]["hash"]
    )
