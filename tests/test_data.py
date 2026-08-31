import numpy as np
from kws_de import config
from kws_de.data import build_dataset


def _clip(rng):
    return rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)


def test_build_dataset_shapes_and_labels():
    rng = np.random.default_rng(0)
    clips = {c: [_clip(rng) for _ in range(3)] for c in config.COMMANDS}
    clips["_unknown_"] = [_clip(rng) for _ in range(4)]
    noises = [rng.standard_normal(8000).astype(np.float32) for _ in range(2)]
    X, y = build_dataset(clips, noises, rng, snrs=(20, 0))
    assert X.ndim == 3 and X.shape[1:] == (config.N_FRAMES, config.N_MFCC)
    assert X.shape[0] == y.shape[0]
    assert set(np.unique(y)).issubset(set(range(config.NUM_CLASSES)))
    # _silence_ class must be present (built from noise)
    assert config.label_index("_silence_") in set(y.tolist())


def test_commands_are_augmented_per_snr():
    rng = np.random.default_rng(1)
    clips = {c: [_clip(rng)] for c in config.COMMANDS}
    clips["_unknown_"] = [_clip(rng)]
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X2, y2 = build_dataset(clips, noises, rng, snrs=(20, 10))
    X1, y1 = build_dataset(clips, noises, rng, snrs=(20,))
    licht = config.label_index("Licht")
    assert (y2 == licht).sum() == 2 * (y1 == licht).sum()
