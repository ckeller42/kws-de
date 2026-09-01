import numpy as np

from kws_de import config
from kws_de.data import build_dataset, command_words


def test_command_words_are_slot_vocab():
    assert command_words() == config.DEVICES + config.ZONES + config.ACTIONS


def test_build_dataset_over_command_labels():
    rng = np.random.default_rng(0)
    clips = {
        w: [rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)] for w in command_words()
    }
    clips["_unknown_"] = [rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)]
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y = build_dataset(clips, noises, rng, snrs=(20,), labels=config.COMMAND_LABELS)
    assert X.shape[1:] == (config.N_FRAMES, config.N_MFCC)
    assert set(np.unique(y)).issubset(set(range(len(config.COMMAND_LABELS))))
