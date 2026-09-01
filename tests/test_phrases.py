import numpy as np

from kws_de import config
from kws_de.ctc import token_id
from kws_de.grammar import Intent
from kws_de.phrases import build_phrase_batch, make_phrase, phrase_features


def _clip(n_samples, val=0.1):
    return (np.ones(n_samples, dtype=np.float32) * val).astype(np.float32)


def test_make_phrase_longer_than_either_clip():
    rng = np.random.default_rng(0)
    word_clips = {"Licht": [_clip(4000)], "an": [_clip(3000)]}
    wav = make_phrase(["Licht", "an"], word_clips, rng)
    assert wav.shape[0] > 4000
    assert wav.shape[0] > 3000


def test_phrase_features_shape_grows_with_duration():
    short = _clip(4000)
    long = _clip(16000)
    f_short = phrase_features(short)
    f_long = phrase_features(long)
    assert f_short.shape[1] == config.N_MFCC
    assert f_long.shape[1] == config.N_MFCC
    assert f_long.shape[0] > f_short.shape[0]


def test_build_phrase_batch_pairs_features_with_target_ids():
    rng = np.random.default_rng(0)
    catalog = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "wärmer")]
    clips_by_word = {
        "Licht": [_clip(4000)],
        "Küche": [_clip(3500)],
        "an": [_clip(3000)],
        "Heizung": [_clip(4200)],
        "wärmer": [_clip(3800)],
    }
    batch = build_phrase_batch(catalog, clips_by_word, rng)
    assert len(batch) == 2
    feat_seq, target_ids = batch[0]
    assert feat_seq.shape[1] == config.N_MFCC
    assert target_ids == [token_id("Licht"), token_id("Küche"), token_id("an")]
    _, target_ids2 = batch[1]
    assert target_ids2 == [token_id("Heizung"), token_id("wärmer")]


def test_build_phrase_batch_skips_missing_words():
    rng = np.random.default_rng(0)
    catalog = [Intent("Licht", None, "an")]
    clips_by_word = {"Licht": [_clip(4000)]}  # "an" missing
    batch = build_phrase_batch(catalog, clips_by_word, rng)
    assert batch == []
