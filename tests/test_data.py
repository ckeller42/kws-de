import numpy as np

from kws_de import config
from kws_de.data import _origin_flags, build_dataset, split_by_speaker


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


def test_split_by_speaker_is_disjoint_and_covers_all_clips():
    rng = np.random.default_rng(2)
    # 10 speakers per label, 2 clips each -> speaker id must fully determine the split.
    clips = {
        "Licht": [(_clip(rng), f"spk{i}") for i in range(10) for _ in range(2)],
        "_unknown_": [(_clip(rng), f"u{i}") for i in range(10) for _ in range(2)],
    }
    train, test = split_by_speaker(clips, rng, test_frac=0.2)
    for label in clips:
        n_total = len(clips[label])
        assert len(train[label]) + len(test[label]) == n_total
        assert len(test[label]) > 0  # held-out speakers actually produced test clips


def test_split_by_speaker_no_speaker_in_both_splits():
    rng = np.random.default_rng(3)
    # Tag each speaker's clips with a distinct constant value so identity survives
    # the split (speaker id itself is dropped from the output).
    tagged = {
        "Wasser": [
            (np.full(config.CLIP_SAMPLES, spk_id, np.float32), f"spk{spk_id}")
            for spk_id in range(4)
            for _ in range(2)
        ]
    }
    train, test = split_by_speaker(tagged, rng, test_frac=0.5)
    train_ids = {float(c[0]) for c in train["Wasser"]}
    test_ids = {float(c[0]) for c in test["Wasser"]}
    assert train_ids.isdisjoint(test_ids)
    assert test_ids  # some speakers held out
    assert train_ids | test_ids == {0.0, 1.0, 2.0, 3.0}


def test_split_by_speaker_keep_speaker_preserves_speaker_id():
    rng = np.random.default_rng(4)
    clips = {"Licht": [(_clip(rng), f"spk{i}") for i in range(6) for _ in range(2)]}
    train, test = split_by_speaker(clips, rng, test_frac=0.5, keep_speaker=True)
    all_items = train["Licht"] + test["Licht"]
    assert len(all_items) == len(clips["Licht"])
    # each returned item is still a (clip, speaker_id) pair
    for clip, spk in all_items:
        assert clip.shape == (config.CLIP_SAMPLES,)
        assert spk.startswith("spk")
    # no speaker straddles both splits
    train_spk = {spk for _, spk in train["Licht"]}
    test_spk = {spk for _, spk in test["Licht"]}
    assert train_spk.isdisjoint(test_spk)


def test_origin_flags_marks_tts_rows_and_mirrors_build_dataset_order():
    clips_ws = {
        "Licht": [(np.zeros(1), "tts:Anna:180"), (np.zeros(1), "real_speaker_1")],
        "_unknown_": [(np.zeros(1), "real_speaker_2")],
    }
    snrs = (20, 10, 0)
    flags = _origin_flags(clips_ws, snrs)
    # Licht: 1 TTS clip x 3 snrs (True) + 1 real clip x 3 snrs (False), then
    # 1 unknown row (False), then n_sil = max(1, 1) = 1 silence row (False).
    expected = [True, True, True, False, False, False, False, False]
    assert flags.tolist() == expected
