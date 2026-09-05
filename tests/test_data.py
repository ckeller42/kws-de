import numpy as np

from kws_de import config, tts
from kws_de.data import (
    _origin_flags,
    _tts_combo_plan,
    build_dataset,
    make_transition_windows,
    split_by_speaker,
    split_three_way,
    tts_engines,
    tts_gate_min_tokens_for_content,
)


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


def test_every_word_class_gets_the_same_audio_domains():
    # The 0.000-catalog bug: commands were noise-only, _unknown_ clean-only, so the model
    # learned "clean audio == _unknown_". Guard: a command clip and an _unknown_ clip must
    # yield the SAME number of samples (1 clean + len(snrs) noise-mixed).
    rng = np.random.default_rng(0)
    clips = {config.COMMANDS[0]: [_clip(rng)], "_unknown_": [_clip(rng)]}
    noises = [rng.standard_normal(8000).astype(np.float32)]
    snrs = (20, 0)
    _X, y = build_dataset(clips, noises, rng, snrs=snrs)
    n_cmd = (y == config.label_index(config.COMMANDS[0])).sum()
    n_unk = (y == config.label_index("_unknown_")).sum()
    assert n_cmd == n_unk == 1 + len(snrs)


def test_commands_are_augmented_per_snr():
    # per-clip row count = 1 clean copy + len(snrs) noise-mixed copies, symmetric
    # between commands and _unknown_ (see data.build_dataset docstring: asymmetric
    # clean/noise domains between them is what made the model learn "clean audio
    # implies _unknown_" instead of the actual words).
    rng = np.random.default_rng(1)
    clips = {c: [_clip(rng)] for c in config.COMMANDS}
    clips["_unknown_"] = [_clip(rng)]
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X2, y2 = build_dataset(clips, noises, rng, snrs=(20, 10))
    X1, y1 = build_dataset(clips, noises, rng, snrs=(20,))
    licht = config.label_index("Licht")
    unknown = config.label_index("_unknown_")
    assert (y2 == licht).sum() == 1 + 2
    assert (y1 == licht).sum() == 1 + 1
    assert (y2 == unknown).sum() == 1 + 2
    assert (y1 == unknown).sum() == 1 + 1


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


def test_split_by_speaker_no_leak_across_labels():
    # A speaker recorded under two labels (mirrors a device speaker whose word takes and
    # `_unknown_` negative windows share one `rec:<spk>` id) must land in exactly one
    # split -- the speaker->split draw must be global, not independent per label.
    rng = np.random.default_rng(9)
    clips = {
        "Licht": [(np.full(config.CLIP_SAMPLES, i, np.float32), f"spk{i}") for i in range(10)],
        "_unknown_": [(np.full(config.CLIP_SAMPLES, i, np.float32), f"spk{i}") for i in range(10)],
    }
    train, test = split_by_speaker(clips, rng, test_frac=0.3, keep_speaker=True)
    train_spk = {s for label in clips for _, s in train[label]}
    test_spk = {s for label in clips for _, s in test[label]}
    assert train_spk.isdisjoint(test_spk)
    assert test_spk  # some speakers actually held out


def test_origin_flags_marks_tts_rows_and_mirrors_build_dataset_order():
    clips_ws = {
        "Licht": [(np.zeros(1), "tts:Anna:180"), (np.zeros(1), "real_speaker_1")],
        "_unknown_": [(np.zeros(1), "real_speaker_2")],
    }
    snrs = (20, 10, 0)
    flags = _origin_flags(clips_ws, snrs)
    # per_clip = 1 + len(snrs) = 4 rows/clip (clean + one per snr).
    # Licht: 1 TTS clip x 4 (True) + 1 real clip x 4 (False), then
    # 1 unknown clip x 4 (False), then n_sil = max(1, 1) = 1 silence row (False),
    # then n_clean_sil = max(1, n_sil // 10) = 1 clean silence row (False).
    expected = [True] * 4 + [False] * 4 + [False] * 4 + [False] * 1 + [False] * 1
    assert flags.tolist() == expected


def test_build_dataset_perturbs_synthetic_clips_only():
    rng = np.random.default_rng(0)
    clips = {config.COMMANDS[0]: [_clip(rng), _clip(rng)], "_unknown_": [_clip(rng)]}
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X0, y0 = build_dataset(clips, noises, np.random.default_rng(1), snrs=(20,))
    X1, y1 = build_dataset(
        clips,
        noises,
        np.random.default_rng(1),
        snrs=(20,),
        synthetic={config.COMMANDS[0]: [True, False]},
    )
    # one flagged clip -> one extra (clean + 1 snr) pair for that label, nothing else
    assert X1.shape[0] == X0.shape[0] + 2
    assert (y1 == 0).sum() == (y0 == 0).sum() + 2
    assert (y1 != 0).sum() == (y0 != 0).sum()


def test_origin_flags_doubles_perturbed_tts_rows():
    clips_ws = {
        "Licht": [(np.zeros(1), "tts:say:Anna"), (np.zeros(1), "real_1")],
        "_unknown_": [(np.zeros(1), "real_2")],
    }
    flags = _origin_flags(clips_ws, (20, 10, 0), perturb_tts=True)
    assert flags.tolist() == [True] * 8 + [False] * 4 + [False] * 4 + [False] + [False]


def test_make_transition_windows_geometry():
    # Two 5000-sample words + a 4000-sample (250ms) gap = 14000 total samples,
    # under CLIP_SAMPLES (16000) -- so regardless of where the boundary-window
    # jitter lands within the gap, the CLIP_SAMPLES cut always overlaps both
    # words (worked out from the window-vs-gap arithmetic, not just observed):
    # min B-overlap is 4000 samples, min A-overlap is 4000 samples. Makes the
    # "straddles both words" assertion below deterministic, not seed-lucky.
    rng = np.random.default_rng(0)
    clips_by_word = {
        "A": [np.full(5000, 1.0, np.float32)],
        "B": [np.full(5000, 2.0, np.float32)],
    }
    n_pairs = 5
    unknown, positives = make_transition_windows(clips_by_word, rng, n_pairs, gap_ms=250)

    assert len(unknown) == n_pairs
    assert len(positives) == 2 * n_pairs

    center = config.CLIP_SAMPLES // 2
    for win in unknown:
        assert win.shape == (config.CLIP_SAMPLES,)
        assert win.dtype == np.float32
        # straddles the boundary: both words' audio present, neither alone
        assert (win == 1.0).any()
        assert (win == 2.0).any()

    for win, label in positives:
        assert win.shape == (config.CLIP_SAMPLES,)
        assert label in ("A", "B")
        expected = 1.0 if label == "A" else 2.0
        # window is cut centered exactly on the labeled word's clip-center sample
        assert win[center] == expected


def test_make_transition_windows_empty_when_no_clips():
    rng = np.random.default_rng(0)
    assert make_transition_windows({}, rng, n_pairs=3) == ([], [])
    assert make_transition_windows({"A": []}, rng, n_pairs=3) == ([], [])


def test_tts_engines_reads_env_override(monkeypatch):
    monkeypatch.setenv("KWS_TTS_ENGINES", "say, piper")
    assert tts_engines() == ["say", "piper"]


def test_tts_engines_defaults_to_available(monkeypatch):
    monkeypatch.delenv("KWS_TTS_ENGINES", raising=False)
    assert tts_engines() == tts.available_engines()


def test_tts_gate_min_tokens_for_content_opts_in_via_env(monkeypatch):
    monkeypatch.delenv("KWS_TTS_GATE", raising=False)
    assert tts_gate_min_tokens_for_content() == 0  # strict by default
    monkeypatch.setenv("KWS_TTS_GATE", "lenient")
    assert tts_gate_min_tokens_for_content() == 3
    monkeypatch.setenv("KWS_TTS_GATE", "0")  # gate disabled entirely: not "lenient"
    assert tts_gate_min_tokens_for_content() == 0


def test_tts_combo_plan_single_engine_matches_voice_combos():
    combos = _tts_combo_plan("Licht", 5, ["say"])
    assert len(combos) == 5
    assert all(e == "say" for e, _, _ in combos)


def test_tts_combo_plan_balances_across_engines_of_unequal_pool_size(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # empty piper cache -> static fallback
    # With an empty piper cache, say (9 voices) has a bigger voice/rate pool than piper's
    # static fallback (4 voices); requesting far more than either pool holds must still
    # come back with EQUAL counts per engine, not say-heavy.
    combos = _tts_combo_plan("Licht", 1000, ["say", "piper"])
    from collections import Counter

    counts = Counter(e for e, _, _ in combos)
    assert set(counts) == {"say", "piper"}
    assert counts["say"] == counts["piper"]


def test_tts_combo_plan_empty_engines():
    assert _tts_combo_plan("Licht", 10, []) == []


def test_tts_combo_plan_cycles_past_pool_exhaustion_still_balanced(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # empty piper cache -> static fallback
    # n bigger than the balanced pool must still return exactly n combos, evenly split.
    combos = _tts_combo_plan("Licht", 2000, ["say", "piper"])
    from collections import Counter

    counts = Counter(e for e, _, _ in combos)
    assert len(combos) == 2000
    assert counts["say"] == counts["piper"] == 1000


def test_split_three_way_is_speaker_disjoint_and_covers_all():
    rng = np.random.default_rng(7)
    # 20 speakers x 2 clips; tag each clip with its speaker's int so identity survives
    clips = {
        "Licht": [
            (np.full(config.CLIP_SAMPLES, s, np.float32), f"spk{s}")
            for s in range(20)
            for _ in range(2)
        ],
    }
    train, val, test = split_three_way(clips, rng, val_frac=0.2, test_frac=0.2)
    tr = {float(c[0]) for c in train["Licht"]}
    va = {float(c[0]) for c in val["Licht"]}
    te = {float(c[0]) for c in test["Licht"]}
    assert tr and va and te  # all three non-empty
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)  # disjoint speakers
    assert tr | va | te == set(float(s) for s in range(20))  # all speakers covered
    assert len(train["Licht"]) + len(val["Licht"]) + len(test["Licht"]) == 40


def test_split_three_way_no_leak_across_labels():
    # Same cross-label leak guard as split_by_speaker's, for the three-way split.
    rng = np.random.default_rng(11)
    clips = {
        "Licht": [(np.full(config.CLIP_SAMPLES, i, np.float32), f"spk{i}") for i in range(20)],
        "_unknown_": [(np.full(config.CLIP_SAMPLES, i, np.float32), f"spk{i}") for i in range(20)],
    }
    train, val, test = split_three_way(clips, rng, val_frac=0.2, test_frac=0.2, keep_speaker=True)
    train_spk = {s for label in clips for _, s in train[label]}
    val_spk = {s for label in clips for _, s in val[label]}
    test_spk = {s for label in clips for _, s in test[label]}
    assert train_spk.isdisjoint(val_spk)
    assert train_spk.isdisjoint(test_spk)
    assert val_spk.isdisjoint(test_spk)
    assert val_spk and test_spk


def test_build_dataset_includes_transition_windows():
    rng = np.random.default_rng(5)
    clips = {c: [_clip(rng) for _ in range(2)] for c in config.COMMANDS}
    clips["_unknown_"] = [_clip(rng) for _ in range(2)]
    noises = [rng.standard_normal(8000).astype(np.float32) for _ in range(2)]
    licht = config.label_index("Licht")
    unknown = config.label_index("_unknown_")

    trans_unknown = [_clip(rng) for _ in range(3)]
    trans_positive = [(_clip(rng), "Licht") for _ in range(2)]
    X, y = build_dataset(
        clips,
        noises,
        rng,
        snrs=(20,),
        transition_unknown=trans_unknown,
        transition_positives=trans_positive,
    )
    X0, y0 = build_dataset(clips, noises, rng, snrs=(20,))

    # per_clip = 1 + len(snrs) = 2 rows per transition window
    assert (y == licht).sum() == (y0 == licht).sum() + 2 * len(trans_positive)
    assert (y == unknown).sum() == (y0 == unknown).sum() + 2 * len(trans_unknown)
