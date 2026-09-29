import pickle

import numpy as np

from kws_de import config, dataset
from kws_de.dataset import assemble, load_split


def _clip(rng):
    return rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)


def test_assemble_returns_features_labels_and_origin():
    rng = np.random.default_rng(0)
    clips_ws = {
        config.COMMANDS[0]: [(_clip(rng), "real1"), (_clip(rng), "tts:say:Anna:180")],
        "_unknown_": [(_clip(rng), "real2")],
    }
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y, is_tts = assemble(clips_ws, noises, rng, labels=config.LABELS, commands=config.COMMANDS)
    assert X.shape[1:] == (config.N_FRAMES, config.N_MFCC)
    assert len(X) == len(y) == len(is_tts)
    assert is_tts.dtype == bool and is_tts.any()  # the tts:* speaker rows are flagged


def test_assemble_rows_match_origin_flags_with_tts():
    rng = np.random.default_rng(0)
    clips_ws = {
        config.COMMANDS[0]: [(_clip(rng), "real1"), (_clip(rng), "tts:say:Anna")],
        "_unknown_": [(_clip(rng), "real2")],
    }
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y, is_tts = assemble(clips_ws, noises, rng, labels=config.LABELS, commands=config.COMMANDS)
    assert len(X) == len(is_tts)
    assert is_tts.sum() == 2 * (1 + 3)  # the TTS clip and its perturbed copy, clean + 3 snrs


def test_assemble_flags_clone_rows_synthetic_and_aligned(monkeypatch, tmp_path):
    # E64: a `clone:` clip is perturbed like TTS (8 rows), so its flags must be 8 x True --
    # with van augmentation on, a "real" reading gives 7 x False and shifts every later row.
    import soundfile as sf

    rng = np.random.default_rng(0)
    for name in ("noise", "rir"):
        d = tmp_path / name
        d.mkdir()
        sf.write(d / "a.wav", rng.standard_normal(16000).astype(np.float32) * 0.1, 16000)
        monkeypatch.setenv(f"KWS_{name.upper()}_DIR", str(d))
    clips_ws = {
        config.COMMANDS[0]: [(_clip(rng), "clone:spk01"), (_clip(rng), "rec:spk01")],
        "_unknown_": [(_clip(rng), "real2")],
    }
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y, is_tts = assemble(clips_ws, noises, rng, labels=config.LABELS, commands=config.COMMANDS)
    assert len(X) == len(y) == len(is_tts)
    assert is_tts[:8].all() and is_tts.sum() == 8  # the clone + its perturbed copy, nothing else


def test_split_for_build_keeps_clones_out_of_the_split_draw():
    # E64: with clones in the draw, three extra speaker ids re-deal everyone else's split.
    rng = np.random.default_rng(0)
    word = config.COMMANDS[0]
    base = {
        word: [(_clip(rng), f"mswc{i}") for i in range(20)] + [(_clip(rng), "rec:spk01")],
        "_unknown_": [(_clip(rng), f"tts:say:v{i}") for i in range(20)],
    }
    clones = [(_clip(rng), f"clone:spk{i:02d}") for i in (1, 2, 22)]
    with_clones = {**base, word: base[word] + clones}

    def ids(ws):
        return {lbl: [s for _, s in items] for lbl, items in ws.items()}

    *splits0, clones0 = dataset.split_for_build(base, seed=0)
    *splits1, clones1 = dataset.split_for_build(with_clones, seed=0)
    assert [ids(s) for s in splits1] == [ids(s) for s in splits0]
    assert clones0 == {} and ids(clones1) == {word: ["clone:spk01", "clone:spk02", "clone:spk22"]}


def test_assemble_clones_gives_word_rows_only_all_synthetic():
    rng = np.random.default_rng(0)
    word = config.COMMANDS[0]
    clones = {word: [(_clip(rng), "clone:spk01"), (_clip(rng), "clone:spk02")]}
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y, is_tts = dataset.assemble_clones(
        clones, noises, rng, labels=config.LABELS, commands=config.COMMANDS
    )
    assert len(X) == len(y) == len(is_tts) == 2 * 2 * (1 + 3)  # clip + perturbed copy each
    assert is_tts.all() and set(y) == {config.LABELS.index(word)}  # no silence rows


def test_load_split_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    X = np.zeros((3, config.N_FRAMES, config.N_MFCC), np.float32)
    y = np.array([0, 1, 2], np.int64)
    is_tts = np.array([True, False, True])
    np.savez(tmp_path / "features_val.npz", X=X, y=y, is_tts=is_tts)
    Xl, yl, tl = load_split("val")
    assert Xl.shape == X.shape and list(yl) == [0, 1, 2] and list(tl) == [True, False, True]


def test_build_persists_tts_filled_clips_to_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DEVICES", ["Licht"])
    monkeypatch.setattr(config, "ZONES", [])
    monkeypatch.setattr(config, "ACTIONS", [])
    monkeypatch.setattr(config, "COMMAND_LABELS", ["Licht", "_unknown_", "_silence_"])

    rng = np.random.default_rng(0)
    cached = {"clips": {"Licht": [(_clip(rng), "real1")]}}
    with open(tmp_path / "raw_clips_merged.pkl", "wb") as fh:
        pickle.dump(cached, fh)
    with open(tmp_path / "noise.pkl", "wb") as fh:
        pickle.dump([rng.standard_normal(8000).astype(np.float32)], fh)

    def fake_fill_with_tts(clips, words):
        clips["Licht"].append((_clip(rng), "tts:say:Anna"))
        return {"Licht": 1}

    monkeypatch.setattr(dataset, "_fill_with_tts", fake_fill_with_tts)

    dataset.build(seed=0)

    with open(tmp_path / "raw_clips_merged.pkl", "rb") as fh:  # noqa: S301
        saved = pickle.load(fh)
    speakers = [spk for _, spk in saved["clips"]["Licht"]]
    assert "tts:say:Anna" in speakers


def test_load_split_prefix_selects_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    np.savez(
        tmp_path / "features_v3_test.npz",
        X=np.zeros((2, 49, 10), np.float32),
        y=np.array([1, 2]),
        is_tts=np.array([False, True]),
    )
    X, y, is_tts = dataset.load_split("test", prefix="features_v3")
    assert X.shape == (2, 49, 10) and list(y) == [1, 2] and list(is_tts) == [False, True]
