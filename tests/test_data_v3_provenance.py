import numpy as np
import soundfile as sf

from kws_de import config, data


def test_recordings_root_prefers_approved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert data.recordings_root() == tmp_path / "recordings"
    (tmp_path / "recordings" / "approved" / "words").mkdir(parents=True)
    assert data.recordings_root() == tmp_path / "recordings" / "approved" / "words"


def test_negative_windows_hop_one_second(tmp_path):
    neg = tmp_path / "negatives" / "spk02"
    neg.mkdir(parents=True)
    sf.write(
        neg / "hallo_001.wav",
        np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.1,
        16000,
        subtype="PCM_16",
    )
    wins = data.negative_windows(tmp_path / "negatives")
    assert (
        len(wins) == 3
        and all(w.shape == (config.CLIP_SAMPLES,) for w, _ in wins)
        and wins[0][1] == "rec:spk02"
    )


def test_negative_windows_warns_and_skips_wrong_sample_rate(tmp_path, caplog):
    neg = tmp_path / "negatives" / "spk03"
    neg.mkdir(parents=True)
    sf.write(
        neg / "hallo_001.wav",
        np.random.default_rng(0).standard_normal(8000 * 3).astype(np.float32) * 0.1,
        8000,  # not config.SAMPLE_RATE
        subtype="PCM_16",
    )
    with caplog.at_level("WARNING"):
        wins = data.negative_windows(tmp_path / "negatives")
    assert wins == []
    assert "sample rate" in caplog.text


def test_negative_windows_warns_and_skips_too_short_file(tmp_path, caplog):
    neg = tmp_path / "negatives" / "spk04"
    neg.mkdir(parents=True)
    sf.write(
        neg / "hallo_001.wav",
        np.random.default_rng(0).standard_normal(8000).astype(np.float32) * 0.1,  # 0.5 s < 1 window
        16000,
        subtype="PCM_16",
    )
    with caplog.at_level("WARNING"):
        wins = data.negative_windows(tmp_path / "negatives")
    assert wins == []  # no all-zero window emitted
    assert "shorter than one window" in caplog.text


def _tone(ms=800, sr=16000):
    t = np.arange(int(sr * ms / 1000)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _approved_tree(root):
    """approved/words/Licht/spk09_001.wav + approved/negatives/spk09/hallo_001.wav."""
    w = root / "approved" / "words" / "Licht"
    w.mkdir(parents=True)
    sf.write(w / "spk09_001.wav", _tone(), 16000, subtype="PCM_16")
    n = root / "approved" / "negatives" / "spk09"
    n.mkdir(parents=True)
    sf.write(n / "hallo_001.wav", _tone(ms=2000), 16000, subtype="PCM_16")


def test_merge_recordings_folds_approved_in_and_never_duplicates(tmp_path):
    """A cache-based build must see recordings approved AFTER the cache was written
    (the loop's whole promise), and a second build must replace them, not stack them."""
    rec = tmp_path / "recordings"
    _approved_tree(rec)
    clips_ws = {"Licht": [(np.zeros(config.CLIP_SAMPLES, np.float32), "mswc:x")], "_unknown_": []}

    merged = data.merge_recordings(clips_ws, rec)
    assert merged["Licht"] == 1 and merged["_unknown_"] == 2
    assert "rec:spk09" in {s for _, s in clips_ws["Licht"]}
    assert {s for _, s in clips_ws["_unknown_"]} == {"rec:spk09"}

    sizes = {k: len(v) for k, v in clips_ws.items()}
    data.merge_recordings(clips_ws, rec)
    assert {k: len(v) for k, v in clips_ws.items()} == sizes


def test_merge_recordings_no_approved_tree_is_a_noop(tmp_path):
    clips_ws = {"Licht": [(np.zeros(config.CLIP_SAMPLES, np.float32), "rec:spk09")]}
    assert data.merge_recordings(clips_ws, tmp_path / "recordings") == {}
    assert clips_ws["Licht"][0][1] == "rec:spk09"  # legacy rec: clips left alone


def test_force_rec_to_train_moves_device_clips_out_of_val_and_test():
    from kws_de.dataset import force_rec_to_train

    c = np.zeros(config.CLIP_SAMPLES, np.float32)
    train = {"Licht": [(c, "mswc:a")]}
    val = {"Licht": [(c, "rec:spk09"), (c, "mswc:b")]}
    test = {"Licht": [(c, "rec:spk09")]}
    assert force_rec_to_train(train, val, test) == 2
    assert [s for _, s in train["Licht"]] == ["mswc:a", "rec:spk09", "rec:spk09"]
    assert [s for _, s in val["Licht"]] == ["mswc:b"]
    assert test["Licht"] == []


def test_build_merges_recordings_and_trains_on_device_speakers(tmp_path, monkeypatch):
    """End to end over `dataset.build`: cache with no rec: clips + an approved tree ->
    the manifest's TRAIN split carries the device speaker and counts it as a recording."""
    import pickle

    from kws_de import dataset

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DEVICES", ["Licht"])
    monkeypatch.setattr(config, "ZONES", [])
    monkeypatch.setattr(config, "ACTIONS", [])
    monkeypatch.setattr(config, "COMMAND_LABELS", ["Licht", "_unknown_", "_silence_"])
    monkeypatch.setattr(dataset, "_fill_with_tts", lambda clips, words: {})

    rng = np.random.default_rng(0)

    def clip():
        return rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)

    cached = {
        "clips": {
            "Licht": [(clip(), f"mswc:{i}") for i in range(6)],
            "_unknown_": [(clip(), f"mswc:{i}") for i in range(6)],
        }
    }
    with open(tmp_path / "raw_clips_v3.pkl", "wb") as fh:
        pickle.dump(cached, fh)
    with open(tmp_path / "noise.pkl", "wb") as fh:
        pickle.dump([rng.standard_normal(8000).astype(np.float32)], fh)
    _approved_tree(tmp_path / "recordings")

    m = dataset.build(seed=0, cache_name="raw_clips_v3.pkl", out_prefix="features_v3")
    assert m["splits"]["train"]["speakers"] == ["spk09"]
    assert m["splits"]["train"]["sources"]["recording"] == 3  # 1 word + 2 negative windows
    assert m["splits"]["val"]["speakers"] == [] and m["splits"]["test"]["speakers"] == []


def test_manifest_records_sources_and_speakers():
    from kws_de.dataset import build_manifest

    X = np.zeros((4, config.N_FRAMES, config.N_MFCC), np.float32)
    y = np.array([0, 1, 0, 1])
    is_tts = np.array([True, False, False, False])
    speakers = ["tts:a", "rec:spk02", "mswc:x", "rec:spk03"]
    m = build_manifest(
        {"train": (X, y, is_tts)},
        seed=0,
        labels=config.COMMAND_LABELS,
        speakers={"train": speakers},
    )
    assert m["splits"]["train"]["sources"] == {"tts": 1, "recording": 2, "mswc": 1}
    assert m["splits"]["train"]["speakers"] == ["spk02", "spk03"]  # numeric ids only, sorted
