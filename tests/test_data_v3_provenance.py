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
    wins = data.negative_windows(tmp_path / "negatives", np.random.default_rng(0))
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
        wins = data.negative_windows(tmp_path / "negatives", np.random.default_rng(0))
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
        wins = data.negative_windows(tmp_path / "negatives", np.random.default_rng(0))
    assert wins == []  # no all-zero window emitted
    assert "shorter than one window" in caplog.text


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
