from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.recordings import centre, load_recordings


def test_centre_pads_short_symmetrically():
    out = centre(np.ones(100, np.float32), n=1000)
    assert out.shape == (1000,)
    assert out[450:550].sum() == 100 and out[:450].sum() == 0 and out[550:].sum() == 0


def test_centre_crops_long_symmetrically():
    sig = np.arange(1000, dtype=np.float32)
    out = centre(sig, n=100)
    assert out.shape == (100,) and out[0] == 450 and out[-1] == 549


def test_load_recordings_speaker_prefix_and_unknown_folders_ignored(tmp_path: Path):
    d = tmp_path / "Aufstelldach"
    d.mkdir()
    t = np.arange(int(0.4 * 16000)) / 16000
    tone = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    padded = np.concatenate([np.zeros(4000, np.float32), tone, np.zeros(4000, np.float32)])
    sf.write(d / "alice_1.wav", padded, 16000)
    sf.write(d / "bob_1.wav", padded, 16000)
    (tmp_path / "Unrelated").mkdir()
    sf.write(tmp_path / "Unrelated" / "x_1.wav", padded, 16000)

    clips = load_recordings(tmp_path, ["Aufstelldach", "Licht"])
    assert sorted(spk for _, spk in clips["Aufstelldach"]) == ["rec:alice", "rec:bob"]
    assert clips["Licht"] == []
    assert "Unrelated" not in clips
    for clip, _ in clips["Aufstelldach"]:
        assert clip.shape == (config.CLIP_SAMPLES,) and clip.dtype == np.float32
        # trimmed+centred: energy sits in the middle, not at the start
        assert np.abs(clip[:2000]).max() < 1e-3 and np.abs(clip[7000:9000]).max() > 0.1
