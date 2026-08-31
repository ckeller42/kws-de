import numpy as np
from kws_de import config
from kws_de.features import mfcc


def test_shape_and_dtype():
    x = np.zeros(config.CLIP_SAMPLES, dtype=np.float32)
    out = mfcc(x)
    assert out.shape == (config.N_FRAMES, config.N_MFCC)
    assert out.dtype == np.float32


def test_pads_and_truncates():
    short = np.ones(1000, dtype=np.float32)
    long = np.ones(config.CLIP_SAMPLES * 2, dtype=np.float32)
    assert mfcc(short).shape == (config.N_FRAMES, config.N_MFCC)
    assert mfcc(long).shape == (config.N_FRAMES, config.N_MFCC)


def test_deterministic_against_golden():
    # A fixed 440 Hz tone must always produce the same coefficients (host/device anchor).
    t = np.arange(config.CLIP_SAMPLES) / config.SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = mfcc(tone)
    golden = np.load("tests/fixtures/mfcc_golden.npz")["mfcc"]
    np.testing.assert_allclose(out, golden, rtol=1e-5, atol=1e-5)
