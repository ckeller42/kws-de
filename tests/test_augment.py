import numpy as np

from kws_de.augment import measure_snr, mix_at_snr


def test_mix_hits_target_snr():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000).astype(np.float32)
    noise = rng.standard_normal(4000).astype(np.float32)  # shorter -> must tile
    for target in (20.0, 10.0, 0.0):
        noisy = mix_at_snr(sig, noise, target, rng)
        assert noisy.shape == sig.shape
        assert abs(measure_snr(sig, noisy) - target) < 0.5


def test_zero_signal_is_safe():
    rng = np.random.default_rng(1)
    sig = np.zeros(16000, dtype=np.float32)
    noise = rng.standard_normal(16000).astype(np.float32)
    out = mix_at_snr(sig, noise, 10.0, rng)
    assert out.shape == sig.shape and np.all(np.isfinite(out))
