import numpy as np

from kws_de.augment import measure_snr, mix_at_snr, perturb


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


def _peak_hz(sig, sr=16000):
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    return float(np.fft.rfftfreq(len(sig), 1 / sr)[np.argmax(spec)])


def test_perturb_shifts_pitch_and_stretches_time():
    sr = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    out = perturb(tone, n_steps=12.0, rate=1.25, sr=sr)
    assert out.dtype == np.float32
    assert abs(len(out) - sr / 1.25) < 0.02 * sr  # 1.25x faster -> ~80 % of the samples
    assert abs(_peak_hz(out) - 880) < 20  # +12 semitones = one octave
    same = perturb(tone, n_steps=0.0, rate=1.0, sr=sr)
    assert len(same) == sr and abs(_peak_hz(same) - 440) < 5
