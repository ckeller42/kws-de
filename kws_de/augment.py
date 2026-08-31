import numpy as np


def _tile_to(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if noise.shape[0] < n:
        reps = int(np.ceil(n / noise.shape[0]))
        noise = np.tile(noise, reps)
    start = 0 if noise.shape[0] == n else int(rng.integers(0, noise.shape[0] - n + 1))
    return noise[start : start + n]


def mix_at_snr(signal, noise, snr_db, rng):
    sig = np.asarray(signal, dtype=np.float32)
    nz = _tile_to(np.asarray(noise, dtype=np.float32), sig.shape[0], rng)
    p_sig = float(np.mean(sig**2))
    p_nz = float(np.mean(nz**2)) or 1e-12
    if p_sig <= 1e-12:  # silence: return scaled noise at a fixed level
        return (nz / np.sqrt(p_nz) * 0.01).astype(np.float32)
    gain = np.sqrt(p_sig / (p_nz * (10 ** (snr_db / 10))))
    return (sig + gain * nz).astype(np.float32)


def measure_snr(signal, noisy):
    sig = np.asarray(signal, dtype=np.float32)
    nz = np.asarray(noisy, dtype=np.float32) - sig
    p_sig = float(np.mean(sig**2))
    p_nz = float(np.mean(nz**2)) or 1e-12
    return 10 * np.log10(p_sig / p_nz)
