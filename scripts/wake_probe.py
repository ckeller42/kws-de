"""Score a wake model on a wav (or a directory of wavs) at the firmware's gate.

Usage: uv run python scripts/wake_probe.py <model.tflite> <wav|dir> [cutoff] [consecutive]
Prints one row per clip plus a summary line: how many clips fire, and the worst peak.
"""

import pathlib
import sys

import numpy as np
import soundfile as sf

from kws_de.wake import gate_fired, stream_wake_probs

model, target = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
consecutive = int(sys.argv[4]) if len(sys.argv) > 4 else 2

wavs = sorted(target.rglob("*.wav")) if target.is_dir() else [target]
fired_total, worst = 0, 0.0
for w in wavs:
    audio, sr = sf.read(w, dtype="float32")
    if sr != 16000:
        raise SystemExit(f"{w}: expected 16 kHz, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    pcm = np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16)
    probs = stream_wake_probs(model, pcm)
    fired = gate_fired(probs, cutoff, consecutive)
    fired_total += fired
    worst = max(worst, float(probs.max()))
    print(f"  {w.name:34} peak {probs.max():.3f}  fired={'Y' if fired else 'n'}")
print(f"fired {fired_total}/{len(wavs)} at {cutoff} x{consecutive}; worst peak {worst:.3f}")
