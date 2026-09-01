"""Multi-engine German TTS for generating diverse KWS training clips.

For keyword spotting, the single biggest quality lever on synthetic data is **voice
diversity** — a model trained on one synth voice memorizes that voice instead of learning
the word. So we stack several offline German TTS engines and rotate across their voices:

- ``say``    — macOS `say` (~9 German voices). Always available on macOS. Permissive.
- ``piper``  — Piper neural voices (thorsten, kerstin, eva_k, ramona, karlsson, …). Permissive.
- ``parler`` — Parler-TTS, voice from a text description (Apache-2.0). Many voices on demand.
- ``xtts``   — Coqui XTTS-v2 zero-shot cloning (non-commercial) — clone many reference speakers.

Each engine is OPTIONAL: it is used only if its backend imports. The actual synthesis calls
are integration code (``# pragma: no cover``); the engine-selection / diversity logic is pure
and unit-tested. Generated audio is 16 kHz mono float32, gitignored training data.
"""

from __future__ import annotations

import itertools
from pathlib import Path

# Per-engine voice pools (extend as more are installed). Kept as data so voice_combos is pure.
ENGINE_VOICES: dict[str, list[str]] = {
    "say": ["Anna", "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"],
    # Voices actually cached under data/piper-voices/ (see _piper_voice_path) — full
    # rhasspy/piper-voices ids "<locale>-<name>-<quality>". To add more, download the
    # matching .onnx + .onnx.json pair from huggingface.co/rhasspy/piper-voices into
    # data/piper-voices/<name>/<quality>/ and list the id here.
    "piper": [
        "de_DE-thorsten-medium",
        "de_DE-eva_k-x_low",
        "de_DE-ramona-low",
        "de_DE-karlsson-low",
    ],
    "parler": [
        "a calm woman",
        "an energetic young man",
        "an older man, deep voice",
        "a cheerful woman, fast",
        "a neutral male narrator",
    ],
    "xtts": [],  # filled at runtime from reference speaker clips (e.g. Common Voice DE)
}
RATES = [120, 140, 160, 180, 200, 220, 240, 260, 280]  # wpm; say maps directly, piper ignores it
# but still benefits — Piper's default noise_scale makes every synthesis call stochastic
# (verified: same voice+rate produces different audio each call), so more (voice, rate)
# labels means more distinct clips even though rate itself isn't wired into Piper's config.


def available_engines() -> list[str]:  # pragma: no cover - probes optional backends
    """Which TTS engines can actually run here. ``say`` is assumed on macOS; the rest are
    included only if their Python backend imports."""
    import importlib.util
    import sys

    engines = []
    if sys.platform == "darwin":
        engines.append("say")
    for name, mod in (("piper", "piper"), ("parler", "parler_tts"), ("xtts", "TTS")):
        if importlib.util.find_spec(mod) is not None:
            engines.append(name)
    return engines


def voice_combos(n: int, engines: list[str]) -> list[tuple[str, str, int]]:
    """Build up to ``n`` diverse ``(engine, voice, rate)`` combos, ROUND-ROBIN across the
    given engines so the set spans as many engines/voices as possible before repeating.
    Pure — no backends touched."""
    per_engine = []
    for e in engines:
        voices = ENGINE_VOICES.get(e) or ["default"]
        per_engine.append([(e, v, r) for v, r in itertools.product(voices, RATES)])
    combos: list[tuple[str, str, int]] = []
    # interleave engines: take one from each in turn until we have n
    idx = 0
    while len(combos) < n and any(idx < len(pe) for pe in per_engine):
        for pe in per_engine:
            if idx < len(pe):
                combos.append(pe[idx])
                if len(combos) >= n:
                    break
        idx += 1
    return combos[:n]


def synthesize(word: str, engine: str, voice: str, rate: int, out_wav):  # pragma: no cover
    """Dispatch one synthesis to the chosen engine → 16 kHz mono float32, or None on failure.
    Integration code (shells out / loads heavy models); see each engine's backend."""
    if engine == "say":
        from kws_de.data import _say_one

        r = _say_one(word, voice, rate, "{w}", out_wav)
        return None if r is None else r[0]
    if engine == "piper":
        return _piper_say(word, voice, out_wav)
    if engine == "parler":
        return _parler_say(word, voice, out_wav)
    if engine == "xtts":
        return _xtts_say(word, voice, out_wav)
    return None


def _piper_voice_path(voice: str) -> Path:
    """Resolve a Piper voice id (e.g. ``de_DE-thorsten-medium``) to its ``.onnx`` model
    path in the local voice cache, mirroring the rhasspy/piper-voices HuggingFace layout:
    ``data/piper-voices/<name>/<quality>/<locale>-<name>-<quality>.onnx``. Pure string/path
    logic — no filesystem access, so it's unit-testable without the voice files present."""
    from kws_de import config

    _locale, rest = voice.split("-", 1)
    name, quality = rest.rsplit("-", 1)
    return config.DATA_DIR / "piper-voices" / name / quality / f"{voice}.onnx"


_PIPER_VOICE_CACHE: dict[str, object] = {}  # voice id -> loaded piper.PiperVoice (memoized)


def _piper_load_voice(voice: str):  # pragma: no cover - loads an onnx model
    """Load (and memoize) a Piper voice model. Raises FileNotFoundError if it hasn't been
    downloaded into data/piper-voices/ yet."""
    if voice not in _PIPER_VOICE_CACHE:
        from piper import PiperVoice

        model_path = _piper_voice_path(voice)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice {voice!r} not cached at {model_path} — download the "
                "matching .onnx + .onnx.json from huggingface.co/rhasspy/piper-voices"
            )
        _PIPER_VOICE_CACHE[voice] = PiperVoice.load(model_path)
    return _PIPER_VOICE_CACHE[voice]


def _piper_say(word, voice, out_wav):  # pragma: no cover - loads model, shells out to onnxruntime
    import numpy as np
    import soundfile as sf

    from kws_de import config

    try:
        pv = _piper_load_voice(voice)
        chunks = list(pv.synthesize(word))
        if not chunks:
            return None
        sr = chunks[0].sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        if sr != config.SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        audio = audio.astype(np.float32)
        sf.write(str(out_wav), audio, config.SAMPLE_RATE)
        out_wav.unlink()
    except Exception:
        return None
    return audio


def _parler_say(word, description, out_wav):  # pragma: no cover
    raise NotImplementedError("wire Parler-TTS: generate 'word' with the voice description @16kHz")


def _xtts_say(word, ref_speaker, out_wav):  # pragma: no cover
    raise NotImplementedError("wire XTTS-v2: clone ref_speaker, synth 'word' @16kHz")
