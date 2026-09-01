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

# Per-engine voice pools (extend as more are installed). Kept as data so voice_combos is pure.
ENGINE_VOICES: dict[str, list[str]] = {
    "say": ["Anna", "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"],
    "piper": [
        "de_DE-thorsten",
        "de_DE-kerstin",
        "de_DE-eva_k",
        "de_DE-ramona",
        "de_DE-karlsson",
        "de_DE-pavoque",
        "de_DE-mls",
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
RATES = [150, 180, 210, 240]  # words-per-minute style variation (engine maps as it can)


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


def _piper_say(word, voice, out_wav):  # pragma: no cover
    raise NotImplementedError("wire Piper (piper-tts): load de_DE voice, synth 'word' @16kHz")


def _parler_say(word, description, out_wav):  # pragma: no cover
    raise NotImplementedError("wire Parler-TTS: generate 'word' with the voice description @16kHz")


def _xtts_say(word, ref_speaker, out_wav):  # pragma: no cover
    raise NotImplementedError("wire XTTS-v2: clone ref_speaker, synth 'word' @16kHz")
