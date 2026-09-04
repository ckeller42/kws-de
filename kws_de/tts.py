"""Multi-engine German TTS for generating diverse KWS training clips.

For keyword spotting, the single biggest quality lever on synthetic data is **voice
diversity** — a model trained on one synth voice memorizes that voice instead of learning
the word. So we stack several offline German TTS engines and rotate across their voices:

- ``say``    — macOS `say` (~9 German voices). Always available on macOS. Permissive.
- ``piper``  — Piper neural voices (cache-discovered, multi-speaker aware). Permissive.
- ``parler`` — Parler-TTS, voice from a text description (Apache-2.0). Many voices on demand.
- ``xtts``   — Coqui XTTS-v2 zero-shot cloning (non-commercial) — clone many reference speakers.

Each engine is OPTIONAL: it is used only if its backend imports. The actual synthesis calls
are integration code (``# pragma: no cover``); the engine-selection / diversity logic is pure
and unit-tested. Generated audio is 16 kHz mono float32, gitignored training data.
"""

from __future__ import annotations

import csv
import functools
import itertools
import json
import re
import subprocess
import threading
from pathlib import Path

# Every synthesis appends a row here, next to the clips it wrote: what the clip was meant
# to say, and which voice said it. Without it nothing downstream can tell whether a wav
# holds its intended text — `kws-tts-check` reads this back and judges each clip against
# its row (kws_de.qc.tts_check).
MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = ["file", "text", "voice", "engine"]
_MANIFEST_LOCK = threading.Lock()  # batch generators synthesize from a thread pool

# Per-engine voice pools (extend as more are installed). Kept as data so voice_combos is pure.
ENGINE_VOICES: dict[str, list[str]] = {
    # Backstop only. At runtime `engine_voices("say")` asks `say -v '?'` which German
    # voices are really installed and uses their FULL names — these bare ones resolve to
    # the English voice of the same name on a machine that has both.
    "say": ["Anna", "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"],
    # Fallback when data/piper-voices/ is empty (fresh checkout, CI). At runtime
    # `engine_voices("piper")` discovers whatever is cached there instead — see
    # README "TTS backstop voices" for how to add one.
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
RATES = [120, 140, 160, 180, 200, 220, 240, 260, 280]
# wpm; say maps directly, piper via length_scale = 160/rate

PIPER_BASE_RATE = 160  # wpm that maps to Piper length_scale 1.0


def piper_voices(root: Path) -> list[str]:
    """Piper voice ids cached under ``root`` (``<name>/<quality>/<id>.onnx`` + ``.onnx.json``),
    sorted. A multi-speaker voice (``num_speakers > 1`` in its json) expands to one id per
    speaker, ``<id>#<speaker_id>``, so every speaker is its own voice for the split. Pure
    directory scan; empty when nothing is cached."""
    out: list[str] = []
    for onnx in root.glob("*/*/*.onnx"):
        meta = onnx.with_suffix(".onnx.json")
        n = json.loads(meta.read_text()).get("num_speakers", 1) if meta.exists() else 1
        out.extend([onnx.stem] if n <= 1 else [f"{onnx.stem}#{i}" for i in range(n)])
    return sorted(out)


def de_say_voices(listing: str) -> list[str]:
    """German (``de_DE``) voice names out of ``say -v '?'`` output, VERBATIM — including the
    parenthesised suffix macOS appends to a name that exists in several languages ("Eddy
    (German (Germany))"). Pure parse.

    The bare name is not enough and that is the whole point: ``say -v Eddy`` picks the
    ENGLISH Eddy on a machine that has both, and says so nowhere — measured, it produced
    the same audio as ``-v Samantha`` for German text. A voice that is not installed at
    all falls back to the system default just as silently."""
    out = []
    for line in listing.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if m and m.group(2) == "de_DE":
            out.append(m.group(1).strip())
    return out


@functools.lru_cache(maxsize=1)
def say_voices() -> list[str]:  # pragma: no cover - shells out to macOS `say`
    """German ``say`` voices really installed here (empty off macOS). Memoized: the voice
    pool is asked for once per word."""
    try:
        r = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=10, check=True
        )
    except Exception:
        return []
    return de_say_voices(r.stdout)


def engine_voices(engine: str) -> list[str]:
    """Voices to draw from for ``engine``. Piper: whatever is cached on disk; ``say``:
    the German voices actually installed on this machine; both fall back to
    ``ENGINE_VOICES`` when discovery comes up empty. Other engines: the static list."""
    if engine == "piper":
        from kws_de import config

        return piper_voices(config.DATA_DIR / "piper-voices") or ENGINE_VOICES["piper"]
    if engine == "say":
        return say_voices() or ENGINE_VOICES["say"]
    return ENGINE_VOICES.get(engine) or ["default"]


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
    given engines so the set spans as many engines/voices as possible before repeating. No
    synthesis backend is touched, but for ``"piper"`` this reads the Piper voice cache under
    ``config.DATA_DIR`` (via ``engine_voices``), so its result depends on what is cached
    locally."""
    per_engine = []
    for e in engines:
        voices = engine_voices(e)
        per_engine.append([(e, v, r) for r, v in itertools.product(RATES, voices)])
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


def append_manifest(out_wav: Path, text: str, voice: str, engine: str) -> None:
    """Append one row to ``manifest.csv`` in ``out_wav``'s directory. Thread-safe (append
    under a lock) because generators synthesize in parallel."""
    out_wav = Path(out_wav)
    man = out_wav.parent / MANIFEST_NAME
    with _MANIFEST_LOCK:
        new = not man.exists()
        with man.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"file": out_wav.name, "text": text, "voice": voice, "engine": engine})


def synthesize(word: str, engine: str, voice: str, rate: int, out_wav):  # pragma: no cover
    """Dispatch one synthesis to the chosen engine → 16 kHz mono float32, or None on failure.
    Integration code (shells out / loads heavy models); see each engine's backend.

    The wav is LEFT at ``out_wav`` and a ``manifest.csv`` row is written beside it: a clip
    nobody can check is a clip nobody can trust, and the check (`kws-tts-check`) needs both
    the file and the text it was supposed to say. Callers that only want the samples own
    the cleanup — delete the file (and the manifest) when done with it."""
    if engine == "say":
        from kws_de.data import _say_one

        r = _say_one(word, voice, rate, "{w}", out_wav)
        audio = None if r is None else r[0]
    elif engine == "piper":
        audio = _piper_say(word, voice, rate, out_wav)
    elif engine == "parler":
        audio = _parler_say(word, voice, out_wav)
    elif engine == "xtts":
        audio = _xtts_say(word, voice, out_wav)
    else:
        return None
    if audio is not None:
        append_manifest(Path(out_wav), word, voice, engine)
    return audio


def _piper_voice_path(voice: str) -> Path:
    """Resolve a Piper voice id (e.g. ``de_DE-thorsten-medium``) to its ``.onnx`` model
    path in the local voice cache, mirroring the rhasspy/piper-voices HuggingFace layout:
    ``data/piper-voices/<name>/<quality>/<locale>-<name>-<quality>.onnx``. Pure string/path
    logic — no filesystem access, so it's unit-testable without the voice files present."""
    from kws_de import config

    voice = voice.split("#", 1)[0]  # "<id>#<speaker_id>" -> the shared model file
    _locale, rest = voice.split("-", 1)
    name, quality = rest.rsplit("-", 1)
    return config.DATA_DIR / "piper-voices" / name / quality / f"{voice}.onnx"


_PIPER_VOICE_CACHE: dict[str, object] = {}  # voice id -> loaded piper.PiperVoice (memoized)


def _piper_load_voice(voice: str):  # pragma: no cover - loads an onnx model
    """Load (and memoize) a Piper voice model. Raises FileNotFoundError if it hasn't been
    downloaded into data/piper-voices/ yet."""
    voice = voice.split("#", 1)[0]
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


def _piper_say(word, voice, rate, out_wav):  # pragma: no cover - loads model, runs onnxruntime
    import numpy as np
    import soundfile as sf
    from piper import SynthesisConfig

    from kws_de import config

    base, _, sid = voice.partition("#")
    cfg = SynthesisConfig(speaker_id=int(sid) if sid else None, length_scale=PIPER_BASE_RATE / rate)
    try:
        pv = _piper_load_voice(base)
        chunks = list(pv.synthesize(word, cfg))
        if not chunks:
            return None
        sr = chunks[0].sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        if sr != config.SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        audio = audio.astype(np.float32)
        sf.write(str(out_wav), audio, config.SAMPLE_RATE)  # left on disk for the gate
    except Exception:
        return None
    return audio


def _parler_say(word, description, out_wav):  # pragma: no cover
    raise NotImplementedError("wire Parler-TTS: generate 'word' with the voice description @16kHz")


def _xtts_say(word, ref_speaker, out_wav):  # pragma: no cover
    raise NotImplementedError("wire XTTS-v2: clone ref_speaker, synth 'word' @16kHz")
