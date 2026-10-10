"""E71 `cnn` arm: the deployed 23-class command model (86b7105e, the harness `baseline()`
path) behind the E69 matcher interface, so it gets an open-set number on the same protocol.
Templates are ignored -- the classifier enrolls nothing.

Pool W (template labels are command words): posteriors over trailing 1 s windows
every 100 ms (`_stream_posteriors`), per command word the max over windows; distance =
1 - that prob, all 21 command words ranked (`_unknown_`/`_silence_` never a label).
Pool P (template labels are intents): the device's assist-window decode (`decode_window`, first_ms =
CLIP_MS, as `baseline()`) -> intent text; distance = 1 - the lowest raw prob among the
fired non-`_unknown_` words. No intent -> [("<reject>", inf)] (never accepted, never a FA).

Self-check: uv run --no-sync python scripts/e69_cnn.py
"""

import math
import os
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

from kws_de import config
from kws_de.eval import _stream_posteriors, intent_text
from kws_de.grammar import Intent
from kws_de.stream import KeywordStream
from kws_de.window_intent import WindowBuffers, in_wake_tail, rescore, runner_up

LABELS = config.COMMAND_LABELS
WORDS = [i for i, lab in enumerate(LABELS) if lab not in ("_unknown_", "_silence_")]
STEP = config.SAMPLE_RATE // 10
REJECT = [("<reject>", math.inf)]
_fn = None


def _predict():
    global _fn
    if _fn is None:  # lazily per worker: tflite is not fork-safe
        from kws_de.codegen import model_bytes
        from kws_de.eval import make_command_predict_fn

        header = Path(__file__).resolve().parents[1] / "firmware/main/gen/model_data.h"
        # one thread per interpreter: the harness runs --jobs processes side by side
        with mock.patch.object(os, "cpu_count", return_value=1):
            _fn = make_command_predict_fn(model_bytes(header))
    return _fn


def decode(steps) -> tuple[object, float]:
    """`decode_window(steps, LABELS, 100, first_ms=CLIP_MS)`, plus the min fired-word prob."""
    ks, buf, ps = KeywordStream(None, LABELS), WindowBuffers(), []
    for k, post in enumerate(steps):
        for fired in ks.push(post):
            if in_wake_tail(config.CLIP_MS + k * 100, fired):
                continue
            p = float(np.asarray(post)[LABELS.index(fired)])
            if buf.add(fired, p, *runner_up(ks.last_smoothed, LABELS, fired)) and (
                fired != "_unknown_"
            ):
                ps.append(p)
    return rescore(buf.intent, buf.seconds)[0], min(ps, default=0.0)


def match_words(sig: np.ndarray) -> list[tuple[str, float]]:
    best = np.max(_stream_posteriors(_predict(), sig, STEP), axis=0)
    return sorted(((LABELS[i], 1.0 - float(best[i])) for i in WORDS), key=lambda x: x[1])


def match_intent(sig: np.ndarray) -> list[tuple[str, float]]:
    intent, p = decode(_stream_posteriors(_predict(), sig, STEP))
    return [(intent_text(intent), 1.0 - p)] if isinstance(intent, Intent) else REJECT


def is_word_pool(templates) -> bool:
    return set(templates) <= set(LABELS)


def make_matcher(
    templates: dict[str, list[np.ndarray]],
) -> Callable[[np.ndarray], list[tuple[str, float]]]:
    return match_words if is_word_pool(templates) else match_intent


if __name__ == "__main__":
    import soundfile as sf

    from kws_de.window_intent import decode_window

    rec = config.DATA_DIR / "recordings" / "approved"
    w = sf.read(sorted((rec / "context" / "Licht").glob("*.wav"))[0], dtype="float32")[0]
    r = match_words(w)
    assert len(r) == 21 and r[0][1] <= r[1][1] and 0 <= r[0][1] <= 1, r
    for f in sorted((rec / "phrases").rglob("*.wav"))[:20]:
        sig = sf.read(f, dtype="float32")[0]
        steps = _stream_posteriors(_predict(), sig, STEP)
        assert decode(steps)[0] == decode_window(steps, LABELS, 100, first_ms=config.CLIP_MS)
    print("e69_cnn selftest ok", r[:2])
