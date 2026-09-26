import pathlib
import re

import numpy as np

from kws_de import config, window_intent
from kws_de.grammar import Intent

FW = pathlib.Path(__file__).resolve().parent.parent / "firmware" / "main"


def _define(header: str, name: str) -> str:
    return re.search(rf"#define {name} (\S+)", (FW / header).read_text()).group(1)


def test_constants_match_the_firmware():
    # The reference is only a reference while its numbers are the device's.
    assert window_intent.WAKE_TAIL_MS == int(_define("assist_gate.h", "ASSIST_WAKE_TAIL_MS"))
    assert window_intent.RESCORE_FLOOR == float(
        _define("intent.h", "INTENT_RESCORE_FLOOR").rstrip("f")
    )
    rh = (FW / "recognise.h").read_text()
    for field, size in (
        ("window_intent", window_intent.INTENT_BYTES),
        ("window_words", window_intent.WORDS_BYTES),
        ("window_seconds", window_intent.SECONDS_BYTES),
    ):
        assert re.search(rf"char {field}\[{size}\];", rh), field
    gate = (FW / "assist_gate.h").read_text()
    for lab in window_intent.WAKE_TAIL_LABELS:
        assert f'strcmp(label, "{lab}")' in gate


def test_wake_tail_drops_only_the_artefact_classes():
    assert window_intent.in_wake_tail(300, "aus")
    assert window_intent.in_wake_tail(300, "_unknown_")
    assert not window_intent.in_wake_tail(300, "Licht")  # a real first word survives
    assert not window_intent.in_wake_tail(450, "aus")


def test_buffers_stay_aligned_when_one_fills():
    buf = window_intent.WindowBuffers()
    while buf.add("fünfundsiebzig", 0.9, "fünfundsiebzig", 0.9):
        pass
    assert len(buf.intent) == len(buf.words) == len(buf.seconds) > 0
    assert len(" ".join(buf.intent).encode()) < window_intent.INTENT_BYTES
    assert len("|".join(buf.seconds).encode()) < window_intent.SECONDS_BYTES


def _post(label, p=0.9, second=None, p2=0.0):
    v = np.full(len(config.COMMAND_LABELS), 0.001)
    v[config.COMMAND_LABELS.index(label)] = p
    if second:
        v[config.COMMAND_LABELS.index(second)] = p2
    return v


def test_decode_window_rescores_an_unknown_slot():
    # "Licht" fires, then the action comes out as _unknown_ with "an" runner-up:
    # plain parse fails (missing action), the device's rescore recovers it.
    labels = config.COMMAND_LABELS
    steps = (
        [_post("_silence_")] * 6 + [_post("Licht")] * 4 + [_post("_unknown_", 0.6, "an", 0.35)] * 4
    )
    got = window_intent.decode_window(steps, labels, step_ms=100)
    assert got == Intent("Licht", None, "an")


def test_decode_window_drops_a_tail_aus():
    # An "aus" fired inside the first 450 ms is the "...Bus" artefact, so the
    # window is left with Licht + an, not the duplicate-action "aus ... an".
    labels = config.COMMAND_LABELS
    steps = [_post("aus")] * 3 + [_post("Licht")] * 4 + [_post("an")] * 4
    assert window_intent.decode_window(steps, labels, step_ms=100) == Intent("Licht", None, "an")
