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
    assert window_intent.ALIGN_FLOOR == float(_define("intent.h", "INTENT_ALIGN_FLOOR").rstrip("f"))
    assert window_intent.ALIGN_TAU == float(_define("intent.h", "INTENT_ALIGN_TAU").rstrip("f"))
    assert window_intent.ALIGN_MAX_STEPS == int(_define("intent.h", "INTENT_ALIGN_MAX_STEPS"))
    assert window_intent.ALIGN_SMOOTH_WIN == int(_define("gen/features_config.h", "KWS_SMOOTH_WIN"))
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
    got = window_intent.decode_window(steps, labels, step_ms=100, align=False)
    assert got == Intent("Licht", None, "an")


def test_decode_window_drops_a_tail_aus():
    # An "aus" fired inside the first 450 ms is the "...Bus" artefact, so the
    # window is left with Licht + an, not the duplicate-action "aus ... an".
    labels = config.COMMAND_LABELS
    steps = [_post("aus")] * 3 + [_post("Licht")] * 4 + [_post("an")] * 4
    for align in (False, True):  # the fire path and the aligner mask the same tail
        assert window_intent.decode_window(steps, labels, step_ms=100, align=align) == Intent(
            "Licht", None, "an"
        )


def test_candidates_are_the_49_valid_intents_and_all_parse():
    from kws_de.grammar import parse

    cands = window_intent.candidates()
    assert len(cands) == 49
    assert len({i for _, i in cands}) == 49
    for toks, intent in cands:
        assert parse(toks) == intent


def test_align_recovers_a_zone_whose_run_is_too_short_to_fire():
    # E41's in-context case: the zone wins one step only (run 1 < KWS_MIN_CONSECUTIVE
    # 2), so the fire path never emits it and parses "Licht an". Aligning explains
    # that step better as Küche (0.6) than as background (0.4) and keeps the zone.
    labels = config.COMMAND_LABELS
    steps = (
        [_post("_silence_")] * 3
        + [_post("Licht")] * 3
        + [_post("Küche", 0.6, "_unknown_", 0.39)]
        + [_post("_unknown_", 0.6, "Küche", 0.39)]
        + [_post("an")] * 3
    )
    assert window_intent.decode_window(steps, labels, step_ms=100, align=False) == Intent(
        "Licht", None, "an"
    )
    got, conf, margin = window_intent.align(steps, labels, tau=0.3, delta=0.0, smooth_win=1)
    assert got == Intent("Licht", "Küche", "an")
    assert 0.3 < conf <= 1.0 and margin > 0
    # Through the device path (3-step smoothing, floor 0.10, tau 0.70) a ONE-step
    # zone is flattened to 0.2 and the path's confidence falls under tau, so the
    # fire path answers: lost, as on the device. E70 measured where the gate
    # lands on real phrases; this pins only that the fallback engages.
    assert window_intent.decode_window(steps, labels, step_ms=100) == Intent("Licht", None, "an")


def test_align_prefers_no_zone_when_the_evidence_says_background():
    # The same window with Küche never above _unknown_: the evidence favours no
    # zone and align agrees with the fire path rather than inventing one.
    labels = config.COMMAND_LABELS
    steps = [_post("Licht")] * 3 + [_post("_unknown_", 0.5, "Küche", 0.45)] * 3 + [_post("an")] * 3
    got, *_ = window_intent.align(steps, labels, tau=0.3, delta=0.0)
    assert got == Intent("Licht", None, "an")


def test_align_rejects_noise_and_ties():
    from kws_de.grammar import Rejection

    labels = config.COMMAND_LABELS
    noise = [np.full(len(labels), 1 / len(labels))] * 10  # nothing clears the 0.25 floor
    got, score, _ = window_intent.align(noise, labels, tau=0.3, delta=0.0)
    assert got == Rejection("no candidate") and score == 0.0

    steps = [_post("Licht")] * 3 + [_post("an", 0.5, "aus", 0.5)] * 3
    got, _, margin = window_intent.align(steps, labels, tau=0.3, delta=0.05)
    assert got == Rejection("ambiguous") and margin == 0.0
    got, _, _ = window_intent.align(steps, labels, tau=0.3, delta=0.0)
    assert got in (Intent("Licht", None, "an"), Intent("Licht", None, "aus"))


def test_align_enforces_order_occupancy_and_tail_mask():
    from kws_de.grammar import Rejection

    labels = config.COMMAND_LABELS
    # action before device: no monotone alignment exists
    steps = [_post("an")] * 3 + [_post("_silence_")] * 3 + [_post("Licht")] * 3
    got, *_ = window_intent.align(steps, labels, tau=0.3, delta=0)
    assert got == Rejection("no candidate")
    # one step cannot carry two tokens: a lone Licht step never pairs with itself
    one = [_post("Licht", 0.6, "an", 0.3)]
    got, *_ = window_intent.align(one, labels, tau=0.3, delta=0)
    assert got == Rejection("no candidate")
    # ...but two adjacent steps can carry two tokens (no background needed between)
    two = [_post("Licht")] + [_post("an")]
    got, *_ = window_intent.align(two, labels, tau=0.3, delta=0)
    assert got == Intent("Licht", None, "an")
    # "aus" in the wake tail is the "...Bus" artefact: masked, so Licht+aus(tail)+an -> Licht an
    steps = [_post("aus")] * 2 + [_post("Licht")] * 3 + [_post("an")] * 3
    got, *_ = window_intent.align(steps, labels, tau=0.3, delta=0)
    assert got == Intent("Licht", None, "an")


def test_decode_window_falls_back_to_the_fire_path_when_unsure():
    # Weak evidence: the aligner's best candidate is under tau, the fire path
    # (which fires on a 0.5 threshold) still answers, as the device does.
    labels = config.COMMAND_LABELS
    steps = [_post("Licht", 0.55, "_unknown_", 0.4)] * 4 + [_post("an", 0.55, "_unknown_", 0.4)] * 4
    got, conf, _ = window_intent.align(steps, labels, tau=window_intent.ALIGN_TAU, delta=0.0)
    assert not isinstance(got, Intent) and conf < window_intent.ALIGN_TAU
    assert window_intent.decode_window(steps, labels, step_ms=100) == Intent("Licht", None, "an")
