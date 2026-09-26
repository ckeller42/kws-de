"""Python reference for the device's assist-window pipeline: fired words ->
intent, exactly as firmware/main/recognise.cc + wake.cc + intent.c run it.

Host sentence numbers used plain `grammar.parse` over the fired words, while
the device also drops "...Bus" tail artefacts, keeps a runner-up word per fire
in fixed-size buffers, and retries a failed parse with one runner-up
substitution. This module is that whole path in one place, so a host number
means a device number (architecture review S2). `rescore` is checked against
the C `intent_rescore` by the generated cases in firmware/test/intent_cases.h.
"""

import numpy as np

from kws_de import config
from kws_de.grammar import Intent, parse

# Mirrors of firmware constants -- tests/test_window_intent.py pins each one to
# the C header it comes from, so a change on either side fails the suite.
WAKE_TAIL_MS = 450  # assist_gate.h ASSIST_WAKE_TAIL_MS
WAKE_TAIL_LABELS = ("aus", "_unknown_")  # assist_gate_in_wake_tail()
RESCORE_FLOOR = 0.25  # intent.h INTENT_RESCORE_FLOOR
INTENT_BYTES = 64  # recognise.h window_intent
WORDS_BYTES = 96  # recognise.h window_words
SECONDS_BYTES = 96  # recognise.h window_seconds
_LINE_BYTES = INTENT_BYTES - 1  # intent_parse() reads at most this many bytes
_MAX_TOKENS = 16  # intent_rescore()'s wtok[16] / stok[16]


def in_wake_tail(ms_since_open: float, label: str) -> bool:
    """assist_gate_in_wake_tail(): drop an `aus`/`_unknown_` fire this soon in."""
    return ms_since_open < WAKE_TAIL_MS and label in WAKE_TAIL_LABELS


def _nbytes(s: str) -> int:
    return len(s.encode())


def runner_up(smoothed, labels, fired: str) -> tuple[str, float]:
    """The stream decoder's second-best command word at a fire (recognise.cc):
    argmax of the smoothed posterior over every label except the fired one,
    `_unknown_` and `_silence_`."""
    best, best_p = None, -1.0
    for i, lab in enumerate(labels):
        if lab in (fired, "_unknown_", "_silence_"):
            continue
        if smoothed[i] > best_p:
            best, best_p = lab, float(smoothed[i])
    return (best, best_p) if best is not None else ("_unknown_", 0.0)


class WindowBuffers:
    """window_intent / window_words / window_seconds: whole entries only, and all
    three take a fire together or not at all (recognise.cc since #104)."""

    def __init__(self):
        self.intent, self.words, self.seconds = [], [], []
        self._len = [0, 0, 0]

    def add(self, fired: str, p: float, second: str, second_p: float) -> bool:
        n = self._len
        ei = (" " if n[0] else "") + fired
        ew = ("|" if n[1] else "") + f"{fired}:{p:.2f}"
        es = ("|" if n[2] else "") + f"{second}:{second_p:.2f}"
        if (
            n[0] + _nbytes(ei) >= INTENT_BYTES
            or n[1] + _nbytes(ew) >= WORDS_BYTES
            or n[2] + _nbytes(es) >= SECONDS_BYTES
        ):
            return False
        self.intent.append(fired)
        self.words.append(f"{fired}:{p:.2f}")
        self.seconds.append(f"{second}:{second_p:.2f}")
        n[0] += _nbytes(ei)
        n[1] += _nbytes(ew)
        n[2] += _nbytes(es)
        return True


def rescore(words: list, seconds: list, floor: float = RESCORE_FLOOR):
    """intent_rescore(): the plain parse, or -- if it fails -- the parse with
    the one `_unknown_` slot whose runner-up clears `floor` replaced by that
    runner-up. Returns (result, (from, to) or None). Bit-for-bit the C logic,
    including its early give-ups: a misaligned `seconds`, a second fixable
    slot, or a substituted line longer than intent_parse() can read."""
    base = parse(words)
    words, seconds = words[:_MAX_TOKENS], seconds[:_MAX_TOKENS]  # C tokenises into [16]
    if isinstance(base, Intent) or len(words) != len(seconds):
        return base, None
    sub = sub_label = None
    for i, (w, s) in enumerate(zip(words, seconds, strict=True)):
        if w != "_unknown_" or ":" not in s:
            continue
        lab, p = s.split(":", 1)
        if float(p) < floor:
            continue
        if sub is not None:
            return base, None  # a second fixable slot: one substitution can't cover both
        if lab not in config.COMMAND_LABELS or lab in ("_unknown_", "_silence_"):
            continue
        sub, sub_label = i, lab
    if sub is None:
        return base, None
    merged = [sub_label if i == sub else w for i, w in enumerate(words)]
    if _nbytes(" ".join(merged)) > _LINE_BYTES:
        return base, None
    got = parse(merged)
    return (got, ("_unknown_", sub_label)) if isinstance(got, Intent) else (base, None)


def decode_window(steps, labels, step_ms: float, first_ms: float | None = None, stream_kwargs=None):
    """Run the device's per-window path over `steps` (one posterior per
    recogniser step; the first `first_ms` after the window opened, default
    `step_ms`) and return the intent (Intent or Rejection) wake.cc would report."""
    from kws_de.stream import KeywordStream

    ks = KeywordStream(None, labels, **(stream_kwargs or {}))
    first_ms = step_ms if first_ms is None else first_ms
    buf = WindowBuffers()
    for k, post in enumerate(steps):
        for fired in ks.push(post):
            if in_wake_tail(first_ms + k * step_ms, fired):
                continue
            second, second_p = runner_up(ks.last_smoothed, labels, fired)
            p = float(np.asarray(post)[labels.index(fired)])
            buf.add(fired, p, second, second_p)
    return rescore(buf.intent, buf.seconds)[0]
