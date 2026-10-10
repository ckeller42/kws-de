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


# Grammar-constrained decoding (architecture review S1, spec
# docs/superpowers/specs/2026-10-10-grammar-constrained-decoding-design.md).
# Instead of firing words and parsing them, every valid intent is aligned
# against the window's smoothed posteriors and the best one is taken on a
# margin. Defaults here are the E70 sweep's starting grid, not device constants
# yet: the firmware port pins them once the sweep has chosen.
ALIGN_SMOOTH_WIN = 3  # stream.c's KWS_SMOOTH_WIN: align over the same smoothed vector
BACKGROUND_LABELS = ("_silence_", "_unknown_")  # what a step not on a token must explain itself as


def candidates() -> list[tuple[list[str], Intent]]:
    """Every valid intent as the token sequence a speaker says it in
    (device, zone for zoned devices only, action). 49 for the current grammar."""
    out = []
    for device in config.DEVICES:
        zones = [None, *config.ZONES] if device in config.ZONED_DEVICES else [None]
        for zone in zones:
            for action in config.DEVICE_ACTIONS[device]:
                toks = [device] + ([zone] if zone else []) + [action]
                out.append((toks, Intent(device, zone, action)))
    return out


def _smoothed(steps, win: int) -> np.ndarray:
    raw = np.asarray(steps, dtype=np.float64)
    out = np.empty_like(raw)
    for t in range(len(raw)):
        out[t] = raw[max(0, t - win + 1) : t + 1].mean(axis=0)
    return out


def _path_score(lp: np.ndarray, bg: np.ndarray, idx: list[int]) -> tuple[float, float]:
    """Viterbi over the left-to-right chain bg0 w1 bg1 w2 ... wL bgL: every step
    is explained either by the token it sits on or as background, so hypotheses
    of different length compete on the whole window. Each token must take >= 1
    step; two tokens may abut (the background between them is optional).
    Returns (total log-prob of the best path, geometric mean of the token-step
    posteriors along it) or (-inf, 0.0) if no path exists."""
    T, L = lp.shape[0], len(idx)
    S = 2 * L + 1
    emit = np.empty((T, S))
    emit[:, 0::2] = bg[:, None]
    for i, k in enumerate(idx):
        emit[:, 2 * i + 1] = lp[:, k]
    NEG = -np.inf
    score = np.full(S, NEG)
    ntok = np.zeros(S)  # token steps on the best path into each state
    tsum = np.zeros(S)  # their summed log-posteriors
    score[0], score[1] = emit[0, 0], emit[0, 1]
    ntok[1], tsum[1] = 1, emit[0, 1]
    for t in range(1, T):
        new = np.full(S, NEG)
        nn, ns = np.zeros(S), np.zeros(S)
        for s in range(S):
            # stay, advance from s-1, or (token state) skip the background from s-2
            srcs = [s, s - 1] + ([s - 2] if s % 2 == 1 and s >= 2 else [])
            srcs = [u for u in srcs if u >= 0]
            u = max(srcs, key=lambda u: score[u])
            if score[u] == NEG:
                continue
            new[s] = score[u] + emit[t, s]
            tok = s % 2 == 1
            nn[s], ns[s] = ntok[u] + tok, tsum[u] + (emit[t, s] if tok else 0.0)
        score, ntok, tsum = new, nn, ns
    end = max((S - 1, S - 2), key=lambda s: score[s])
    if score[end] == NEG:
        return NEG, 0.0
    return float(score[end]), float(np.exp(tsum[end] / ntok[end]))


def align_scores(
    steps,
    labels,
    *,
    floor: float = RESCORE_FLOOR,
    step_ms: float = 100.0,
    first_ms: float | None = None,
    smooth_win: int = ALIGN_SMOOTH_WIN,
) -> list[tuple[float, float, Intent]]:
    """Every valid intent aligned against the window's smoothed posteriors,
    best first: (path score = exp(mean per-step log-prob), confidence = geometric
    mean of the token-step posteriors, intent). Tokens below `floor` at a step
    cannot sit there; `aus` is masked inside the wake tail like the fire path.
    Independent of tau/delta so a sweep scores once and decides many times."""
    labels = list(labels)
    sm = _smoothed(steps, smooth_win)
    if sm.ndim != 2 or sm.shape[0] == 0:
        return []
    first_ms = step_ms if first_ms is None else first_ms
    with np.errstate(divide="ignore"):
        lp = np.log(np.where(sm >= floor, sm, 0.0))
        bg = np.log(sum(sm[:, labels.index(b)] for b in BACKGROUND_LABELS))
    for lab in WAKE_TAIL_LABELS:
        if lab in labels:
            k = labels.index(lab)
            tail = np.array([in_wake_tail(first_ms + t * step_ms, lab) for t in range(len(sm))])
            lp[tail, k] = -np.inf
    T = len(sm)
    scored = []
    for toks, intent in candidates():
        s, conf = _path_score(lp, bg, [labels.index(t) for t in toks])
        if np.isfinite(s):
            scored.append((float(np.exp(s / T)), conf, intent))
    scored.sort(key=lambda x: -x[0])
    return scored


def decide(scored, tau: float, delta: float):
    """The accept rule over `align_scores()` output: best intent if its
    confidence >= tau and its path-score margin over the best *different*
    intent >= delta. Returns (Intent | Rejection, confidence, margin)."""
    from kws_de.grammar import Rejection

    if not scored:
        return Rejection("no candidate"), 0.0, 0.0
    best_score, conf, best = scored[0]
    second = next((s for s, _, i in scored[1:] if i != best), 0.0)
    margin = best_score - second
    if conf < tau:
        return Rejection("below tau"), conf, margin
    if margin < delta:
        return Rejection("ambiguous"), conf, margin
    return best, conf, margin


def align(steps, labels, *, tau: float, delta: float, **kw):
    """Grammar-constrained decode of one window; see align_scores() + decide()."""
    return decide(align_scores(steps, labels, **kw), tau, delta)


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
