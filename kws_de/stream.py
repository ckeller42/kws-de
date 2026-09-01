from collections import deque

import numpy as np


class KeywordStream:
    """Smooths per-step posteriors and decodes them into discrete keyword
    events using edge-triggered, run-based decoding.

    Each step's *candidate* is the top-1 label of the smoothed posterior
    (trailing mean over `smooth_win` raw pushes) if its probability is
    >= `threshold` and it isn't `_silence_`, else `None`. Consecutive steps
    with the same candidate form a *run*; a run fires its label exactly
    once, as soon as it reaches `min_consecutive` steps (never again while
    that same candidate persists — no global cooldown, so a different
    label's run can fire immediately after its own run qualifies). A run
    of the SAME label that most recently fired may fire again only after
    at least `gap_steps` consecutive non-matching steps have elapsed since
    its previous run ended.
    """

    def __init__(
        self, predict_fn, labels, smooth_win=3, threshold=0.5, min_consecutive=2, gap_steps=2
    ):
        self.predict_fn = predict_fn
        self.labels = list(labels)
        self.smooth_win = smooth_win
        self.threshold = threshold
        self.min_consecutive = min_consecutive
        self.gap_steps = gap_steps
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=self.smooth_win)
        self._run_label = None
        self._run_len = 0
        self._run_fired = False
        self._last_fired_label = None
        self._gap_since_last_fired = 0

    def push(self, posterior) -> list:
        self._hist.append(np.asarray(posterior, dtype=np.float64))
        smoothed = np.mean(self._hist, axis=0)
        idx = int(np.argmax(smoothed))
        label = self.labels[idx]
        candidate = label if (smoothed[idx] >= self.threshold and label != "_silence_") else None

        if candidate == self._run_label:
            self._run_len += 1
        else:
            self._run_label = candidate
            self._run_len = 1
            self._run_fired = False

        events = []
        if candidate is not None and self._run_len >= self.min_consecutive and not self._run_fired:
            gap_ok = (
                candidate != self._last_fired_label or self._gap_since_last_fired >= self.gap_steps
            )
            if gap_ok:
                events.append(candidate)
                self._run_fired = True
                self._last_fired_label = candidate
                self._gap_since_last_fired = 0

        if self._last_fired_label is not None and candidate != self._last_fired_label:
            self._gap_since_last_fired += 1

        return events
