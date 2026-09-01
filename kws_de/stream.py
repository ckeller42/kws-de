from collections import deque

import numpy as np


class KeywordStream:
    """Smooths per-step posteriors, thresholds, and debounces into discrete
    keyword events. `refractory` counts full pushes suppressed after a fire
    (the firing push itself doesn't consume a decrement, so a sustained
    signal needs `refractory + 1` pushes before it can fire again)."""

    def __init__(self, predict_fn, labels, smooth_win=3, threshold=0.6, refractory=5):
        self.predict_fn = predict_fn
        self.labels = list(labels)
        self.smooth_win = smooth_win
        self.threshold = threshold
        self.refractory = refractory
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=self.smooth_win)
        self._cooldown = 0

    def push(self, posterior) -> list:
        self._hist.append(np.asarray(posterior, dtype=np.float64))
        smoothed = np.mean(self._hist, axis=0)
        idx = int(np.argmax(smoothed))
        label = self.labels[idx]
        if self._cooldown == 0 and smoothed[idx] >= self.threshold and label != "_silence_":
            self._cooldown = self.refractory + 1
            return [label]
        if self._cooldown > 0:
            self._cooldown -= 1
        return []
