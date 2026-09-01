import numpy as np

from kws_de.stream import KeywordStream

LABELS = ["Licht", "an", "_silence_"]


def _one_hot(i, n=3, p=0.9):
    v = np.full(n, (1 - p) / (n - 1))
    v[i] = p
    return v


def test_fires_once_per_sustained_word():
    ks = KeywordStream(None, LABELS, smooth_win=3, threshold=0.6, refractory=4)
    events = []
    for _ in range(6):  # 'Licht' sustained
        events += ks.push(_one_hot(0))
    assert events == ["Licht"]  # exactly one event despite 6 frames


def test_refractory_blocks_immediate_refire_then_allows_new_word():
    ks = KeywordStream(None, LABELS, smooth_win=2, threshold=0.6, refractory=3)
    seq = [0, 0, 0, 1, 1, 1]  # Licht then an
    out = []
    for i in seq:
        out += ks.push(_one_hot(i))
    assert out == ["Licht", "an"]


def test_silence_never_fires():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.5, refractory=1)
    out = []
    for _ in range(5):
        out += ks.push(_one_hot(2))  # _silence_
    assert out == []
