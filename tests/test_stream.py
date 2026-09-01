import numpy as np

from kws_de.stream import KeywordStream

LABELS = ["Licht", "an", "_silence_"]


def _one_hot(i, n=3, p=0.9):
    v = np.full(n, (1 - p) / (n - 1))
    v[i] = p
    return v


def _weak_one_hot(i, n=3, p=0.65):
    v = np.full(n, (1 - p) / (n - 1))
    v[i] = p
    return v


def _push_seq(ks, indices, weak=False):
    events = []
    for i in indices:
        v = _weak_one_hot(i) if weak else _one_hot(i)
        events += ks.push(v)
    return events


def test_fires_once_per_sustained_word():
    ks = KeywordStream(None, LABELS, smooth_win=3, threshold=0.6, min_consecutive=2, gap_steps=2)
    events = _push_seq(ks, [0] * 6)  # 'Licht' sustained
    assert events == ["Licht"]  # exactly one event despite 6 frames


def test_two_words_back_to_back_no_gap_no_swallowing():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.6, min_consecutive=2, gap_steps=2)
    seq = [0, 0, 0, 1, 1, 1]  # Licht then an, no gap
    out = _push_seq(ks, seq)
    assert out == ["Licht", "an"]


def test_one_step_ghost_between_words_is_filtered():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.6, min_consecutive=2, gap_steps=2)
    # A A A ghost(silence, i.e. non-A/B) B B B
    seq = [0, 0, 0, 2, 1, 1, 1]
    out = _push_seq(ks, seq)
    assert out == ["Licht", "an"]


def test_same_word_twice_with_none_gap_between_runs():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.6, min_consecutive=2, gap_steps=2)
    # Licht run, then silence gap >= gap_steps, then Licht run again
    seq = [0, 0, 0, 2, 2, 0, 0, 0]
    out = _push_seq(ks, seq)
    assert out == ["Licht", "Licht"]


def test_silence_never_fires_and_resets_runs():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.5, min_consecutive=2, gap_steps=2)
    out = _push_seq(ks, [2] * 5)  # _silence_
    assert out == []


def test_below_threshold_never_fires_and_resets_runs():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.9, min_consecutive=2, gap_steps=2)
    out = _push_seq(ks, [0, 0, 0], weak=True)  # top-1 Licht @0.65 < threshold 0.9
    assert out == []


def test_lingering_weaker_but_still_top_only_one_event():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.5, min_consecutive=2, gap_steps=2)
    # A A A A(weaker but still top-1 and above threshold) -> still just 1 event
    events = []
    events += ks.push(_one_hot(0))
    events += ks.push(_one_hot(0))
    events += ks.push(_one_hot(0))
    events += ks.push(_weak_one_hot(0))
    assert events == ["Licht"]


def test_fresh_run_of_same_label_before_gap_elapses_does_not_refire():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.6, min_consecutive=2, gap_steps=2)
    # Licht run, one None step (gap not yet elapsed), Licht run again -> no immediate refire
    seq = [0, 0, 0, 2, 0, 0]
    out = _push_seq(ks, seq)
    assert out == ["Licht"]
