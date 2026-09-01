import numpy as np

from kws_de.ctc import CTC_TOKENS, greedy_decode, logits_to_intent, token_id
from kws_de.grammar import Intent, Rejection


def _onehot_seq(ids, n):
    L = np.full((len(ids), n), -9.0)
    for i, t in enumerate(ids):
        L[i, t] = 9.0
    return L


def test_greedy_collapses_repeats_and_drops_blank():
    n = len(CTC_TOKENS)
    b = 0
    licht, an = token_id("Licht"), token_id("an")
    # frames: Licht Licht blank an an  -> ["Licht","an"]
    logits = _onehot_seq([licht, licht, b, an, an], n)
    assert greedy_decode(logits) == ["Licht", "an"]


def test_blank_only_is_empty():
    n = len(CTC_TOKENS)
    assert greedy_decode(_onehot_seq([0, 0, 0], n)) == []


def test_token_id_and_vocab():
    assert CTC_TOKENS[0] == "_blank_"
    assert token_id("_blank_") == 0
    assert token_id("Licht") == CTC_TOKENS.index("Licht")


def test_logits_to_intent_full_phrase():
    n = len(CTC_TOKENS)
    ids = [token_id("Licht"), token_id("Küche"), token_id("an")]
    # blank between each token so nothing collapses across word boundaries
    seq = []
    for i, t in enumerate(ids):
        if i > 0:
            seq.append(0)
        seq.append(t)
        seq.append(t)
    logits = _onehot_seq(seq, n)
    assert logits_to_intent(logits) == Intent("Licht", "Küche", "an")


def test_logits_to_intent_blank_only_is_rejection():
    n = len(CTC_TOKENS)
    logits = _onehot_seq([0, 0, 0], n)
    assert isinstance(logits_to_intent(logits), Rejection)
