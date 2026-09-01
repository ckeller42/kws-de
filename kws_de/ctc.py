"""CTC token vocabulary + greedy decode -> grammar intent (Phase 2, Tasks 1-2).

Blank at index 0; the remaining tokens are the slot-command words (devices,
zones, actions) -- the same vocabulary `kws_de.grammar.parse` already
understands, so `logits_to_intent` is decode composed straight into it.
"""

import numpy as np

from kws_de import config
from kws_de.grammar import Intent, Rejection, parse

CTC_TOKENS: list[str] = ["_blank_"] + config.DEVICES + config.ZONES + config.ACTIONS


def token_id(token: str) -> int:
    return CTC_TOKENS.index(token)


def greedy_decode(logits: np.ndarray) -> list[str]:
    """Per-frame argmax, collapse consecutive repeats, drop blank (index 0),
    map surviving ids to token strings."""
    ids = np.asarray(logits).argmax(-1)
    collapsed = [t for i, t in enumerate(ids) if i == 0 or t != ids[i - 1]]
    return [CTC_TOKENS[int(t)] for t in collapsed if t != 0]


def logits_to_intent(logits: np.ndarray) -> Intent | Rejection:
    return parse(greedy_decode(logits))
