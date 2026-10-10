"""E71 `agree` cascade: MFCC-DTW plus a second opinion -- the deployed classifier (`cnn`
arm) on pool W, the deployed-model embedding (`embed` arm) on pool P, where the classifier
gets almost no intent right. Accept only when both arms' top-1 labels agree; the label is
that shared label. Score = max of the two arms' d1/d2 margins (best over second-best label,
scale-free, so no per-arm normalisation constant has to be fitted); disagreement -> inf
(never accepted, never a false accept). E71_AGREE_SCORE=dtw keeps DTW's raw top-1 distance
as the score instead (agreement is then a pure gate).
"""

import math
import os
from collections.abc import Callable

import e69_cnn
import e69_dtw
import e69_embed
import numpy as np

SCORE = os.environ.get("E71_AGREE_SCORE", "margin")
MODE = None if SCORE == "margin" else f"score{SCORE}"  # harness result-file name tag


def _margin(ranked) -> float:
    return ranked[0][1] / max(ranked[1][1], 1e-12) if len(ranked) > 1 else 1.0


def make_matcher(
    templates: dict[str, list[np.ndarray]],
) -> Callable[[np.ndarray], list[tuple[str, float]]]:
    dtw = e69_dtw.make_matcher(templates)
    other = (e69_cnn if e69_cnn.is_word_pool(templates) else e69_embed).make_matcher(templates)

    def match(sig: np.ndarray) -> list[tuple[str, float]]:
        a, b = dtw(sig), other(sig)
        if a[0][0] != b[0][0]:
            return [(a[0][0], math.inf)]
        s = a[0][1] if SCORE == "dtw" else max(_margin(a), _margin(b))
        return [(a[0][0], s)]

    return match
