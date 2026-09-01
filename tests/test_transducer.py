import numpy as np

from kws_de import config
from kws_de.transducer import ctc_loss_for, ctc_train


def _seq(token_value, t=24, n_mfcc=config.N_MFCC, noise=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    base = np.full((t, n_mfcc), token_value, dtype=np.float32)
    return base + rng.normal(0, noise, size=base.shape).astype(np.float32)


def _synthetic_batches(n_per_class=6, seed=0):
    """Two trivially separable token classes (ids 1 and 2; 0 is blank) so a
    tiny model should reduce CTC loss within a handful of epochs."""
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_per_class):
        batches.append((_seq(+2.0, rng=rng), [1]))
        batches.append((_seq(-2.0, rng=rng), [2]))
    return batches


def test_ctc_train_smoke_loss_decreases():
    n_tokens = 3  # blank + 2 classes
    batches = _synthetic_batches()

    history: list[float] = []
    model = ctc_train(
        batches,
        n_tokens,
        encoder="matchboxnet",
        epochs=25,
        seed=0,
        learning_rate=5e-3,
        history=history,
    )

    assert len(history) == 25
    assert history[-1] < history[0]
    # sanity: ctc_loss_for is a finite, callable eval-mode loss on the trained model
    eval_loss = ctc_loss_for(model, batches)
    assert eval_loss == eval_loss  # not NaN
    assert eval_loss > 0.0
