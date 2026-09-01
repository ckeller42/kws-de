import numpy as np

from kws_de import budgets, config
from kws_de.architectures.ctc_encoder import build_ctc_encoder
from kws_de.ctc import CTC_TOKENS
from kws_de.export import to_int8_tflite
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


def test_ctc_encoder_fixed_t_int8_exports_tflm_clean():
    """The E8 export blocker, as a gate: a fixed-T CTC encoder must INT8-export
    with zero tensor-list ops. The old TimeDistributed(Dense) head unrolled into
    a tf.while loop -> TensorListReserve, which to_int8_tflite (INT8 builtins
    only, no SELECT_TF_OPS) cannot even convert. A Conv1D(1) per-frame head is a
    single static op, so the graph stays inside the TFLM builtin set."""
    n_tokens = len(CTC_TOKENS)
    t_frames = 48
    m = build_ctc_encoder(n_tokens, encoder="matchboxnet", t_frames=t_frames)
    # fixed T + batch 1 = a fully concrete input shape (full-INT8 quant needs
    # static tensors, and it is the honest on-device shape: one chunk at a time,
    # fixed chunk + ring buffer).
    assert m.input_shape == (1, t_frames, config.N_MFCC, 1)

    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, t_frames, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)  # must NOT raise ConverterError (E8's failure)

    # Every op must be in the TFLM builtin set the on-device resolver registers
    # -- no TensorListReserve (the E8 blocker), and no dynamic-shape ops either.
    tflm_ops = {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "MEAN",
        "SOFTMAX",
        "RESHAPE",
        "ADD",
        "DELEGATE",
    }
    ops = budgets.tflite_op_types(blob)
    assert ops <= tflm_ops, f"non-TFLM ops present: {sorted(ops - tflm_ops)}"
    assert budgets.is_full_int8(blob)


def test_ctc_encoder_default_is_variable_length():
    """The training path is unchanged: default (t_frames=None) still takes a
    variable-length time axis."""
    n_tokens = len(CTC_TOKENS)
    m = build_ctc_encoder(n_tokens, encoder="matchboxnet")
    assert m.input_shape == (None, None, config.N_MFCC, 1)
    assert m.output_shape == (None, None, n_tokens)
