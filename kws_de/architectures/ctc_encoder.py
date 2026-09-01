"""Streaming CTC encoder (Phase 2, Task 4): the matchboxnet conv body (all
stride-1 in time, no global pool -- see kws_de/architectures/matchboxnet.py)
feeding a per-frame Conv1D(1) head, so it emits per-frame logits
`(batch, T, n_tokens)` instead of collapsing time into one classification per
clip.

Two shapes, one graph: `t_frames=None` (default) keeps the variable-length
time axis for training with `tf.nn.ctc_loss`; a concrete `t_frames` builds a
fixed-length clone for INT8 export -- full-INT8 quantisation needs static
tensor shapes, and a fixed chunk + ring buffer is the honest on-device
streaming shape anyway. Every layer is a static conv/reshape (no
`TimeDistributed` while-loop -> no `TensorListReserve`), so the exported graph
stays inside the TFLM INT8 builtin set. All weights are time-independent, so a
fixed-T clone loads a variable-T model's weights via `set_weights`."""

import tensorflow as tf

from kws_de import config
from kws_de.architectures.matchboxnet import _tcs_conv

_ENCODER_BODIES = ("matchboxnet",)


def build_ctc_encoder(
    n_tokens: int,
    encoder: str = "matchboxnet",
    *,
    t_frames: int | None = None,
    n_mfcc: int = config.N_MFCC,
    channels: int = 32,
    b_blocks: int = 3,
    r_subblocks: int = 2,
    kernel_time: int = 5,
) -> tf.keras.Model:
    if encoder not in _ENCODER_BODIES:
        raise ValueError(f"unsupported CTC encoder body: {encoder!r} (have {_ENCODER_BODIES})")

    L = tf.keras.layers
    # Training path: (batch=None, T=None) for variable-length CTC. Export path:
    # a concrete t_frames AND batch=1 -- on-device inference is one chunk at a
    # time, and a fixed batch is what lets every RESHAPE below fold to a static
    # op (a None batch makes TFLite recompute shapes at runtime via
    # SHAPE / STRIDED_SLICE / PACK). Weights are batch/T-independent, so the
    # export clone loads a variable-shape model's weights with set_weights.
    if t_frames is not None:
        inp = L.Input(batch_shape=(1, t_frames, n_mfcc, 1))
    else:
        inp = L.Input((None, n_mfcc, 1))
    # When t_frames is concrete every reshape below is fully static (a plain
    # RESHAPE); only the training path (t_frames=None) leaves T as -1. That is
    # what keeps the *exported* graph free of dynamic-shape ops (SHAPE /
    # STRIDED_SLICE / PACK) -- they only run on-device via the fixed-T export.
    t_dim = t_frames if t_frames is not None else -1
    # (T, F, 1) -> (T, 1, F) as a pure reshape (freq bins -> channels, time stays
    # spatial): the (F, 1) tail flattens to (1, F) contiguously, so no TRANSPOSE.
    x = L.Reshape((t_dim, 1, n_mfcc))(inp)

    x = _tcs_conv(x, channels, kernel_time=3)  # prologue
    x = L.ReLU()(x)

    for _ in range(b_blocks):
        block_in = x
        for r in range(r_subblocks):
            x = _tcs_conv(x, channels, kernel_time=kernel_time)
            if r < r_subblocks - 1:
                x = L.ReLU()(x)
        if block_in.shape[-1] != channels:
            block_in = L.Conv2D(channels, (1, 1), padding="same", use_bias=False)(block_in)
            block_in = L.BatchNormalization()(block_in)
        x = L.Add()([x, block_in])
        x = L.ReLU()(x)

    x = L.Conv2D(channels * 2, (1, 1), padding="same", use_bias=False)(x)  # epilogue
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # No GlobalAveragePooling2D here (unlike build_matchboxnet) -- CTC needs a
    # per-frame prediction, not one pooled clip-level vector. Every conv above
    # is stride-1/"same" in time, so T is preserved end to end.
    #
    # Head: a 1x1 Conv2D on the 4-D (T, 1, C) tensor == a Dense applied at every
    # timestep, as ONE static CONV_2D -- not TimeDistributed's per-frame tf.while
    # loop (which forced TensorListReserve / SELECT_TF_OPS and blocked E8), and
    # not Conv1D (whose TFLite lowering injects EXPAND_DIMS / SHAPE / STRIDED_SLICE).
    x = L.Conv2D(n_tokens, (1, 1), name="ctc_head")(x)  # (T, 1, n_tokens); logits, no softmax
    # (T, 1, n_tokens) -> (T, n_tokens); static reshape when t_frames is set
    out = L.Reshape((t_dim, n_tokens))(x)
    return tf.keras.Model(inp, out, name=f"ctc_encoder_{encoder}")
