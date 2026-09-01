"""Streaming CTC encoder (Phase 2, Task 4): the matchboxnet conv body (all
stride-1 in time, no global pool -- see kws_de/architectures/matchboxnet.py)
feeding a time-distributed Dense head, so it emits per-frame logits
`(batch, T, n_tokens)` for a variable-length input `(batch, None, N_MFCC, 1)`
instead of collapsing time into one classification per clip."""

import tensorflow as tf

from kws_de import config
from kws_de.architectures.matchboxnet import _tcs_conv

_ENCODER_BODIES = ("matchboxnet",)


def build_ctc_encoder(
    n_tokens: int,
    encoder: str = "matchboxnet",
    *,
    n_mfcc: int = config.N_MFCC,
    channels: int = 32,
    b_blocks: int = 3,
    r_subblocks: int = 2,
    kernel_time: int = 5,
) -> tf.keras.Model:
    if encoder not in _ENCODER_BODIES:
        raise ValueError(f"unsupported CTC encoder body: {encoder!r} (have {_ENCODER_BODIES})")

    L = tf.keras.layers
    inp = L.Input((None, n_mfcc, 1))  # (T, N_MFCC, 1), T variable
    # (T, F, 1) -> (T, 1, F): frequency bins become channels, time stays spatial
    # -- same layout matchboxnet's body expects, but T read dynamically so any
    # length works (mirrors build_matchboxnet's reshape, minus the fixed n_frames).
    x = L.Lambda(lambda z: tf.reshape(z, [tf.shape(z)[0], tf.shape(z)[1], 1, n_mfcc]))(inp)

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
    x = L.Lambda(lambda z: tf.squeeze(z, axis=2))(x)  # (T, 1, C) -> (T, C)
    out = L.TimeDistributed(L.Dense(n_tokens))(x)  # per-frame logits, no softmax (tf.nn.ctc_loss)
    return tf.keras.Model(inp, out, name=f"ctc_encoder_{encoder}")
