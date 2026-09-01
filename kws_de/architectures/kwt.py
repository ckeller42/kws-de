"""Keyword-Transformer (arXiv:2104.00769) for KWS.

Reference-only (not device-runnable): builds, trains, and INT8-exports
(int8 I/O) fine, but `MultiHeadAttention`/`LayerNormalization` lower to
BATCH_MATMUL/TRANSPOSE/GATHER/CONCATENATION/TILE plus float DEQUANTIZE/
QUANTIZE bridges around the un-fused softmax/layernorm math — none of
which TFLM/ESP-NN on the ESP32-S3 support. Those ops fall outside the
CONV_2D/DEPTHWISE_CONV_2D/FULLY_CONNECTED/MEAN/SOFTMAX/RESHAPE/ADD set the
other three architectures stay within. See
`tests/test_architectures.py::test_kwt_is_not_tflm_device_runnable`.
"""

import tensorflow as tf

from kws_de import config


def _encoder_block(x, d_model, num_heads, mlp_dim):
    L = tf.keras.layers
    h = L.LayerNormalization(epsilon=1e-6)(x)
    attn = L.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(h, h)
    x = L.Add()([x, attn])
    h2 = L.LayerNormalization(epsilon=1e-6)(x)
    h2 = L.Dense(mlp_dim, activation="relu")(h2)
    h2 = L.Dense(d_model)(h2)
    return L.Add()([x, h2])


def build_kwt(
    input_shape=(config.N_FRAMES, config.N_MFCC, 1),
    n_classes=config.NUM_CLASSES,
    *,
    d_model: int = 64,
    depth: int = 3,
    num_heads: int = 4,
    mlp_dim: int = 128,
) -> tf.keras.Model:
    L = tf.keras.layers
    n_frames, n_mfcc = input_shape[0], input_shape[1]
    inp = L.Input(input_shape)
    x = L.Lambda(lambda z: tf.reshape(z, [-1, n_frames, n_mfcc]))(inp)
    x = L.Dense(d_model)(x)  # per-frame linear projection ("patch" embedding)

    # Class token: Embedding lookup on a constant zero index broadcasts over batch
    # without a custom trainable-weight layer.
    zero_idx = L.Lambda(lambda z: tf.zeros((tf.shape(z)[0], 1), dtype=tf.int32))(x)
    cls = L.Embedding(1, d_model)(zero_idx)
    x = L.Concatenate(axis=1)([cls, x])

    # Positional embedding: Embedding lookup on constant range indices.
    seq_len = n_frames + 1
    pos_idx = L.Lambda(
        lambda z: tf.tile(tf.range(seq_len, dtype=tf.int32)[None, :], [tf.shape(z)[0], 1])
    )(x)
    pos = L.Embedding(seq_len, d_model)(pos_idx)
    x = L.Add()([x, pos])

    for _ in range(depth):
        x = _encoder_block(x, d_model, num_heads, mlp_dim)
    x = L.LayerNormalization(epsilon=1e-6)(x)
    cls_out = L.Lambda(lambda z: z[:, 0, :])(x)
    out = L.Dense(n_classes, activation="softmax")(cls_out)
    return tf.keras.Model(inp, out, name="kwt_kws_de")
