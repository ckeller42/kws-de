"""Compact BC-ResNet (arXiv:2106.04140) for KWS.

Broadcasted-residual blocks: a frequency-depthwise conv branch stays full
resolution; a temporal-depthwise conv branch runs on the frequency-averaged
features and is broadcast-added back across the frequency axis. All ops are
plain Conv2D/DepthwiseConv2D/BatchNorm/ReLU/Add/Mean — INT8-exportable and
within the TFLM op set.
"""

import tensorflow as tf

from kws_de import config


def _broadcast_residual_block(x, width):
    L = tf.keras.layers
    residual = x

    freq = L.DepthwiseConv2D((1, 3), padding="same", use_bias=False)(x)
    freq = L.BatchNormalization()(freq)
    freq = L.ReLU()(freq)

    # Average over the frequency axis, then a temporal-only depthwise conv,
    # broadcast-added back over frequency (ADD supports numpy-style broadcast).
    freq_avg = L.Lambda(lambda t: tf.reduce_mean(t, axis=2, keepdims=True))(freq)
    temporal = L.DepthwiseConv2D((3, 1), padding="same", use_bias=False)(freq_avg)
    temporal = L.BatchNormalization()(temporal)
    temporal = L.ReLU()(temporal)

    x = L.Add()([freq, temporal])
    x = L.Conv2D(width, (1, 1), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return L.Add()([x, residual])


def build_bc_resnet(
    input_shape=(config.N_FRAMES, config.N_MFCC, 1),
    n_classes=config.NUM_CLASSES,
    *,
    width: int = 24,
    n_blocks: int = 4,
) -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input(input_shape)
    x = L.Conv2D(width, (3, 3), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for _ in range(n_blocks):
        x = _broadcast_residual_block(x, width)
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="bc_resnet_kws_de")
