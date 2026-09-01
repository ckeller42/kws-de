"""Compact MatchboxNet (arXiv:2004.08531) for KWS.

Treats the 10 MFCCs as channels over the 49 time steps and runs
time-channel-separable ("TCS") convolutions: a depthwise conv over time
followed by a pointwise (1x1) conv across channels. Implemented as Conv2D
ops on a (T, 1, C) tensor (a 1-D conv in disguise) so every op stays in the
TFLM set (CONV_2D/DEPTHWISE_CONV_2D/ADD/MEAN/RESHAPE/FULLY_CONNECTED/
SOFTMAX) instead of the unsupported CONV_1D/DEPTHWISE_CONV_1D.
"""

import tensorflow as tf

from kws_de import config


def _tcs_conv(x, filters, kernel_time):
    L = tf.keras.layers
    x = L.DepthwiseConv2D((kernel_time, 1), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return x


def build_matchboxnet(
    input_shape=(config.N_FRAMES, config.N_MFCC, 1),
    n_classes=config.NUM_CLASSES,
    *,
    channels: int = 32,
    b_blocks: int = 3,
    r_subblocks: int = 2,
    kernel_time: int = 5,
) -> tf.keras.Model:
    L = tf.keras.layers
    n_frames, n_mfcc = input_shape[0], input_shape[1]
    inp = L.Input(input_shape)
    # (T, F, 1) -> (T, 1, F): frequency bins become channels, time stays spatial.
    x = L.Lambda(lambda z: tf.reshape(z, [-1, n_frames, 1, n_mfcc]))(inp)

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
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="matchboxnet_kws_de")
