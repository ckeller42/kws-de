import tensorflow as tf

from kws_de import config


def build_dscnn(num_classes: int | None = None, width: int = 32) -> tf.keras.Model:
    """Build the DS-CNN classifier. `num_classes` defaults to the live command
    vocabulary, `len(config.COMMAND_LABELS)`; the retired v1 set
    (`config.NUM_CLASSES`, 7 classes) must now be asked for explicitly.
    `width` is the channel count of every conv/depthwise-separable block
    (default 32; the deployed command model uses 48) -- narrower widths trade
    accuracy for fewer MACs/params on device."""
    num_classes = num_classes if num_classes is not None else len(config.COMMAND_LABELS)
    L = tf.keras.layers
    inp = L.Input((config.N_FRAMES, config.N_MFCC, 1))
    x = L.Conv2D(width, (3, 3), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for _ in range(3):
        x = L.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.Conv2D(width, (1, 1), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="dscnn_kws_de")
