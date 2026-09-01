import tensorflow as tf

from kws_de import config


def build_dscnn(num_classes: int | None = None) -> tf.keras.Model:
    """Build the DS-CNN classifier. `num_classes` defaults to `config.NUM_CLASSES`
    (v1, 7 classes); pass `len(config.COMMAND_LABELS)` for v2 (23 classes)."""
    num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
    L = tf.keras.layers
    inp = L.Input((config.N_FRAMES, config.N_MFCC, 1))
    x = L.Conv2D(32, (3, 3), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for _ in range(3):
        x = L.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.Conv2D(32, (1, 1), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="dscnn_kws_de")
