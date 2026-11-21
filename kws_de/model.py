import random
import sys

import tensorflow as tf

from kws_de import config

# Grow GPU memory on demand instead of TF's default of reserving the whole card
# at first use. On a shared GPU (thinky runs an LLM server that holds ~21 GB of
# the 24 GB card) TF's grab either fails outright or starves anything that needs
# CUDA in the same process afterwards -- faster-whisper for the TTS gate, the
# voice-clone TTS engine. Growth is the standard coexistence setting; it costs
# nothing on macOS/Metal or CPU-only, and must be set before the first GPU op,
# which is why it lives next to the import every training path goes through.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:  # already initialised elsewhere -- too late to change, not fatal
        pass


class _IntRandom(random.Random):
    """`random.Random` whose `randint` accepts integral floats. tf_keras 2.18's
    `_create_seed` calls `randint(1, 1e9)`, which Python 3.12 rejects; on 3.11
    the float was coerced with the same result, so the sequence is unchanged."""

    def randint(self, a, b):
        return super().randint(int(a), int(b))


def set_random_seed(seed: int) -> None:
    """`tf.keras.utils.set_random_seed` plus the Python 3.12 shim for tf_keras's
    seed generator. Use this instead of calling Keras directly."""
    tf.keras.utils.set_random_seed(seed)
    tfk = sys.modules.get("tf_keras.src.backend")
    if tfk is not None:
        tfk._SEED_GENERATOR.generator = _IntRandom(seed)


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
