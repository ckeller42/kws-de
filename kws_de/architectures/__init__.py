"""Pluggable architecture registry for the Phase-1 KWS benchmark.

Every builder has the interface `build(input_shape, n_classes) -> tf.keras.Model`,
takes `input_shape=(config.N_FRAMES, config.N_MFCC, 1)`, and outputs `n_classes`
softmax. `ds_cnn` wraps the existing Phase-0 `kws_de.model.build_dscnn`.
"""

from collections.abc import Callable

import tensorflow as tf

from kws_de.architectures.bc_resnet import build_bc_resnet
from kws_de.architectures.kwt import build_kwt
from kws_de.architectures.matchboxnet import build_matchboxnet
from kws_de.model import build_dscnn


def _build_ds_cnn(input_shape, n_classes) -> tf.keras.Model:
    del input_shape  # build_dscnn's input shape is fixed by config
    return build_dscnn(num_classes=n_classes)


ARCHITECTURES: dict[str, Callable[..., tf.keras.Model]] = {
    "ds_cnn": _build_ds_cnn,
    "bc_resnet": build_bc_resnet,
    "matchboxnet": build_matchboxnet,
    "kwt": build_kwt,
}


def get(name: str) -> Callable[..., tf.keras.Model]:
    return ARCHITECTURES[name]
