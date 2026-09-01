"""The `metal` extra must actually give TensorFlow a GPU on Apple silicon.

tensorflow-metal is a binary plugin built against one TensorFlow ABI; a
too-new TensorFlow makes the plugin fail to load at import time. This test
pins the contract: metal extra installed on darwin/arm64 => a GPU device is
listed. Skipped everywhere else (CI Linux, or metal extra not installed).
"""

import importlib.metadata
import platform

import pytest


def _metal_installed() -> bool:
    try:
        importlib.metadata.version("tensorflow-metal")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64" or not _metal_installed(),
    reason="tensorflow-metal only exists for Apple silicon",
)
def test_metal_gpu_visible():
    import tensorflow as tf  # import itself fails when the plugin ABI mismatches

    assert tf.config.list_physical_devices("GPU"), "metal extra installed but no GPU listed"
