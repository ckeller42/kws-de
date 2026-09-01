import os

import numpy as np
import tensorflow as tf

from kws_de import config


def estimate_macs(model) -> int:
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, cout = layer.output.shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            cin = layer.input.shape[-1]
            total += h * w * cout * cin * k
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            _, h, w, c = layer.output.shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            total += h * w * c * k
        elif isinstance(layer, tf.keras.layers.Dense):
            total += layer.input.shape[-1] * layer.units
    return int(total)


def _interp(tflite: bytes):
    itp = tf.lite.Interpreter(model_content=tflite, num_threads=os.cpu_count())
    itp.allocate_tensors()
    return itp


def tflite_op_types(tflite: bytes) -> set:
    itp = _interp(tflite)
    return {d["op_name"] for d in itp._get_ops_details()}  # noqa: SLF001


def is_full_int8(tflite: bytes) -> bool:
    itp = _interp(tflite)
    return (
        itp.get_input_details()[0]["dtype"] == np.int8
        and itp.get_output_details()[0]["dtype"] == np.int8
    )


def is_8bit_quantized(tflite: bytes) -> bool:
    """8-bit quantized I/O (int8 OR uint8) — both run on TFLM/ESP-NN. microWakeWord
    exports uint8 I/O, so the wake model is device-runnable but not strictly int8."""
    itp = _interp(tflite)
    ok = {np.int8, np.uint8}
    return itp.get_input_details()[0]["dtype"] in ok and itp.get_output_details()[0]["dtype"] in ok


def check_budgets(tflite: bytes, model) -> dict:
    report = {
        "model_bytes": len(tflite),
        "macs": estimate_macs(model),
        "int8": is_full_int8(tflite),
        "ops": sorted(tflite_op_types(tflite)),
    }
    assert report["model_bytes"] <= config.MAX_MODEL_BYTES, "model too large"
    assert report["macs"] <= config.MAX_MACS, "MAC budget exceeded"
    assert report["int8"], "model is not full-INT8"
    return report


# Wake stage budget: always-on, so held tighter than the command model. The wake
# tflite is an external microWakeWord artifact (no Keras graph to hand), so this
# checks bytes/ops only — no MACs estimate (see check_budgets for the command model).
MAX_WAKE_MODEL_BYTES = 150_000


def check_wake_budgets(tflite: bytes) -> dict:
    report = {
        "model_bytes": len(tflite),
        "int8": is_full_int8(tflite),
        "quantized_8bit": is_8bit_quantized(tflite),
        "ops": sorted(tflite_op_types(tflite)),
    }
    assert report["model_bytes"] <= MAX_WAKE_MODEL_BYTES, "wake model too large"
    # microWakeWord exports uint8 I/O — 8-bit quantized is the device requirement, not int8.
    assert report["quantized_8bit"], "wake model is not 8-bit quantized"
    return report
