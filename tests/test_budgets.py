import numpy as np

from kws_de import config
from kws_de.budgets import check_budgets, estimate_macs, is_full_int8, tflite_op_types
from kws_de.export import to_int8_tflite
from kws_de.model import build_dscnn

# ops TFLM + ESP-NN run on the ESP32-S3 (DELEGATE is a host-interpreter artifact, not on-device).
TFLM_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "MEAN",
    "SOFTMAX",
    "RESHAPE",
    "DELEGATE",
}


def test_macs_and_budgets_pass_for_shipping_model():
    # Build at the real command-model class count so the gate proves the *shipping* model fits.
    m = build_dscnn(num_classes=len(config.COMMAND_LABELS))
    assert 0 < estimate_macs(m) <= config.MAX_MACS
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)
    assert is_full_int8(blob)
    report = check_budgets(blob, m)
    assert report["model_bytes"] <= config.MAX_MODEL_BYTES
    assert report["int8"] is True
    # every op must be device-runnable — an unsupported op won't run on the MCU.
    assert tflite_op_types(blob) <= TFLM_OPS
