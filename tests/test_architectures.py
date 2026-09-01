import numpy as np
import pytest

from kws_de import config
from kws_de.architectures import ARCHITECTURES, get
from kws_de.architectures.ctc_encoder import build_ctc_encoder
from kws_de.budgets import check_budgets, tflite_op_types
from kws_de.export import to_int8_tflite

# Architectures proven to INT8-export within the TFLM op set (see
# test_device_runnable_architectures_export_int8_and_fit_budget below). kwt is
# reference-only — see kws_de/architectures/kwt.py and
# test_kwt_is_not_tflm_device_runnable.
DEVICE_RUNNABLE_ARCHITECTURES = ("ds_cnn", "bc_resnet", "matchboxnet")

# Ops TFLM + ESP-NN run on the ESP32-S3 (DELEGATE is a host-interpreter artifact,
# not on-device) — mirrors tests/test_budgets.py's TFLM_OPS.
TFLM_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "MEAN",
    "SOFTMAX",
    "RESHAPE",
    "ADD",
    "DELEGATE",
}


def test_registry_has_expected_names():
    assert {"ds_cnn", "bc_resnet", "matchboxnet", "kwt"} <= set(ARCHITECTURES)


def test_ds_cnn_builder_shape():
    m = get("ds_cnn")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)


def test_bc_resnet_builds_and_is_small():
    m = get("bc_resnet")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 60_000


def test_matchboxnet_builds():
    m = get("matchboxnet")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 150_000


def test_kwt_builds():
    m = get("kwt")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 300_000


@pytest.mark.xfail(
    reason=(
        "kwt is reference-only: MultiHeadAttention/LayerNormalization lower to "
        "BATCH_MATMUL/TRANSPOSE/GATHER/CONCATENATION/TILE plus float DEQUANTIZE/"
        "QUANTIZE bridges, outside the TFLM op set — not device-runnable even "
        "though it does INT8-export with int8 I/O."
    ),
    strict=True,
)
def test_kwt_is_not_tflm_device_runnable():
    m = get("kwt")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)
    assert tflite_op_types(blob) <= TFLM_OPS


def test_device_runnable_architectures_export_int8_and_fit_budget():
    n_classes = len(config.COMMAND_LABELS)
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    for name in DEVICE_RUNNABLE_ARCHITECTURES:
        m = get(name)((config.N_FRAMES, config.N_MFCC, 1), n_classes=n_classes)
        blob = to_int8_tflite(m, rep)
        check_budgets(blob, m)  # must not raise


def test_ctc_encoder_variable_length_logits():
    from kws_de.ctc import CTC_TOKENS

    n_tokens = len(CTC_TOKENS)
    m = build_ctc_encoder(n_tokens, encoder="matchboxnet")
    assert m.input_shape == (None, None, config.N_MFCC, 1)
    assert m.output_shape == (None, None, n_tokens)

    rng = np.random.default_rng(0)
    x60 = rng.standard_normal((1, 60, config.N_MFCC, 1)).astype(np.float32)
    x90 = rng.standard_normal((1, 90, config.N_MFCC, 1)).astype(np.float32)
    out60 = m(x60, training=False).numpy()
    out90 = m(x90, training=False).numpy()
    assert out60.shape == (1, 60, n_tokens)
    assert out90.shape == (1, 90, n_tokens)


def test_ctc_encoder_rejects_unknown_encoder():
    with pytest.raises(ValueError):
        build_ctc_encoder(10, encoder="not-a-real-encoder")
