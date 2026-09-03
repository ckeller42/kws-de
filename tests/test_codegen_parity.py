"""Byte-for-byte parity between the generated C and the TFLite interpreter.

These tests compile the generated C against esp-nn's ANSI-C kernels and run
the interpreter on the same inputs. Zero LSB difference is the requirement.
They need the models (KWS_DATA_ROOT) and a working `cc`, so they skip cleanly
where either is missing.
"""

import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from kws_de import codegen, config, tflite_graph

tf = pytest.importorskip("tensorflow")

REPO = pathlib.Path(__file__).resolve().parents[1]
TEST_DIR = REPO / "firmware" / "test"
GEN_DIR = REPO / "firmware" / "main" / "gen"
WAKE = config.MODELS_DIR / "hey_bus.tflite"
COMMAND = config.MODELS_DIR / "command.tflite"

# The layers the spec names: the command model's first 3x3 conv over a single
# input channel, its 3x3 depthwise, its 1x1 conv and its (per-channel) FC; the
# wake model's 5x1 stem conv, a 21x1 depthwise, a 1x1 conv and the 1088->1 FC.
LAYERS = [
    ("command", COMMAND, 0, "CONV_2D 3x3x1"),
    ("command", COMMAND, 1, "DEPTHWISE_CONV_2D 3x3"),
    ("command", COMMAND, 2, "CONV_2D 1x1"),
    ("command", COMMAND, 8, "FULLY_CONNECTED 32->23 per-channel"),
    ("wake", WAKE, 14, "CONV_2D 5x1 stem"),
    ("wake", WAKE, 32, "DEPTHWISE_CONV_2D 21x1"),
    ("wake", WAKE, 18, "CONV_2D 1x1"),
    ("wake", WAKE, 36, "FULLY_CONNECTED 1088->1"),
]

needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")


def _make(target: str) -> None:
    subprocess.run(["make", "-C", str(TEST_DIR), target], check=True)


def _interpreter_output(blob: bytes, op, inputs):
    """The op's exact answer from the reference kernels, via a one-op probe
    model -- the same route the LOGISTIC table will take in a later task.

    BUILTIN_REF is not a detail: the desktop interpreter's default resolver
    hands int8 convolutions to the XNNPACK delegate, which requantises with a
    single rounding step. TFLM's reference kernels and esp-nn both use
    gemmlowp's double rounding (SaturatingRoundingDoublingHighMul followed by
    RoundingDivideByPOT), so the delegate disagrees with the device by 1 LSB on
    roughly 0.2% of outputs. The reference kernels are what the device runs.
    """
    itp = tf.lite.Interpreter(
        model_content=tflite_graph.probe_model(blob, op),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    itp.allocate_tensors()
    detail_in, detail_out = itp.get_input_details()[0], itp.get_output_details()[0]
    itp.set_tensor(detail_in["index"], inputs)
    itp.invoke()
    return itp.get_tensor(detail_out["index"]).astype(np.int8).ravel()


@needs_cc
@pytest.mark.parametrize(
    ("model", "path", "op_index", "what"),
    LAYERS,
    ids=[f"{m}-op{i}" for m, _, i, _ in LAYERS],
)
def test_layer_is_byte_identical(model, path, op_index, what):
    """Generate one real layer, compile it against esp-nn's ANSI kernels and
    demand every output byte match the interpreter's."""
    if not path.exists():
        pytest.skip(f"{path} absent (KWS_DATA_ROOT)")
    blob = path.read_bytes()
    graph = tflite_graph.read_graph(blob)
    op = graph.ops[op_index]
    assert op.name in codegen.EMITTERS, f"op {op_index} is {op.name}, not {what}"
    in_t = graph.tensors[op.inputs[0]]
    rng = np.random.default_rng(op_index)
    inputs = rng.integers(-128, 128, size=in_t.shape, dtype=np.int8)

    expect = _interpreter_output(blob, op, inputs)
    codegen.write_probe_vectors(blob, op, inputs, expect, GEN_DIR)
    _make("test_infer_parity")
    result = subprocess.run(
        [str(TEST_DIR / "test_infer_parity")], capture_output=True, text=True, check=True
    )
    assert f"conv parity: 0/{expect.size} bytes differ" in result.stdout, result.stdout
    assert "test_infer_parity OK" in result.stdout
