import dataclasses

import flatbuffers
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensorflow.lite.python import schema_py_generated as schema  # noqa: E402

from kws_de import tflite_graph  # noqa: E402 -- import after importorskip, on purpose


@pytest.fixture(scope="module")
def tiny_tflite() -> bytes:
    """A 2-op int8 model: Conv2D(3 filters, 3x3, relu) -> Dense(4). Small enough
    to assert exact structure, real enough to carry per-channel weight scales."""
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((8, 4, 1)),
            tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4),
        ]
    )
    rep = np.random.default_rng(0).normal(size=(16, 8, 4, 1)).astype(np.float32)

    def rep_gen():
        for i in range(rep.shape[0]):
            yield [rep[i : i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def test_ops_in_execution_order_with_options(tiny_tflite):
    g = tflite_graph.read_graph(tiny_tflite)
    names = [op.name for op in g.ops]
    assert names[0] == "CONV_2D"
    assert "FULLY_CONNECTED" in names
    conv = g.ops[0]
    assert conv.options["padding"] == "SAME"
    assert conv.options["stride_w"] == 1
    assert conv.options["stride_h"] == 1
    assert conv.options["fused_activation_function"] == "RELU"
    assert conv.options["dilation_w_factor"] == 1


def test_tensor_quantisation_and_constants(tiny_tflite):
    g = tflite_graph.read_graph(tiny_tflite)
    conv = g.ops[0]
    weights = g.tensors[conv.inputs[1]]
    assert weights.dtype == "int8"
    assert weights.shape == (3, 3, 3, 1)  # [out_c, kh, kw, in_c]
    assert len(weights.scales) == 3  # per-channel
    assert weights.quantized_dimension == 0
    assert weights.data is not None and len(weights.data) == 27
    out = g.tensors[conv.outputs[0]]
    assert out.data is None  # activation, not a constant
    assert len(out.scales) == 1 and out.scales[0] > 0
    bias = tflite_graph.constant(g, conv.inputs[2])
    assert bias.dtype == np.int32 and bias.shape == (3,)


def test_graph_io_and_no_variables(tiny_tflite):
    g = tflite_graph.read_graph(tiny_tflite)
    assert len(g.inputs) == 1 and len(g.outputs) == 1
    assert g.tensors[g.inputs[0]].dtype == "int8"
    assert g.variables == {}
    assert g.init_subgraph is None


def test_opname_reports_custom_code():
    """Custom ops (e.g. TFLite-Micro's PLACEHOLDER_FOR_GREATER_OP_CODES escape
    hatch) must name the op so refusals are actionable, not just "CUSTOM"."""
    model = schema.ModelT()
    model.version = 3
    code = schema.OperatorCodeT()
    code.builtinCode = schema.BuiltinOperator.CUSTOM
    code.customCode = b"MyCustomOp"
    model.operatorCodes = [code]
    t = schema.TensorT()
    t.shape = [1]
    t.type = schema.TensorType.INT8
    t.name = b"t0"
    t.buffer = 0
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [0]
    sg = schema.SubGraphT()
    sg.tensors = [t]
    sg.operators = [op]
    sg.inputs = [0]
    sg.outputs = [0]
    model.subgraphs = [sg]
    model.buffers = [schema.BufferT()]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b"TFL3")
    g = tflite_graph.read_graph(bytes(builder.Output()))
    assert g.ops[0].name == "CUSTOM:MyCustomOp"


def test_probe_model_reproduces_one_op(tiny_tflite):
    """A single-op model rebuilt from one op runs on the interpreter with the
    same quantisation -- this is how bit-exact activation tables are derived."""
    g = tflite_graph.read_graph(tiny_tflite)
    conv = g.ops[0]
    probe = tflite_graph.probe_model(tiny_tflite, conv)
    itp = tf.lite.Interpreter(model_content=probe)
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    assert tuple(int(d) for d in inp["shape"]) == g.tensors[conv.inputs[0]].shape
    assert tuple(int(d) for d in out["shape"]) == g.tensors[conv.outputs[0]].shape
    assert float(out["quantization"][0]) == pytest.approx(g.tensors[conv.outputs[0]].scales[0])


def test_probe_model_handles_missing_output(tiny_tflite):
    """The -1 sentinel (an op's optional output not produced) must pass through
    untouched, the same way probe_model already handles it on inputs."""
    g = tflite_graph.read_graph(tiny_tflite)
    conv = g.ops[0]
    op = dataclasses.replace(conv, outputs=(-1,))
    probe = tflite_graph.probe_model(tiny_tflite, op)
    m = schema.ModelT.InitFromPackedBuf(bytearray(probe), 0)
    assert list(m.subgraphs[0].operators[0].outputs) == [-1]
