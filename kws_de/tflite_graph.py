"""Read a .tflite flatbuffer into plain dataclasses.

`tf.lite.Interpreter._get_ops_details()` gives op names and tensor indices but
no builtin options (stride, padding, fused activation) and no raw buffers, and
it hides the CALL_ONCE init subgraph behind XNNPACK's DELEGATE wrappers. The
code generator needs all of it exactly as the flatbuffer stores it, so we parse
the schema directly. `kws-model-graph` and `kws-codegen` share this reader so a
diagram and the generated C can never disagree about the same model.
"""

import dataclasses

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

DTYPES = {
    schema.TensorType.FLOAT32: "float32",
    schema.TensorType.INT32: "int32",
    schema.TensorType.UINT8: "uint8",
    schema.TensorType.INT64: "int64",
    schema.TensorType.INT16: "int16",
    schema.TensorType.INT8: "int8",
    schema.TensorType.RESOURCE: "resource",
    schema.TensorType.BOOL: "bool",
}
PADDING = {schema.Padding.SAME: "SAME", schema.Padding.VALID: "VALID"}
ACTIVATIONS = {
    schema.ActivationFunctionType.NONE: "NONE",
    schema.ActivationFunctionType.RELU: "RELU",
    schema.ActivationFunctionType.RELU_N1_TO_1: "RELU_N1_TO_1",
    schema.ActivationFunctionType.RELU6: "RELU6",
}
_NUMPY = {
    "float32": np.float32,
    "int32": np.int32,
    "uint8": np.uint8,
    "int64": np.int64,
    "int16": np.int16,
    "int8": np.int8,
    "bool": np.bool_,
}
_OP_NAMES = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}

# Builtin-option fields we care about, flatbuffers camelCase -> our snake_case.
_OPTION_FIELDS = {
    "padding": ("padding", lambda v: PADDING[v]),
    "strideW": ("stride_w", int),
    "strideH": ("stride_h", int),
    "dilationWFactor": ("dilation_w_factor", int),
    "dilationHFactor": ("dilation_h_factor", int),
    "depthMultiplier": ("depth_multiplier", int),
    "fusedActivationFunction": ("fused_activation_function", lambda v: ACTIVATIONS[v]),
    "filterWidth": ("filter_width", int),
    "filterHeight": ("filter_height", int),
    "axis": ("axis", int),
    "beta": ("beta", float),
    "keepDims": ("keep_dims", bool),
    "beginMask": ("begin_mask", int),
    "endMask": ("end_mask", int),
    "ellipsisMask": ("ellipsis_mask", int),
    "newAxisMask": ("new_axis_mask", int),
    "shrinkAxisMask": ("shrink_axis_mask", int),
    "initSubgraphIndex": ("init_subgraph_index", int),
    "sharedName": ("shared_name", lambda v: bytes(v).decode()),
}


@dataclasses.dataclass(frozen=True)
class Tensor:
    index: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    quantized_dimension: int
    data: bytes | None


@dataclasses.dataclass(frozen=True)
class Op:
    index: int
    name: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    options: dict[str, object]


@dataclasses.dataclass(frozen=True)
class Graph:
    ops: tuple[Op, ...]
    tensors: dict[int, Tensor]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    variables: dict[int, str]
    init_subgraph: int | None


def _opname(model, opcode_index: int) -> str:
    code = model.operatorCodes[opcode_index]
    return _OP_NAMES[max(int(code.builtinCode), int(code.deprecatedBuiltinCode))]


def _options(op) -> dict[str, object]:
    opts = op.builtinOptions
    if opts is None:
        return {}
    out: dict[str, object] = {}
    for raw, (name, cast) in _OPTION_FIELDS.items():
        if hasattr(opts, raw):
            value = getattr(opts, raw)
            if value is not None:
                out[name] = cast(value)
    return out


def _buffer(model, tensor) -> bytes | None:
    data = model.buffers[int(tensor.buffer)].data
    if data is None or len(data) == 0:
        return None
    return bytes(bytearray(data))


def _tensor(model, index: int, t) -> Tensor:
    q = t.quantization
    scales = tuple(float(s) for s in (q.scale if q is not None and q.scale is not None else ()))
    zps = tuple(int(z) for z in (q.zeroPoint if q is not None and q.zeroPoint is not None else ()))
    dim = int(q.quantizedDimension) if q is not None else 0
    dtype = DTYPES.get(int(t.type))
    if dtype is None:
        raise ValueError(f"tensor {index} ({bytes(t.name).decode()}): unsupported dtype {t.type}")
    return Tensor(
        index=index,
        name=bytes(t.name).decode() if t.name is not None else f"t{index}",
        shape=tuple(int(d) for d in (t.shape if t.shape is not None else ())),
        dtype=dtype,
        scales=scales,
        zero_points=zps,
        quantized_dimension=dim,
        data=_buffer(model, t),
    )


def read_graph(tflite: bytes, subgraph: int = 0) -> Graph:
    """Parse `tflite` into a Graph. Ops come back in execution order."""
    model = schema.ModelT.InitFromPackedBuf(bytearray(tflite), 0)
    if subgraph >= len(model.subgraphs):
        raise ValueError(f"model has {len(model.subgraphs)} subgraphs, asked for {subgraph}")
    sg = model.subgraphs[subgraph]
    tensors = {i: _tensor(model, i, t) for i, t in enumerate(sg.tensors)}
    ops = []
    variables: dict[int, str] = {}
    init_subgraph = None
    for i, op in enumerate(sg.operators):
        name = _opname(model, int(op.opcodeIndex))
        opts = _options(op)
        parsed = Op(
            index=i,
            name=name,
            inputs=tuple(int(t) for t in (op.inputs if op.inputs is not None else ())),
            outputs=tuple(int(t) for t in (op.outputs if op.outputs is not None else ())),
            options=opts,
        )
        if name == "VAR_HANDLE":
            variables[parsed.outputs[0]] = str(opts.get("shared_name", f"var{parsed.outputs[0]}"))
        elif name == "CALL_ONCE":
            init_subgraph = int(opts["init_subgraph_index"])
        ops.append(parsed)
    return Graph(
        ops=tuple(ops),
        tensors=tensors,
        inputs=tuple(int(t) for t in sg.inputs),
        outputs=tuple(int(t) for t in sg.outputs),
        variables=variables,
        init_subgraph=init_subgraph,
    )


def constant(graph: Graph, index: int, dtype: str | None = None) -> np.ndarray:
    """The constant buffer of tensor `index`, shaped and typed. Raises if the
    tensor is an activation (no buffer) -- never returns a silent zero array."""
    t = graph.tensors[index]
    if t.data is None:
        raise ValueError(f"tensor {index} ({t.name}) has no constant buffer")
    array = np.frombuffer(t.data, dtype=_NUMPY[dtype or t.dtype])
    return array.reshape(t.shape) if t.shape else array


def probe_model(tflite: bytes, op: Op, subgraph: int = 0) -> bytes:
    """A one-op .tflite holding just `op` and every tensor it touches (inputs,
    outputs, and any constant weights/bias among them), with the original
    quantisation, buffer contents and builtin options. Running this through
    tf.lite.Interpreter gives the reference kernel's exact answer for that op --
    which is how the LOGISTIC lookup table is derived bit-exactly instead of
    reimplementing gemmlowp's fixed-point sigmoid.

    Constant tensors (e.g. CONV_2D's weights/bias) are copied with their real
    buffer data and left out of the subgraph's inputs; only true activation
    tensors become subgraph inputs -- matching how the original model is laid
    out and satisfying kernels (e.g. conv.cc) that require the bias tensor to
    be present, not just the data tensor.
    """
    model = schema.ModelT.InitFromPackedBuf(bytearray(tflite), 0)
    sg = model.subgraphs[subgraph]
    src_op = sg.operators[op.index]
    src_code = model.operatorCodes[int(src_op.opcodeIndex)]

    out = schema.ModelT()
    out.version = 3
    out.description = b"kws-codegen probe"
    code = schema.OperatorCodeT()
    code.builtinCode = src_code.builtinCode
    code.deprecatedBuiltinCode = src_code.deprecatedBuiltinCode
    code.version = src_code.version
    out.operatorCodes = [code]

    buffers = [schema.BufferT()]  # buffer 0 is always the reserved empty buffer
    tensors: list = []
    is_constant: list[bool] = []
    remap: dict[int, int] = {}

    def add_tensor(src_index: int) -> int:
        if src_index in remap:
            return remap[src_index]
        src_t = sg.tensors[src_index]
        t = schema.TensorT()
        t.shape = list(src_t.shape)
        t.type = src_t.type
        t.name = src_t.name
        if src_t.quantization is not None:
            q = schema.QuantizationParametersT()
            scale = src_t.quantization.scale
            zero_point = src_t.quantization.zeroPoint
            q.scale = list(scale) if scale is not None else []
            q.zeroPoint = list(zero_point) if zero_point is not None else []
            q.quantizedDimension = src_t.quantization.quantizedDimension
            t.quantization = q
        src_data = model.buffers[int(src_t.buffer)].data
        if src_data is not None and len(src_data) > 0:
            buf = schema.BufferT()
            buf.data = list(bytearray(src_data))
            buffers.append(buf)
            t.buffer = len(buffers) - 1
            is_constant.append(True)
        else:
            t.buffer = 0
            is_constant.append(False)
        idx = len(tensors)
        tensors.append(t)
        remap[src_index] = idx
        return idx

    new_inputs = [add_tensor(int(i)) if int(i) >= 0 else -1 for i in op.inputs]
    new_outputs = [add_tensor(int(i)) for i in op.outputs]

    new_sg = schema.SubGraphT()
    new_sg.tensors = tensors
    seen: set[int] = set()
    new_sg.inputs = [
        idx
        for idx in new_inputs
        if idx != -1 and not is_constant[idx] and not (idx in seen or seen.add(idx))
    ]
    new_sg.outputs = new_outputs

    new_op = schema.OperatorT()
    new_op.opcodeIndex = 0
    new_op.inputs = new_inputs
    new_op.outputs = new_outputs
    new_op.builtinOptionsType = src_op.builtinOptionsType
    new_op.builtinOptions = src_op.builtinOptions
    new_sg.operators = [new_op]

    out.buffers = buffers
    out.subgraphs = [new_sg]
    builder = flatbuffers.Builder(1024)
    builder.Finish(out.Pack(builder), b"TFL3")
    return bytes(builder.Output())
