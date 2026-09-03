"""Generate C inference from a .tflite graph, calling esp-nn directly.

The device today runs the TFLite-Micro interpreter: per-op dispatch, tensor
bookkeeping and an arena walk around every kernel call. Both our models are
overhead-bound, not arithmetic-bound, so this module emits the call sequence
as straight-line C with static buffers and precomputed requantisation, and the
interpreter stays only as the build-time fallback and the parity reference.

Everything here is bit-exactness-first: multipliers are prepared with TFLM's
own integer math, activation ranges with TFLM's own rounding, and anything the
generator does not understand is an error naming the op and tensor, never a
silent approximation.
"""

import dataclasses

from kws_de import tflite_graph

SUPPORTED_OPS = frozenset(
    {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "AVERAGE_POOL_2D",
        "MEAN",
        "SOFTMAX",
        "LOGISTIC",
        "QUANTIZE",
        "RESHAPE",
        "CONCATENATION",
        "STRIDED_SLICE",
        "VAR_HANDLE",
        "READ_VARIABLE",
        "ASSIGN_VARIABLE",
        "CALL_ONCE",
    }
)


class UnsupportedGraph(ValueError):
    """The generator refuses this graph. Message names the op and tensor."""


@dataclasses.dataclass(frozen=True)
class Ring:
    name: str
    var_tensor: int
    buffer_tensor: int
    new_tensor: int
    rows: int  # persistent ring size: what STRIDED_SLICE keeps, i.e. history rows
    new_rows: int
    channels: int
    bytes: int


@dataclasses.dataclass(frozen=True)
class Plan:
    graph: tflite_graph.Graph
    ops: tuple[tflite_graph.Op, ...]
    rings: tuple[Ring, ...]
    alias: dict[int, int]


def _rows_and_channels(shape: tuple[int, ...], tensor: int) -> tuple[int, int]:
    """A ring tensor is [1, rows, 1, C] (microWakeWord) or [1, rows, C]."""
    if len(shape) == 4 and shape[0] == 1 and shape[2] == 1:
        return shape[1], shape[3]
    if len(shape) == 3 and shape[0] == 1:
        return shape[1], shape[2]
    raise UnsupportedGraph(f"tensor t{tensor}: ring shape {shape} is not [1, rows, 1, C]")


def rewrite_streaming(graph: tflite_graph.Graph) -> Plan:
    """Collapse every resource variable's READ/CONCAT/SLICE/ASSIGN idiom into
    one ring buffer, and return the ops that actually compute something.

    Equivalence is checked, not assumed: the CONCATENATION must put history
    first and this step's rows last on axis 1, and the STRIDED_SLICE must drop
    exactly the oldest `new_rows` rows of that same buffer.
    """
    for op in graph.ops:
        if op.name not in SUPPORTED_OPS:
            out = op.outputs[0] if op.outputs else -1
            raise UnsupportedGraph(
                f"op {op.index} {op.name} (-> t{out}) is not in the supported set"
            )
        for t in (*op.inputs, *op.outputs):
            tensor = graph.tensors[t]
            if any(d <= 0 for d in tensor.shape):
                raise UnsupportedGraph(
                    f"op {op.index} {op.name}: tensor t{t} ({tensor.name}) has "
                    f"dynamic shape {tensor.shape}"
                )
            if tensor.dtype not in ("int8", "int32", "uint8", "resource"):
                raise UnsupportedGraph(
                    f"op {op.index} {op.name}: tensor t{t} ({tensor.name}) is "
                    f"{tensor.dtype}, only int8/int32/uint8 are supported"
                )

    def _index_once(name: str, value_of) -> dict[int, int]:
        """Map resource-variable tensor -> value tensor for every `name` op,
        raising if a variable is touched by more than one such op."""
        out: dict[int, int] = {}
        for op in graph.ops:
            if op.name != name:
                continue
            var = op.inputs[0]
            if var in out:
                raise UnsupportedGraph(
                    f"resource t{var} ({graph.variables.get(var, '?')}) has more than "
                    f"one {name} (op {op.index} duplicates an earlier one)"
                )
            out[var] = value_of(op)
        return out

    reads = _index_once("READ_VARIABLE", lambda op: op.outputs[0])
    assigns = _index_once("ASSIGN_VARIABLE", lambda op: op.inputs[1])
    concat_of = {op.inputs[0]: op for op in graph.ops if op.name == "CONCATENATION"}
    slice_of = {op.outputs[0]: op for op in graph.ops if op.name == "STRIDED_SLICE"}

    rings: list[Ring] = []
    alias: dict[int, int] = {}
    for i, var in enumerate(sorted(graph.variables)):
        if var not in reads or var not in assigns:
            raise UnsupportedGraph(
                f"resource t{var} ({graph.variables[var]}) is not read and assigned once each"
            )
        history = reads[var]
        concat = concat_of.get(history)
        if concat is None or concat.options.get("axis") != 1 or len(concat.inputs) != 2:
            raise UnsupportedGraph(
                f"resource t{var}: t{history} does not feed a 2-input CONCATENATION on axis 1"
            )
        buffer_tensor = concat.outputs[0]
        new_tensor = concat.inputs[1]
        sliced = assigns[var]
        cut = slice_of.get(sliced)
        if cut is None or cut.inputs[0] != buffer_tensor:
            raise UnsupportedGraph(
                f"resource t{var}: ASSIGN_VARIABLE source t{sliced} is not a "
                f"STRIDED_SLICE of the ring buffer t{buffer_tensor}"
            )
        # `rows` is the ring's persistent size -- what STRIDED_SLICE keeps and
        # hands back to ASSIGN_VARIABLE, i.e. what the resource variable holds
        # between steps. The CONCATENATION output is transient: history (rows)
        # plus this step's new_rows, consumed by the compute op and then
        # trimmed straight back down to rows.
        rows, channels = _rows_and_channels(graph.tensors[sliced].shape, sliced)
        total_rows, total_c = _rows_and_channels(graph.tensors[buffer_tensor].shape, buffer_tensor)
        new_rows, new_c = _rows_and_channels(graph.tensors[new_tensor].shape, new_tensor)
        if channels != total_c or channels != new_c or rows + new_rows != total_rows:
            raise UnsupportedGraph(
                f"resource t{var}: ring buffer t{buffer_tensor} {total_rows}x{total_c} "
                f"does not decompose into kept {rows}x{channels} + new {new_rows}x{new_c}"
            )
        begin = tflite_graph.constant(graph, cut.inputs[1], "int32")
        strides = tflite_graph.constant(graph, cut.inputs[3], "int32")
        # begin_mask bit d set => begin[d] ignored (start at 0). The shift is
        # only equivalent when axis 1 starts at new_rows and every other axis is
        # taken whole with stride 1. Per the STRIDED_SLICE spec, a negative begin
        # counts back from the end of the axis (begin[1] = -12 on a 13-row buffer
        # means row 1), so normalise against the buffer's total row count before
        # comparing to new_rows.
        begin1 = int(begin[1]) + total_rows if int(begin[1]) < 0 else int(begin[1])
        begin_mask = int(cut.options.get("begin_mask", 0))
        if (begin_mask >> 1) & 1 or begin1 != new_rows or not all(int(s) == 1 for s in strides):
            raise UnsupportedGraph(
                f"resource t{var}: STRIDED_SLICE t{sliced} begin={list(begin)} "
                f"mask={begin_mask} strides={list(strides)} is not a shift by {new_rows} rows"
            )
        rings.append(
            Ring(
                name=f"ring{i}",
                var_tensor=var,
                buffer_tensor=buffer_tensor,
                new_tensor=new_tensor,
                rows=rows,
                new_rows=new_rows,
                channels=channels,
                bytes=rows * channels,
            )
        )

    ring_concat_outputs = {r.buffer_tensor for r in rings}
    ring_slice_outputs = {assigns[r.var_tensor] for r in rings}

    ops = []
    for op in graph.ops:
        if op.name in ("VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE", "CALL_ONCE"):
            continue
        if op.name == "CONCATENATION":
            if op.outputs[0] not in ring_concat_outputs:
                raise UnsupportedGraph(
                    f"op {op.index} CONCATENATION (-> t{op.outputs[0]}) is not part of "
                    "a detected ring; the emitters only support ring concatenation"
                )
            continue
        if op.name == "STRIDED_SLICE":
            if op.outputs[0] not in ring_slice_outputs:
                raise UnsupportedGraph(
                    f"op {op.index} STRIDED_SLICE (-> t{op.outputs[0]}) is not part of "
                    "a detected ring; the emitters only support ring slicing"
                )
            continue
        if op.name == "RESHAPE":
            # Same bytes, different shape: the output shares the input's storage.
            alias[op.outputs[0]] = alias.get(op.inputs[0], op.inputs[0])
        ops.append(op)
    return Plan(graph=graph, ops=tuple(ops), rings=tuple(rings), alias=alias)
