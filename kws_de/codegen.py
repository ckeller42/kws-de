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
import math
import pathlib
import re

import numpy as np

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


_ITEMSIZE = {"int8": 1, "uint8": 1, "int32": 4, "int16": 2, "float32": 4}
_ALIGN = 16  # esp-nn's S3 kernels want 16-byte-aligned operands


@dataclasses.dataclass(frozen=True)
class Arena:
    offsets: dict[int, int]  # tensor index -> byte offset in the arena
    size: int  # total arena bytes (16-byte aligned)


def tensor_bytes(graph: tflite_graph.Graph, index: int) -> int:
    t = graph.tensors[index]
    return math.prod(t.shape) * _ITEMSIZE[t.dtype]


def _round_up(value: int, align: int = _ALIGN) -> int:
    return -(-value // align) * align


def plan_arena(plan: Plan, scratch_bytes: int = 0) -> Arena:
    """Greedy first-fit over tensor lifetimes.

    A tensor is live from the op that writes it to the last op that reads it
    (aliases -- RESHAPE's zero-copy outputs -- are resolved to the tensor
    that actually owns the storage). Constants live in flash, ring buffers
    are their own static storage, and the graph input is a caller-supplied
    buffer outside the arena -- so the arena holds exactly the intermediate
    activations, including the graph's output (the caller reads it before the
    next call can reuse the slot). Scratch (esp-nn's conv workspace) is a
    single block at the end, live for the whole call, because two kernels
    never run at once.
    """
    graph = plan.graph
    fixed = {r.buffer_tensor for r in plan.rings} | set(graph.inputs)

    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for step, op in enumerate(plan.ops):
        for t in op.outputs:
            first.setdefault(t, step)
            last[t] = max(last.get(t, step), step)
        for t in op.inputs:
            t = plan.alias.get(t, t)
            if graph.tensors[t].data is None:
                last[t] = max(last.get(t, step), step)

    # A graph output has no consumer op -- its interval would otherwise
    # collapse to [producer_step, producer_step] and free its slot the
    # instant it's written, letting a later op's result land on the same
    # bytes and clobber the output before the caller reads it.
    end_step = len(plan.ops) - 1
    for t in graph.outputs:
        t = plan.alias.get(t, t)
        if t in first:
            last[t] = end_step

    candidates = [
        t
        for t in sorted(first, key=lambda t: (-tensor_bytes(graph, t), t))
        if t not in fixed and t not in plan.alias and graph.tensors[t].data is None
    ]

    offsets: dict[int, int] = {}
    placed: list[tuple[int, int, int, int]] = []  # (offset, end, first, last)
    for t in candidates:
        size = _round_up(tensor_bytes(graph, t))
        lo, hi = first[t], last[t]
        overlapping = sorted((off, end) for off, end, f, lst in placed if not (lst < lo or f > hi))
        offset = 0
        for off, end in overlapping:
            if offset + size <= off:
                break
            offset = max(offset, end)
        offsets[t] = offset
        placed.append((offset, offset + size, lo, hi))

    used = max((end for _, end, _, _ in placed), default=0)
    return Arena(offsets=offsets, size=_round_up(used + scratch_bytes))


_MACRO_RE = re.compile(r"^#define\s+(\w+)\s+(\d+)\s*$", re.MULTILINE)


def tflm_arena_bytes(header: pathlib.Path, macro: str) -> int:
    """The arena the firmware allocates for TFLM today, read from the committed
    gen/ header -- the ceiling the generated arena must not exceed."""
    for name, value in _MACRO_RE.findall(pathlib.Path(header).read_text()):
        if name == macro:
            return int(value)
    raise UnsupportedGraph(f"{header} does not define {macro}")


_QMIN, _QMAX = -128, 127


def _round_half_away(value: float) -> int:
    """TfLiteRound == std::round: ties away from zero. Python's round() is
    banker's rounding and would differ on exactly-.5 multipliers."""
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def quantize_multiplier(double_multiplier: float) -> tuple[int, int]:
    """Port of tflite::QuantizeMultiplier -> (quantized_multiplier, shift)."""
    if double_multiplier == 0.0:
        return 0, 0
    q, shift = math.frexp(double_multiplier)
    q_fixed = _round_half_away(q * (1 << 31))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return int(q_fixed), int(shift)


def _effective_scale(in_scale: float, filter_scale: float, out_scale: float) -> float:
    # TFLM reads all three as float32 and widens to double before dividing.
    return (
        float(np.float32(in_scale)) * float(np.float32(filter_scale)) / float(np.float32(out_scale))
    )


def per_channel_multipliers(graph, op) -> tuple[list[int], list[int]]:
    """(multipliers, shifts) per output channel, exactly as TFLM prepares them.
    A per-tensor filter scale is broadcast over the channels, as TFLM does."""
    in_t = graph.tensors[op.inputs[0]]
    w_t = graph.tensors[op.inputs[1]]
    out_t = graph.tensors[op.outputs[0]]
    channels = out_t.shape[-1]
    scales = w_t.scales if len(w_t.scales) > 1 else (w_t.scales[0],) * channels
    if len(scales) != channels:
        raise UnsupportedGraph(
            f"op {op.index} {op.name}: filter t{w_t.index} has {len(scales)} scales "
            f"for {channels} output channels"
        )
    mults, shifts = [], []
    for scale in scales:
        m, s = quantize_multiplier(_effective_scale(in_t.scales[0], scale, out_t.scales[0]))
        mults.append(m)
        shifts.append(s)
    return mults, shifts


def activation_range(graph, op) -> tuple[int, int]:
    """Port of CalculateActivationRangeQuantizedImpl for int8 outputs."""
    out = graph.tensors[op.outputs[0]]
    scale, zp = float(np.float32(out.scales[0])), int(out.zero_points[0])
    activation = str(op.options.get("fused_activation_function", "NONE"))

    def q(value: float) -> int:
        return zp + _round_half_away(value / scale)

    if activation == "RELU":
        return max(_QMIN, q(0.0)), _QMAX
    if activation == "RELU6":
        return max(_QMIN, q(0.0)), min(_QMAX, q(6.0))
    if activation == "RELU_N1_TO_1":
        return max(_QMIN, q(-1.0)), min(_QMAX, q(1.0))
    if activation != "NONE":
        raise UnsupportedGraph(f"op {op.index} {op.name}: activation {activation}")
    return _QMIN, _QMAX


def padding_hw(in_h, in_w, k_h, k_w, s_h, s_w, padding) -> tuple[int, int, int, int]:
    """(pad_h, pad_w, out_h, out_w) -- ComputeOutSize + ComputePadding, dilation 1."""
    if padding == "SAME":
        out_h, out_w = -(-in_h // s_h), -(-in_w // s_w)
    elif padding == "VALID":
        out_h, out_w = (in_h - k_h) // s_h + 1, (in_w - k_w) // s_w + 1
    else:
        raise UnsupportedGraph(f"unknown padding {padding!r}")
    pad_h = max(0, ((out_h - 1) * s_h + k_h - in_h) // 2)
    pad_w = max(0, ((out_w - 1) * s_w + k_w - in_w) // 2)
    return pad_h, pad_w, out_h, out_w


@dataclasses.dataclass
class Emitter:
    """Accumulates the pieces of one generated .c file."""

    prefix: str  # "wake" or "command"
    plan: Plan
    arena: Arena
    consts: list[str] = dataclasses.field(default_factory=list)
    body: list[str] = dataclasses.field(default_factory=list)
    scratch_bytes: int = 0

    def const_i8(self, name: str, values) -> str:
        flat = ", ".join(str(int(v)) for v in np.ravel(values))
        self.consts.append(f"static const int8_t {name}[{np.size(values)}] = {{{flat}}};")
        return name

    def const_i32(self, name: str, values) -> str:
        flat = ", ".join(str(int(v)) for v in np.ravel(values))
        self.consts.append(f"static const int32_t {name}[{np.size(values)}] = {{{flat}}};")
        return name

    def ref(self, tensor: int) -> str:
        """C expression for tensor `tensor`'s storage."""
        tensor = self.plan.alias.get(tensor, tensor)
        for ring in self.plan.rings:
            if ring.buffer_tensor == tensor:
                return ring.name
        if tensor in self.plan.graph.inputs:
            return "in"
        if tensor in self.plan.graph.outputs:
            return "out"
        return f"(arena + {self.arena.offsets[tensor]})"

    def emit(self, line: str) -> None:
        self.body.append("    " + line)


def _dims(name: str, width: int, height: int, channels: int, extra: int) -> str:
    return (
        f"const data_dims_t {name} = {{ .width = {width}, .height = {height}, "
        f".channels = {channels}, .extra = {extra} }};"
    )


def _nhwc(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 4 or shape[0] != 1:
        raise UnsupportedGraph(f"expected a [1, H, W, C] tensor, got {shape}")
    return shape[1], shape[2], shape[3]


def _conv_geometry(g, op):
    """Shapes, stride and padding shared by CONV_2D and DEPTHWISE_CONV_2D,
    with the tensor's own output shape used as the check on our padding math."""
    in_h, in_w, in_c = _nhwc(g.tensors[op.inputs[0]].shape)
    out_h, out_w, out_c = _nhwc(g.tensors[op.outputs[0]].shape)
    _, k_h, k_w, w_c = g.tensors[op.inputs[1]].shape
    if int(op.options.get("dilation_h_factor", 1)) != 1 or (
        int(op.options.get("dilation_w_factor", 1)) != 1
    ):
        # esp-tflite-micro sends dilated convolutions to the TFLM reference
        # kernel instead of esp-nn, so the esp-nn call would not be equivalent.
        raise UnsupportedGraph(f"op {op.index} {op.name}: dilation != 1 is not emitted")
    s_h, s_w = int(op.options["stride_h"]), int(op.options["stride_w"])
    pad_h, pad_w, exp_h, exp_w = padding_hw(
        in_h, in_w, k_h, k_w, s_h, s_w, str(op.options["padding"])
    )
    if (exp_h, exp_w) != (out_h, out_w):
        raise UnsupportedGraph(
            f"op {op.index} {op.name}: padding gives {exp_h}x{exp_w}, tensor "
            f"t{op.outputs[0]} says {out_h}x{out_w}"
        )
    return (in_h, in_w, in_c), (out_h, out_w, out_c), (k_h, k_w, w_c), (s_h, s_w), (pad_h, pad_w)


def _conv_constants(ctx: Emitter, op) -> tuple[str, str, str, str, str]:
    """weights, bias, multiplier and shift tables in flash, plus the op's tag."""
    g = ctx.plan.graph
    tag = f"op{op.index}"
    weights = ctx.const_i8(f"{tag}_w", tflite_graph.constant(g, op.inputs[1]))
    bias = ctx.const_i32(f"{tag}_b", tflite_graph.constant(g, op.inputs[2], "int32"))
    mults, shifts = per_channel_multipliers(g, op)
    return (
        tag,
        weights,
        bias,
        ctx.const_i32(f"{tag}_mult", mults),
        ctx.const_i32(f"{tag}_shift", shifts),
    )


def emit_conv(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    (in_h, in_w, in_c), (out_h, out_w, out_c), (k_h, k_w, w_c), (s_h, s_w), (pad_h, pad_w) = (
        _conv_geometry(g, op)
    )
    tag, weights, bias, mult, shift = _conv_constants(ctx, op)
    act_min, act_max = activation_range(g, op)
    in_zp = int(g.tensors[op.inputs[0]].zero_points[0])
    out_zp = int(g.tensors[op.outputs[0]].zero_points[0])
    ctx.emit("{")
    ctx.emit("  " + _dims(f"{tag}_in", in_w, in_h, in_c, 1))
    ctx.emit("  " + _dims(f"{tag}_out", out_w, out_h, out_c, 1))
    ctx.emit("  " + _dims(f"{tag}_flt", k_w, k_h, w_c, 0))
    ctx.emit(
        f"  const conv_params_t {tag}_p = {{ .in_offset = {-in_zp}, "
        f".out_offset = {out_zp}, .stride = {{ {s_w}, {s_h} }}, "
        f".padding = {{ {pad_w}, {pad_h} }}, .dilation = {{ 0, 0 }}, "
        f".activation = {{ {act_min}, {act_max} }} }};"
    )
    ctx.emit(
        f"  const quant_data_t {tag}_q = {{ .shift = (int32_t *){shift}, "
        f".mult = (int32_t *){mult} }};"
    )
    ctx.emit("  esp_nn_set_conv_scratch_buf(scratch);")
    ctx.emit(
        f"  esp_nn_conv_s8(&{tag}_in, {ctx.ref(op.inputs[0])}, &{tag}_flt, {weights}, "
        f"{bias}, &{tag}_out, {ctx.ref(op.outputs[0])}, &{tag}_p, &{tag}_q);"
    )
    ctx.emit("}")


def emit_depthwise(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    (in_h, in_w, in_c), (out_h, out_w, out_c), (k_h, k_w, w_c), (s_h, s_w), (pad_h, pad_w) = (
        _conv_geometry(g, op)
    )
    ch_mult = int(op.options.get("depth_multiplier", 1))
    if in_c * ch_mult != out_c:
        raise UnsupportedGraph(
            f"op {op.index} DEPTHWISE_CONV_2D: {in_c} x {ch_mult} != {out_c} output channels"
        )
    tag, weights, bias, mult, shift = _conv_constants(ctx, op)
    act_min, act_max = activation_range(g, op)
    in_zp = int(g.tensors[op.inputs[0]].zero_points[0])
    out_zp = int(g.tensors[op.outputs[0]].zero_points[0])
    ctx.emit("{")
    ctx.emit("  " + _dims(f"{tag}_in", in_w, in_h, in_c, 1))
    ctx.emit("  " + _dims(f"{tag}_out", out_w, out_h, out_c, 1))
    ctx.emit("  " + _dims(f"{tag}_flt", k_w, k_h, w_c, 0))
    ctx.emit(
        f"  const dw_conv_params_t {tag}_p = {{ .in_offset = {-in_zp}, "
        f".out_offset = {out_zp}, .ch_mult = {ch_mult}, "
        f".stride = {{ {s_w}, {s_h} }}, .padding = {{ {pad_w}, {pad_h} }}, "
        f".dilation = {{ 0, 0 }}, .activation = {{ {act_min}, {act_max} }} }};"
    )
    ctx.emit(
        f"  const quant_data_t {tag}_q = {{ .shift = (int32_t *){shift}, "
        f".mult = (int32_t *){mult} }};"
    )
    ctx.emit("  esp_nn_set_depthwise_conv_scratch_buf(scratch);")
    ctx.emit(
        f"  esp_nn_depthwise_conv_s8(&{tag}_in, {ctx.ref(op.inputs[0])}, &{tag}_flt, "
        f"{weights}, {bias}, &{tag}_out, {ctx.ref(op.outputs[0])}, &{tag}_p, &{tag}_q);"
    )
    ctx.emit("}")


def emit_fully_connected(ctx: Emitter, op) -> None:
    """A per-tensor filter scale takes esp_nn_fully_connected_s8; more than one
    scale is TFLM's `is_per_channel` path, which takes the _per_ch_ variant with
    multiplier/shift tables. Both branches exist in esp-tflite-micro's kernel and
    the model decides which -- our command model's classifier is per-channel."""
    g = ctx.plan.graph
    in_t = g.tensors[op.inputs[0]]
    w = g.tensors[op.inputs[1]]
    out_t = g.tensors[op.outputs[0]]
    out_depth = out_t.shape[-1]
    accum_depth = w.shape[-1]
    batches = math.prod(out_t.shape) // out_depth
    if batches != 1:
        raise UnsupportedGraph(
            f"op {op.index} FULLY_CONNECTED: output {out_t.shape} is {batches} batches; "
            "only a single batch is emitted"
        )
    tag = f"op{op.index}"
    weights = ctx.const_i8(f"{tag}_w", tflite_graph.constant(g, op.inputs[1]))
    bias = ctx.const_i32(f"{tag}_b", tflite_graph.constant(g, op.inputs[2], "int32"))
    act_min, act_max = activation_range(g, op)
    in_zp = int(in_t.zero_points[0])
    w_zp = int(w.zero_points[0])
    out_zp = int(out_t.zero_points[0])
    head = (
        f"{ctx.ref(op.inputs[0])}, {-in_zp}, {accum_depth}, {weights}, {-w_zp}, "
        f"{bias}, {ctx.ref(op.outputs[0])}, {out_depth}, {out_zp}, "
    )
    tail = f"{act_min}, {act_max});"
    if len(w.scales) > 1:
        mults, shifts = per_channel_multipliers(g, op)
        mult = ctx.const_i32(f"{tag}_mult", mults)
        shift = ctx.const_i32(f"{tag}_shift", shifts)
        ctx.emit(
            f"esp_nn_fully_connected_per_ch_s8({head}(int32_t *){shift}, (int32_t *){mult}, {tail}"
        )
        return
    mult, shift = quantize_multiplier(
        _effective_scale(in_t.scales[0], w.scales[0], out_t.scales[0])
    )
    ctx.emit(f"esp_nn_fully_connected_s8({head}{shift}, {mult}, {tail}")


EMITTERS = {
    "CONV_2D": emit_conv,
    "DEPTHWISE_CONV_2D": emit_depthwise,
    "FULLY_CONNECTED": emit_fully_connected,
}

_PROBE_SCRATCH = 8192  # host-only: esp-nn's ANSI kernels ask for no scratch at all


def write_probe_vectors(tflite: bytes, op, inputs, expect, gen_dir) -> None:
    """Emit gen/conv_probe.c plus gen/conv_probe_vectors.h: one op wrapped in a
    `conv_probe` function, with the interpreter's input and expected output as
    C arrays -- the smallest thing that proves the emitters' parameter
    preparation matches TFLM's, before a whole model is generated."""
    graph = tflite_graph.read_graph(tflite)
    # The probe's graph is one op: its activation input is the function's `in`
    # and its output is `out`, so Emitter.ref resolves both without an arena.
    probe = dataclasses.replace(graph, inputs=(op.inputs[0],), outputs=(op.outputs[0],))
    plan = Plan(graph=probe, ops=(op,), rings=(), alias={})
    ctx = Emitter(prefix="conv_probe", plan=plan, arena=plan_arena(plan))
    EMITTERS[op.name](ctx, op)
    gen_dir = pathlib.Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "conv_probe.c").write_text(
        "/* generated by kws-codegen -- do not edit */\n"
        '#include <stdint.h>\n#include "esp_nn.h"\n\n'
        + "\n".join(ctx.consts)
        + "\n\nvoid conv_probe(const int8_t *in, int8_t *out, void *scratch)\n{\n"
        # FULLY_CONNECTED takes no scratch buffer; the harness passes one anyway.
        + "    (void) scratch;\n"
        + "\n".join(ctx.body)
        + "\n}\n"
    )
    flat_in = ", ".join(str(int(v)) for v in np.ravel(inputs))
    flat_expect = ", ".join(str(int(v)) for v in np.ravel(expect))
    (gen_dir / "conv_probe_vectors.h").write_text(
        "/* generated by kws-codegen -- do not edit */\n#pragma once\n#include <stdint.h>\n"
        f"#define CONV_PROBE_OUT_LEN {np.size(expect)}\n"
        f"#define CONV_PROBE_SCRATCH {_PROBE_SCRATCH}\n"
        f"static const int8_t CONV_PROBE_IN[{np.size(inputs)}] = {{{flat_in}}};\n"
        f"static const int8_t CONV_PROBE_EXPECT[{np.size(expect)}] = {{{flat_expect}}};\n"
    )
