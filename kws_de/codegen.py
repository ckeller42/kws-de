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

import argparse
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


def _check_single_io(graph: tflite_graph.Graph) -> None:
    """Emitter.ref and _render only ever look at graph.inputs[0]/outputs[0]; a
    second input or output would silently alias onto the first one's pointer
    instead of erroring, the "silent approximation" this module refuses."""
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise UnsupportedGraph(
            f"graph has {len(graph.inputs)} input(s) {list(graph.inputs)} and "
            f"{len(graph.outputs)} output(s) {list(graph.outputs)}; only exactly "
            "one of each is emitted"
        )


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
            if t < 0:
                continue  # TFLite's "absent optional tensor"; the emitters refuse it
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
    scratch_offset: int = 0  # where esp-nn's scratch block starts


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
            if t >= 0 and graph.tensors[t].data is None:
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

    used = _round_up(max((end for _, end, _, _ in placed), default=0))
    return Arena(offsets=offsets, size=used + _round_up(scratch_bytes), scratch_offset=used)


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


# esp-nn's ESP32-S3 kernels do not allocate: the caller hands them a scratch
# buffer and esp_nn_get_{conv,depthwise_conv}_scratch_size_esp32s3() says how
# big. The host build links the ANSI kernels, whose _ansi variants return 0, so
# the host can never discover the device's requirement -- an arena sized by the
# host would overrun on the S3. These are ports of the S3 functions (esp-nn
# v1.3.0 `src/convolution/esp_nn_conv_esp32s3.c` and
# `src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c`), branch for branch, so
# the generated arena is sized for what actually runs on the device.
_ESP_NN_ALIGN_MARGIN = 64  # alignment (16) + assembly pre/post access margin (48)


def _conv_scratch(in_hw_c, out_hw_c, k_hw, stride, pad) -> int:
    """Port of esp_nn_get_conv_scratch_size_esp32s3."""
    in_h, in_w, in_c = in_hw_c
    out_h, out_w, out_c = out_hw_c
    k_h, k_w = k_hw
    s_h, s_w = stride
    pad_h, pad_w = pad
    if (k_w, k_h, pad_w, pad_h, s_w, s_h) == (1, 1, 0, 0, 1, 1):
        new_c = (in_c + 7) & ~7
        transpose = 0 if in_w * in_h < 8 else 2 * (8 * new_c)
        input_scratch = in_w * in_h * new_c if in_c % 8 else 0
        return input_scratch + new_c * out_c + transpose + _ESP_NN_ALIGN_MARGIN
    filter_row = k_w * in_c
    window = k_w * k_h * in_c
    if filter_row < 16 <= window:  # im2col path
        aligned = (window + 15) & ~15
        return out_c * 4 + 16 + out_c * aligned + 16 + aligned + _ESP_NN_ALIGN_MARGIN
    new_c = (in_c + 15) & ~15
    pad_right = max(0, (out_w - 1) * s_w + k_w - pad_w - in_w)
    pad_bottom = max(0, (out_h - 1) * s_h + k_h - pad_h - in_h)
    if (pad_w, pad_h, pad_right, pad_bottom) == (0, 0, 0, 0):
        input_scratch = 0
    else:
        input_scratch = (in_w + pad_w + pad_right) * (in_h + pad_h + pad_bottom) * in_c
    filter_scratch = k_w * k_h * new_c * out_c
    aligned_row = ((filter_row + 15) // 16) * 16
    return (
        input_scratch
        + filter_scratch
        + aligned_row * k_h * out_c
        + _ESP_NN_ALIGN_MARGIN
        + out_c * 4
    )


def _depthwise_scratch(in_hw_c, out_hw_c, k_hw, stride, pad, ch_mult) -> int:
    """Port of esp_nn_get_depthwise_conv_scratch_size_esp32s3."""
    in_h, in_w, channels = in_hw_c
    out_h, out_w, _ = out_hw_c
    k_h, k_w = k_hw
    s_h, s_w = stride
    pad_h, pad_w = pad
    filter_size = k_w * k_h * channels * ch_mult
    input_size = in_w * in_h * channels
    if ch_mult == 1 and channels % 8 == 0:
        if (k_w, k_h) != (3, 3):
            total_s16 = 2 * (filter_size + input_size)
            if total_s16 <= 48 * 1024:
                return total_s16 + 32
            return 2 * filter_size + 2 * in_w * k_h * channels + 32  # tiled
        if channels % 16 == 0 and (pad_w, pad_h) in ((1, 1), (0, 0)):
            if pad_w or pad_h:
                pad_width, pad_height = pad_w * 2, pad_h * 2
            else:
                pad_width = (out_w * s_w + k_w - 1) - in_w
                pad_height = (out_h * s_h + k_h - 1) - in_h
            if not (pad_width or pad_height):
                return filter_size + 16
            full_input = (in_w + pad_width) * (in_h + pad_height) * channels
            if full_input <= 40 * 1024:
                return filter_size + full_input + 16
            return filter_size + (in_w + pad_width) * k_h * channels + 16  # tiled
        if channels >= 12:
            new_ch = (channels + 15) & ~15
            total_pad_wd = pad_w * 2 + max(0, (out_w * s_w + 2) - in_w)
            total_pad_ht = pad_h * 2 + max(0, (out_h * s_h + 2) - in_h)
            new_input = (in_w + total_pad_wd) * (in_h + total_pad_ht) * new_ch
            return 9 * new_ch + new_input + out_w * out_h * new_ch + 64
        return 2 * (filter_size + input_size) + 32  # channels == 8: s16 path
    if ch_mult == 1 and channels > 3:
        padded = (channels + 7) & ~7
        input_start = (k_w * k_h * padded * 2 + 15) & ~15
        out_start = (input_start + in_w * in_h * padded * 2 + 15) & ~15
        bias_start = (out_start + out_w * out_h * padded + 15) & ~15
        return bias_start + 3 * padded * 4 + 16
    if ch_mult % 4 == 0:
        return 2 * (filter_size + input_size) + 32
    return 32


def scratch_bytes(graph, op) -> int:
    """Scratch bytes esp-nn's ESP32-S3 kernels need for `op` (0 if none)."""
    if op.name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        in_hw_c, out_hw_c, (k_h, k_w, _), stride, pad = _conv_geometry(graph, op)
        if op.name == "CONV_2D":
            return _conv_scratch(in_hw_c, out_hw_c, (k_h, k_w), stride, pad)
        ch_mult = int(op.options.get("depth_multiplier", 1))
        return _depthwise_scratch(in_hw_c, out_hw_c, (k_h, k_w), stride, pad, ch_mult)
    if op.name == "SOFTMAX":
        # esp_nn_get_softmax_scratch_size_esp32s3: one int32 per depth column.
        return 4 * int(graph.tensors[op.inputs[0]].shape[-1])
    return 0


def _check_u16(op, **values: int) -> None:
    """esp-nn narrows several kernel arguments to uint16_t. A model that
    overflows one would wrap silently at the call boundary, so refuse it here."""
    for name, value in values.items():
        if not 0 < int(value) <= 0xFFFF:
            raise UnsupportedGraph(
                f"op {op.index} {op.name}: {name}={value} does not fit the "
                "uint16_t esp-nn narrows it to"
            )


def _check_int8(op, **tensors) -> None:
    """LOGISTIC/MEAN/AVERAGE_POOL_2D read their operands through a plain
    int8_t* (unlike QUANTIZE, which is the one op that legitimately crosses
    int8/uint8). A uint8 tensor read that way is silently misinterpreted --
    refuse it by name instead."""
    for name, t in tensors.items():
        if t.dtype != "int8":
            raise UnsupportedGraph(
                f"op {op.index} {op.name}: {name} t{t.index} is {t.dtype}, only int8 is emitted"
            )


def _bias(ctx: "Emitter", op, tag: str) -> str:
    """The op's bias table. TFLite encodes an absent optional bias as tensor
    index -1, which would index Python's *last* tensor -- somebody else's
    buffer emitted as this op's bias. Refuse instead."""
    if len(op.inputs) < 3 or op.inputs[2] < 0:
        raise UnsupportedGraph(
            f"op {op.index} {op.name}: no bias tensor (input 2 is absent); "
            "only biased ops are emitted"
        )
    return ctx.const_i32(f"{tag}_b", tflite_graph.constant(ctx.plan.graph, op.inputs[2], "int32"))


# esp-nn's Kconfig can turn on CONFIG_NN_SKIP_NUDGE ("use fast (non-bit-exact)
# requantization"), which silently stops matching TFLM's reference arithmetic.
# Every generated translation unit carries this so an sdkconfig flip is a build
# failure on the device, not a quiet loss of parity.
NUDGE_GUARD = (
    "#if defined(SKIP_NUDGE) || defined(CONFIG_NN_SKIP_NUDGE)\n"
    '#error "esp-nn SKIP_NUDGE requantisation is not bit-exact; '
    'this model was generated for the exact path"\n'
    "#endif\n"
)


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
    bias = _bias(ctx, op, tag)
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
    _check_u16(op, in_h=in_h, in_w=in_w, in_c=in_c, out_h=out_h, out_w=out_w, out_c=out_c)
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
    _check_u16(op, in_h=in_h, in_w=in_w, in_c=in_c, out_h=out_h, out_w=out_w, out_c=out_c)
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
    _check_u16(op, accum_depth=accum_depth, out_depth=out_depth)
    tag = f"op{op.index}"
    weights = ctx.const_i8(f"{tag}_w", tflite_graph.constant(g, op.inputs[1]))
    bias = _bias(ctx, op, tag)
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


_SCALED_DIFF_INTEGER_BITS = 5


def softmax_params_from_scale(input_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    """PreprocessSoftmaxScaling + diff_min, as CalculateSoftmaxParams does."""
    bits = _SCALED_DIFF_INTEGER_BITS
    real = min(beta * float(np.float32(input_scale)) * (1 << (31 - bits)), float((1 << 31) - 1))
    mult, shift = quantize_multiplier(real)
    # CalculateInputRadius(bits, shift, 31), floor of an exactly representable
    # quotient, so the shift is the same thing without the double round-trip.
    return mult, shift, -(((1 << bits) - 1) * (1 << (31 - bits)) >> shift)


def softmax_params(graph, op) -> tuple[int, int, int]:
    """(input_multiplier, input_left_shift, diff_min) for an int8 SOFTMAX."""
    in_t, out_t = graph.tensors[op.inputs[0]], graph.tensors[op.outputs[0]]
    if int(out_t.zero_points[0]) != -128 or float(np.float32(out_t.scales[0])) != 1.0 / 256:
        raise UnsupportedGraph(
            f"op {op.index} SOFTMAX: output t{op.outputs[0]} must be int8 with scale "
            f"1/256 and zero point -128, got {out_t.scales[0]}/{out_t.zero_points[0]}"
        )
    mult, shift, diff_min = softmax_params_from_scale(
        in_t.scales[0], float(op.options.get("beta", 1.0))
    )
    if shift < 0:
        # esp_nn_softmax_s8 computes `1 << shift`; a negative one is undefined.
        raise UnsupportedGraph(
            f"op {op.index} SOFTMAX: input scale {in_t.scales[0]} gives left shift {shift}"
        )
    return mult, shift, diff_min


def logistic_lut(tflite: bytes, op) -> list[int]:
    """The reference LOGISTIC kernel's answer for every possible int8 input.

    gemmlowp's fixed-point sigmoid is not worth reimplementing: the input is one
    byte wide, so run the reference kernel itself over all 256 values through a
    one-op probe model and freeze the result as a table. Bit-exact by
    construction, and it cannot drift from the kernel it was taken from.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_content=tflite_graph.probe_model(tflite, op),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    detail_in = interpreter.get_input_details()[0]
    detail_out = interpreter.get_output_details()[0]
    lut = []
    for q in range(-128, 128):
        interpreter.set_tensor(detail_in["index"], np.full(detail_in["shape"], q, np.int8))
        interpreter.invoke()
        lut.append(int(interpreter.get_tensor(detail_out["index"]).flat[0]))
    return lut


def emit_logistic(ctx: Emitter, op, lut) -> None:
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    _check_int8(op, input=in_t, output=out_t)
    table = ctx.const_i8(f"op{op.index}_lut", lut)
    count = math.prod(in_t.shape)
    src, dst = ctx.ref(op.inputs[0]), ctx.ref(op.outputs[0])
    ctx.emit(f"for (int i = 0; i < {count}; i++)")
    ctx.emit(f"    {dst}[i] = {table}[(uint8_t)({src}[i] + 128)];")


def emit_quantize(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    count = math.prod(in_t.shape)
    in_zp, out_zp = int(in_t.zero_points[0]), int(out_t.zero_points[0])
    mult, shift = quantize_multiplier(
        float(np.float32(in_t.scales[0])) / float(np.float32(out_t.scales[0]))
    )
    cast = "uint8_t" if out_t.dtype == "uint8" else "int8_t"
    in_cast = "uint8_t" if in_t.dtype == "uint8" else "int8_t"
    src, dst = ctx.ref(op.inputs[0]), ctx.ref(op.outputs[0])
    same_scale = (mult, shift) == (1 << 30, 1)  # reference/requantize.h's own test
    mixed = (in_t.dtype, out_t.dtype) in (("int8", "uint8"), ("uint8", "int8"))
    if same_scale and mixed and abs(in_zp - out_zp) == 128:
        # reference/requantize.h fast path: a pure 128 shift is a sign-bit flip,
        # byte-identical whether the source is read as int8_t or uint8_t.
        ctx.emit(f"for (int i = 0; i < {count}; i++)")
        ctx.emit(f"    (({cast} *){dst})[i] = ({cast})({src}[i] ^ 0x80);")
        return
    lo, hi = (0, 255) if out_t.dtype == "uint8" else (-128, 127)
    ctx.emit(f"for (int i = 0; i < {count}; i++) {{")
    ctx.emit(
        f"    int32_t v = kws_requantize(((const {in_cast} *){src})[i] - {in_zp}, "
        f"{mult}, {shift}) + {out_zp};"
    )
    ctx.emit(f"    (({cast} *){dst})[i] = (v < {lo}) ? {lo} : ((v > {hi}) ? {hi} : v);")
    ctx.emit("}")


def emit_softmax(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    shape = g.tensors[op.inputs[0]].shape
    depth = shape[-1]
    outer = math.prod(shape) // depth
    mult, shift, diff_min = softmax_params(g, op)
    ctx.emit("esp_nn_set_softmax_scratch_buf(scratch);")
    ctx.emit(
        f"esp_nn_softmax_s8({ctx.ref(op.inputs[0])}, {outer}, {depth}, "
        f"{mult}, {shift}, {diff_min}, {ctx.ref(op.outputs[0])});"
    )


def emit_mean(ctx: Emitter, op) -> None:
    """MEAN over (H, W). esp_nn_mean_nhwc_s8 is TFLM's QuantizedMeanOrSum with
    the 1/(H*W) already folded into the multiplier -- which is where the
    precision lives, so the fold is done exactly as the reducer does it (an
    AVERAGE_POOL_2D export of the same layer drifts by up to 90 LSB)."""
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    _check_int8(op, input=in_t, output=out_t)
    in_h, in_w, in_c = _nhwc(in_t.shape)
    axes = sorted(int(a) for a in tflite_graph.constant(g, op.inputs[1], "int32"))
    if axes != [1, 2] or out_t.shape[-1] != in_c or math.prod(out_t.shape) != in_c:
        raise UnsupportedGraph(
            f"op {op.index} MEAN: only reduction over axes [1, 2] of a [1, H, W, C] "
            f"tensor is emitted, got axes {axes} and output shape {out_t.shape}"
        )
    mult, shift = quantize_multiplier(
        float(np.float32(in_t.scales[0])) / float(np.float32(out_t.scales[0]))
    )
    # Readapt the rescale to fold in 1 / (H * W), exactly as QuantizedMeanOrSum:
    #   s = min(63 - clz64(n), 32, 31 + output_shift); mult = (mult << s) / n
    # 63 - clz64(n) is floor(log2(n)), i.e. n.bit_length() - 1 for n >= 1.
    n = in_h * in_w
    s = min(n.bit_length() - 1, 32, 31 + shift)
    mult, shift = (mult << s) // n, shift - s
    ctx.emit(
        f"esp_nn_mean_nhwc_s8({ctx.ref(op.inputs[0])}, {ctx.ref(op.outputs[0])}, "
        f"{in_h}, {in_w}, {in_c}, {int(in_t.zero_points[0])}, "
        f"{int(out_t.zero_points[0])}, {mult}, {shift});"
    )


def emit_average_pool(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    _check_int8(op, input=in_t, output=out_t)
    if float(np.float32(in_t.scales[0])) != float(np.float32(out_t.scales[0])) or int(
        in_t.zero_points[0]
    ) != int(out_t.zero_points[0]):
        raise UnsupportedGraph(
            f"op {op.index} AVERAGE_POOL_2D: esp_nn_avg_pool_s8 does not requantise, "
            f"but t{op.inputs[0]} and t{op.outputs[0]} have different quantisation"
        )
    in_h, in_w, in_c = _nhwc(in_t.shape)
    out_h, out_w, _ = _nhwc(out_t.shape)
    k_h, k_w = int(op.options["filter_height"]), int(op.options["filter_width"])
    s_h, s_w = int(op.options["stride_h"]), int(op.options["stride_w"])
    pad_h, pad_w, exp_h, exp_w = padding_hw(
        in_h, in_w, k_h, k_w, s_h, s_w, str(op.options["padding"])
    )
    if (exp_h, exp_w) != (out_h, out_w):
        raise UnsupportedGraph(
            f"op {op.index} AVERAGE_POOL_2D: padding gives {exp_h}x{exp_w}, tensor "
            f"t{op.outputs[0]} says {out_h}x{out_w}"
        )
    _check_u16(op, in_h=in_h, in_w=in_w, in_c=in_c, out_h=out_h, out_w=out_w)
    act_min, act_max = activation_range(g, op)
    ctx.emit(
        f"esp_nn_avg_pool_s8({ctx.ref(op.inputs[0])}, {in_w}, {in_h}, "
        f"{ctx.ref(op.outputs[0])}, {out_w}, {out_h}, {s_w}, {s_h}, {k_w}, {k_h}, "
        f"{pad_w}, {pad_h}, {act_min}, {act_max}, {in_c});"
    )


EMITTERS = {
    "CONV_2D": emit_conv,
    "DEPTHWISE_CONV_2D": emit_depthwise,
    "FULLY_CONNECTED": emit_fully_connected,
    "MEAN": emit_mean,
    "SOFTMAX": emit_softmax,
}


def initial_states(tflite: bytes, graph: tflite_graph.Graph) -> dict[int, int]:
    """resource tensor -> the byte its CALL_ONCE init subgraph fills it with.

    microWakeWord's ring states start at the quantised zero, which is -128
    (0x80) and not 0 for these tensors -- a reset that zeroed them would
    disagree with the interpreter for the whole first second of every clip.
    Read the value out of the model instead of assuming one.
    """
    if graph.init_subgraph is None:
        return {}
    init = tflite_graph.read_graph(tflite, graph.init_subgraph)
    assigns = {op.inputs[0]: op.inputs[1] for op in init.ops if op.name == "ASSIGN_VARIABLE"}
    fill: dict[int, int] = {}
    for var, name in graph.variables.items():
        source = next((t for t, n in init.variables.items() if n == name), None)
        if source is None or source not in assigns:
            raise UnsupportedGraph(f"resource t{var} ({name}) is not initialised by CALL_ONCE")
        data = init.tensors[assigns[source]].data
        if data is None or len(set(data)) != 1:
            raise UnsupportedGraph(
                f"resource t{var} ({name}): CALL_ONCE initialiser is not a constant fill"
            )
        fill[var] = data[0]
    return fill


_PROLOGUE = """/* generated by kws-codegen from {model} -- do not edit */
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_nn.h"

{guard}"""

# TFLM's MultiplyByQuantizedMultiplier for the ops esp-nn does not cover: the
# same gemmlowp double-rounding path esp_nn_requantize takes when SKIP_NUDGE is
# not defined (which the guard above makes sure of).
_REQUANTIZE = """
static inline int32_t kws_requantize(int32_t x, int32_t mult, int32_t shift)
{
    int32_t left = shift > 0 ? shift : 0;
    int32_t right = shift > 0 ? 0 : -shift;
    int64_t ab = (int64_t)(x * (1 << left)) * (int64_t)mult;
    int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t high = (int32_t)((ab + nudge) / (1LL << 31));
    if (right == 0) return high;
    int32_t mask = (1 << right) - 1;
    int32_t rem = high & mask;
    int32_t threshold = (mask >> 1) + (high < 0 ? 1 : 0);
    return (high >> right) + (rem > threshold ? 1 : 0);
}
"""


def generate(tflite: bytes, name: str) -> dict[str, str]:
    """Return {"<name>_infer.c": source, "<name>_infer.h": header}."""
    graph = tflite_graph.read_graph(tflite)
    _check_single_io(graph)
    plan = rewrite_streaming(graph)
    scratch = max((scratch_bytes(graph, op) for op in plan.ops), default=0)
    ctx = Emitter(
        prefix=name,
        plan=plan,
        arena=plan_arena(plan, scratch_bytes=scratch),
        scratch_bytes=scratch,
    )

    # The CONCATENATION that built each ring is gone, so the rows it appended
    # are copied into the ring's tail right where the op producing them ran.
    feeds: dict[int, list[Ring]] = {}
    for ring in plan.rings:
        feeds.setdefault(plan.alias.get(ring.new_tensor, ring.new_tensor), []).append(ring)

    def _fill(tensor: int) -> None:
        for ring in feeds.get(tensor, ()):
            ctx.emit(
                f"memcpy({ring.name} + {ring.bytes}, {ctx.ref(ring.new_tensor)}, "
                f"{ring.new_rows * ring.channels});"
            )

    for tensor in graph.inputs:
        _fill(tensor)
    for op in plan.ops:
        if op.name == "RESHAPE":
            pass  # an alias, no code
        elif op.name == "CONV_2D":
            emit_conv(ctx, op)
        elif op.name == "DEPTHWISE_CONV_2D":
            emit_depthwise(ctx, op)
        elif op.name == "FULLY_CONNECTED":
            emit_fully_connected(ctx, op)
        elif op.name == "AVERAGE_POOL_2D":
            emit_average_pool(ctx, op)
        elif op.name == "MEAN":
            emit_mean(ctx, op)
        elif op.name == "SOFTMAX":
            emit_softmax(ctx, op)
        elif op.name == "LOGISTIC":
            emit_logistic(ctx, op, logistic_lut(tflite, op))
        elif op.name == "QUANTIZE":
            emit_quantize(ctx, op)
        else:
            raise UnsupportedGraph(f"op {op.index} {op.name}: no emitter")
        for t in op.outputs:
            _fill(t)
    return _render(ctx, name, initial_states(tflite, graph))


def _render(ctx: Emitter, name: str, fill: dict[int, int]) -> dict[str, str]:
    graph = ctx.plan.graph
    in_t = graph.tensors[graph.inputs[0]]
    out_t = graph.tensors[graph.outputs[0]]
    in_len, out_len = math.prod(in_t.shape), math.prod(out_t.shape)
    out_ctype = "uint8_t" if out_t.dtype == "uint8" else "int8_t"
    rings = ctx.plan.rings

    # esp-nn's kernels want 16-byte-aligned operands; the generated constant
    # tables are copied into the (aligned) scratch by the S3 kernels themselves,
    # so only the buffers the kernels read and write directly are declared so.
    statics = [f"static int8_t arena[{ctx.arena.size}] __attribute__((aligned(16)));"]
    if ctx.scratch_bytes:
        statics.append(f"static int8_t *const scratch = arena + {ctx.arena.scratch_offset};")
    statics += [
        f"static int8_t {r.name}[{r.bytes + r.new_rows * r.channels}] __attribute__((aligned(16)));"
        for r in rings
    ]

    lines = [_PROLOGUE.format(model=name, guard=NUDGE_GUARD)]
    if any("kws_requantize" in line for line in ctx.body):
        lines.append(_REQUANTIZE)
    lines += ["\n".join(statics), "", "\n".join(ctx.consts), ""]

    # One name, one number: ARENA_BYTES/arena_bytes() is the transient planner
    # arena (activations + esp-nn scratch), reused every call; STATE_BYTES/
    # state_bytes() is the ring storage that persists *between* calls (0 for a
    # stateless model like command). Task 6 compares the arena number, not the
    # sum of both, against TFLM's arena ceiling.
    state_bytes = " + ".join(f"sizeof {r.name}" for r in rings) or "0"

    reset = [f"void {name}_infer_reset(void)", "{"]
    reset += [f"    memset({r.name}, {fill[r.var_tensor]}, sizeof {r.name});" for r in rings]
    reset += [
        "}",
        "",
        f"void {name}_infer_init(void)",
        "{",
        f"    {name}_infer_reset();",
        "}",
        "",
        f"size_t {name}_infer_arena_bytes(void)",
        "{",
        "    return sizeof arena;",
        "}",
        "",
        f"size_t {name}_infer_state_bytes(void)",
        "{",
        f"    return {state_bytes};",
        "}",
    ]
    lines.append("\n".join(reset))

    if rings:
        signature = f"void {name}_infer_step(const int8_t in[{in_len}], {out_ctype} *out)"
        shifts = [
            f"    memmove({r.name}, {r.name} + {r.new_rows * r.channels}, {r.bytes});"
            for r in rings
        ]
        decl = (
            f"void {name}_infer_step(const int8_t in[{in_t.shape[-2]} * {in_t.shape[-1]}], "
            f"{out_ctype} *prob_q);\n"
        )
        body = [signature, "{", *ctx.body, *shifts, "}"]
    else:
        signature = f"void {name}_infer(const int8_t in[{in_len}], {out_ctype} out[{out_len}])"
        decl = (
            "/* `in` and `out` must be 16-byte aligned: esp-nn's S3 kernels take "
            "them as direct operands, with no arena copy in front of either "
            "(checked on `in` in debug builds; `out` is never read back here to "
            "check against). */\n"
            f"void {name}_infer(const int8_t in[{in_len}], {out_ctype} out[{out_len}]);\n"
        )
        # Cheap, debug-only: the caller-owned `in` buffer is not one of the
        # generator's own aligned statics (arena/rings), and nothing else here
        # would catch a misaligned pointer before esp-nn silently mis-reads it.
        body = [
            signature,
            "{",
            "#ifndef NDEBUG",
            "    assert(((uintptr_t) in & 15) == 0);",
            "#endif",
            *ctx.body,
            "}",
        ]
    lines.append("\n".join(body))

    header = (
        f"/* generated by kws-codegen from {name} -- do not edit */\n"
        "#pragma once\n#include <stddef.h>\n#include <stdint.h>\n\n"
        f"#define {name.upper()}_INFER_INPUT_LEN {in_len}\n"
        f"#define {name.upper()}_INFER_OUTPUT_LEN {out_len}\n"
        "/* Transient arena: activations + esp-nn scratch, live only for the "
        "duration of one call. */\n"
        f"#define {name.upper()}_INFER_ARENA_BYTES {ctx.arena.size}\n"
        "/* Persistent state: ring-buffer history that must survive between "
        f"calls (0 if {name} is stateless). Separate from the arena above -- "
        "add both for the model's total static footprint. */\n"
        f"#define {name.upper()}_INFER_STATE_BYTES "
        f"{sum(r.bytes + r.new_rows * r.channels for r in rings)}\n\n"
        '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
        f"void {name}_infer_init(void);\n"
        f"void {name}_infer_reset(void);\n" + decl + f"size_t {name}_infer_arena_bytes(void);\n"
        f"size_t {name}_infer_state_bytes(void);\n"
        "\n#ifdef __cplusplus\n}\n#endif\n"
    )
    return {f"{name}_infer.c": "\n".join(lines) + "\n", f"{name}_infer.h": header}


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
        + NUDGE_GUARD
        + "\n"
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
        # The size the device's S3 kernel would ask for, not a host guess: the
        # ANSI kernels linked here need none, so only this makes the harness
        # exercise a realistic buffer.
        f"#define CONV_PROBE_SCRATCH {max(scratch_bytes(graph, op), 16)}\n"
        f"static const int8_t CONV_PROBE_IN[{np.size(inputs)}] = {{{flat_in}}};\n"
        f"static const int8_t CONV_PROBE_EXPECT[{np.size(expect)}] = {{{flat_expect}}};\n"
    )


def _c_rows(name: str, ctype: str, rows) -> str:
    rows = np.asarray(rows)
    body = ",\n".join("  {" + ", ".join(str(int(v)) for v in row) + "}" for row in rows)
    return f"static const {ctype} {name}[{rows.shape[0]}][{rows.shape[1]}] = {{\n{body}\n}};\n"


def write_infer_vectors(name: str, clips, gen_dir) -> None:
    """gen/<name>_infer_vectors.h: every step of every clip and the answer the
    interpreter gave for it.

    `clips` is a sequence of (inputs, expect) pairs, each [steps, len] -- one
    clip per state reset, so the harness can check a whole streaming sequence
    and not just a single call. Regenerate whenever the model changes; the
    header and the model are read together by the host parity harness.
    """
    inputs = np.concatenate([np.asarray(i, np.int8) for i, _ in clips])
    expect = np.concatenate([np.asarray(e) for _, e in clips])
    ctype = "uint8_t" if expect.dtype == np.uint8 else "int8_t"
    steps = ", ".join(str(len(i)) for i, _ in clips)
    upper = name.upper()
    pathlib.Path(gen_dir, f"{name}_infer_vectors.h").write_text(
        "/* generated by kws-codegen -- do not edit */\n#pragma once\n#include <stdint.h>\n"
        f"#define {upper}_CLIPS {len(clips)}\n"
        f"#define {upper}_STEPS {len(inputs)}\n"
        f"static const uint16_t {upper}_CLIP_STEPS[{len(clips)}] = {{{steps}}};\n"
        + _c_rows(f"{upper}_IN", "int8_t", inputs)
        + _c_rows(f"{upper}_EXPECT", ctype, expect)
    )


def smoke_vectors_text(tflite: bytes, name: str, steps: int = 64, seed: int = 0) -> str | None:
    """<name>_smoke_vectors.h: a small, deterministic, synthetic streaming
    fixture for the committed <name>_infer.c -- `None` for a stateless model
    (nothing streaming to smoke-test).

    `steps` fixed-seed PRNG int8 feature windows, run once through the
    model's own BUILTIN_REF reference interpreter (the same dependency
    `generate()` already has for LOGISTIC's LUT -- see `logistic_lut` above)
    and frozen here as C arrays. No user recordings, no real model behaviour
    beyond "the reference kernels' own answer to synthetic noise" -- this is
    not a substitute for the real-clip parity in tests/test_codegen_parity.py
    (KWS_DATA_ROOT), it exists so the *committed* wake_infer.c can be proven
    to still match some known-good answer with no model or data present at
    all, which is what `firmware/test/test_wake_smoke` builds against in
    `$(TESTS)` (see firmware/test/Makefile).
    """
    graph = tflite_graph.read_graph(tflite)
    plan = rewrite_streaming(graph)
    if not plan.rings:
        return None
    import tensorflow as tf

    in_t = graph.tensors[graph.inputs[0]]
    win, feat = in_t.shape[-2], in_t.shape[-1]
    rows = np.random.default_rng(seed).integers(-128, 128, size=(steps * win, feat), dtype=np.int8)

    itp = tf.lite.Interpreter(
        model_content=tflite,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    itp.allocate_tensors()
    detail_in = itp.get_input_details()[0]
    detail_out = itp.get_output_details()[0]
    itp.reset_all_variables()
    inputs, probs = [], []
    for start in range(0, len(rows) - win + 1, win):
        window = rows[start : start + win]
        itp.set_tensor(detail_in["index"], window[None, ...].astype(np.int8))
        itp.invoke()
        inputs.append(window.ravel())
        probs.append(itp.get_tensor(detail_out["index"]).ravel())
    inputs = np.array(inputs, np.int8)
    out_dtype = np.uint8 if detail_out["dtype"] == np.uint8 else np.int8
    expect = np.array(probs, out_dtype)
    ctype = "uint8_t" if out_dtype == np.uint8 else "int8_t"
    upper = f"{name.upper()}_SMOKE"
    return (
        "/* generated by kws-codegen -- do not edit */\n#pragma once\n#include <stdint.h>\n"
        f"#define {upper}_STEPS {len(inputs)}\n"
        + _c_rows(f"{upper}_IN", "int8_t", inputs)
        + _c_rows(f"{upper}_EXPECT", ctype, expect)
    )


def write(tflite_path, name: str, out_dir) -> dict[str, int]:
    """Generate <name>_infer.{c,h} (and, for a streaming model,
    <name>_smoke_vectors.h) into out_dir. Returns a small report.

    Deliberately does not stamp the model id into the header comment: the
    existing whole-model parity tests (tests/test_codegen_parity.py) also
    write straight from `generate()` into this same directory as a build
    step, with no stamping. Anything `write()` added on top of plain
    `generate()` output would flip between "stamped" and "not" depending on
    which of these two callers ran last, which is exactly the kind of
    non-determinism `check()` exists to catch -- so both paths emit the same
    bytes. A stale header is already visible the strong way: `check()` fails.
    """
    blob = pathlib.Path(tflite_path).read_bytes()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = generate(blob, name)
    for filename, text in files.items():
        (out_dir / filename).write_text(text)
    smoke = smoke_vectors_text(blob, name)
    if smoke is not None:
        (out_dir / f"{name}_smoke_vectors.h").write_text(smoke)
    graph = tflite_graph.read_graph(blob)
    plan = rewrite_streaming(graph)
    scratch = max((scratch_bytes(graph, op) for op in plan.ops), default=0)
    arena = plan_arena(plan, scratch_bytes=scratch)
    return {
        "arena_bytes": arena.size,
        # Persistent-history bytes only (r.bytes), not the full C array size
        # (r.bytes + r.new_rows*r.channels) that WAKE_INFER_STATE_BYTES /
        # state_bytes() report -- see kws_de/codegen.py's _render for that
        # split (task-5-fix1-report.md S-6). Both are real numbers for
        # different questions; this one is what the brief's roundtrip test
        # pins (3792 for the wake model).
        "ring_bytes": sum(r.bytes for r in plan.rings),
        "state_bytes": sum(r.bytes + r.new_rows * r.channels for r in plan.rings),
        "ops": len(plan.ops),
    }


def check(tflite_path, name: str, committed_dir) -> list[str]:
    """Names of committed generated files that differ from a fresh generation.

    Byte-exact, unlike kws-fwgen's tolerance-based check: everything here is
    integer arithmetic on the model's own bytes, so any difference is a real
    difference -- a changed model, a changed generator, or a stale commit.
    """
    blob = pathlib.Path(tflite_path).read_bytes()
    committed_dir = pathlib.Path(committed_dir)
    files = dict(generate(blob, name))
    smoke = smoke_vectors_text(blob, name)
    if smoke is not None:
        files[f"{name}_smoke_vectors.h"] = smoke
    stale = []
    for filename, text in files.items():
        path = committed_dir / filename
        if not path.exists() or path.read_text() != text:
            stale.append(filename)
    return stale


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser(
        description="Generate C inference (esp-nn calls) from a .tflite model."
    )
    ap.add_argument("model", help="path to a .tflite file")
    ap.add_argument(
        "--name",
        required=True,
        choices=("wake", "command"),
        help="generated symbol prefix and file stem",
    )
    ap.add_argument("--out", default="firmware/main/gen", help="output directory")
    ap.add_argument(
        "--check",
        metavar="DIR",
        help="verify committed generated files in DIR are current "
        "instead of writing; exit 1 on mismatch",
    )
    args = ap.parse_args()
    model = pathlib.Path(args.model)
    if not model.exists():
        print(f"WARNING: {model} absent -- skipping (models are not committed)")
        return
    if args.check:
        stale = check(model, args.name, args.check)
        if stale:
            raise SystemExit("stale generated inference: " + ", ".join(stale))
        return
    info = write(model, args.name, args.out)
    print(
        f"{args.name}: {info['ops']} ops, arena {info['arena_bytes']} B, "
        f"rings {info['ring_bytes']} B"
    )
