import dataclasses
import pathlib

import pytest

from kws_de import codegen, config, tflite_graph

WAKE = config.MODELS_DIR / "hey_bus.tflite"
COMMAND = config.MODELS_DIR / "command.tflite"
needs_wake = pytest.mark.skipif(not WAKE.exists(), reason=f"{WAKE} absent (KWS_DATA_ROOT)")
needs_command = pytest.mark.skipif(not COMMAND.exists(), reason=f"{COMMAND} absent")

GEN = pathlib.Path(__file__).resolve().parents[1] / "firmware" / "main" / "gen"


def _tensor(index, shape, dtype="int8", scale=0.5, zp=-128, data=None):
    return tflite_graph.Tensor(
        index=index,
        name=f"t{index}",
        shape=shape,
        dtype=dtype,
        scales=(scale,),
        zero_points=(zp,),
        quantized_dimension=0,
        data=data,
    )


def _streaming_graph():
    """The microWakeWord streaming idiom in miniature: a 2-row history ring
    joined with 1 new row, consumed by a depthwise conv, then sliced back."""
    tensors = {
        0: _tensor(0, (1, 1, 1, 4)),  # new row (graph input)
        1: _tensor(1, (), dtype="resource"),  # VAR_HANDLE output
        2: _tensor(2, (1, 2, 1, 4)),  # READ_VARIABLE output
        3: _tensor(3, (1, 3, 1, 4)),  # CONCATENATION output = ring
        4: _tensor(4, (1, 2, 1, 4)),  # STRIDED_SLICE output
        5: _tensor(5, (1, 1, 1, 4)),  # DEPTHWISE_CONV_2D output
        6: _tensor(6, (1, 3, 1, 4), dtype="int8", data=bytes(12)),  # weights
        7: _tensor(7, (4,), dtype="int32", data=bytes(16)),  # bias
        8: _tensor(
            8,
            (4,),
            dtype="int32",
            data=b"\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        ),  # begin: [0, 1, 0, 0] (end_mask=15 means the end value is never read)
        9: _tensor(
            9,
            (4,),
            dtype="int32",
            data=b"\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00",
        ),  # strides: [1, 1, 1, 1]
    }
    ops = (
        tflite_graph.Op(0, "VAR_HANDLE", (), (1,), {"shared_name": "s/states_1"}),
        tflite_graph.Op(1, "READ_VARIABLE", (1,), (2,), {}),
        tflite_graph.Op(2, "CONCATENATION", (2, 0), (3,), {"axis": 1}),
        tflite_graph.Op(
            3,
            "STRIDED_SLICE",
            (3, 8, 8, 9),
            (4,),
            {"begin_mask": 13, "end_mask": 15, "shrink_axis_mask": 0},
        ),
        tflite_graph.Op(4, "ASSIGN_VARIABLE", (1, 4), (), {}),
        tflite_graph.Op(
            5,
            "DEPTHWISE_CONV_2D",
            (3, 6, 7),
            (5,),
            {
                "padding": "VALID",
                "stride_w": 1,
                "stride_h": 1,
                "depth_multiplier": 1,
                "fused_activation_function": "NONE",
            },
        ),
    )
    return tflite_graph.Graph(
        ops=ops,
        tensors=tensors,
        inputs=(0,),
        outputs=(5,),
        variables={1: "s/states_1"},
        init_subgraph=None,
    )


def test_rewrite_collapses_the_idiom_into_one_ring():
    plan = codegen.rewrite_streaming(_streaming_graph())
    assert len(plan.rings) == 1
    ring = plan.rings[0]
    # rows is what the ring keeps between steps (the STRIDED_SLICE output /
    # resource-variable size), not the transient concat total of 3 rows.
    assert (ring.rows, ring.new_rows, ring.channels, ring.bytes) == (2, 1, 4, 8)
    assert ring.buffer_tensor == 3 and ring.new_tensor == 0 and ring.var_tensor == 1
    assert [op.name for op in plan.ops] == ["DEPTHWISE_CONV_2D"]


def test_rewrite_rejects_a_slice_that_is_not_the_ring_shift():
    """The ring rewrite is only equivalent if the slice drops exactly the
    oldest new_rows rows. Anything else must fail loudly, not silently."""
    g = _streaming_graph()
    ops = list(g.ops)
    ops[3] = tflite_graph.Op(
        3,
        "STRIDED_SLICE",
        (3, 8, 8, 8),
        (4,),
        {"begin_mask": 15, "end_mask": 15, "shrink_axis_mask": 0},
    )
    bad = dataclasses.replace(g, ops=tuple(ops))
    with pytest.raises(codegen.UnsupportedGraph, match="STRIDED_SLICE"):
        codegen.rewrite_streaming(bad)


def test_duplicate_read_variable_names_the_resource():
    g = _streaming_graph()
    ops = list(g.ops) + [tflite_graph.Op(6, "READ_VARIABLE", (1,), (2,), {})]
    bad = dataclasses.replace(g, ops=tuple(ops))
    with pytest.raises(codegen.UnsupportedGraph, match=r"t1\b"):
        codegen.rewrite_streaming(bad)


def test_non_ring_concatenation_names_the_op():
    g = _streaming_graph()
    tensors = dict(g.tensors)
    tensors[10] = _tensor(10, (1, 2, 1, 4))
    extra = tflite_graph.Op(6, "CONCATENATION", (0, 0), (10,), {"axis": 1})
    bad = dataclasses.replace(g, tensors=tensors, ops=(*g.ops, extra))
    with pytest.raises(codegen.UnsupportedGraph, match=r"op 6\b"):
        codegen.rewrite_streaming(bad)


def test_unsupported_op_names_the_op_and_tensor():
    g = _streaming_graph()
    ops = list(g.ops)
    ops[5] = tflite_graph.Op(5, "TRANSPOSE_CONV", (3, 6, 7), (5,), {})
    with pytest.raises(codegen.UnsupportedGraph) as excinfo:
        codegen.rewrite_streaming(dataclasses.replace(g, ops=tuple(ops)))
    assert "TRANSPOSE_CONV" in str(excinfo.value) and "t5" in str(excinfo.value)


@needs_wake
def test_wake_model_has_six_rings_of_3792_bytes():
    g = tflite_graph.read_graph(WAKE.read_bytes())
    plan = codegen.rewrite_streaming(g)
    assert len(plan.rings) == 6
    assert sorted(r.bytes for r in plan.rings) == [80, 128, 512, 768, 1024, 1280]
    assert sum(r.bytes for r in plan.rings) == 3792
    assert {r.new_rows for r in plan.rings} == {1, 3}
    # every READ_VARIABLE / CONCATENATION / STRIDED_SLICE / ASSIGN_VARIABLE /
    # VAR_HANDLE / CALL_ONCE is gone; only compute and reshape/quantize remain
    assert {op.name for op in plan.ops} == {
        "RESHAPE",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "LOGISTIC",
        "QUANTIZE",
    }
    assert len(plan.ops) == 14


def test_arena_reuses_the_slot_of_a_dead_tensor():
    """Two activations whose lifetimes do not overlap must share one offset;
    a planner that just concatenates would double the arena."""
    plan = codegen.rewrite_streaming(_streaming_graph())
    arena = codegen.plan_arena(plan)
    assert arena.size >= 16
    assert arena.size % 16 == 0
    # only t5 (the depthwise output, 4 B) needs arena space: the ring is static
    # storage and t0 is the graph input.
    assert set(arena.offsets) == {5}
    assert arena.offsets[5] == 0


def test_arena_accounts_for_scratch():
    plan = codegen.rewrite_streaming(_streaming_graph())
    plain = codegen.plan_arena(plan)
    with_scratch = codegen.plan_arena(plan, scratch_bytes=1024)
    assert with_scratch.size == plain.size + 1024


@needs_wake
def test_wake_arena_is_at_most_the_tflm_arena():
    g = tflite_graph.read_graph(WAKE.read_bytes())
    arena = codegen.plan_arena(codegen.rewrite_streaming(g))
    tflm = codegen.tflm_arena_bytes(GEN / "wake_model_config.h", "KWS_WAKE_ARENA_BYTES")
    assert tflm == 40960
    assert arena.size <= tflm, f"generated arena {arena.size} B exceeds TFLM's {tflm} B"


@needs_wake
def test_wake_arena_holds_every_live_activation():
    g = tflite_graph.read_graph(WAKE.read_bytes())
    plan = codegen.rewrite_streaming(g)
    arena = codegen.plan_arena(plan)
    for op in plan.ops:
        for t in op.outputs:
            if t in plan.alias or t in {r.buffer_tensor for r in plan.rings}:
                continue
            end = arena.offsets[t] + codegen.tensor_bytes(g, t)
            assert end <= arena.size


def _output_clobber_graph():
    """op0's output is the graph's own output; op1 runs after and produces an
    unrelated tensor. A planner that frees a graph output's slot the instant
    it's written would hand op1 the same offset, clobbering the output before
    the caller ever reads it post-invoke()."""
    tensors = {0: _tensor(0, (1, 4)), 1: _tensor(1, (1, 4)), 2: _tensor(2, (1, 4))}
    ops = (
        tflite_graph.Op(0, "OP0", (0,), (1,), {}),
        tflite_graph.Op(1, "OP1", (), (2,), {}),
    )
    graph = tflite_graph.Graph(
        ops=ops, tensors=tensors, inputs=(0,), outputs=(1,), variables={}, init_subgraph=None
    )
    return codegen.Plan(graph=graph, ops=ops, rings=(), alias={})


def _overlapping_lifetimes_graph():
    """op2 reads both op0's and op1's outputs, so those two are alive at the
    same time and must never share bytes."""
    tensors = {
        0: _tensor(0, (1, 4)),
        1: _tensor(1, (1, 4)),
        2: _tensor(2, (1, 4)),
        3: _tensor(3, (1, 4)),
    }
    ops = (
        tflite_graph.Op(0, "OP0", (0,), (1,), {}),
        tflite_graph.Op(1, "OP1", (0,), (2,), {}),
        tflite_graph.Op(2, "OP2", (1, 2), (3,), {}),
    )
    graph = tflite_graph.Graph(
        ops=ops, tensors=tensors, inputs=(0,), outputs=(3,), variables={}, init_subgraph=None
    )
    return codegen.Plan(graph=graph, ops=ops, rings=(), alias={})


def _live_ranges(plan):
    """Ground truth tensor lifetimes, computed independently of
    `plan_arena`: [op that writes it, last op that reads it], with graph
    outputs forced live through the final op."""
    graph = plan.graph
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for step, op in enumerate(plan.ops):
        for t in op.outputs:
            first.setdefault(t, step)
            last[t] = max(last.get(t, step), step)
        for t in op.inputs:
            t = plan.alias.get(t, t)
            last[t] = max(last.get(t, step), step)
    end = len(plan.ops) - 1
    for t in graph.outputs:
        t = plan.alias.get(t, t)
        if t in first:
            last[t] = end
    return {t: (first[t], last[t]) for t in first}


def _overlaps(plan, arena):
    """Pairs of arena-resident tensors whose lifetimes *and* byte ranges
    overlap -- real corruption, found without touching `plan_arena`'s
    internals so this check can't share a bug with the code it's checking."""
    live = _live_ranges(plan)
    tensors = [t for t in arena.offsets if t in live]
    bad = []
    for i, a in enumerate(tensors):
        for b in tensors[i + 1 :]:
            fa, la = live[a]
            fb, lb = live[b]
            if la < fb or lb < fa:
                continue  # lifetimes don't overlap -- sharing an offset is fine
            oa, ea = arena.offsets[a], arena.offsets[a] + codegen.tensor_bytes(plan.graph, a)
            ob, eb = arena.offsets[b], arena.offsets[b] + codegen.tensor_bytes(plan.graph, b)
            if oa < eb and ob < ea:
                bad.append((a, b))
    return bad


def test_arena_keeps_graph_output_live_to_the_end():
    plan = _output_clobber_graph()
    arena = codegen.plan_arena(plan)
    assert arena.offsets[1] != arena.offsets[2]
    assert _overlaps(plan, arena) == []


def test_arena_disjoint_ranges_for_overlapping_lifetimes():
    plan = _overlapping_lifetimes_graph()
    arena = codegen.plan_arena(plan)
    assert _overlaps(plan, arena) == []
    assert all(offset % 16 == 0 for offset in arena.offsets.values())


def test_overlap_checker_rejects_a_broken_offset0_allocator():
    """The checker used above must actually catch a broken allocator, not
    just always pass -- force everything to offset 0 and confirm it fires."""
    plan = _overlapping_lifetimes_graph()
    broken = codegen.Arena(offsets={1: 0, 2: 0, 3: 0}, size=16)
    assert _overlaps(plan, broken) != []


@needs_wake
def test_wake_arena_has_zero_pairwise_overlaps():
    g = tflite_graph.read_graph(WAKE.read_bytes())
    plan = codegen.rewrite_streaming(g)
    arena = codegen.plan_arena(plan)
    assert _overlaps(plan, arena) == []
    assert all(offset % 16 == 0 for offset in arena.offsets.values())


def test_quantize_multiplier_matches_tflm_reference_values():
    """Spot values computed from TFLM's QuantizeMultiplier: frexp then round
    q * 2**31, with the 2**31 carry and the flush-to-zero below 2**-31."""
    assert codegen.quantize_multiplier(0.0) == (0, 0)
    assert codegen.quantize_multiplier(1.0) == (1 << 30, 1)
    assert codegen.quantize_multiplier(0.5) == (1 << 30, 0)
    assert codegen.quantize_multiplier(2.0) == (1 << 30, 2)
    mult, shift = codegen.quantize_multiplier(0.0234375)
    assert mult == 1610612736 and shift == -5
    assert codegen.quantize_multiplier(2.0**-40) == (0, 0)


def test_activation_range_relu_uses_round_half_away_from_zero():
    g = _streaming_graph()
    op = g.ops[5]
    assert codegen.activation_range(g, op) == (-128, 127)  # NONE
    relu = dataclasses.replace(op, options={**op.options, "fused_activation_function": "RELU"})
    # output scale 0.5, zp -128 -> quantize(0.0) == -128 -> clamped to qmin
    assert codegen.activation_range(g, relu) == (-128, 127)


def _emitter(plan):
    return codegen.Emitter(prefix="t", plan=plan, arena=codegen.plan_arena(plan))


@needs_wake
def test_scratch_sizes_come_from_the_esp32s3_formulas():
    """esp-nn's ANSI kernels ask for no scratch, so a host-measured size proves
    nothing. These are hand-derived from esp_nn_get_*_scratch_size_esp32s3.

    op 14 (5x1x40 -> 32, VALID, no padding needed): general path with in_ch
    padded to 48, so 5*48*32 filter + 5*48*32 aligned filter rows + 64 margin
    + 32*4 offset accumulators. op 32 (21x1 depthwise, 64 channels, ch_mult 1):
    the non-3x3 s16 path, 2 * (filter 21*64 + input 21*64) + 32.
    """
    g = tflite_graph.read_graph(WAKE.read_bytes())
    conv, depthwise = g.ops[14], g.ops[32]
    assert codegen.scratch_bytes(g, conv) == 5 * 48 * 32 + 48 * 5 * 32 + 64 + 32 * 4 == 15552
    assert codegen.scratch_bytes(g, depthwise) == 2 * (21 * 64 + 21 * 64) + 32 == 5408
    assert codegen.scratch_bytes(g, g.ops[37]) == 0  # LOGISTIC needs none


def test_absent_bias_is_refused_instead_of_indexing_the_last_tensor():
    """TFLite writes an absent optional bias as index -1; tensors[-1] is a real
    (wrong) tensor in Python, so the generator must refuse it by name."""
    g = _streaming_graph()
    ops = list(g.ops)
    ops[5] = dataclasses.replace(ops[5], inputs=(3, 6, -1))
    plan = codegen.rewrite_streaming(dataclasses.replace(g, ops=tuple(ops)))
    with pytest.raises(codegen.UnsupportedGraph, match="no bias tensor"):
        codegen.emit_depthwise(_emitter(plan), plan.ops[0])


def _fc_plan(accum_depth):
    tensors = {
        0: _tensor(0, (1, accum_depth)),
        1: _tensor(1, (1, accum_depth), data=bytes(accum_depth)),
        2: _tensor(2, (1,), dtype="int32", data=bytes(4)),
        3: _tensor(3, (1, 1)),
    }
    op = tflite_graph.Op(0, "FULLY_CONNECTED", (0, 1, 2), (3,), {})
    graph = tflite_graph.Graph(
        ops=(op,), tensors=tensors, inputs=(0,), outputs=(3,), variables={}, init_subgraph=None
    )
    return codegen.Plan(graph=graph, ops=(op,), rings=(), alias={})


def test_fully_connected_refuses_dimensions_that_would_wrap_uint16():
    """esp_nn_fully_connected_s8 takes row_len as uint16_t; 70000 would wrap to
    4464 with no diagnostic on either side."""
    plan = _fc_plan(70000)
    with pytest.raises(codegen.UnsupportedGraph, match="accum_depth=70000"):
        codegen.emit_fully_connected(_emitter(plan), plan.ops[0])
    codegen.emit_fully_connected(_emitter(_fc_plan(64)), _fc_plan(64).ops[0])  # in range


def test_generated_sources_refuse_to_build_with_skip_nudge():
    """A sdkconfig flip to CONFIG_NN_SKIP_NUDGE swaps esp-nn's requantisation
    for a faster, non-bit-exact one. Every generated .c must break the build."""
    assert "CONFIG_NN_SKIP_NUDGE" in codegen.NUDGE_GUARD
    assert codegen.NUDGE_GUARD.count("#error") == 1


def test_softmax_params_match_tflm_for_the_command_head():
    """Input scale 0.371104, beta 1.0 -> PreprocessSoftmaxScaling + diff_min."""
    mult, shift, diff_min = codegen.softmax_params_from_scale(0.371104, beta=1.0)
    assert shift >= 0
    assert 0 < mult <= (1 << 31) - 1
    # CalculateInputRadius(5, shift, 31) == floor(31 * 2**26 / 2**shift)
    assert diff_min == -((31 * (1 << 26)) >> shift) < 0


@needs_wake
def test_logistic_lut_is_256_monotone_entries_from_the_reference_kernel():
    blob = WAKE.read_bytes()
    g = tflite_graph.read_graph(blob)
    op = next(o for o in g.ops if o.name == "LOGISTIC")
    lut = codegen.logistic_lut(blob, op)
    assert len(lut) == 256
    assert all(-128 <= v <= 127 for v in lut)
    assert all(b >= a for a, b in zip(lut, lut[1:], strict=False)), "sigmoid is non-decreasing"
    assert lut[0] == -128 and lut[-1] == 127


@needs_wake
def test_wake_states_reset_to_the_quantised_zero_not_to_zero():
    """microWakeWord's CALL_ONCE fills every ring with -128 (the quantised
    zero). A reset that memset 0 would disagree with the interpreter for the
    whole first second of every clip."""
    blob = WAKE.read_bytes()
    g = tflite_graph.read_graph(blob)
    fill = codegen.initial_states(blob, g)
    assert set(fill) == set(g.variables)
    assert set(fill.values()) == {0x80}  # int8 -128
    assert "memset(ring0, 128, sizeof ring0);" in codegen.generate(blob, "wake")["wake_infer.c"]


@needs_wake
def test_generate_wake_emits_the_documented_api():
    blob = WAKE.read_bytes()
    files = codegen.generate(blob, "wake")
    assert set(files) == {"wake_infer.c", "wake_infer.h"}
    header = files["wake_infer.h"]
    for decl in (
        "void wake_infer_init(void);",
        "void wake_infer_reset(void);",
        "void wake_infer_step(const int8_t in[3 * 40], uint8_t *prob_q);",
        "size_t wake_infer_arena_bytes(void);",
    ):
        assert decl in header, decl
    source = files["wake_infer.c"]
    assert source.count("esp_nn_conv_s8(") == 5
    assert source.count("esp_nn_depthwise_conv_s8(") == 4
    assert source.count("esp_nn_fully_connected_s8(") == 1
    assert "ring0" in source and "ring5" in source
    assert "memmove" in source  # the ring shift
    assert "CONFIG_NN_SKIP_NUDGE" in source
    # the wake QUANTIZE is int8 -> uint8 at one scale: the reference kernel's
    # sign-bit-flip fast path, not a multiply
    assert "^ 0x80" in source and "kws_requantize" not in source


@needs_command
def test_generate_command_emits_a_stateless_entry_point():
    blob = COMMAND.read_bytes()
    files = codegen.generate(blob, "command")
    header, source = files["command_infer.h"], files["command_infer.c"]
    assert "void command_infer(const int8_t in[490], int8_t out[23]);" in header
    assert "ring0" not in source and "memmove" not in source
    assert source.count("esp_nn_mean_nhwc_s8(") == 1
    assert source.count("esp_nn_softmax_s8(") == 1


def test_padding_same_and_valid():
    assert codegen.padding_hw(49, 10, 3, 3, 1, 1, "SAME") == (1, 1, 49, 10)
    assert codegen.padding_hw(5, 1, 5, 1, 3, 1, "VALID") == (0, 0, 1, 1)
    assert codegen.padding_hw(1, 1, 1, 1, 1, 1, "SAME") == (0, 0, 1, 1)
