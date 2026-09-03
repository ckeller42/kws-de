import dataclasses

import pytest

from kws_de import codegen, config, tflite_graph

WAKE = config.MODELS_DIR / "hey_bus.tflite"
needs_wake = pytest.mark.skipif(not WAKE.exists(), reason=f"{WAKE} absent (KWS_DATA_ROOT)")


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
