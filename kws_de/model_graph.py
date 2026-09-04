"""Render a .tflite model's compute graph as Graphviz DOT, read straight
from the flatbuffer via the TFLite interpreter -- so the diagram can't
drift from the model that actually ships. See docs/sphinx/models.rst for
the rendered figures (`kws-model-graph <model.tflite> --out <file.dot>`).

One node per compute op (conv/depthwise/dense/logistic/softmax/mean/pool).
A *streaming* model (microWakeWord's Hey Bus) keeps history between
Invokes in TFLite resource variables: each VAR_HANDLE/READ_VARIABLE/
ASSIGN_VARIABLE triple (plus the CONCATENATION/STRIDED_SLICE around it
that splices the ring into the data path) collapses into one "ring N×C"
node. A non-streaming model (the command DS-CNN) has none of that -- every
frame is a fresh forward pass, no ring nodes appear.

Housekeeping ops (RESHAPE, QUANTIZE, CALL_ONCE, the XNNPACK-delegate
partition wrapper) never become nodes; edges are traced straight through
them to the nearest real (compute or ring) node.
"""

import argparse
import pathlib

import numpy as np

AMBER = "#C98A1E"
AMBER_SOFT = "#F3E3BF"
TEAL = "#2F9E8F"
TEAL_SOFT = "#CFEBE6"
CORAL = "#C94B32"
CORAL_SOFT = "#F6D3CB"

# Ops that become their own graph node.
COMPUTE_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "LOGISTIC",
    "SOFTMAX",
    "MEAN",
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
}
# For every "other" (housekeeping) op, input[0] is the real data tensor and
# the rest are constants (shape/begin/end/stride/axis) with no producer --
# except CONCATENATION, where every listed input is a real data tensor
# (that's exactly how a ring buffer's history rejoins the new frame).
_ALL_INPUTS_ARE_DATA = {"CONCATENATION"}


def _shape(tensor) -> list[int]:
    return [int(x) for x in tensor["shape"]]


def _squeeze(shape) -> list[int]:
    s = [d for d in shape if d != 1]
    return s or [1]


def _ring_label(shape) -> str:
    return "x".join(str(d) for d in _squeeze(shape))


def _nbytes(shape, dtype) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * np.dtype(dtype).itemsize


def _compute_info(op, tensors) -> dict:
    """Kernel/channel/MACs for one compute op, from its tensor shapes alone.
    Every conv-family op here runs once per Invoke (no batching), so MACs =
    weight count x output spatial size (H x W for conv, 1 for dense)."""
    name = op["op_name"]
    out_shape = _shape(tensors[int(op["outputs"][0])])
    info = {"macs": None, "weights": 0, "cin": None, "cout": None, "kernel": None}
    if name in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
        w_shape = _shape(tensors[int(op["inputs"][1])])
        weights = 1
        for d in w_shape:
            weights *= d
        spatial = 1
        for d in out_shape[1:-1]:
            spatial *= d
        info["weights"] = weights
        info["macs"] = weights * spatial
        if name == "CONV_2D":
            info["cout"], kh, kw, info["cin"] = w_shape
            info["kernel"] = (kh, kw)
        elif name == "DEPTHWISE_CONV_2D":
            _, kh, kw, c = w_shape
            info["cin"] = info["cout"] = c
            info["kernel"] = (kh, kw)
        else:  # FULLY_CONNECTED
            info["cout"], info["cin"] = w_shape[0], w_shape[-1]
    else:
        info["cout"] = out_shape[-1] if out_shape else None
    return info


def _node_label(op, info: dict) -> str:
    name = op["op_name"]
    macs = f"{info['macs']:,} MAC" if info["macs"] else None
    if name == "CONV_2D":
        kh, kw = info["kernel"]
        first = f"conv {kh}x{kw}, {info['cin']}->{info['cout']}"
    elif name == "DEPTHWISE_CONV_2D":
        kh, kw = info["kernel"]
        first = f"depthwise {kh}x{kw} x{info['cin']}"
    elif name == "FULLY_CONNECTED":
        first = f"dense {info['cin']}->{info['cout']}"
    elif name == "MEAN":
        first = f"mean (global avg pool) -> {info['cout']}"
    elif name == "LOGISTIC":
        first = "sigmoid"
    elif name == "SOFTMAX":
        first = "softmax"
    else:
        first = name.lower().replace("_", " ")
    return f"{first}\\n{macs}" if macs else first


def analyze(tflite: bytes) -> dict:
    """Read a .tflite flatbuffer through kws_de.tflite_graph and return
    the structure `to_dot` renders: compute ops with their MACs, ring-state
    variables, per-op pipeline stage, and the edges between all of it."""
    from kws_de import tflite_graph

    g = tflite_graph.read_graph(tflite)
    # The rest of this function speaks the old dict shape; adapt once, here.
    tensors = {
        i: {"index": i, "shape": list(t.shape), "dtype": np.dtype(t.dtype)}
        for i, t in g.tensors.items()
    }
    real_ops = [
        {
            "index": op.index,
            "op_name": op.name,
            "inputs": list(op.inputs),
            "outputs": list(op.outputs),
        }
        for op in g.ops
    ]
    in_detail = tensors[g.inputs[0]]
    out_detail = tensors[g.outputs[0]]
    producer = {}
    for op in real_ops:
        for t in op["outputs"]:
            producer[int(t)] = op

    # --- group VAR_HANDLE/READ_VARIABLE/ASSIGN_VARIABLE by resource tensor ---
    groups: dict[int, dict] = {}
    for op in real_ops:
        if op["op_name"] == "VAR_HANDLE":
            groups[int(op["outputs"][0])] = {"reads": [], "assigns": []}
    for op in real_ops:
        if op["op_name"] == "READ_VARIABLE":
            groups[int(op["inputs"][0])]["reads"].append(op)
        elif op["op_name"] == "ASSIGN_VARIABLE":
            groups[int(op["inputs"][0])]["assigns"].append(op)

    rings = []
    ring_of_read_tensor: dict[int, str] = {}
    for i, grp in enumerate(groups.values()):
        sample = grp["reads"][0]["outputs"][0] if grp["reads"] else grp["assigns"][0]["inputs"][1]
        shape = _shape(tensors[int(sample)])
        node_id = f"ring{i}"
        rings.append(
            {
                "id": node_id,
                "label": f"ring {_ring_label(shape)}",
                "bytes": _nbytes(shape, tensors[int(sample)]["dtype"]),
                "assigns": grp["assigns"],
            }
        )
        for r in grp["reads"]:
            ring_of_read_tensor[int(r["outputs"][0])] = node_id

    def trace(tensor_idx: int, seen: frozenset = frozenset()) -> set:
        if tensor_idx in ring_of_read_tensor:
            return {("state", ring_of_read_tensor[tensor_idx])}
        if tensor_idx in seen:
            return set()
        op = producer.get(tensor_idx)
        if op is None:
            return {("data", "IN")}
        if op["op_name"] in COMPUTE_OPS:
            return {("data", f"op{op['index']}")}
        ins = op["inputs"] if op["op_name"] in _ALL_INPUTS_ARE_DATA else op["inputs"][:1]
        out = set()
        for t in ins:
            out |= trace(int(t), seen | {tensor_idx})
        return out

    compute_ops = [o for o in real_ops if o["op_name"] in COMPUTE_OPS]
    infos = {int(o["index"]): _compute_info(o, tensors) for o in compute_ops}

    # Stage per compute op: stem (before any depthwise) / "Block N -- depthwise
    # KHxKW" (that depthwise conv plus whatever pointwise conv follows it) /
    # Head (from the first non-conv/depthwise compute op to the end).
    stages: dict[int, str] = {}
    stage, block_n = "Stem", 0
    for o in compute_ops:
        name = o["op_name"]
        if name == "DEPTHWISE_CONV_2D":
            block_n += 1
            kh, kw = infos[int(o["index"])]["kernel"]
            stage = f"Block {block_n} -- depthwise {kh}x{kw}"
        elif name not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            stage = "Head"
        stages[int(o["index"])] = stage

    edges: set[tuple[str, str, str]] = set()
    for o in compute_ops:
        for kind, src in trace(int(o["inputs"][0])):
            edges.add((src, f"op{o['index']}", kind))
    for ring in rings:
        for a in ring["assigns"]:
            for kind, src in trace(int(a["inputs"][1])):
                if kind == "data":  # only the new data written into the ring, not its own echo
                    edges.add((src, ring["id"], "state"))
    for _kind, src in trace(int(out_detail["index"])):
        edges.add((src, "OUT", "out"))

    return {
        "n_ops": len(real_ops),
        "n_tensors": len(tensors),
        "compute_ops": compute_ops,
        "infos": infos,
        "stages": stages,
        "rings": rings,
        "edges": edges,
        "in_detail": in_detail,
    }


def to_dot(tflite: bytes, title: str | None = None) -> str:
    g = analyze(tflite)

    # A ring's "home" stage is whichever block *reads* it (the state that
    # block's concat pulls history from) -- not whichever block last wrote
    # it. Read edges are applied first and always win; the write-edge loop
    # only fills in rings that (hypothetically) have no read at all. `edges`
    # is a set, so iteration order can't be relied on to prefer read edges.
    ring_stage: dict[str, str] = {}
    for src, dst, kind in g["edges"]:
        if kind == "state" and src.startswith("ring") and dst.startswith("op"):
            ring_stage[src] = g["stages"][int(dst[2:])]
    for src, dst, kind in g["edges"]:
        if kind == "state" and dst.startswith("ring") and src.startswith("op"):
            ring_stage.setdefault(dst, g["stages"][int(src[2:])])

    by_stage: dict[str, list[str]] = {}
    for o in g["compute_ops"]:
        idx = int(o["index"])
        label = _node_label(o, g["infos"][idx])
        by_stage.setdefault(g["stages"][idx], []).append(f'    op{idx} [label="{label}"];')
    for ring in g["rings"]:
        stage = ring_stage.get(ring["id"], "Stem")
        by_stage.setdefault(stage, []).append(
            f'    {ring["id"]} [label="{ring["label"]}", shape=cylinder, style=filled, '
            f'fillcolor="{TEAL_SOFT}", color="{TEAL}"];'
        )

    in_shape = "x".join(str(d) for d in _shape(g["in_detail"]))
    lines = [
        "digraph model {",
        "  rankdir=TB;",
        f'  node [shape=box, style=filled, fillcolor="{AMBER_SOFT}", color="{AMBER}", '
        'fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        f'  IN [label="input {in_shape}\\n{g["in_detail"]["dtype"].name}", shape=oval, '
        f'style=filled, fillcolor=white, color="{AMBER}"];',
        f'  OUT [label="output", shape=oval, style=filled, fillcolor="{CORAL_SOFT}", '
        f'color="{CORAL}"];',
    ]
    for i, (stage, node_lines) in enumerate(by_stage.items()):
        lines.append(f"  subgraph cluster_{i} {{")
        lines.append(f'    label="{stage}"; style=dashed; color="{AMBER}"; fontsize=11;')
        lines.extend(node_lines)
        lines.append("  }")
    for src, dst, kind in sorted(g["edges"]):
        if kind == "state":
            lines.append(f'  {src} -> {dst} [style=dashed, color="{TEAL}"];')
        elif kind == "out":
            lines.append(f'  {src} -> {dst} [color="{CORAL}", penwidth=2];')
        else:
            lines.append(f'  {src} -> {dst} [color="{AMBER}"];')

    total_weights = sum(i["weights"] for i in g["infos"].values())
    total_macs = sum(i["macs"] or 0 for i in g["infos"].values())
    state_bytes = sum(r["bytes"] for r in g["rings"])
    stats = (
        f"{g['n_ops']} ops, {g['n_tensors']} tensors, {total_weights:,} weights, "
        f"{total_macs:,} MACs, {state_bytes:,} B state, {len(tflite):,} B file"
    )
    footer = f"{title}\\n\\n{stats}" if title else stats
    lines.append(f'  labelloc="b"; fontsize=11; fontname="Helvetica"; label="{footer}";')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser(description="Render a .tflite model's graph as Graphviz DOT.")
    ap.add_argument("model", help="path to a .tflite file")
    ap.add_argument("--out", required=True, help="output .dot path")
    ap.add_argument("--title", default=None, help="figure title, embedded in the footer label")
    args = ap.parse_args()
    tflite = pathlib.Path(args.model).read_bytes()
    pathlib.Path(args.out).write_text(to_dot(tflite, title=args.title) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
