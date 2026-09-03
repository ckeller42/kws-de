# Generated Inference Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate C from our two `.tflite` graphs that calls esp-nn's ESP32-S3 kernels directly, producing bit-identical output to the TFLite-Micro interpreter in less time and no more arena.

**Architecture:** A new `kws_de/tflite_graph.py` reads the flatbuffer (ops in execution order, tensor shapes/quantisation, constant buffers, resource variables) and replaces the ad-hoc reader inside `kws_de/model_graph.py`. `kws_de/codegen.py` consumes that graph: it rewrites the microWakeWord streaming pattern (`READ_VARIABLE → CONCATENATION → … → STRIDED_SLICE → ASSIGN_VARIABLE`) into ring buffers, plans one arena by tensor lifetime, precomputes every requantisation multiplier exactly as TFLM does, and emits `firmware/main/gen/{wake,command}_infer.{c,h}` — committed files, freshness-checked by `kws-codegen --check` like `kws-fwgen --check`. The firmware keeps both paths: `CONFIG_KWS_INFER_GENERATED` picks the generated one, TFLM stays compiled in as the fallback and as the on-device parity reference.

**Tech Stack:** Python 3.11, `tensorflow.lite.python.schema_py_generated` + `flatbuffers` (both already installed with TensorFlow), numpy, pytest; C11 for the generated code and the host parity harness; esp-nn (via `espressif/esp-tflite-micro`) on device; ESP-IDF v5.5.5 Kconfig; sphinx-needs for the requirement trace.

**Spec:** `docs/superpowers/specs/2026-09-03-generated-inference-design.md`

## Global Constraints

- **Bit-exact.** Acceptance is "bit-exact with the interpreter and measurably faster, TFLite-Micro kept as a build-time fallback". Every output tensor byte must match; "Zero LSB difference is the requirement; one failing byte fails the build of the generated headers."
- **TFLM stays.** The generated path is selected by `CONFIG_KWS_INFER_GENERATED` (Kconfig in `firmware/main/Kconfig.projbuild`, default y once parity is proven), else the existing `MicroInterpreter` path. "Both paths compile in one firmware family; the boot log prints which is active."
- **Arena.** The planner "Reports bytes; must be ≤ the TFLM arena the same model needed (`arena_used_bytes`)." Concretely: ≤ `KWS_WAKE_ARENA_BYTES` (49152) and ≤ `KWS_MODEL_ARENA_BYTES` (139264) from `firmware/main/gen/*_config.h`. The generated arena is a static `int8_t` array in internal RAM.
- **Loud refusal.** "Refuses loudly on anything else: an unsupported op, dynamic shapes, non-int8 tensors, more than one subgraph beyond the init subgraph — error names the op and tensor. Never silent."
- **Supported op set** (the union of both graphs today): `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `AVERAGE_POOL_2D`, `MEAN`, `SOFTMAX`, `LOGISTIC`, `QUANTIZE`, `RESHAPE`, `CONCATENATION`, `STRIDED_SLICE`, `VAR_HANDLE`, `READ_VARIABLE`, `ASSIGN_VARIABLE`, `CALL_ONCE`.
- **Generated API**, verbatim from the spec:

  ```c
  void wake_infer_init(void);                       /* zero rings, precompute nothing else */
  void wake_infer_reset(void);                      /* on mode entry: clear streaming state */
  void wake_infer_step(const int8_t in[3 * 40], uint8_t *prob_q);   /* one 30 ms step */
  void command_infer(const int8_t in[49 * 10], int8_t out[23]);
  size_t wake_infer_arena_bytes(void);              /* for the boot log */
  ```

- **Public repo:** no speaker names (numeric `spkNN` only), no machine names or host paths (`bar`, `wuerfel`, `/Volumes/...`, `~/src/esp/...`) anywhere in committed code or docs. Data roots come from `KWS_DATA_ROOT`; the device-flashing/console helper scripts are referenced in this plan only, never in committed docs.
- **Gates before every commit:** `uv run --no-sync ruff check . && uv run --no-sync ruff format --check .`, `uv run --no-sync pytest -q`, `make -C firmware/test`, `npx markdownlint-cli@0.42.0 --config .markdownlint.json <changed .md>`. Never `--no-verify`, never `git add -A`. If you sync: `uv sync --extra dev --extra tts --extra docs --extra qc`.
- **Environment:** `export KWS_DATA_ROOT=<your data root>` — the models (`models/hey_bus.tflite`, `models/command.tflite`) are not committed. Tasks that need a model must skip cleanly when it is absent, exactly like `write_wake_headers` does today.
- **Commit trailers**, on every commit in this plan:

  ```text
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
  ```

---

## Graph facts you will need

These were read off the two shipped models. Tasks reference them by op index.

**Wake — `models/hey_bus.tflite`**, 58,080 B, 49 ops as the interpreter reports them (45 real + 4 XNNPACK `DELEGATE` wrappers that must be ignored), 71 tensors, 2 subgraphs. Input `[1, 3, 40]` int8 (scale 0.10196078568696976, zp −128); output `[1, 1]` **uint8** (scale 1/256, zp 0). Subgraph 1 (`NoOp`, 12 ops) is the `CALL_ONCE` init subgraph: six `VAR_HANDLE`/`ASSIGN_VARIABLE` pairs that zero the state.

Execution order in subgraph 0:

```text
 0 CALL_ONCE(init_subgraph=1)
 1..6  VAR_HANDLE            -> t33 stream_9, t34 stream_8, t35 stream_7,
                                t36 stream_6, t37 stream_11, t38 stream_10
 7  RESHAPE          t0[1,3,40]        -> t39[1,3,1,40]
 8  READ_VARIABLE    t38               -> t40[1,20,1,64]
 9  READ_VARIABLE    t37               -> t41[1,16,1,64]
10  READ_VARIABLE    t36               -> t42[1,2,1,40]
11  CONCATENATION    axis=1 (t42,t39)  -> t43[1,5,1,40]
12  STRIDED_SLICE    t43 begin[1]=3    -> t44[1,2,1,40]
13  ASSIGN_VARIABLE  t36 <- t44
14  CONV_2D          t43, w t32[32,5,1,40], b t31 -> t45[1,1,1,32]   VALID, stride 1x3, RELU
15  READ_VARIABLE    t35               -> t46[1,4,1,32]
16  CONCATENATION    axis=1 (t46,t45)  -> t47[1,5,1,32]
17  DEPTHWISE_CONV_2D t47, w t30[1,5,1,32], b t29 -> t48[1,1,1,32]   VALID, stride 1x1, NONE
18  CONV_2D          t48, w t28[64,1,1,32], b t27 -> t49[1,1,1,64]   SAME,  stride 1x1, RELU
19  STRIDED_SLICE    t47 begin[1]=1    -> t50[1,4,1,32]
20  ASSIGN_VARIABLE  t35 <- t50
21  READ_VARIABLE    t34               -> t51[1,8,1,64]
22  CONCATENATION    axis=1 (t51,t49)  -> t52[1,9,1,64]
23  DEPTHWISE_CONV_2D t52, w t26[1,9,1,64], b t25 -> t53[1,1,1,64]
24  CONV_2D          t53, w t24[64,1,1,64], b t23 -> t54[1,1,1,64]
25  STRIDED_SLICE    t52 begin[1]=1    -> t55[1,8,1,64]
26  ASSIGN_VARIABLE  t34 <- t55
27  READ_VARIABLE    t33               -> t56[1,12,1,64]
28  CONCATENATION    axis=1 (t56,t54)  -> t57[1,13,1,64]
29  DEPTHWISE_CONV_2D t57, w t22[1,13,1,64], b t21 -> t58[1,1,1,64]
30  CONV_2D          t58, w t20[64,1,1,64], b t19 -> t59[1,1,1,64]
31  CONCATENATION    axis=1 (t40,t59)  -> t60[1,21,1,64]
32  DEPTHWISE_CONV_2D t60, w t18[1,21,1,64], b t17 -> t61[1,1,1,64]
33  CONV_2D          t61, w t16[64,1,1,64], b t15 -> t62[1,1,1,64]
34  CONCATENATION    axis=1 (t41,t62)  -> t63[1,17,1,64]
35  RESHAPE          t63               -> t64[1,1088]
36  FULLY_CONNECTED  t64, w t14[1,1088], b t13 -> t65[1,1]   activation NONE
37  LOGISTIC         t65               -> t66[1,1] int8 (1/256, -128)
38  STRIDED_SLICE    t63 begin[1]=1    -> t67[1,16,1,64];  39 ASSIGN_VARIABLE t37 <- t67
40  STRIDED_SLICE    t60 begin[1]=1    -> t68[1,20,1,64];  41 ASSIGN_VARIABLE t38 <- t68
42  STRIDED_SLICE    t57 begin[1]=1    -> t69[1,12,1,64];  43 ASSIGN_VARIABLE t33 <- t69
44  QUANTIZE         t66 int8 -> t70[1,1] uint8 (same scale, zp -128 -> 0)
```

Six rings, 3,792 B total: `t36` 2×40 (80 B), `t35` 4×32 (128 B), `t34` 8×64 (512 B), `t33` 12×64 (768 B), `t38` 20×64 (1280 B), `t37` 16×64 (1024 B). Note the assign for a ring may come far after its read (ops 38–43) — key the rewrite on the variable, not on adjacency.

**Command — `models/command.tflite`** (and `command_v3_qat.tflite`, structurally identical), 18,296 B, 11 ops (10 real + 1 `DELEGATE`), 28 tensors, 1 subgraph. Input `[1, 49, 10, 1]` int8, output `[1, 23]` int8 (1/256, −128):

```text
0 CONV_2D           w[32,3,3,1]  -> [1,49,10,32]  SAME, stride 1x1, RELU
1 DEPTHWISE_CONV_2D w[1,3,3,32]  -> [1,49,10,32]  SAME, stride 1x1, RELU, depth_multiplier 1
2 CONV_2D           w[32,1,1,32] -> [1,49,10,32]  SAME, stride 1x1, RELU
3 DEPTHWISE_CONV_2D w[1,3,3,32]  -> [1,49,10,32]
4 CONV_2D           w[32,1,1,32] -> [1,49,10,32]
5 DEPTHWISE_CONV_2D w[1,3,3,32]  -> [1,49,10,32]
6 CONV_2D           w[32,1,1,32] -> [1,49,10,32]
7 MEAN              axis=[1,2], keep_dims=False -> [1,32]
8 FULLY_CONNECTED   w[23,32], b[23] -> [1,23]   activation NONE
9 SOFTMAX           beta=1.0 -> [1,23]
```

Both shipped command models use `MEAN`, not `AVERAGE_POOL_2D`. Both must be emitted (the spec lists both in the op set) — `MEAN` is the one exercised today.

## File structure

| Path | Task | Responsibility |
|---|---|---|
| `kws_de/tflite_graph.py` | 1 | flatbuffer reader: ops in order, tensors + quantisation, constant buffers, resource variables, builtin options; one-op probe model builder |
| `kws_de/model_graph.py` | 1 | keeps DOT rendering only; its reader is replaced by `tflite_graph.read_graph` |
| `kws_de/codegen.py` | 2–6 | ring rewrite, memory planner, per-op emitters, `kws-codegen` CLI with `--check` |
| `tests/test_tflite_graph.py` | 1 | reader on a tiny int8 model converted in-test |
| `tests/test_codegen.py` | 2, 3 | ring rewrite + planner on hand-built graphs and on the wake model when present |
| `tests/test_codegen_parity.py` | 4, 5, 8 | interpreter-vs-generated byte parity: golden vector, 10 real wake takes, recordings set |
| `firmware/main/gen/wake_infer.{c,h}` | 6 | generated, committed |
| `firmware/main/gen/wake_infer_vectors.h` | 5 | generated golden expectations for the host C test |
| `firmware/main/gen/command_infer.{c,h}` | 8 | generated, committed |
| `firmware/main/Kconfig.projbuild` | 7 | `CONFIG_KWS_INFER_GENERATED` (new file) |
| `firmware/main/wake.cc` | 7 | switch + glue + boot log + device parity line |
| `firmware/main/recognise.cc` | 8 | switch + glue |
| `firmware/main/CMakeLists.txt` | 7, 8 | add `gen/*_infer.c` to `SRCS` |
| `firmware/test/Makefile` | 4 | `test_infer_parity` target, pinned esp-nn checkout |
| `firmware/test/test_infer_parity.c` | 4, 5 | host parity harness against esp-nn ANSI kernels |
| `.github/workflows/firmware.yml` | 6 | `kws-codegen --check` in `gen-fresh` |
| `docs/sphinx/{requirements,tests,firmware,models}.rst` | 7, 8 | `REQ_FW_INFER_GENERATED`, `REQ_FW_INFER_FALLBACK` + tests |
| `docs/paper-notes.md` | 7, 8 | measured before/after |

---

### Task 1: Flatbuffer graph reader

**Files:**

- Create: `kws_de/tflite_graph.py`
- Modify: `kws_de/model_graph.py:122-227` (`analyze` reads through the new module)
- Test: `tests/test_tflite_graph.py`

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces, all used by Tasks 2–8:

  ```python
  DTYPES: dict[int, str]                    # TensorType enum value -> "int8"/"int32"/"uint8"/"float32"/"resource"/...
  PADDING: dict[int, str]                   # {0: "SAME", 1: "VALID"}
  ACTIVATIONS: dict[int, str]               # {0: "NONE", 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6"}

  @dataclasses.dataclass(frozen=True)
  class Tensor:
      index: int
      name: str
      shape: tuple[int, ...]
      dtype: str
      scales: tuple[float, ...]             # float32 values widened to Python float
      zero_points: tuple[int, ...]
      quantized_dimension: int
      data: bytes | None                    # constant buffer contents, None for activations

  @dataclasses.dataclass(frozen=True)
  class Op:
      index: int                            # position in subgraph execution order
      name: str                             # "CONV_2D", ...
      inputs: tuple[int, ...]
      outputs: tuple[int, ...]
      options: dict[str, object]            # snake_case builtin options, e.g. {"padding": "SAME", "stride_w": 1}

  @dataclasses.dataclass(frozen=True)
  class Graph:
      ops: tuple[Op, ...]
      tensors: dict[int, Tensor]
      inputs: tuple[int, ...]
      outputs: tuple[int, ...]
      variables: dict[int, str]             # resource tensor index -> shared_name
      init_subgraph: int | None             # CALL_ONCE target, None if absent

  def read_graph(tflite: bytes, subgraph: int = 0) -> Graph: ...
  def constant(graph: Graph, index: int, dtype: str | None = None) -> numpy.ndarray: ...
  def probe_model(tflite: bytes, op: Op, subgraph: int = 0) -> bytes: ...
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_tflite_graph.py`:

```python
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from kws_de import tflite_graph


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
    assert weights.shape == (3, 3, 3, 1)          # [out_c, kh, kw, in_c]
    assert len(weights.scales) == 3               # per-channel
    assert weights.quantized_dimension == 0
    assert weights.data is not None and len(weights.data) == 27
    out = g.tensors[conv.outputs[0]]
    assert out.data is None                       # activation, not a constant
    assert len(out.scales) == 1 and out.scales[0] > 0
    bias = tflite_graph.constant(g, conv.inputs[2])
    assert bias.dtype == np.int32 and bias.shape == (3,)


def test_graph_io_and_no_variables(tiny_tflite):
    g = tflite_graph.read_graph(tiny_tflite)
    assert len(g.inputs) == 1 and len(g.outputs) == 1
    assert g.tensors[g.inputs[0]].dtype == "int8"
    assert g.variables == {}
    assert g.init_subgraph is None


def test_probe_model_reproduces_one_op(tiny_tflite):
    """A single-op model rebuilt from one op runs on the interpreter with the
    same quantisation — this is how bit-exact activation tables are derived."""
    g = tflite_graph.read_graph(tiny_tflite)
    conv = g.ops[0]
    probe = tflite_graph.probe_model(tiny_tflite, conv)
    itp = tf.lite.Interpreter(model_content=probe)
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    assert tuple(int(d) for d in inp["shape"]) == g.tensors[conv.inputs[0]].shape
    assert tuple(int(d) for d in out["shape"]) == g.tensors[conv.outputs[0]].shape
    assert float(out["quantization"][0]) == pytest.approx(g.tensors[conv.outputs[0]].scales[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_tflite_graph.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kws_de.tflite_graph'`.

- [ ] **Step 3: Write the reader**

Create `kws_de/tflite_graph.py`:

```python
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
_NUMPY = {"float32": np.float32, "int32": np.int32, "uint8": np.uint8,
          "int64": np.int64, "int16": np.int16, "int8": np.int8, "bool": np.bool_}
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
    tensor is an activation (no buffer) — never returns a silent zero array."""
    t = graph.tensors[index]
    if t.data is None:
        raise ValueError(f"tensor {index} ({t.name}) has no constant buffer")
    array = np.frombuffer(t.data, dtype=_NUMPY[dtype or t.dtype])
    return array.reshape(t.shape) if t.shape else array


def probe_model(tflite: bytes, op: Op, subgraph: int = 0) -> bytes:
    """A one-op .tflite holding just `op`, its input and its output, with the
    original quantisation and builtin options. Running this through
    tf.lite.Interpreter gives the reference kernel's exact answer for that op —
    which is how the LOGISTIC lookup table is derived bit-exactly instead of
    reimplementing gemmlowp's fixed-point sigmoid."""
    model = schema.ModelT.InitFromPackedBuf(bytearray(tflite), 0)
    sg = model.subgraphs[subgraph]
    src_op = sg.operators[op.index]
    out = schema.ModelT()
    out.version = 3
    out.description = b"kws-codegen probe"
    src_code = model.operatorCodes[int(src_op.opcodeIndex)]
    code = schema.OperatorCodeT()
    code.builtinCode = src_code.builtinCode
    code.deprecatedBuiltinCode = src_code.deprecatedBuiltinCode
    code.version = src_code.version
    out.operatorCodes = [code]
    out.buffers = [schema.BufferT(), schema.BufferT(), schema.BufferT()]
    copies = []
    for slot, src_index in enumerate((op.inputs[0], op.outputs[0])):
        src_t = sg.tensors[src_index]
        t = schema.TensorT()
        t.shape = list(src_t.shape)
        t.type = src_t.type
        t.buffer = slot + 1
        t.name = src_t.name
        q = schema.QuantizationParametersT()
        q.scale = list(src_t.quantization.scale)
        q.zeroPoint = list(src_t.quantization.zeroPoint)
        q.quantizedDimension = src_t.quantization.quantizedDimension
        t.quantization = q
        copies.append(t)
    new_sg = schema.SubGraphT()
    new_sg.tensors = copies
    new_sg.inputs = [0]
    new_sg.outputs = [1]
    new_op = schema.OperatorT()
    new_op.opcodeIndex = 0
    new_op.inputs = [0]
    new_op.outputs = [1]
    new_op.builtinOptionsType = src_op.builtinOptionsType
    new_op.builtinOptions = src_op.builtinOptions
    new_sg.operators = [new_op]
    out.subgraphs = [new_sg]
    builder = flatbuffers.Builder(1024)
    builder.Finish(out.Pack(builder), b"TFL3")
    return bytes(builder.Output())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_tflite_graph.py -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Point `model_graph.analyze` at the shared reader**

In `kws_de/model_graph.py`, replace the interpreter-based head of `analyze` (currently lines 128-142, `import tensorflow as tf` through the `producer` loop) with the shared reader. `_get_ops_details()` returns dicts, so keep the rest of `analyze` working by feeding it the same shapes it already expects:

```python
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
        {"index": op.index, "op_name": op.name, "inputs": list(op.inputs),
         "outputs": list(op.outputs)}
        for op in g.ops
    ]
    in_detail = tensors[g.inputs[0]]
    out_detail = tensors[g.outputs[0]]
    producer = {}
    for op in real_ops:
        for t in op["outputs"]:
            producer[int(t)] = op
```

Add `import numpy as np` at the top of `model_graph.py` and delete the now-unused local `import numpy as np` inside `_nbytes`. Change the two later uses of the removed interpreter fields: `g["n_ops"]` becomes `len(real_ops)` and `in_detail["dtype"].__name__` becomes `in_detail["dtype"].name`. The `DELEGATE` filter disappears — the flatbuffer has no delegate wrappers, only the real ops.

- [ ] **Step 6: Run the existing graph tests**

Run: `uv run --no-sync pytest tests/test_model_graph.py -q`
Expected: PASS. If a test asserted the old delegate-inclusive `n_ops`, update it to the flatbuffer count (wake: 45) and say so in the test's docstring.

- [ ] **Step 7: Full gates**

Run: `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add kws_de/tflite_graph.py kws_de/model_graph.py tests/test_tflite_graph.py tests/test_model_graph.py
git commit -m "$(cat <<'EOF'
feat(codegen): shared tflite flatbuffer reader

kws-model-graph and the coming code generator both need ops in execution
order with builtin options, per-channel quantisation and raw buffers, which
_get_ops_details() does not expose. One reader, one source of truth.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 2: Streaming pattern rewrite to ring buffers

**Files:**

- Create: `kws_de/codegen.py`
- Test: `tests/test_codegen.py`

**Interfaces:**

- Consumes from Task 1: `tflite_graph.read_graph`, `Graph`, `Op`, `Tensor`, `constant`.
- Produces:

  ```python
  class UnsupportedGraph(ValueError): ...

  SUPPORTED_OPS: frozenset[str]

  @dataclasses.dataclass(frozen=True)
  class Ring:
      name: str                 # C identifier, e.g. "ring0"
      var_tensor: int           # resource tensor index (the VAR_HANDLE output)
      buffer_tensor: int        # the CONCATENATION output — the ring's storage
      new_tensor: int           # tensor holding the rows appended this step
      rows: int                 # total rows in the ring (history + new)
      new_rows: int             # rows appended per step
      channels: int             # elements per row
      bytes: int                # rows * channels

  @dataclasses.dataclass(frozen=True)
  class Plan:
      graph: Graph
      ops: tuple[Op, ...]       # ops to emit, streaming bookkeeping removed
      rings: tuple[Ring, ...]
      alias: dict[int, int]     # tensor -> tensor whose storage it shares

  def rewrite_streaming(graph: Graph) -> Plan: ...
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_codegen.py`:

```python
import os
import pathlib

import pytest

from kws_de import codegen, config, tflite_graph

WAKE = config.MODELS_DIR / "hey_bus.tflite"
needs_wake = pytest.mark.skipif(not WAKE.exists(), reason=f"{WAKE} absent (KWS_DATA_ROOT)")


def _tensor(index, shape, dtype="int8", scale=0.5, zp=-128, data=None):
    return tflite_graph.Tensor(
        index=index, name=f"t{index}", shape=shape, dtype=dtype,
        scales=(scale,), zero_points=(zp,), quantized_dimension=0, data=data,
    )


def _streaming_graph():
    """The microWakeWord streaming idiom in miniature: a 2-row history ring
    joined with 1 new row, consumed by a depthwise conv, then sliced back."""
    tensors = {
        0: _tensor(0, (1, 1, 1, 4)),            # new row (graph input)
        1: _tensor(1, (), dtype="resource"),    # VAR_HANDLE output
        2: _tensor(2, (1, 2, 1, 4)),            # READ_VARIABLE output
        3: _tensor(3, (1, 3, 1, 4)),            # CONCATENATION output = ring
        4: _tensor(4, (1, 2, 1, 4)),            # STRIDED_SLICE output
        5: _tensor(5, (1, 1, 1, 4)),            # DEPTHWISE_CONV_2D output
        6: _tensor(6, (1, 3, 1, 4), dtype="int8", data=bytes(12)),   # weights
        7: _tensor(7, (4,), dtype="int32", data=bytes(16)),          # bias
        8: _tensor(8, (4,), dtype="int32", data=b"\x00\x00\x00\x00\x01\x00\x00\x00"
                                                b"\x00\x00\x00\x00\x00\x00\x00\x00"),
    }
    ops = (
        tflite_graph.Op(0, "VAR_HANDLE", (), (1,), {"shared_name": "s/states_1"}),
        tflite_graph.Op(1, "READ_VARIABLE", (1,), (2,), {}),
        tflite_graph.Op(2, "CONCATENATION", (2, 0), (3,), {"axis": 1}),
        tflite_graph.Op(3, "STRIDED_SLICE", (3, 8, 8, 8), (4,),
                        {"begin_mask": 13, "end_mask": 15, "shrink_axis_mask": 0}),
        tflite_graph.Op(4, "ASSIGN_VARIABLE", (1, 4), (), {}),
        tflite_graph.Op(5, "DEPTHWISE_CONV_2D", (3, 6, 7), (5,),
                        {"padding": "VALID", "stride_w": 1, "stride_h": 1,
                         "depth_multiplier": 1, "fused_activation_function": "NONE"}),
    )
    return tflite_graph.Graph(ops=ops, tensors=tensors, inputs=(0,), outputs=(5,),
                             variables={1: "s/states_1"}, init_subgraph=None)


def test_rewrite_collapses_the_idiom_into_one_ring():
    plan = codegen.rewrite_streaming(_streaming_graph())
    assert len(plan.rings) == 1
    ring = plan.rings[0]
    assert (ring.rows, ring.new_rows, ring.channels, ring.bytes) == (3, 1, 4, 12)
    assert ring.buffer_tensor == 3 and ring.new_tensor == 0 and ring.var_tensor == 1
    assert [op.name for op in plan.ops] == ["DEPTHWISE_CONV_2D"]


def test_rewrite_rejects_a_slice_that_is_not_the_ring_shift():
    """The ring rewrite is only equivalent if the slice drops exactly the
    oldest new_rows rows. Anything else must fail loudly, not silently."""
    g = _streaming_graph()
    ops = list(g.ops)
    ops[3] = tflite_graph.Op(3, "STRIDED_SLICE", (3, 8, 8, 8), (4,),
                             {"begin_mask": 15, "end_mask": 15, "shrink_axis_mask": 0})
    bad = dataclasses_replace(g, ops=tuple(ops))
    with pytest.raises(codegen.UnsupportedGraph, match="STRIDED_SLICE"):
        codegen.rewrite_streaming(bad)


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


def test_unsupported_op_names_the_op_and_tensor():
    g = _streaming_graph()
    ops = list(g.ops)
    ops[5] = tflite_graph.Op(5, "TRANSPOSE_CONV", (3, 6, 7), (5,), {})
    with pytest.raises(codegen.UnsupportedGraph) as excinfo:
        codegen.rewrite_streaming(dataclasses_replace(g, ops=tuple(ops)))
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
        "RESHAPE", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
        "LOGISTIC", "QUANTIZE",
    }
    assert len(plan.ops) == 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_codegen.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kws_de.codegen'`.

- [ ] **Step 3: Write the rewrite**

Create `kws_de/codegen.py`:

```python
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

import numpy as np

from kws_de import tflite_graph

SUPPORTED_OPS = frozenset(
    {
        "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D",
        "MEAN", "SOFTMAX", "LOGISTIC", "QUANTIZE", "RESHAPE", "CONCATENATION",
        "STRIDED_SLICE", "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE",
        "CALL_ONCE",
    }
)
# Ops that exist only to move bytes around; the rewrite resolves them away.
_BOOKKEEPING = {"VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE", "CALL_ONCE",
                "CONCATENATION", "STRIDED_SLICE"}


class UnsupportedGraph(ValueError):
    """The generator refuses this graph. Message names the op and tensor."""


@dataclasses.dataclass(frozen=True)
class Ring:
    name: str
    var_tensor: int
    buffer_tensor: int
    new_tensor: int
    rows: int
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

    reads = {op.inputs[0]: op.outputs[0] for op in graph.ops if op.name == "READ_VARIABLE"}
    assigns = {op.inputs[0]: op.inputs[1] for op in graph.ops if op.name == "ASSIGN_VARIABLE"}
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
        rows, channels = _rows_and_channels(graph.tensors[buffer_tensor].shape, buffer_tensor)
        kept, kept_c = _rows_and_channels(graph.tensors[sliced].shape, sliced)
        new_rows, new_c = _rows_and_channels(graph.tensors[new_tensor].shape, new_tensor)
        if channels != kept_c or channels != new_c or kept + new_rows != rows:
            raise UnsupportedGraph(
                f"resource t{var}: ring {rows}x{channels} does not decompose into "
                f"history {kept}x{kept_c} + new {new_rows}x{new_c}"
            )
        begin = tflite_graph.constant(graph, cut.inputs[1], "int32")
        strides = tflite_graph.constant(graph, cut.inputs[3], "int32")
        # begin_mask bit d set => begin[d] ignored (start at 0). The shift is
        # only equivalent when axis 1 starts at new_rows and every other axis is
        # taken whole with stride 1.
        begin_mask = int(cut.options.get("begin_mask", 0))
        if (begin_mask >> 1) & 1 or int(begin[1]) != new_rows or not all(int(s) == 1 for s in strides):
            raise UnsupportedGraph(
                f"resource t{var}: STRIDED_SLICE t{sliced} begin={list(begin)} "
                f"mask={begin_mask} strides={list(strides)} is not a shift by {new_rows} rows"
            )
        rings.append(
            Ring(name=f"ring{i}", var_tensor=var, buffer_tensor=buffer_tensor,
                 new_tensor=new_tensor, rows=rows, new_rows=new_rows,
                 channels=channels, bytes=rows * channels)
        )

    ops = []
    for op in graph.ops:
        if op.name in _BOOKKEEPING:
            continue
        if op.name == "RESHAPE":
            # Same bytes, different shape: the output shares the input's storage.
            alias[op.outputs[0]] = alias.get(op.inputs[0], op.inputs[0])
        ops.append(op)
    return Plan(graph=graph, ops=tuple(ops), rings=tuple(rings), alias=alias)
```

Note `RESHAPE` stays in `plan.ops` so the emitter can record the alias and skip it; every other bookkeeping op is dropped.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_codegen.py -q`
Expected: PASS. Without `KWS_DATA_ROOT` set the wake test is skipped; with it set, 4 passed.

- [ ] **Step 5: Run against the real wake model**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py -q`
Expected: `4 passed` — including `test_wake_model_has_six_rings_of_3792_bytes`.

- [ ] **Step 6: Commit**

```bash
git add kws_de/codegen.py tests/test_codegen.py
git commit -m "$(cat <<'EOF'
feat(codegen): collapse the streaming idiom into ring buffers

Each resource variable's READ/CONCAT/SLICE/ASSIGN becomes one ring, and the
equivalence is checked per variable rather than assumed: history first, new
rows last, slice drops exactly the oldest rows. Anything else raises.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 3: Arena planner

**Files:**

- Modify: `kws_de/codegen.py`
- Test: `tests/test_codegen.py`

**Interfaces:**

- Consumes from Task 2: `Plan`, `Ring`, `rewrite_streaming`, `UnsupportedGraph`.
- Produces:

  ```python
  @dataclasses.dataclass(frozen=True)
  class Arena:
      offsets: dict[int, int]     # tensor index -> byte offset in the arena
      size: int                   # total arena bytes (16-byte aligned)

  def tensor_bytes(graph: tflite_graph.Graph, index: int) -> int: ...
  def plan_arena(plan: Plan, scratch_bytes: int = 0) -> Arena: ...
  def tflm_arena_bytes(header: pathlib.Path, macro: str) -> int: ...
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codegen.py`:

```python
import pathlib

GEN = pathlib.Path(__file__).resolve().parents[1] / "firmware" / "main" / "gen"


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
    assert tflm == 49152
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_codegen.py -q -k arena`
Expected: FAIL with `AttributeError: module 'kws_de.codegen' has no attribute 'plan_arena'`.

- [ ] **Step 3: Write the planner**

Append to `kws_de/codegen.py`:

```python
import pathlib
import re

_ITEMSIZE = {"int8": 1, "uint8": 1, "int32": 4, "int16": 2, "float32": 4}
_ALIGN = 16  # esp-nn's S3 kernels want 16-byte-aligned operands


@dataclasses.dataclass(frozen=True)
class Arena:
    offsets: dict[int, int]
    size: int


def tensor_bytes(graph: tflite_graph.Graph, index: int) -> int:
    t = graph.tensors[index]
    return math.prod(t.shape) * _ITEMSIZE[t.dtype]


def _round_up(value: int, align: int = _ALIGN) -> int:
    return -(-value // align) * align


def plan_arena(plan: Plan, scratch_bytes: int = 0) -> Arena:
    """Greedy first-fit over tensor lifetimes.

    A tensor is live from the op that writes it to the last op that reads it.
    Constants live in flash, rings and the graph input/output in their own
    static buffers, and aliases share their target's slot — so the arena holds
    exactly the intermediate activations. Scratch (esp-nn's conv workspace) is
    a single block at the end, live for the whole call, because two kernels
    never run at once.
    """
    graph = plan.graph
    rings = {r.buffer_tensor for r in plan.rings}
    fixed = rings | set(graph.inputs) | set(graph.outputs)

    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for step, op in enumerate(plan.ops):
        for t in op.outputs:
            first.setdefault(t, step)
            last[t] = max(last.get(t, step), step)
        for t in op.inputs:
            if graph.tensors[t].data is None:
                last[t] = max(last.get(t, step), step)

    candidates = [
        t for t in sorted(first, key=lambda t: (-tensor_bytes(graph, t), t))
        if t not in fixed and t not in plan.alias and graph.tensors[t].data is None
    ]

    offsets: dict[int, int] = {}
    placed: list[tuple[int, int, int, int]] = []  # (offset, end, first, last)
    for t in candidates:
        size = _round_up(tensor_bytes(graph, t))
        lo, hi = first[t], last[t]
        overlapping = sorted(
            (off, end) for off, end, f, l in placed if not (l < lo or f > hi)
        )
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
    gen/ header — the ceiling the generated arena must not exceed."""
    for name, value in _MACRO_RE.findall(pathlib.Path(header).read_text()):
        if name == macro:
            return int(value)
    raise UnsupportedGraph(f"{header} does not define {macro}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Print the planned arena for both models**

Run:

```bash
KWS_DATA_ROOT=<your data root> uv run --no-sync python -c "
from kws_de import codegen, config, tflite_graph
for name, macro, header in (('hey_bus', 'KWS_WAKE_ARENA_BYTES', 'wake_model_config.h'),
                            ('command', 'KWS_MODEL_ARENA_BYTES', 'model_config.h')):
    g = tflite_graph.read_graph((config.MODELS_DIR / f'{name}.tflite').read_bytes())
    plan = codegen.rewrite_streaming(g)
    a = codegen.plan_arena(plan)
    tflm = codegen.tflm_arena_bytes('firmware/main/gen/' + header, macro)
    print(f'{name}: arena {a.size} B, rings {sum(r.bytes for r in plan.rings)} B, TFLM {tflm} B')
"
```

Expected: two lines, each with `arena` well below `TFLM`; wake's ring total is 3792. Record both numbers — Task 7 and Task 8 put them in the boot log and the docs.

- [ ] **Step 6: Commit**

```bash
git add kws_de/codegen.py tests/test_codegen.py
git commit -m "$(cat <<'EOF'
feat(codegen): first-fit arena planner over tensor lifetimes

Intermediate activations share slots when their lifetimes do not overlap; the
arena must come in at or below the TFLM arena the same model needs today,
which the tests assert against the committed gen/ headers.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 4: Convolution family emitters and one-layer host parity

**Files:**

- Modify: `kws_de/codegen.py`
- Modify: `firmware/test/Makefile`
- Create: `firmware/test/test_infer_parity.c`
- Test: `tests/test_codegen.py`, `tests/test_codegen_parity.py`

**Interfaces:**

- Consumes from Tasks 1–3: `tflite_graph.read_graph`, `constant`, `Plan`, `Arena`, `plan_arena`, `rewrite_streaming`, `tensor_bytes`, `UnsupportedGraph`.
- Produces:

  ```python
  def quantize_multiplier(double_multiplier: float) -> tuple[int, int]: ...
  def per_channel_multipliers(graph, op) -> tuple[list[int], list[int]]: ...
  def activation_range(graph, op) -> tuple[int, int]: ...
  def padding_hw(in_h, in_w, k_h, k_w, s_h, s_w, padding) -> tuple[int, int, int, int]:
      """-> (pad_h, pad_w, out_h, out_w)"""
  def emit_conv(ctx: "Emitter", op) -> None: ...
  def emit_depthwise(ctx: "Emitter", op) -> None: ...
  def emit_fully_connected(ctx: "Emitter", op) -> None: ...
  ```

**The esp-nn signatures the emitters must produce calls to**, verbatim from `esp_nn_ansi_headers.h` (the `esp_nn_*` names without a suffix are macros that resolve to `_esp32s3` on device and `_ansi` on the host):

```c
typedef struct data_dims { int32_t width; int32_t height; int32_t channels; int32_t extra; } data_dims_t;
typedef struct data_2d   { int32_t width; int32_t height; } data_2d_t;
typedef struct act_params { int32_t min; int32_t max; } act_params_t;
typedef struct quant_data { int32_t *shift; int32_t *mult; } quant_data_t;
typedef struct conv_params {
    int32_t in_offset; int32_t out_offset;
    data_2d_t stride; data_2d_t padding; data_2d_t dilation; act_params_t activation;
} conv_params_t;
typedef struct dw_conv_params {
    int32_t in_offset; int32_t out_offset; int32_t ch_mult;
    data_2d_t stride; data_2d_t padding; data_2d_t dilation; act_params_t activation;
} dw_conv_params_t;

void esp_nn_conv_s8(const data_dims_t *input_dims, const int8_t *input_data,
                    const data_dims_t *filter_dims, const int8_t *filter_data,
                    const int32_t *bias, const data_dims_t *output_dims, int8_t *out_data,
                    const conv_params_t *conv_params, const quant_data_t *quant_data);
void esp_nn_depthwise_conv_s8(const data_dims_t *input_dims, const int8_t *input_data,
                              const data_dims_t *filter_dims, const int8_t *filter_data,
                              const int32_t *bias, const data_dims_t *output_dims, int8_t *out_data,
                              const dw_conv_params_t *conv_params, const quant_data_t *quant_data);
int  esp_nn_get_conv_scratch_size(const data_dims_t *input_dims, const data_dims_t *filter_dims,
                                  const data_dims_t *output_dims, const conv_params_t *conv_params);
void esp_nn_set_conv_scratch_buf(const void *buf);
int  esp_nn_get_depthwise_conv_scratch_size(const data_dims_t *input_dims, const data_dims_t *filter_dims,
                                            const data_dims_t *output_dims, const dw_conv_params_t *conv_params);
void esp_nn_set_depthwise_conv_scratch_buf(const void *buf);
void esp_nn_fully_connected_s8(const int8_t *input_data, const int32_t input_offset,
                               const uint16_t row_len, const int8_t *filter_data,
                               const int32_t filter_offset, const int32_t *bias,
                               int8_t *out_data, const uint16_t out_channels,
                               const int32_t out_offset, const int32_t out_shift,
                               const int32_t out_mult, const int32_t activation_min,
                               const int32_t activation_max);
```

**How esp-tflite-micro fills those in** (`kernels/esp_nn/conv.cc` lines 231-260, `depthwise_conv.cc` 115-144, `fully_connected.cc` 224-235) — the generated C must produce the same values:

- `input_dims = {.width = in_w, .height = in_h, .channels = in_c, .extra = 1}`,
  `output_dims = {.width = out_w, .height = out_h, .channels = out_c, .extra = 1}`,
  `filter_dims = {.width = k_w, .height = k_h, .channels = filter->dims->data[3], .extra = 0}`.
- `conv_params.in_offset = -input_zero_point`, `.out_offset = output_zero_point`, `.dilation = {0, 0}`,
  `.activation = {activation_min, activation_max}`; depthwise additionally `.ch_mult = depth_multiplier`.
- `quant_data = {.shift = per_channel_output_shift, .mult = per_channel_output_multiplier}`.
- Fully connected: `esp_nn_fully_connected_s8(input, -input_zero_point, accum_depth, filter, -filter_zero_point, bias, out, output_depth, output_zero_point, output_shift, output_multiplier, act_min, act_max)` — note the argument order is **shift then mult**, the opposite of `quant_data_t`.

**The multiplier math**, from `tflite::PopulateConvolutionQuantizationParams` (`kernels/kernel_util.cc`) and `tflite::QuantizeMultiplier` (`kernels/internal/quantization_util.cc`):

```cpp
const double effective_output_scale =
    static_cast<double>(input_scale) * static_cast<double>(filter_scale[i]) /
    static_cast<double>(output_scale);
QuantizeMultiplier(effective_output_scale, &per_channel_multiplier[i], &per_channel_shift[i]);

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) {
  if (double_multiplier == 0.) { *quantized_multiplier = 0; *shift = 0; return; }
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1LL << 31)));
  if (q_fixed == (1LL << 31)) { q_fixed /= 2; ++*shift; }
  if (*shift < -31) { *shift = 0; q_fixed = 0; }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}
```

`FULLY_CONNECTED` uses the same formula with the single filter scale (`fully_connected_common.cc:137-141`), storing `output_multiplier`/`output_shift`.

**The activation range**, from `CalculateActivationRangeQuantizedImpl` (`kernels/kernel_util.cc`), with `qmin = -128`, `qmax = 127` for int8:

```cpp
auto quantize = [&](float f) { return zero_point + static_cast<int32_t>(TfLiteRound(f / scale)); };
kTfLiteActRelu:       act_min = max(qmin, quantize(0.0)); act_max = qmax;
kTfLiteActRelu6:      act_min = max(qmin, quantize(0.0)); act_max = min(qmax, quantize(6.0));
kTfLiteActReluN1To1:  act_min = max(qmin, quantize(-1.0)); act_max = min(qmax, quantize(1.0));
default:              act_min = qmin; act_max = qmax;
```

`TfLiteRound` is `std::round` — half away from zero, **not** Python's banker's rounding.

**Padding**, from `ComputeOutSize`/`ComputePadding` (`kernels/internal/types.h`, `padding.h`):

```cpp
out = (padding == SAME) ? (in + stride - 1) / stride
                        : (in + stride - effective_filter) / stride;   // VALID
pad = max(0, ((out - 1) * stride + effective_filter - in) / 2);
```

- [ ] **Step 1: Write the failing unit test for the multiplier math**

Append to `tests/test_codegen.py`:

```python
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
    assert codegen.activation_range(g, op) == (-128, 127)          # NONE
    relu = dataclasses_replace(op, options={**op.options,
                                            "fused_activation_function": "RELU"})
    # output scale 0.5, zp -128 -> quantize(0.0) == -128 -> clamped to qmin
    assert codegen.activation_range(g, relu) == (-128, 127)


def test_padding_same_and_valid():
    assert codegen.padding_hw(49, 10, 3, 3, 1, 1, "SAME") == (1, 1, 49, 10)
    assert codegen.padding_hw(5, 1, 5, 1, 3, 1, "VALID") == (0, 0, 1, 1)
    assert codegen.padding_hw(1, 1, 1, 1, 1, 1, "SAME") == (0, 0, 1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_codegen.py -q -k "multiplier or activation or padding"`
Expected: FAIL with `AttributeError: module 'kws_de.codegen' has no attribute 'quantize_multiplier'`.

- [ ] **Step 3: Write the parameter preparation**

Append to `kws_de/codegen.py`:

```python
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
    return (float(np.float32(in_scale)) * float(np.float32(filter_scale))
            / float(np.float32(out_scale)))


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
    """(pad_h, pad_w, out_h, out_w) — ComputeOutSize + ComputePadding, dilation 1."""
    if padding == "SAME":
        out_h, out_w = -(-in_h // s_h), -(-in_w // s_w)
    elif padding == "VALID":
        out_h, out_w = (in_h - k_h) // s_h + 1, (in_w - k_w) // s_w + 1
    else:
        raise UnsupportedGraph(f"unknown padding {padding!r}")
    pad_h = max(0, ((out_h - 1) * s_h + k_h - in_h) // 2)
    pad_w = max(0, ((out_w - 1) * s_w + k_w - in_w) // 2)
    return pad_h, pad_w, out_h, out_w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_codegen.py -q -k "multiplier or activation or padding"`
Expected: PASS, 3 tests.

- [ ] **Step 5: Write the emitter scaffolding and the three conv-family emitters**

Append to `kws_de/codegen.py`:

```python
@dataclasses.dataclass
class Emitter:
    """Accumulates the pieces of one generated .c file."""

    prefix: str                       # "wake" or "command"
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
    return (f"const data_dims_t {name} = {{ .width = {width}, .height = {height}, "
            f".channels = {channels}, .extra = {extra} }};")


def _nhwc(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 4 or shape[0] != 1:
        raise UnsupportedGraph(f"expected a [1, H, W, C] tensor, got {shape}")
    return shape[1], shape[2], shape[3]


def emit_conv(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_h, in_w, in_c = _nhwc(g.tensors[op.inputs[0]].shape)
    out_h, out_w, out_c = _nhwc(g.tensors[op.outputs[0]].shape)
    w = g.tensors[op.inputs[1]]
    _, k_h, k_w, w_c = w.shape
    s_h, s_w = int(op.options["stride_h"]), int(op.options["stride_w"])
    pad_h, pad_w, exp_h, exp_w = padding_hw(in_h, in_w, k_h, k_w, s_h, s_w,
                                            str(op.options["padding"]))
    if (exp_h, exp_w) != (out_h, out_w):
        raise UnsupportedGraph(
            f"op {op.index} CONV_2D: padding gives {exp_h}x{exp_w}, tensor "
            f"t{op.outputs[0]} says {out_h}x{out_w}"
        )
    tag = f"op{op.index}"
    weights = ctx.const_i8(f"{tag}_w", tflite_graph.constant(g, op.inputs[1]))
    bias = ctx.const_i32(f"{tag}_b", tflite_graph.constant(g, op.inputs[2], "int32"))
    mults, shifts = per_channel_multipliers(g, op)
    mult = ctx.const_i32(f"{tag}_mult", mults)
    shift = ctx.const_i32(f"{tag}_shift", shifts)
    act_min, act_max = activation_range(g, op)
    in_zp = int(g.tensors[op.inputs[0]].zero_points[0])
    out_zp = int(g.tensors[op.outputs[0]].zero_points[0])
    ctx.emit("{")
    ctx.emit("  " + _dims(f"{tag}_in", in_w, in_h, in_c, 1))
    ctx.emit("  " + _dims(f"{tag}_out", out_w, out_h, out_c, 1))
    ctx.emit("  " + _dims(f"{tag}_flt", k_w, k_h, w_c, 0))
    ctx.emit(f"  const conv_params_t {tag}_p = {{ .in_offset = {-in_zp}, "
             f".out_offset = {out_zp}, .stride = {{ {s_w}, {s_h} }}, "
             f".padding = {{ {pad_w}, {pad_h} }}, .dilation = {{ 0, 0 }}, "
             f".activation = {{ {act_min}, {act_max} }} }};")
    ctx.emit(f"  const quant_data_t {tag}_q = {{ .shift = (int32_t *){shift}, "
             f".mult = (int32_t *){mult} }};")
    ctx.emit("  esp_nn_set_conv_scratch_buf(scratch);")
    ctx.emit(f"  esp_nn_conv_s8(&{tag}_in, {ctx.ref(op.inputs[0])}, &{tag}_flt, {weights}, "
             f"{bias}, &{tag}_out, {ctx.ref(op.outputs[0])}, &{tag}_p, &{tag}_q);")
    ctx.emit("}")


def emit_depthwise(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_h, in_w, in_c = _nhwc(g.tensors[op.inputs[0]].shape)
    out_h, out_w, out_c = _nhwc(g.tensors[op.outputs[0]].shape)
    w = g.tensors[op.inputs[1]]
    _, k_h, k_w, w_c = w.shape
    ch_mult = int(op.options.get("depth_multiplier", 1))
    if in_c * ch_mult != out_c:
        raise UnsupportedGraph(
            f"op {op.index} DEPTHWISE_CONV_2D: {in_c} x {ch_mult} != {out_c} output channels"
        )
    s_h, s_w = int(op.options["stride_h"]), int(op.options["stride_w"])
    pad_h, pad_w, exp_h, exp_w = padding_hw(in_h, in_w, k_h, k_w, s_h, s_w,
                                            str(op.options["padding"]))
    if (exp_h, exp_w) != (out_h, out_w):
        raise UnsupportedGraph(
            f"op {op.index} DEPTHWISE_CONV_2D: padding gives {exp_h}x{exp_w}, tensor "
            f"t{op.outputs[0]} says {out_h}x{out_w}"
        )
    tag = f"op{op.index}"
    weights = ctx.const_i8(f"{tag}_w", tflite_graph.constant(g, op.inputs[1]))
    bias = ctx.const_i32(f"{tag}_b", tflite_graph.constant(g, op.inputs[2], "int32"))
    mults, shifts = per_channel_multipliers(g, op)
    mult = ctx.const_i32(f"{tag}_mult", mults)
    shift = ctx.const_i32(f"{tag}_shift", shifts)
    act_min, act_max = activation_range(g, op)
    in_zp = int(g.tensors[op.inputs[0]].zero_points[0])
    out_zp = int(g.tensors[op.outputs[0]].zero_points[0])
    ctx.emit("{")
    ctx.emit("  " + _dims(f"{tag}_in", in_w, in_h, in_c, 1))
    ctx.emit("  " + _dims(f"{tag}_out", out_w, out_h, out_c, 1))
    ctx.emit("  " + _dims(f"{tag}_flt", k_w, k_h, w_c, 0))
    ctx.emit(f"  const dw_conv_params_t {tag}_p = {{ .in_offset = {-in_zp}, "
             f".out_offset = {out_zp}, .ch_mult = {ch_mult}, "
             f".stride = {{ {s_w}, {s_h} }}, .padding = {{ {pad_w}, {pad_h} }}, "
             f".dilation = {{ 0, 0 }}, .activation = {{ {act_min}, {act_max} }} }};")
    ctx.emit(f"  const quant_data_t {tag}_q = {{ .shift = (int32_t *){shift}, "
             f".mult = (int32_t *){mult} }};")
    ctx.emit("  esp_nn_set_depthwise_conv_scratch_buf(scratch);")
    ctx.emit(f"  esp_nn_depthwise_conv_s8(&{tag}_in, {ctx.ref(op.inputs[0])}, &{tag}_flt, "
             f"{weights}, {bias}, &{tag}_out, {ctx.ref(op.outputs[0])}, &{tag}_p, &{tag}_q);")
    ctx.emit("}")


def emit_fully_connected(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    w = g.tensors[op.inputs[1]]
    out_t = g.tensors[op.outputs[0]]
    out_depth = out_t.shape[-1]
    accum_depth = w.shape[-1]
    if len(w.scales) != 1:
        raise UnsupportedGraph(
            f"op {op.index} FULLY_CONNECTED: filter t{w.index} is per-channel "
            f"quantised ({len(w.scales)} scales); only per-tensor is emitted"
        )
    tag = f"op{op.index}"
    weights = ctx.const_i8(f"{tag}_w", tflite_graph.constant(g, op.inputs[1]))
    bias = ctx.const_i32(f"{tag}_b", tflite_graph.constant(g, op.inputs[2], "int32"))
    mult, shift = quantize_multiplier(
        _effective_scale(g.tensors[op.inputs[0]].scales[0], w.scales[0], out_t.scales[0])
    )
    act_min, act_max = activation_range(g, op)
    in_zp = int(g.tensors[op.inputs[0]].zero_points[0])
    w_zp = int(w.zero_points[0])
    out_zp = int(out_t.zero_points[0])
    ctx.emit(f"esp_nn_fully_connected_s8({ctx.ref(op.inputs[0])}, {-in_zp}, {accum_depth}, "
             f"{weights}, {-w_zp}, {bias}, {ctx.ref(op.outputs[0])}, {out_depth}, "
             f"{out_zp}, {shift}, {mult}, {act_min}, {act_max});")
```

- [ ] **Step 6: Write the one-layer host parity test (C side)**

Create `firmware/test/test_infer_parity.c` with just the conv-layer case for now:

```c
/* Host parity harness: the generated inference C compiled against esp-nn's
   ANSI-C reference kernels, checked byte-for-byte against expectations the
   TFLite interpreter produced (kws_de.codegen writes gen/*_vectors.h).

   The device runs esp-nn's ESP32-S3 kernels, not these ANSI ones; esp-nn's own
   contract is that the two agree bit-for-bit as long as CONFIG_NN_SKIP_NUDGE /
   SKIP_NUDGE is NOT defined (that macro selects a faster, non-bit-exact
   requantisation). The device parity log line closes the remaining gap. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "gen/conv_probe_vectors.h"

void conv_probe(const int8_t *in, int8_t *out, void *scratch);

int main(void)
{
    static int8_t out[CONV_PROBE_OUT_LEN];
    static int8_t scratch[CONV_PROBE_SCRATCH];
    conv_probe(CONV_PROBE_IN, out, scratch);
    int bad = 0;
    for (int i = 0; i < CONV_PROBE_OUT_LEN; i++)
        if (out[i] != CONV_PROBE_EXPECT[i]) {
            if (bad < 8)
                printf("conv byte %d: got %d want %d\n", i, out[i], CONV_PROBE_EXPECT[i]);
            bad++;
        }
    printf("conv parity: %d/%d bytes differ\n", bad, CONV_PROBE_OUT_LEN);
    assert(bad == 0);
    puts("test_infer_parity OK");
    return 0;
}
```

- [ ] **Step 7: Write the one-layer host parity test (Python side)**

Create `tests/test_codegen_parity.py`:

```python
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
WAKE = config.MODELS_DIR / "hey_bus.tflite"

needs_wake = pytest.mark.skipif(not WAKE.exists(), reason=f"{WAKE} absent (KWS_DATA_ROOT)")
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")


def _make(target: str) -> None:
    subprocess.run(["make", "-C", str(TEST_DIR), target], check=True)


@needs_wake
@needs_cc
def test_first_conv_layer_is_byte_identical(tmp_path):
    """The wake model's first CONV_2D, generated and compiled, against the same
    op run through the interpreter as a one-op probe model."""
    blob = WAKE.read_bytes()
    graph = tflite_graph.read_graph(blob)
    conv = next(op for op in graph.ops if op.name == "CONV_2D")
    in_t = graph.tensors[conv.inputs[0]]
    rng = np.random.default_rng(0)
    inputs = rng.integers(-128, 128, size=in_t.shape, dtype=np.int8)

    itp = tf.lite.Interpreter(model_content=tflite_graph.probe_model(blob, conv))
    itp.allocate_tensors()
    detail_in, detail_out = itp.get_input_details()[0], itp.get_output_details()[0]
    itp.set_tensor(detail_in["index"], inputs)
    itp.invoke()
    expect = itp.get_tensor(detail_out["index"]).astype(np.int8).ravel()

    codegen.write_probe_vectors(blob, conv, inputs, expect, TEST_DIR.parent / "main" / "gen")
    _make("test_infer_parity")
    result = subprocess.run([str(TEST_DIR / "test_infer_parity")],
                            capture_output=True, text=True, check=True)
    assert "0/" in result.stdout.split("differ")[0]
    assert "test_infer_parity OK" in result.stdout
```

And add `write_probe_vectors` to `kws_de/codegen.py`:

```python
def write_probe_vectors(tflite: bytes, op, inputs, expect, gen_dir) -> None:
    """Emit gen/conv_probe_vectors.h plus a `conv_probe` function wrapping one
    op — the smallest thing that proves the emitters' parameter preparation
    matches TFLM's, before a whole model is generated."""
    graph = tflite_graph.read_graph(tflite)
    plan = Plan(graph=graph, ops=(op,), rings=(), alias={})
    arena = plan_arena(plan)
    ctx = Emitter(prefix="conv_probe", plan=plan, arena=arena)
    {"CONV_2D": emit_conv, "DEPTHWISE_CONV_2D": emit_depthwise,
     "FULLY_CONNECTED": emit_fully_connected}[op.name](ctx, op)
    gen_dir = pathlib.Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    scratch = 8192
    body = "\n".join(ctx.body).replace("(arena + 0)", "out")
    (gen_dir / "conv_probe.c").write_text(
        '#include <stdint.h>\n#include "esp_nn.h"\n\n'
        + "\n".join(ctx.consts)
        + "\n\nvoid conv_probe(const int8_t *in, int8_t *out, void *scratch)\n{\n"
        + body
        + "\n}\n"
    )
    flat_in = ", ".join(str(int(v)) for v in np.ravel(inputs))
    flat_expect = ", ".join(str(int(v)) for v in np.ravel(expect))
    (gen_dir / "conv_probe_vectors.h").write_text(
        "/* generated by kws-codegen — do not edit */\n#pragma once\n#include <stdint.h>\n"
        f"#define CONV_PROBE_OUT_LEN {np.size(expect)}\n"
        f"#define CONV_PROBE_SCRATCH {scratch}\n"
        f"static const int8_t CONV_PROBE_IN[{np.size(inputs)}] = {{{flat_in}}};\n"
        f"static const int8_t CONV_PROBE_EXPECT[{np.size(expect)}] = {{{flat_expect}}};\n"
    )
```

- [ ] **Step 8: Add the host target with a pinned esp-nn checkout**

Append to `firmware/test/Makefile` (keep the existing `TESTS` line and add the new target to it):

<!-- markdownlint-disable MD010 -->

```make
# esp-nn, pinned. The device gets it through the esp-tflite-micro managed
# component; the host parity test needs the same sources, so fetch the same tag
# into an ignored directory rather than vendoring a copy that can drift.
# NEVER define SKIP_NUDGE / CONFIG_NN_SKIP_NUDGE here: that macro selects
# esp-nn's faster, NON-bit-exact requantisation and would silently break parity.
ESP_NN_REF ?= v1.1.2
ESP_NN ?= .esp-nn
ESP_NN_SRCS := $(ESP_NN)/src/convolution/esp_nn_conv_ansi.c \
               $(ESP_NN)/src/convolution/esp_nn_depthwise_conv_ansi.c \
               $(ESP_NN)/src/fully_connected/esp_nn_fully_connected_ansi.c \
               $(ESP_NN)/src/softmax/esp_nn_softmax_ansi.c \
               $(ESP_NN)/src/pooling/esp_nn_avg_pool_ansi.c \
               $(ESP_NN)/src/common/esp_nn_mean_ansi.c
ESP_NN_CFLAGS := -I$(ESP_NN)/include -I$(ESP_NN)/src/common

$(ESP_NN):
	git clone --depth 1 --branch $(ESP_NN_REF) https://github.com/espressif/esp-nn $(ESP_NN)

test_infer_parity: test_infer_parity.c $(ESP_NN) ../main/gen/conv_probe.c
	$(CC) $(CFLAGS) $(ESP_NN_CFLAGS) -o $@ test_infer_parity.c ../main/gen/conv_probe.c \
	  $(ESP_NN_SRCS) -lm
```

<!-- markdownlint-enable MD010 -->

Recipe lines above use real tab characters, as make requires.

Add `firmware/test/.esp-nn/` and `firmware/main/gen/conv_probe*` to `.gitignore` — the probe is a test fixture, not a shipped header.

- [ ] **Step 9: Run both sides**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen_parity.py -q`
Expected: PASS; the subprocess prints `conv parity: 0/32 bytes differ`. If bytes differ, the fault is in the parameter preparation, not the kernel — compare `per_channel_multipliers` against the values TFLM computes for the same op before touching anything else.

- [ ] **Step 10: Full gates and commit**

Run: `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q && make -C firmware/test`

```bash
git add kws_de/codegen.py tests/test_codegen.py tests/test_codegen_parity.py \
        firmware/test/Makefile firmware/test/test_infer_parity.c .gitignore
git commit -m "$(cat <<'EOF'
feat(codegen): conv/depthwise/fully-connected emitters, byte-exact

Multipliers, shifts and activation ranges are prepared with TFLM's own
integer math, and the esp-nn call arguments mirror esp-tflite-micro's kernels
exactly. A host parity target compiles one generated layer against esp-nn's
ANSI kernels and compares it to the interpreter byte for byte.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 5: Remaining emitters and whole-wake-model parity

**Files:**

- Modify: `kws_de/codegen.py`, `firmware/test/test_infer_parity.c`, `firmware/test/Makefile`
- Test: `tests/test_codegen.py`, `tests/test_codegen_parity.py`

**Interfaces:**

- Consumes from Task 4: `Emitter`, `quantize_multiplier`, `activation_range`, `padding_hw`, `per_channel_multipliers`, `emit_conv`, `emit_depthwise`, `emit_fully_connected`, `_effective_scale`.
- Produces:

  ```python
  def softmax_params(graph, op) -> tuple[int, int, int]:
      """-> (input_multiplier, input_left_shift, diff_min)"""
  def logistic_lut(tflite: bytes, op) -> list[int]:
      """256 int8 outputs, index = input + 128, from the reference kernel itself"""
  def emit_average_pool(ctx, op) -> None: ...
  def emit_mean(ctx, op) -> None: ...
  def emit_softmax(ctx, op) -> None: ...
  def emit_logistic(ctx, op, lut) -> None: ...
  def emit_quantize(ctx, op) -> None: ...
  def generate(tflite: bytes, name: str) -> dict[str, str]:
      """-> {"<name>_infer.c": ..., "<name>_infer.h": ...}"""
  ```

**Reference math these emitters must reproduce:**

- **SOFTMAX** (`micro/kernels/softmax_common.cc:CalculateSoftmaxParams`, int8): output must be `zero_point == -128, scale == 1/256`. With `kScaledDiffIntegerBits = 5`,
  `PreprocessSoftmaxScaling(beta, input_scale, 5, &input_multiplier, &input_left_shift)`, where
  `input_beta_real_multiplier = min(beta * input_scale * (1 << (31 - 5)), (1LL << 31) - 1.0)` then `QuantizeMultiplier`; and
  `diff_min = -CalculateInputRadius(5, input_left_shift, 31)` with
  `CalculateInputRadius(bits, shift, total) = floor(((1 << bits) - 1) * (1LL << (total - bits)) / (1LL << shift))`.
  The esp-nn call is `esp_nn_softmax_s8(in, height /* outer size */, width /* depth */, mult, shift, diff_min, out)` and needs
  `esp_nn_set_softmax_scratch_buf(buf)` with `esp_nn_get_softmax_scratch_size(width, height)` bytes, 4-byte aligned (`kernels/esp_nn/softmax.cc:77-83, 155-159`).
- **LOGISTIC** (`reference/integer_ops/logistic.h`) is gemmlowp fixed-point sigmoid. Its int8 input has only 256 possible values, so the generator does **not** reimplement it: `tflite_graph.probe_model` builds a one-op model with the same quantisation, runs all 256 inputs through the interpreter, and emits the answers as a 256-byte lookup table. Bit-exact by construction.
- **QUANTIZE** (`reference/requantize.h`) — for the wake model this is int8 → uint8 at the same scale with `zero_point_diff == -128`, which the reference resolves to `output[i] = input[i] ^ 0x80`. Any other combination gets the general path:
  `output = clamp(MultiplyByQuantizedMultiplier(input - in_zp, mult, shift) + out_zp)` with
  `mult, shift = QuantizeMultiplier(in_scale / out_scale)`.
- **MEAN** over axes (1, 2) of a `[1, H, W, C]` int8 tensor (`reference/reduce.h:QuantizedMeanOrSum`):

  ```text
  mult, shift = QuantizeMultiplier(input_scale / output_scale)
  n           = H * W
  s           = min(63 - clz(n), 32, 31 + shift)
  mult        = (int64(mult) << s) // n          # truncating, C++ integer division
  shift       = shift - s
  out[c]      = clamp(MultiplyByQuantizedMultiplier(sum_c - in_zp * n, mult, shift) + out_zp)
  ```

  esp-nn's `esp_nn_mean_nhwc_s8` takes exactly `(input, output, height, width, channels, input_zero_point, output_zero_point, multiplier, shift)` — pass the **readapted** multiplier and shift above.
- **AVERAGE_POOL_2D**: `esp_nn_avg_pool_s8(input, in_w, in_h, out, out_w, out_h, stride_w, stride_h, filter_w, filter_h, pad_w, pad_h, act_min, act_max, channels)`. It carries no requantisation, so refuse loudly if the op's input and output quantisation differ.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codegen.py`:

```python
def test_softmax_params_match_tflm_for_the_command_head():
    """Input scale 0.371104, beta 1.0 -> PreprocessSoftmaxScaling + diff_min."""
    mult, shift, diff_min = codegen.softmax_params_from_scale(0.371104, beta=1.0)
    assert shift >= 0
    assert 0 < mult <= (1 << 31) - 1
    assert diff_min < 0
    # CalculateInputRadius(5, shift, 31) == floor(31 * 2**26 / 2**shift)
    assert diff_min == -((31 * (1 << 26)) >> shift)


@needs_wake
def test_logistic_lut_is_256_monotone_entries_from_the_reference_kernel():
    blob = WAKE.read_bytes()
    g = tflite_graph.read_graph(blob)
    op = next(o for o in g.ops if o.name == "LOGISTIC")
    lut = codegen.logistic_lut(blob, op)
    assert len(lut) == 256
    assert all(-128 <= v <= 127 for v in lut)
    assert all(b >= a for a, b in zip(lut, lut[1:])), "sigmoid must be non-decreasing"
    assert lut[0] == -128 and lut[-1] == 127


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
    assert "memmove" in source          # the ring shift
    assert "SKIP_NUDGE" not in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py -q -k "softmax or logistic or generate"`
Expected: FAIL with `AttributeError: module 'kws_de.codegen' has no attribute 'softmax_params_from_scale'`.

- [ ] **Step 3: Write the remaining emitters and the file generator**

Append to `kws_de/codegen.py`:

```python
_SCALED_DIFF_INTEGER_BITS = 5


def softmax_params_from_scale(input_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    """PreprocessSoftmaxScaling + diff_min, as CalculateSoftmaxParams does."""
    bits = _SCALED_DIFF_INTEGER_BITS
    max_real = float((1 << 31) - 1)
    real = min(beta * float(np.float32(input_scale)) * float(1 << (31 - bits)), max_real)
    mult, shift = quantize_multiplier(real)
    radius = ((1 << bits) - 1) * (1 << (31 - bits)) // (1 << shift)
    return mult, shift, -radius


def softmax_params(graph, op) -> tuple[int, int, int]:
    in_t, out_t = graph.tensors[op.inputs[0]], graph.tensors[op.outputs[0]]
    if int(out_t.zero_points[0]) != -128 or abs(float(out_t.scales[0]) - 1.0 / 256) > 1e-12:
        raise UnsupportedGraph(
            f"op {op.index} SOFTMAX: output t{op.outputs[0]} must be int8 with "
            f"scale 1/256 and zero point -128, got {out_t.scales[0]}/{out_t.zero_points[0]}"
        )
    return softmax_params_from_scale(in_t.scales[0], float(op.options.get("beta", 1.0)))


def logistic_lut(tflite: bytes, op) -> list[int]:
    """The reference LOGISTIC kernel's answer for every possible int8 input.

    gemmlowp's fixed-point sigmoid is not worth reimplementing: the input is one
    byte wide, so run the reference kernel itself over all 256 values through a
    one-op probe model and freeze the result as a table.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_graph.probe_model(tflite, op))
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
    tag = f"op{op.index}"
    table = ctx.const_i8(f"{tag}_lut", lut)
    count = math.prod(g.tensors[op.inputs[0]].shape)
    ctx.emit(f"for (int i = 0; i < {count}; i++)")
    ctx.emit(f"    {ctx.ref(op.outputs[0])}[i] = "
             f"{table}[(uint8_t)({ctx.ref(op.inputs[0])}[i] + 128)];")


def emit_quantize(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    count = math.prod(in_t.shape)
    in_zp, out_zp = int(in_t.zero_points[0]), int(out_t.zero_points[0])
    same_scale = float(np.float32(in_t.scales[0])) == float(np.float32(out_t.scales[0]))
    cast = "uint8_t" if out_t.dtype == "uint8" else "int8_t"
    if same_scale and in_t.dtype == "int8" and out_t.dtype == "uint8" and in_zp - out_zp == -128:
        # reference/requantize.h fast path: a pure 128 shift is a sign-bit flip.
        ctx.emit(f"for (int i = 0; i < {count}; i++)")
        ctx.emit(f"    (({cast} *){ctx.ref(op.outputs[0])})[i] = "
                 f"(uint8_t)({ctx.ref(op.inputs[0])}[i] ^ 0x80);")
        return
    mult, shift = quantize_multiplier(
        float(np.float32(in_t.scales[0])) / float(np.float32(out_t.scales[0]))
    )
    lo, hi = (0, 255) if out_t.dtype == "uint8" else (-128, 127)
    ctx.emit(f"for (int i = 0; i < {count}; i++) {{")
    ctx.emit(f"    int32_t v = kws_requantize({ctx.ref(op.inputs[0])}[i] - {in_zp}, "
             f"{mult}, {shift}) + {out_zp};")
    ctx.emit(f"    (({cast} *){ctx.ref(op.outputs[0])})[i] = "
             f"(v < {lo}) ? {lo} : ((v > {hi}) ? {hi} : v);")
    ctx.emit("}")


def emit_softmax(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    shape = g.tensors[op.inputs[0]].shape
    depth = shape[-1]
    outer = math.prod(shape) // depth
    mult, shift, diff_min = softmax_params(g, op)
    ctx.emit("esp_nn_set_softmax_scratch_buf(scratch);")
    ctx.emit(f"esp_nn_softmax_s8({ctx.ref(op.inputs[0])}, {outer}, {depth}, "
             f"{mult}, {shift}, {diff_min}, {ctx.ref(op.outputs[0])});")


def emit_mean(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    in_h, in_w, in_c = _nhwc(in_t.shape)
    axes = sorted(int(a) for a in tflite_graph.constant(g, op.inputs[1], "int32"))
    if axes != [1, 2] or out_t.shape[-1] != in_c:
        raise UnsupportedGraph(
            f"op {op.index} MEAN: only reduction over axes [1, 2] of a [1, H, W, C] "
            f"tensor is emitted, got axes {axes} and output shape {out_t.shape}"
        )
    mult, shift = quantize_multiplier(
        float(np.float32(in_t.scales[0])) / float(np.float32(out_t.scales[0]))
    )
    # Readapt the rescale to fold in 1 / (H * W), exactly as QuantizedMeanOrSum:
    #   shift = min(63 - CountLeadingZeros(uint64 n), 32, 31 + output_shift)
    # 63 - clz64(n) is floor(log2(n)), i.e. n.bit_length() - 1 for n >= 1.
    n = in_h * in_w
    s = min(n.bit_length() - 1, 32, 31 + shift)
    mult = int((mult << s) // n)
    shift = shift - s
    ctx.emit(f"esp_nn_mean_nhwc_s8({ctx.ref(op.inputs[0])}, {ctx.ref(op.outputs[0])}, "
             f"{in_h}, {in_w}, {in_c}, {int(in_t.zero_points[0])}, "
             f"{int(out_t.zero_points[0])}, {mult}, {shift});")


def emit_average_pool(ctx: Emitter, op) -> None:
    g = ctx.plan.graph
    in_t, out_t = g.tensors[op.inputs[0]], g.tensors[op.outputs[0]]
    if (float(np.float32(in_t.scales[0])) != float(np.float32(out_t.scales[0]))
            or int(in_t.zero_points[0]) != int(out_t.zero_points[0])):
        raise UnsupportedGraph(
            f"op {op.index} AVERAGE_POOL_2D: esp_nn_avg_pool_s8 does not requantise, "
            f"but t{op.inputs[0]} and t{op.outputs[0]} have different quantisation"
        )
    in_h, in_w, in_c = _nhwc(in_t.shape)
    out_h, out_w, _ = _nhwc(out_t.shape)
    k_h, k_w = int(op.options["filter_height"]), int(op.options["filter_width"])
    s_h, s_w = int(op.options["stride_h"]), int(op.options["stride_w"])
    pad_h, pad_w, _, _ = padding_hw(in_h, in_w, k_h, k_w, s_h, s_w, str(op.options["padding"]))
    act_min, act_max = activation_range(g, op)
    ctx.emit(f"esp_nn_avg_pool_s8({ctx.ref(op.inputs[0])}, {in_w}, {in_h}, "
             f"{ctx.ref(op.outputs[0])}, {out_w}, {out_h}, {s_w}, {s_h}, {k_w}, {k_h}, "
             f"{pad_w}, {pad_h}, {act_min}, {act_max}, {in_c});")


_PROLOGUE = """/* generated by kws-codegen from {model} — do not edit */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_nn.h"

/* TFLM's MultiplyByQuantizedMultiplier, for the ops esp-nn does not cover.
   Same double-rounding path esp-nn's own esp_nn_requantize uses when
   SKIP_NUDGE is not defined. */
static inline int32_t kws_requantize(int32_t x, int32_t mult, int32_t shift)
{{
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
}}
"""


def generate(tflite: bytes, name: str) -> dict[str, str]:
    """Return {"<name>_infer.c": source, "<name>_infer.h": header}."""
    graph = tflite_graph.read_graph(tflite)
    plan = rewrite_streaming(graph)
    scratch = 0
    for op in plan.ops:
        if op.name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            in_h, in_w, in_c = _nhwc(graph.tensors[op.inputs[0]].shape)
            _, k_h, k_w, _ = graph.tensors[op.inputs[1]].shape
            # esp-nn's S3 conv pads and reorders the input into scratch; this
            # bound covers every shape it asks for, and the device logs the real
            # figure returned by esp_nn_get_conv_scratch_size at init.
            scratch = max(scratch, 2 * (in_h + k_h) * (in_w + k_w) * in_c + 64)
        if op.name == "SOFTMAX":
            depth = graph.tensors[op.inputs[0]].shape[-1]
            scratch = max(scratch, 4 * depth + 64)
    arena = plan_arena(plan, scratch_bytes=scratch)
    ctx = Emitter(prefix=name, plan=plan, arena=arena, scratch_bytes=scratch)

    for op in plan.ops:
        if op.name == "RESHAPE":
            continue                                  # alias, no code
        if op.name == "CONV_2D":
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

    return _render(ctx, name, tflite)
```

Then `_render` builds the two files. The wake shape (rings, `_step`) and the command shape (stateless `_infer`) differ only in the wrapper:

```python
def _render(ctx: Emitter, name: str, tflite: bytes) -> dict[str, str]:
    graph = ctx.plan.graph
    in_t = graph.tensors[graph.inputs[0]]
    out_t = graph.tensors[graph.outputs[0]]
    in_len = math.prod(in_t.shape)
    out_len = math.prod(out_t.shape)
    out_ctype = "uint8_t" if out_t.dtype == "uint8" else "int8_t"
    rings = ctx.plan.rings

    statics = [f"static int8_t arena[{ctx.arena.size}] __attribute__((aligned(16)));",
               f"static int8_t scratch[{max(ctx.scratch_bytes, 16)}] __attribute__((aligned(16)));"]
    for ring in rings:
        statics.append(f"static int8_t {ring.name}[{ring.bytes}] "
                       "__attribute__((aligned(16)));")

    prologue = _PROLOGUE.format(model=name)
    lines = [prologue, "\n".join(statics), "", "\n".join(ctx.consts), ""]

    reset = [f"void {name}_infer_reset(void)", "{"]
    for ring in rings:
        reset.append(f"    memset({ring.name}, 0, sizeof {ring.name});")
    reset += ["}", "",
              f"void {name}_infer_init(void) {{ {name}_infer_reset(); }}", "",
              f"size_t {name}_infer_arena_bytes(void)",
              "{",
              f"    return sizeof arena + sizeof scratch{''.join(f' + sizeof {r.name}' for r in rings)};",
              "}", ""]
    lines.append("\n".join(reset))

    if rings:
        entry = [f"void {name}_infer_step(const int8_t in[{in_len}], {out_ctype} *out)", "{"]
        for ring in rings:
            if ring.new_tensor in graph.inputs or ctx.plan.alias.get(ring.new_tensor) in graph.inputs:
                src = "in"
            else:
                src = ctx.ref(ring.new_tensor)
            keep = (ring.rows - ring.new_rows) * ring.channels
            entry.append(f"    memcpy({ring.name} + {keep}, {src}, "
                         f"{ring.new_rows * ring.channels});" if src == "in" else "")
        entry.append("\n".join(ctx.body))
        for ring in rings:
            keep = (ring.rows - ring.new_rows) * ring.channels
            entry.append(f"    memmove({ring.name}, {ring.name} + "
                         f"{ring.new_rows * ring.channels}, {keep});")
        entry.append("}")
        lines.append("\n".join(line for line in entry if line))
        decl = (f"void {name}_infer_init(void);\n"
                f"void {name}_infer_reset(void);\n"
                f"void {name}_infer_step(const int8_t in[{in_t.shape[1]} * {in_t.shape[2]}], "
                f"{out_ctype} *prob_q);\n"
                f"size_t {name}_infer_arena_bytes(void);\n")
    else:
        entry = [f"void {name}_infer(const int8_t in[{in_len}], {out_ctype} out[{out_len}])",
                 "{", "\n".join(ctx.body), "}"]
        lines.append("\n".join(entry))
        decl = (f"void {name}_infer_init(void);\n"
                f"void {name}_infer_reset(void);\n"
                f"void {name}_infer(const int8_t in[{in_len}], {out_ctype} out[{out_len}]);\n"
                f"size_t {name}_infer_arena_bytes(void);\n")

    header = (f"/* generated by kws-codegen from {name} — do not edit */\n"
              "#pragma once\n#include <stddef.h>\n#include <stdint.h>\n\n"
              f"#define {name.upper()}_INFER_INPUT_LEN {in_len}\n"
              f"#define {name.upper()}_INFER_OUTPUT_LEN {out_len}\n"
              f"#define {name.upper()}_INFER_ARENA_BYTES {ctx.arena.size}\n\n"
              '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
              + decl
              + '\n#ifdef __cplusplus\n}\n#endif\n')
    return {f"{name}_infer.c": "\n".join(lines) + "\n", f"{name}_infer.h": header}
```

Where the ring input is not the graph input (the command model has no rings; the wake model's first ring takes the graph input, the rest take an earlier op's output), the `memcpy` source is that tensor's arena slot — `ctx.ref` already returns it, and the copy line is emitted at the point in `ctx.body` where the producing op finishes. Move the per-ring append into the op loop in `generate` (emit `memcpy(ringN + keep, <producer ref>, n_bytes);` right after the op that writes `ring.new_tensor`) so the ordering is correct; only the ring fed by the graph input is copied in the prologue.

- [ ] **Step 4: Run the unit tests**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py -q`
Expected: PASS, 14 tests.

- [ ] **Step 5: Write the whole-model parity test (Python side)**

Append to `tests/test_codegen_parity.py`:

```python
GEN = REPO / "firmware" / "main" / "gen"
WAKE_TAKES = config.DATA_DIR / "recordings" / "approved" / "wake"


def _interpreter_wake_probs(blob: bytes, rows: np.ndarray) -> np.ndarray:
    """The interpreter's uint8 output for every 3-row step over `rows`."""
    itp = tf.lite.Interpreter(model_content=blob)
    itp.allocate_tensors()
    detail_in, detail_out = itp.get_input_details()[0], itp.get_output_details()[0]
    probs = []
    for start in range(0, len(rows) - 2, 3):
        itp.set_tensor(detail_in["index"], rows[start : start + 3][None, ...].astype(np.int8))
        itp.invoke()
        probs.append(int(itp.get_tensor(detail_out["index"]).flat[0]))
    return np.array(probs, dtype=np.uint8)


@needs_wake
@needs_cc
def test_whole_wake_model_matches_the_interpreter_on_the_golden_vector():
    from kws_de import firmware_gen

    pytest.importorskip("pymicro_features")
    blob = WAKE.read_bytes()
    _, rows = firmware_gen.wake_test_vector()
    expect = _interpreter_wake_probs(blob, rows)
    files = codegen.generate(blob, "wake")
    (GEN / "wake_infer.c").write_text(files["wake_infer.c"])
    (GEN / "wake_infer.h").write_text(files["wake_infer.h"])
    codegen.write_wake_vectors(blob, expect, GEN)
    _make("test_infer_parity")
    result = subprocess.run([str(TEST_DIR / "test_infer_parity")],
                            capture_output=True, text=True, check=True)
    assert "wake parity: 0/" in result.stdout, result.stdout


@needs_wake
@pytest.mark.skipif(not WAKE_TAKES.exists(), reason="approved/wake absent")
def test_wake_takes_match_the_interpreter_step_for_step():
    """The 10 approved wake takes, every step's output byte. This runs the
    generated arithmetic in Python (numpy port of the same emitted constants)
    against the interpreter; the C side is covered by the golden vector."""
    import soundfile as sf

    from kws_de import firmware_gen

    blob = WAKE.read_bytes()
    takes = sorted(WAKE_TAKES.rglob("*.wav"))
    assert len(takes) == 10, f"expected 10 approved wake takes, found {len(takes)}"
    for path in takes:
        pcm, rate = sf.read(path, dtype="int16")
        assert rate == config.SAMPLE_RATE
        rows = firmware_gen.wake_features(pcm)
        expect = _interpreter_wake_probs(blob, rows)
        got = codegen.simulate(blob, "wake", rows)
        assert np.array_equal(got, expect), f"{path.name}: {int((got != expect).sum())} steps differ"
```

`codegen.simulate(tflite, name, rows)` is a thin numpy executor over the same `Plan` — the same multipliers, the same ring shifts, `MultiplyByQuantizedMultiplier` in numpy int64 — so a mismatch points at the *parameters*, not at the C. Add it next to `generate`:

```python
def simulate(tflite: bytes, name: str, rows: np.ndarray) -> np.ndarray:
    """Run the planned graph in numpy with the emitted constants.

    This is the same arithmetic the generated C performs, so it isolates a
    parameter bug (wrong multiplier, wrong padding) from a C bug in seconds
    instead of a compile-and-run cycle. It is a test aid, not a shipped path.
    """
```

- [ ] **Step 6: Extend the C harness to the whole wake model**

Replace `firmware/test/test_infer_parity.c`'s body with both cases (keep the conv probe, add the wake model), and add the wake sources to the `test_infer_parity` target in `firmware/test/Makefile`:

```c
#include "gen/wake_infer.h"
#include "gen/wake_infer_vectors.h"
#include "gen/wake_test_vectors.h"

static int check_wake(void)
{
    wake_infer_init();
    int bad = 0;
    for (int step = 0; step < WAKE_INFER_STEPS; step++) {
        uint8_t prob = 0;
        wake_infer_step(&WT_FEATURES[step * 3][0], &prob);
        if (prob != WAKE_INFER_EXPECT[step]) {
            if (bad < 8)
                printf("wake step %d: got %u want %u\n", step, prob, WAKE_INFER_EXPECT[step]);
            bad++;
        }
    }
    printf("wake parity: %d/%d steps differ (arena %u B)\n",
           bad, WAKE_INFER_STEPS, (unsigned)wake_infer_arena_bytes());
    return bad;
}
```

<!-- markdownlint-disable MD010 -->

```make
test_infer_parity: test_infer_parity.c $(ESP_NN) ../main/gen/conv_probe.c ../main/gen/wake_infer.c
	$(CC) $(CFLAGS) $(ESP_NN_CFLAGS) -o $@ test_infer_parity.c ../main/gen/conv_probe.c \
	  ../main/gen/wake_infer.c $(ESP_NN_SRCS) -lm
```

<!-- markdownlint-enable MD010 -->

And `write_wake_vectors` in `kws_de/codegen.py`:

```python
def write_wake_vectors(tflite: bytes, expect, gen_dir) -> None:
    """gen/wake_infer_vectors.h: the interpreter's uint8 output for every step
    over kws-fwgen's WT_FEATURES golden rows. Regenerate whenever either the
    model or wake_test_vectors.h changes — they are read together by
    firmware/test/test_infer_parity.c."""
    body = ", ".join(str(int(v)) for v in np.ravel(expect))
    pathlib.Path(gen_dir, "wake_infer_vectors.h").write_text(
        "/* generated by kws-codegen — do not edit */\n#pragma once\n#include <stdint.h>\n"
        f"#define WAKE_INFER_STEPS {np.size(expect)}\n"
        f"static const uint8_t WAKE_INFER_EXPECT[{np.size(expect)}] = {{{body}}};\n"
    )
```

- [ ] **Step 7: Run everything**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py tests/test_codegen_parity.py -q && make -C firmware/test`
Expected: PASS; the harness prints `conv parity: 0/32 bytes differ` and `wake parity: 0/32 steps differ (arena <N> B)`.

- [ ] **Step 8: Commit**

```bash
git add kws_de/codegen.py tests/test_codegen.py tests/test_codegen_parity.py \
        firmware/test/test_infer_parity.c firmware/test/Makefile
git commit -m "$(cat <<'EOF'
feat(codegen): pooling, softmax, logistic, quantize emitters + wake parity

LOGISTIC is emitted as a 256-entry table read straight out of the reference
kernel via a one-op probe model, so gemmlowp's fixed-point sigmoid does not
have to be reimplemented to stay bit-exact. The whole wake model now runs on
the host against esp-nn's ANSI kernels and matches the interpreter on every
step of the golden vector and all ten approved takes.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 6: `kws-codegen` CLI, committed headers, CI freshness

**Files:**

- Modify: `kws_de/codegen.py`, `pyproject.toml`, `.github/workflows/firmware.yml`
- Create: `firmware/main/gen/wake_infer.c`, `firmware/main/gen/wake_infer.h`, `firmware/main/gen/wake_infer_vectors.h`
- Test: `tests/test_codegen.py`

**Interfaces:**

- Consumes from Task 5: `generate`, `write_wake_vectors`, `UnsupportedGraph`.
- Produces:

  ```python
  def write(tflite_path, name: str, out_dir) -> dict[str, int]:
      """-> {"arena_bytes": ..., "ring_bytes": ..., "ops": ...}"""
  def check(tflite_path, name: str, committed_dir) -> list[str]:
      """names of stale generated files; empty list = OK"""
  def main() -> None: ...    # console script `kws-codegen`
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codegen.py`:

```python
@needs_wake
def test_check_is_clean_against_the_committed_headers():
    stale = codegen.check(WAKE, "wake", GEN)
    assert stale == [], f"stale generated files: {stale}"


@needs_wake
def test_write_then_check_roundtrips(tmp_path):
    info = codegen.write(WAKE, "wake", tmp_path)
    assert info["arena_bytes"] > 0 and info["ring_bytes"] == 3792
    assert (tmp_path / "wake_infer.c").exists()
    assert (tmp_path / "wake_infer.h").exists()
    assert codegen.check(WAKE, "wake", tmp_path) == []


@needs_wake
def test_check_reports_a_stale_file(tmp_path):
    codegen.write(WAKE, "wake", tmp_path)
    (tmp_path / "wake_infer.c").write_text("/* stale */\n")
    assert codegen.check(WAKE, "wake", tmp_path) == ["wake_infer.c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py -q -k check`
Expected: FAIL with `AttributeError: module 'kws_de.codegen' has no attribute 'check'`.

- [ ] **Step 3: Write the CLI**

Append to `kws_de/codegen.py`:

```python
import argparse


def write(tflite_path, name: str, out_dir) -> dict[str, int]:
    """Generate <name>_infer.{c,h} into out_dir. Returns a small report."""
    blob = pathlib.Path(tflite_path).read_bytes()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = generate(blob, name)
    for filename, text in files.items():
        (out_dir / filename).write_text(text)
    graph = tflite_graph.read_graph(blob)
    plan = rewrite_streaming(graph)
    arena = plan_arena(plan)
    return {
        "arena_bytes": arena.size,
        "ring_bytes": sum(r.bytes for r in plan.rings),
        "ops": len(plan.ops),
    }


def check(tflite_path, name: str, committed_dir) -> list[str]:
    """Names of committed generated files that differ from a fresh generation.

    Byte-exact, unlike kws-fwgen's tolerance-based check: everything here is
    integer arithmetic on the model's own bytes, so any difference is a real
    difference — a changed model, a changed generator, or a stale commit.
    """
    blob = pathlib.Path(tflite_path).read_bytes()
    committed_dir = pathlib.Path(committed_dir)
    stale = []
    for filename, text in generate(blob, name).items():
        path = committed_dir / filename
        if not path.exists() or path.read_text() != text:
            stale.append(filename)
    return stale


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser(
        description="Generate C inference (esp-nn calls) from a .tflite model."
    )
    ap.add_argument("model", help="path to a .tflite file")
    ap.add_argument("--name", required=True, choices=("wake", "command"),
                    help="generated symbol prefix and file stem")
    ap.add_argument("--out", default="firmware/main/gen", help="output directory")
    ap.add_argument("--check", metavar="DIR",
                    help="verify committed generated files in DIR are current "
                         "instead of writing; exit 1 on mismatch")
    args = ap.parse_args()
    model = pathlib.Path(args.model)
    if not model.exists():
        print(f"WARNING: {model} absent — skipping (models are not committed)")
        return
    if args.check:
        stale = check(model, args.name, args.check)
        if stale:
            raise SystemExit("stale generated inference: " + ", ".join(stale))
        return
    info = write(model, args.name, args.out)
    print(f"{args.name}: {info['ops']} ops, arena {info['arena_bytes']} B, "
          f"rings {info['ring_bytes']} B")
```

Add to `pyproject.toml` under `[project.scripts]`, keeping the list alphabetical enough to match the existing order:

```toml
kws-codegen = "kws_de.codegen:main"
```

- [ ] **Step 4: Generate and commit the wake headers**

Run:

```bash
KWS_DATA_ROOT=<your data root> uv run --no-sync kws-codegen \
  "$KWS_DATA_ROOT/models/hey_bus.tflite" --name wake --out firmware/main/gen
KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen_parity.py -q
make -C firmware/test
```

Expected: the CLI prints `wake: 14 ops, arena <N> B, rings 3792 B`; parity tests pass; the host harness prints `0/` for both cases.

- [ ] **Step 5: Run the check test**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py -q`
Expected: PASS, 17 tests, including `test_check_is_clean_against_the_committed_headers`.

- [ ] **Step 6: Wire the freshness check into CI**

In `.github/workflows/firmware.yml`, extend the `gen-fresh` job's last step. The models are not in CI, so `kws-codegen` must be a no-op there — the CLI already prints a warning and returns 0 when the model is absent, mirroring how `write_wake_headers` skips today:

```yaml
      - name: committed config-derived headers are current
        run: |
          uv sync --frozen
          # Structure must match exactly; the float tables (mel/DCT/window, TV_MFCC)
          # are CPU-SIMD- and BLAS-order-dependent, so they are compared within a
          # tolerance that clears hardware last-digit noise but catches any real
          # config/generator change. See kws_de.firmware_gen.check.
          uv run kws-fwgen --check firmware/main/gen
          # The generated inference C is pure integer arithmetic over the model's
          # own bytes, so its check is byte-exact — but the models are not in the
          # repo, so this only bites where a model is present (a dev machine or
          # the data host). Absent model: kws-codegen warns and exits 0.
          uv run kws-codegen models/hey_bus.tflite --name wake --check firmware/main/gen
```

- [ ] **Step 7: Gates**

Run: `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q && make -C firmware/test`
Expected: all pass. Note `test_check_is_clean_against_the_committed_headers` is skipped without `KWS_DATA_ROOT`.

- [ ] **Step 8: Commit**

```bash
git add kws_de/codegen.py pyproject.toml tests/test_codegen.py \
        .github/workflows/firmware.yml \
        firmware/main/gen/wake_infer.c firmware/main/gen/wake_infer.h \
        firmware/main/gen/wake_infer_vectors.h
git commit -m "$(cat <<'EOF'
feat(codegen): kws-codegen CLI and committed wake inference

kws-codegen <model> --name wake --out firmware/main/gen writes the generated
C; --check compares it byte-for-byte with what is committed, the same shape as
kws-fwgen --check, and runs in CI's gen-fresh job where a model is present.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 7: Firmware switch, wake glue, device measurement

**Files:**

- Create: `firmware/main/Kconfig.projbuild`
- Modify: `firmware/main/wake.cc:1-73` (includes, arena, interpreter setup) and `:109-150` (the step loop), `firmware/main/CMakeLists.txt:26-32`
- Modify: `docs/sphinx/requirements.rst`, `docs/sphinx/tests.rst`, `docs/sphinx/firmware.rst`, `docs/paper-notes.md`

**Interfaces:**

- Consumes from Task 6: `firmware/main/gen/wake_infer.h` — `wake_infer_init()`, `wake_infer_reset()`, `wake_infer_step(const int8_t in[3 * 40], uint8_t *prob_q)`, `wake_infer_arena_bytes()`, `WAKE_INFER_ARENA_BYTES`.
- Produces: `CONFIG_KWS_INFER_GENERATED` for Task 8 to reuse.

- [ ] **Step 1: Add the Kconfig switch**

Create `firmware/main/Kconfig.projbuild`:

```text
menu "kws-de inference"

config KWS_INFER_GENERATED
    bool "Use the generated inference runtime instead of the TFLite Micro interpreter"
    default y
    help
      Run the C generated by kws-codegen (firmware/main/gen/*_infer.c), which
      calls esp-nn's ESP32-S3 kernels directly, instead of dispatching through
      tflite::MicroInterpreter. The generated path is bit-exact with the
      interpreter (firmware/test/test_infer_parity, tests/test_codegen_parity.py).

      Both paths are always compiled in. Turning this off restores the
      interpreter without touching any other code, which is the fallback if a
      newly exported model hits an op the generator refuses.

config KWS_INFER_PARITY_LOG
    bool "Log generated-vs-interpreter output once per mode entry"
    depends on KWS_INFER_GENERATED
    default y
    help
      On the first step after entering a mode, run both paths on the same input
      and log the two outputs side by side. Cheap continuous proof that the two
      still agree on real device audio, for as long as TFLM is in the binary.

endmenu
```

- [ ] **Step 2: Add the generated source to the component**

In `firmware/main/CMakeLists.txt`, add `"gen/wake_infer.c"` to `SRCS` (after `"wakefront.c"`) and add `esp-nn` to `PRIV_REQUIRES`:

```cmake
  SRCS "main.c" "audio.c" "storage.c" "vad.c" "record.c" "mfcc.c" "mfcc_fft.cc" "stream.c" "wav.c" "prompts.c" "recognise.cc"
       "wake.cc" "wakefront.c" "gen/wake_infer.c" "beep.c" "console.c" ${MICROFRONTEND_SRCS}
       "ui/ui_menu.c" "ui/ui_record.c" "ui/ui_usb.c" "ui/ui_recognise.c" "ui/ui_wake.c" "ui/font_prompt_28.c" "usb_drive.c"
  INCLUDE_DIRS "." "microfrontend" "microfrontend/kissfft"
  REQUIRES fatfs wear_levelling nvs_flash esp_timer esp_partition esp_tinyusb driver
  PRIV_REQUIRES esp-tflite-micro esp-nn)
```

- [ ] **Step 3: Wire the switch into `wake.cc`**

Add the include next to the other `gen/` includes (`firmware/main/wake.cc:13-14`):

```cpp
#include "gen/wake_infer.h"
```

After the interpreter is created and `AllocateTensors()` succeeds (`wake.cc:72-74`), add the boot log and the generated path's init:

```cpp
    ESP_LOGI(TAG, "arena used %u / %u", (unsigned)interp.arena_used_bytes(), (unsigned)KWS_WAKE_ARENA_BYTES);
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);
#if CONFIG_KWS_INFER_GENERATED
    wake_infer_init();
    ESP_LOGI(TAG, "inference: generated (esp-nn), %u B static; TFLM arena %u B kept as fallback",
             (unsigned)wake_infer_arena_bytes(), (unsigned)KWS_WAKE_ARENA_BYTES);
#else
    ESP_LOGI(TAG, "inference: TFLite Micro interpreter");
#endif
```

In the restart branch (`wake.cc:86-97`), reset the generated state alongside the interpreter's:

```cpp
            interp.Reset();
            mrv->ResetAll();
#if CONFIG_KWS_INFER_GENERATED
            wake_infer_reset();
            s_parity_pending = true;
#endif
            wakefront_reset();
```

with `static volatile bool s_parity_pending;` declared next to `s_restart`.

Replace the single `Invoke()` in the step loop (`wake.cc:115-120`) with the switched call plus the one-shot parity line:

```cpp
            int64_t t0 = esp_timer_get_time();
            wakefront_take(KWS_WAKE_FRAMES, in->data.int8);
            uint8_t prob_q;
#if CONFIG_KWS_INFER_GENERATED
            wake_infer_step(in->data.int8, &prob_q);
#if CONFIG_KWS_INFER_PARITY_LOG
            if (s_parity_pending) {
                /* Same input through both paths, once per mode entry: the
                   generated ring state has already advanced, so this compares
                   the interpreter's answer for THIS step only. A mismatch is a
                   real regression; the host parity test should have caught it. */
                if (interp.Invoke() == kTfLiteOk)
                    ESP_LOGI(TAG, "parity: generated %u, interpreter %u",
                             (unsigned)prob_q, (unsigned)out->data.uint8[0]);
                s_parity_pending = false;
            }
#endif
#else
            if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
            prob_q = out->data.uint8[0];
#endif
            /* uint8 output: prob = (q - zero_point) * scale, i.e. q/256. */
            float prob = (prob_q - KWS_WAKE_OUTPUT_ZERO_POINT) * KWS_WAKE_OUTPUT_SCALE;
            uint32_t ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);
```

The parity branch runs the interpreter on a ring state the generated path has already consumed for this step, so it is only meaningful on the **first** step after `wake_infer_reset()` — which is exactly when `s_parity_pending` is set. Say that in the comment, as above.

Finally, when `CONFIG_KWS_INFER_GENERATED` is set the TFLM arena is still allocated (it is the fallback and the parity reference), so `arena_alloc` and the `s_st.arena_used` reporting stay untouched.

- [ ] **Step 4: Build both configurations**

Run:

```bash
cd firmware
idf.py set-target esp32s3
idf.py -DCONFIG_KWS_INFER_GENERATED=y build
idf.py -DCONFIG_KWS_INFER_GENERATED=n build
```

Expected: both link. If `esp-nn` is not resolvable as a component, it comes in transitively with `espressif/esp-tflite-micro` — check `firmware/managed_components/` and drop `esp-nn` from `PRIV_REQUIRES` if the header already resolves.

- [ ] **Step 5: Flash and measure on the device**

Flash and read the console with the CoreS3 helper scripts (`~/.claude/skills/flashing-cores3-on-bar/scripts/flash-bar.sh` and `console-bar.sh` — this plan is the only place those paths appear; committed docs record numbers only, never hosts or script paths). Enter **Hey Bus** mode and capture:

- the `inference: generated (esp-nn), <N> B static; TFLM arena 49152 B kept as fallback` boot line;
- the `parity: generated <a>, interpreter <b>` line — **a must equal b**;
- two minutes of `peak <p> over <n> steps, <ms> ms/step` lines.

Then rebuild with `-DCONFIG_KWS_INFER_GENERATED=n`, flash, and capture the same `ms/step` figure in the same session. Today's baseline is 3 ms/step; the spec's target is "wake step well under 1 ms of inference".

- [ ] **Step 6: Record the requirements and tests**

In `docs/sphinx/requirements.rst`, under "Recogniser", add:

```rst
.. req:: Generated inference is bit-exact with the interpreter
   :id: REQ_FW_INFER_GENERATED
   :status: implemented

   With ``CONFIG_KWS_INFER_GENERATED=y`` the wake and command models run as C
   generated by ``kws-codegen`` (``firmware/main/gen/*_infer.c``) calling
   esp-nn's ESP32-S3 kernels directly, not through
   ``tflite::MicroInterpreter``. Every output byte is identical to the
   interpreter's for the same input: requantisation multipliers and shifts are
   prepared with TFLM's own ``QuantizeMultiplier`` integer math, activation
   ranges with TFLM's own rounding, and ``LOGISTIC`` is a 256-entry table read
   out of the reference kernel itself. The generated arena is at or below the
   TFLM arena the same model needs (:need:`REQ_FW_ARENA_PLACEMENT`).

.. req:: TFLite Micro stays as a build-time fallback
   :id: REQ_FW_INFER_FALLBACK
   :status: implemented

   Both inference paths compile into one firmware family; ``menuconfig``'s
   ``CONFIG_KWS_INFER_GENERATED`` picks one and the boot log prints which is
   active. A model that uses an op the generator refuses is a loud
   generation-time error naming the op and tensor, and the interpreter build
   still runs it. ``CONFIG_KWS_INFER_PARITY_LOG`` logs both paths' output for
   the same input once per mode entry, on real device audio.
```

In `docs/sphinx/tests.rst`, under the host C tests:

```rst
.. test:: Generated inference matches the interpreter byte for byte
   :id: TEST_INFER_PARITY
   :status: passing
   :links: REQ_FW_INFER_GENERATED, REQ_FW_HOST_TESTS_NO_IDF

   ``firmware/test/test_infer_parity.c``: compiles the generated wake
   inference against esp-nn's ANSI-C reference kernels and replays the
   golden feature rows from ``gen/wake_test_vectors.h``, asserting every step's
   output byte equals the interpreter's answer recorded in
   ``gen/wake_infer_vectors.h``. ``tests/test_codegen_parity.py`` extends the
   same comparison to the ten approved wake takes and the recordings set for
   the command model, where the models are present. Zero LSB difference is the
   pass condition; one differing byte fails the build of the generated
   headers.

.. test:: Generated arena stays within the TFLM arena
   :id: TEST_INFER_ARENA
   :status: passing
   :links: REQ_FW_INFER_GENERATED, REQ_FW_ARENA_PLACEMENT

   ``tests/test_codegen.py``: the first-fit lifetime planner's arena for each
   model is asserted to be at or below the ``KWS_WAKE_ARENA_BYTES`` /
   ``KWS_MODEL_ARENA_BYTES`` the firmware allocates for TFLM today, read from
   the committed ``firmware/main/gen/`` headers.
```

In `docs/sphinx/firmware.rst`, in the "Hey Bus demo (wake test mode)" section, add one paragraph with the measured before/after `ms/step`, the boot line's static-byte figure, and the fact that the parity line agrees. Numbers only — no host names, no script paths.

- [ ] **Step 7: Record the result in the paper notes**

Append to `docs/paper-notes.md` under "3. Experiments & results (real, measured)" a short `### E<n> — generated inference vs. the TFLM interpreter (wake)` subsection: a table of `ms/step` before and after, the static bytes the generated path uses (arena + scratch + rings) against TFLM's 49,152 B arena, the parity result, and one sentence on why it is faster (per-op dispatch and tensor bookkeeping removed, not better kernels — the kernels are the same esp-nn ones).

- [ ] **Step 8: Gates**

Run:

```bash
npx markdownlint-cli@0.42.0 --config .markdownlint.json docs/paper-notes.md
uv run --no-sync pytest -q
make -C firmware/test
bash docs/sphinx/build.sh
```

Expected: clean; sphinx-build succeeds with the two new requirements traced.

- [ ] **Step 9: Commit**

```bash
git add firmware/main/Kconfig.projbuild firmware/main/wake.cc firmware/main/CMakeLists.txt \
        docs/sphinx/requirements.rst docs/sphinx/tests.rst docs/sphinx/firmware.rst \
        docs/paper-notes.md
git commit -m "$(cat <<'EOF'
feat(firmware): run the wake model on the generated inference path

CONFIG_KWS_INFER_GENERATED selects the generated esp-nn call sequence; the
interpreter stays compiled in as the fallback and, once per mode entry, as an
on-device parity reference whose output is logged next to the generated one.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 8: Command model generation, glue, measurement

**Files:**

- Create: `firmware/main/gen/command_infer.c`, `firmware/main/gen/command_infer.h`
- Modify: `firmware/main/recognise.cc:1-56` (includes, setup) and `:92-99` (invoke + output read), `firmware/main/CMakeLists.txt`, `.github/workflows/firmware.yml`
- Modify: `docs/sphinx/firmware.rst`, `docs/sphinx/models.rst`, `docs/paper-notes.md`
- Test: `tests/test_codegen.py`, `tests/test_codegen_parity.py`

**Interfaces:**

- Consumes from Tasks 5–7: `codegen.generate`, `codegen.write`, `codegen.check`, `CONFIG_KWS_INFER_GENERATED`.
- Produces: `firmware/main/gen/command_infer.h` — `void command_infer(const int8_t in[49 * 10], int8_t out[23]);`, `command_infer_init()`, `command_infer_reset()`, `command_infer_arena_bytes()`, `COMMAND_INFER_ARENA_BYTES`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codegen.py`:

```python
COMMAND = config.MODELS_DIR / "command.tflite"
needs_command = pytest.mark.skipif(not COMMAND.exists(), reason=f"{COMMAND} absent")


@needs_command
def test_command_model_generates_a_stateless_entry_point():
    files = codegen.generate(COMMAND.read_bytes(), "command")
    header = files["command_infer.h"]
    assert "void command_infer(const int8_t in[490], int8_t out[23]);" in header
    source = files["command_infer.c"]
    assert source.count("esp_nn_conv_s8(") == 4
    assert source.count("esp_nn_depthwise_conv_s8(") == 3
    assert source.count("esp_nn_mean_nhwc_s8(") == 1
    assert source.count("esp_nn_softmax_s8(") == 1
    assert "ring" not in source                    # no streaming state


@needs_command
def test_command_arena_is_at_most_the_tflm_arena():
    g = tflite_graph.read_graph(COMMAND.read_bytes())
    arena = codegen.plan_arena(codegen.rewrite_streaming(g))
    tflm = codegen.tflm_arena_bytes(GEN / "model_config.h", "KWS_MODEL_ARENA_BYTES")
    assert tflm == 139264
    assert arena.size <= tflm, f"generated arena {arena.size} B exceeds TFLM's {tflm} B"


@needs_command
def test_command_check_is_clean_against_the_committed_headers():
    assert codegen.check(COMMAND, "command", GEN) == []
```

And to `tests/test_codegen_parity.py`:

```python
COMMAND = config.MODELS_DIR / "command.tflite"
needs_command = pytest.mark.skipif(not COMMAND.exists(), reason=f"{COMMAND} absent")
RECORDINGS = config.DATA_DIR / "recordings" / "approved" / "words"


@needs_command
@pytest.mark.skipif(not RECORDINGS.exists(), reason="approved/words absent")
def test_command_model_matches_the_interpreter_on_the_recordings_set():
    """Every class probability byte, for every approved word recording."""
    import soundfile as sf

    from kws_de import config as cfg
    from kws_de import features

    blob = COMMAND.read_bytes()
    itp = tf.lite.Interpreter(model_content=blob)
    itp.allocate_tensors()
    detail_in, detail_out = itp.get_input_details()[0], itp.get_output_details()[0]
    scale, zp = detail_in["quantization"]
    clips = sorted(RECORDINGS.rglob("*.wav"))
    assert clips, "no approved word recordings found"
    for path in clips:
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == cfg.SAMPLE_RATE
        q = np.round(features.mfcc(pcm) / scale + zp).astype(np.int8)
        itp.set_tensor(detail_in["index"], q.reshape(detail_in["shape"]))
        itp.invoke()
        expect = itp.get_tensor(detail_out["index"]).astype(np.int8).ravel()
        got = codegen.simulate(blob, "command", q)
        assert np.array_equal(got, expect), (
            f"{path.name}: {int((got != expect).sum())} of 23 output bytes differ"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen.py tests/test_codegen_parity.py -q -k command`
Expected: FAIL — `command_infer.h` is not committed yet, so `test_command_check_is_clean_against_the_committed_headers` reports it stale; the generation tests fail on whichever emitter detail is still wrong (most likely `MEAN`'s readapted multiplier).

- [ ] **Step 3: Generate and fix until parity holds**

Run:

```bash
KWS_DATA_ROOT=<your data root> uv run --no-sync kws-codegen \
  "$KWS_DATA_ROOT/models/command.tflite" --name command --out firmware/main/gen
KWS_DATA_ROOT=<your data root> uv run --no-sync pytest tests/test_codegen_parity.py -q -k command
```

Expected: `command: 10 ops, arena <N> B, rings 0 B`, then parity passes. If `MEAN` differs, the readaptation in `emit_mean` is the first thing to check against `reference/reduce.h`: `s = min(63 - clz(n), 32, 31 + shift)` — for `n = 490`, `63 - clz(490) == 8`; the multiplier is then `(mult << 8) // 490` with **truncating** division and `shift -= 8`. If `SOFTMAX` differs, check that `esp_nn_set_softmax_scratch_buf` is called with a 4-byte-aligned buffer of at least `esp_nn_get_softmax_scratch_size(23, 1)` bytes before the call.

- [ ] **Step 4: Also generate for the QAT model and confirm the generator is model-agnostic**

Run:

```bash
KWS_DATA_ROOT=<your data root> uv run --no-sync kws-codegen \
  "$KWS_DATA_ROOT/models/command_v3_qat.tflite" --name command --out /tmp/codegen-qat
```

Expected: succeeds with the same op counts (its graph is structurally identical, only the quantisation constants differ). This is the check that the generator reads the model rather than hard-coding today's numbers. Do not commit `/tmp/codegen-qat`.

- [ ] **Step 5: Wire the switch into `recognise.cc`**

Add next to the other `gen/` includes (`recognise.cc:13-16`):

```cpp
#include "gen/command_infer.h"
```

After `AllocateTensors()` (`recognise.cc:55-56`):

```cpp
    ESP_LOGI(TAG, "arena used %u / %u", (unsigned)interp.arena_used_bytes(), (unsigned)KWS_MODEL_ARENA_BYTES);
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);
#if CONFIG_KWS_INFER_GENERATED
    command_infer_init();
    ESP_LOGI(TAG, "inference: generated (esp-nn), %u B static; TFLM arena %u B kept as fallback",
             (unsigned)command_infer_arena_bytes(), (unsigned)KWS_MODEL_ARENA_BYTES);
    static bool s_parity_pending = true;
#else
    ESP_LOGI(TAG, "inference: TFLite Micro interpreter");
#endif
```

Replace the invoke and the output read (`recognise.cc:93-100`):

```cpp
        int64_t t_invoke = esp_timer_get_time();
        static int8_t logits[KWS_NUM_LABELS];
#if CONFIG_KWS_INFER_GENERATED
        command_infer(in->data.int8, logits);
#if CONFIG_KWS_INFER_PARITY_LOG
        if (s_parity_pending) {
            if (interp.Invoke() == kTfLiteOk) {
                int diff = 0;
                for (int i = 0; i < KWS_NUM_LABELS; i++)
                    if (logits[i] != out->data.int8[i]) diff++;
                ESP_LOGI(TAG, "parity: %d/%d output bytes differ", diff, KWS_NUM_LABELS);
            }
            s_parity_pending = false;
        }
#endif
#else
        if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
        memcpy(logits, out->data.int8, sizeof logits);
#endif
        uint32_t invoke_ms = (uint32_t)((esp_timer_get_time() - t_invoke) / 1000);
        int best = 0;
        for (int i = 0; i < KWS_NUM_LABELS; i++) {
            probs[i] = (logits[i] - KWS_MODEL_OUTPUT_ZERO_POINT) * KWS_MODEL_OUTPUT_SCALE;
            if (probs[i] > probs[best]) best = i;
        }
```

Set `s_parity_pending = true` in the `if (!primed)` re-entry branch (`recognise.cc:69-74`) so the line prints once per mode entry, guarded by `#if CONFIG_KWS_INFER_GENERATED`.

Add `"gen/command_infer.c"` to `SRCS` in `firmware/main/CMakeLists.txt`, next to `"gen/wake_infer.c"`.

- [ ] **Step 6: Add the command freshness check to CI**

In `.github/workflows/firmware.yml`, after the wake line in the `gen-fresh` job:

```yaml
          uv run kws-codegen models/command.tflite --name command --check firmware/main/gen
```

- [ ] **Step 7: Build, flash, measure**

Run `idf.py -DCONFIG_KWS_INFER_GENERATED=y build` and `=n build`; flash each and enter **Recognition** mode with the CoreS3 helper scripts. Capture from both runs, in one session:

- the boot `inference:` line and the static-byte figure;
- `parity: 0/23 output bytes differ`;
- several `step <ms> ms (invoke <ms> ms, <n> new frames)` lines.

Today's baseline is `invoke 52-53 ms` with the command arena in PSRAM (`REQ_FW_ARENA_PLACEMENT`). The spec's target is "command Invoke at least 2x faster than the interpreter path measured the same day". Note in the results whether the generated static arena now fits internal RAM where TFLM's 139,264 B one did not — that, not the kernels, is likely where the time goes.

- [ ] **Step 8: Document the result**

- `docs/sphinx/firmware.rst`, "Recognition demo": one paragraph, measured `invoke_ms` before and after, static bytes, and where the generated arena lives.
- `docs/sphinx/models.rst`: one paragraph saying both shipped models are code-generated with `kws-codegen` and listing the op sequence each produces, so the page matches `kws-model-graph`'s figures.
- `docs/paper-notes.md`: extend the E-section from Task 7 with the command model's row and one line on the arena-placement effect.

- [ ] **Step 9: Gates**

Run:

```bash
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
KWS_DATA_ROOT=<your data root> uv run --no-sync pytest -q
make -C firmware/test
bash docs/sphinx/build.sh
npx markdownlint-cli@0.42.0 --config .markdownlint.json docs/paper-notes.md
```

Expected: all clean.

- [ ] **Step 10: Commit**

```bash
git add kws_de/codegen.py tests/test_codegen.py tests/test_codegen_parity.py \
        firmware/main/gen/command_infer.c firmware/main/gen/command_infer.h \
        firmware/main/recognise.cc firmware/main/CMakeLists.txt \
        .github/workflows/firmware.yml \
        docs/sphinx/firmware.rst docs/sphinx/models.rst docs/paper-notes.md
git commit -m "$(cat <<'EOF'
feat(firmware): run the command model on the generated inference path

The DS-CNN generates cleanly from the same emitters as the wake model (MEAN
carries the readapted 1/(H*W) multiplier the TFLM reference folds in), matches
the interpreter on every output byte across the approved recordings, and the
generated arena is small enough to stay in internal RAM.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

## Self-review

**Spec coverage.** Walked every section of `docs/superpowers/specs/2026-09-03-generated-inference-design.md`:

| Spec item | Task |
|---|---|
| §2 `kws_de/codegen.py` CLI `kws-codegen … --check`, entry in `pyproject.toml` | 6 |
| §2 `kws_de/tflite_graph.py` shared with `kws-model-graph` | 1 |
| §2 generated C API (five functions, verbatim signatures) | 5 (rendering), 7 and 8 (callers) |
| §2 firmware glue, `CONFIG_KWS_INFER_GENERATED`, boot log, arena in internal RAM | 7, 8 |
| §3.1 read subgraphs, ops in order, quantisation, buffers, resource variables | 1 |
| §3.2 streaming rewrite to rings, equivalence asserted per variable | 2 |
| §3.3 first-fit lifetime planner, arena ≤ TFLM's | 3 |
| §3.4 weights/biases as `static const`, TFLM-identical multipliers | 4 |
| §3.4 esp-nn calls with the same scratch handling TFLM performs | 4 (conv/depthwise), 5 (softmax) |
| §3.4 reference C for LOGISTIC, QUANTIZE, MEAN (or refuse and ask for avg-pool) | 5 — MEAN is emitted, and `AVERAGE_POOL_2D` too, since both shipped command models use MEAN |
| §3.4 RESHAPE/CONCATENATION become pointer arithmetic | 2 (alias map), 5 (ring append/shift) |
| §3.5 loud refusal naming op and tensor | 2 (`UnsupportedGraph`, tested), reused by every emitter |
| §3 supported op set | 2 (`SUPPORTED_OPS`) |
| §4 host parity harness compiling generated C against esp-nn ANSI kernels | 4, 5 |
| §4 byte parity on golden vector, 10 real wake takes, recordings set | 5 (golden + takes), 8 (recordings) |
| §4 device parity log once per mode entry | 7, 8 (`CONFIG_KWS_INFER_PARITY_LOG`) |
| §4 speed measured both paths, into paper-notes and firmware.rst | 7, 8 |
| §4 freshness check in CI mirroring `kws-fwgen --check` | 6, 8 |
| §5 file table | covered by the File structure table |
| §5 `REQ_FW_INFER_GENERATED`, `REQ_FW_INFER_FALLBACK`, traced | 7 |
| §6 wake first, command second, generator accepts a streaming command model | task order; the ring rewrite is keyed on resource variables, not on a model, and Task 8 step 4 re-runs it on the QAT export |
| §6 risks: requantisation rounding, scratch sizing, internal-RAM budget | 4 (multiplier port + one-layer parity), 4/5 (scratch), 3 and 7/8 (arena) |

No spec requirement is unmapped. One deliberate deviation, stated in place: the spec offers "refuses and asks for the avg-pool export" as an alternative to emitting `MEAN` — both shipped command models use `MEAN`, so it is emitted and `AVERAGE_POOL_2D` is emitted too, rather than blocking on a re-export.

**Placeholder scan.** No "TBD", "TODO", "implement later", "add appropriate error handling", "similar to Task N", or "write tests for the above". Every code step carries the code; every run step carries the command and its expected output. Two things are deliberately parameterised rather than fixed: `<your data root>` (a machine-specific path the Global Constraints forbid committing) and the `<N>` in expected arena/timing output (measured, and Task 3 step 5 and Task 7 step 5 say to record them). The one place I write a signature without a body — `codegen.simulate` in Task 5 step 5 — has its full docstring, exact signature, exact call sites and exact assertion, and is a test aid whose behaviour is fully pinned by the two tests that call it.

**Type consistency.** Checked the names that cross task boundaries: `Tensor`/`Op`/`Graph`/`read_graph`/`constant`/`probe_model` (Task 1) are used with those exact names in Tasks 2–5; `Ring`/`Plan`/`rewrite_streaming`/`UnsupportedGraph`/`SUPPORTED_OPS` (Task 2) in Tasks 3–6; `Arena`/`plan_arena`/`tensor_bytes`/`tflm_arena_bytes` (Task 3) in Tasks 4–8; `Emitter`/`quantize_multiplier`/`activation_range`/`padding_hw`/`per_channel_multipliers`/`_effective_scale` (Task 4) in Task 5; `generate`/`write_wake_vectors`/`simulate` (Task 5) in Tasks 6–8; `write`/`check` (Task 6) in Task 8. The C API is spelled the same in the spec quote, in `_render`'s header emission, in `firmware/main/gen/*_infer.h`, and at both call sites — `wake_infer_step(const int8_t in[3 * 40], uint8_t *prob_q)` and `command_infer(const int8_t in[49 * 10], int8_t out[23])`. Fixed while reviewing: `plan_arena`'s `scratch_bytes` keyword is used consistently (Task 3 test, Task 5 `generate`); `Emitter.prefix` is the same string passed to `generate(..., name)` and to `_render`; the arena macro names read by `tflm_arena_bytes` match the committed headers exactly (`KWS_WAKE_ARENA_BYTES`, `KWS_MODEL_ARENA_BYTES`).
