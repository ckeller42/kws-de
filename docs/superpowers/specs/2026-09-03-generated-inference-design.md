# Generated inference runtime — design

Date: 2026-09-03. Status: approved in chat, to be implemented after perf wave 2.

## 1. Goal

Replace the TFLite-Micro interpreter on the device with C generated from our two `.tflite`
graphs, calling esp-nn's ESP32-S3 kernels directly, so inference time is spent in kernels rather
than in per-op dispatch and tensor bookkeeping. The wake model does 24,736 MACs in 49 ops and
takes 3 ms per 30 ms step; the command DS-CNN does 2.07 M MACs and its Invoke share is being
measured by wave 2. Both are overhead- and memory-bound, not arithmetic-bound.

Acceptance (decided): **bit-exact with the interpreter and measurably faster, TFLite-Micro kept
as a build-time fallback.**

Non-goals: new kernels or assembly, training changes, removing TFLM, supporting arbitrary graphs.

## 2. Architecture

```
models/hey_bus.tflite ──┐                     firmware/main/gen/wake_infer.{c,h}
models/command*.tflite ─┼─ kws-codegen ─────▶ firmware/main/gen/command_infer.{c,h}
                        │  (kws_de/codegen.py)  const int8 weights, static arena, ring buffers,
                        └─ tflite graph reader   fixed call sequence into esp-nn
```

- `kws_de/codegen.py` — CLI `kws-codegen <model.tflite> --name <wake|command> --out <dir>`
  (and `--check <dir>` for freshness, like `kws-fwgen --check`). Entry in `pyproject.toml`.
- `kws_de/tflite_graph.py` — the graph reader already used by `kws-model-graph`, extended with
  quantisation parameters, buffers and resource-variable bookkeeping, shared by both tools.
- Generated API (C, ESP-IDF component `main`):

  ```c
  void wake_infer_init(void);                       /* zero rings, precompute nothing else */
  void wake_infer_reset(void);                      /* on mode entry: clear streaming state */
  void wake_infer_step(const int8_t in[3 * 40], uint8_t *prob_q);   /* one 30 ms step */
  void command_infer(const int8_t in[49 * 10], int8_t out[23]);
  size_t wake_infer_arena_bytes(void);              /* for the boot log */
  ```

- Firmware glue: `wake.cc` and `recognise.cc` call the generated functions when
  `CONFIG_KWS_INFER_GENERATED=y` (Kconfig in `firmware/main/Kconfig.projbuild`, default y once
  parity is proven), else the existing `MicroInterpreter` path. Both paths compile in one
  firmware family; the boot log prints which is active. Arena for the generated path is a static
  `int8_t` array placed in internal RAM (`.dram0` via a normal static, or `heap_caps_aligned_alloc`
  at init if it must be conditional).

## 3. What the generator does

1. Reads the flatbuffer: subgraphs, ops in execution order, tensors (shape, dtype, quantisation
   scale/zero point, per-channel scales for weights), buffers, resource variables
   (`VAR_HANDLE`/`READ_VARIABLE`/`ASSIGN_VARIABLE`, initialised by `CALL_ONCE`).
2. Rewrites the streaming pattern `READ_VARIABLE → CONCATENATION(new row) → op … STRIDED_SLICE →
   ASSIGN_VARIABLE` into one ring buffer per variable: a static `int8_t ring_k[rows][C]`; the
   step writes the new row in, runs the op over the full window, then shifts (`memmove`) so the
   oldest row drops. Equivalent to the graph's slice/assign by construction; the equivalence is
   asserted per variable by the parity test.
3. Plans memory: greedy first-fit over tensor lifetimes (op index of last use) into one arena;
   emits the arena size and per-tensor offsets as constants. Reports bytes; must be ≤ the TFLM
   arena the same model needed (`arena_used_bytes`).
4. Emits one C file per model:
   - weights and biases as `static const int8_t/int32_t` arrays, per-channel multipliers and
     shifts precomputed exactly as TFLM's `QuantizeMultiplier` does (same integer math, same
     rounding), so requantisation is identical;
   - the op sequence as direct calls: `esp_nn_conv_s8`, `esp_nn_depthwise_conv_s8`,
     `esp_nn_fully_connected_s8`, `esp_nn_avg_pool_s8`, `esp_nn_softmax_s8` (with the same
     `esp_nn_get_*_scratch_size` handling TFLM's esp kernels perform), reference C for
     `LOGISTIC` (int8 → lookup or the TFLM reference), `QUANTIZE`, `MEAN` if a model still has
     it (or the generator refuses and asks for the avg-pool export);
   - `RESHAPE` and `CONCATENATION` of contiguous buffers become pointer arithmetic.
5. Refuses loudly on anything else: an unsupported op, dynamic shapes, non-int8 tensors, more
   than one subgraph beyond the init subgraph — error names the op and tensor. Never silent.

Supported op set (the union of both graphs today): CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED,
AVERAGE_POOL_2D, MEAN, SOFTMAX, LOGISTIC, QUANTIZE, RESHAPE, CONCATENATION, STRIDED_SLICE,
VAR_HANDLE, READ_VARIABLE, ASSIGN_VARIABLE, CALL_ONCE.

## 4. Parity and measurement

- **Host parity harness**: `firmware/test/` gains a target that compiles the generated C against
  esp-nn's ANSI-C reference kernels (the esp-nn component builds for the host with
  `CONFIG_NN_ANSI_C`-equivalent defines) and a Python test that runs the interpreter on the same
  inputs and compares **every output tensor byte-for-byte**: the MFCC golden vector, the 10 real
  wake takes (`approved/wake`), and the approved recordings set for the command model. Zero LSB
  difference is the requirement; one failing byte fails the build of the generated headers.
- **Device parity**: the firmware logs, on the first step after boot, the generated path's
  output next to the interpreter's for the same input (both compiled in), once per mode entry —
  a cheap continuous check while TFLM is still in the binary.
- **Speed**: wave 2's kernel timers plus `invoke_ms` are printed for both paths in one session
  on the device (`mode wake`, `mode recognise`): before/after per model, into `docs/paper-notes.md`
  and `docs/sphinx/firmware.rst`. Target: wake step well under 1 ms of inference; command Invoke
  at least 2× faster than the interpreter path measured the same day.
- **Freshness**: `kws-codegen --check firmware/main/gen` in CI, mirroring `kws-fwgen --check`
  (structure exact; the models are not in CI, so the check runs only when the model is present
  locally, like today's wake headers).

## 5. Files

| Path | Change |
|---|---|
| `kws_de/tflite_graph.py` | shared reader (quantisation, buffers, variables) |
| `kws_de/codegen.py` | planner + emitters + CLI |
| `tests/test_codegen.py` | unit tests on an in-test tiny model; parity test on the real models when present |
| `firmware/main/gen/{wake,command}_infer.{c,h}` | generated, committed |
| `firmware/main/wake.cc`, `recognise.cc`, `Kconfig.projbuild`, `CMakeLists.txt` | switch + glue |
| `firmware/test/Makefile`, `firmware/test/test_infer_parity.c` | host parity harness |
| `docs/sphinx/requirements.rst`, `tests.rst`, `firmware.rst`, `models.rst` | `REQ_FW_INFER_GENERATED` (bit-exact), `REQ_FW_INFER_FALLBACK` (TFLM switch), traced |
| `docs/paper-notes.md` | measured before/after |

## 6. Order of work and risks

1. Wake model first (all streaming machinery, clearest win), interpreter kept and compared.
2. Command model second, in whatever shape wave 2 leaves it (avg-pool export), generator must
   also accept the planned streaming command model (same variable pattern as the wake model).
3. Risks: requantisation rounding differences between TFLM's reference path and the esp-nn call
   signature (caught by the byte parity test; the fix is to replicate TFLM's exact parameter
   preparation); esp-nn scratch-buffer sizing (query the same size functions TFLM uses); internal
   RAM budget (arena ≤ TFLM's, rings ≈ 3.8 KB) — checked on the boot log.
