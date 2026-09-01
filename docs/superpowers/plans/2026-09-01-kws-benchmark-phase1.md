# KWS Architecture Benchmark (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible, paper-style benchmark of small-footprint KWS architectures (DS-CNN, BC-ResNet, MatchboxNet, Keyword-Transformer) on the frozen dataset, scored on isolated-word accuracy, end-to-end catalog accuracy, and on-device fit → `docs/benchmark.md`.

**Architecture:** A pluggable `kws_de/architectures/` package (one builder per model, common interface) plus a benchmark runner that trains each on the Phase-0 splits (same seed/epochs/aug), selects on val, and reports on test + on-device budgets. Every model INT8-exportable and budget-gated.

**Tech Stack:** Python 3.11 (uv), TensorFlow/Keras, numpy, pytest, ruff. Consumes Phase-0 `kws_de.dataset.load_split` and the existing `kws_de.{export,budgets,eval,stream,grammar}`.

**Spec:** `docs/superpowers/specs/2026-09-01-kws-research-plan-design.md` (Phase 1, §5).

## Global Constraints

- Python `>=3.11`; `uv`; `ruff` (line-length 100, `E,F,I,UP,B`). Full suite stays green.
- Every architecture builder has the interface `build(input_shape=(N_FRAMES, N_MFCC, 1), n_classes) -> tf.keras.Model`, outputs `n_classes` softmax, and must INT8-export + pass `kws_de.budgets` (≤ 500 KB / ≤ 3 M MACs / INT8 / TFLM ops).
- All models train on the SAME frozen split (Phase-0 `load_split`), same seed, same epochs, same augmentation. **Selection on val, reporting on test** — never tune on test.
- Metrics per architecture: isolated-word test accuracy; end-to-end catalog full-intent (reuse `kws_de.eval` catalog machinery); params; MACs (`kws_de.budgets.estimate_macs`); INT8 size; estimated latency.
- Output committed: `docs/benchmark.md` (human table) + `benchmark.json` (machine). Full training is local; CI runs a tiny-fixture smoke (each arch builds + INT8-exports + passes budgets).
- No data/model binaries committed. Commit trailers as in the repo. Branch → PR → CI + CodeRabbit.

---

### Task 1: Architecture registry + DS-CNN baseline adapter

**Files:**
- Create: `kws_de/architectures/__init__.py`
- Test: `tests/test_architectures.py`

**Interfaces:**
- Produces: `ARCHITECTURES: dict[str, Callable]` mapping name → `build(input_shape, n_classes) -> keras.Model`; `get(name) -> builder`. `"ds_cnn"` wraps the existing `kws_de.model.build_dscnn`.

- [ ] **Step 1: failing test**

```python
# tests/test_architectures.py
from kws_de import config
from kws_de.architectures import ARCHITECTURES, get

def test_registry_has_expected_names():
    assert {"ds_cnn", "bc_resnet", "matchboxnet", "kwt"} <= set(ARCHITECTURES)

def test_ds_cnn_builder_shape():
    m = get("ds_cnn")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
```

- [ ] **Step 2: fail** → `ModuleNotFoundError`. **Step 3: implement** the registry; `ds_cnn` = `lambda shape, n_classes: build_dscnn(num_classes=n_classes)` (input shape is fixed by config). Register the other three names pointing at builders added in Tasks 2–4 (import lazily or define stubs that Tasks 2–4 fill). **Step 4: pass** the ds_cnn part. **Step 5: commit** `feat(bench): architecture registry + ds_cnn adapter`.

---

### Task 2: BC-ResNet

**Files:** Create `kws_de/architectures/bc_resnet.py`; Test: `tests/test_architectures.py` (extend).

**Interfaces:** Produces `build_bc_resnet(input_shape, n_classes) -> keras.Model` (broadcasted-residual blocks: frequency-depthwise conv + temporal-depthwise conv with a broadcasted residual; a small width, ~10–30 k params).

- [ ] **Step 1: failing test**

```python
def test_bc_resnet_builds_and_is_small():
    from kws_de.architectures import get
    from kws_de import config
    m = get("bc_resnet")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 60_000
```

- [ ] **Step 2: fail. Step 3: implement** a compact BC-ResNet (arXiv:2106.04140): stem conv → N broadcasted-residual blocks (each: 3×1 freq-depthwise → BN → temporal 1×3 depthwise on the freq-averaged branch, broadcast-added back → pointwise) → GAP → dense. Keep ops in the TFLM set (Conv2D/DepthwiseConv2D/BN→folds/ReLU/Mean/Dense). **Step 4: pass. Step 5: commit** `feat(bench): BC-ResNet`.

---

### Task 3: MatchboxNet

**Files:** Create `kws_de/architectures/matchboxnet.py`; Test: extend.

**Interfaces:** `build_matchboxnet(input_shape, n_classes) -> keras.Model` — 1-D time-channel-separable conv blocks (treat the 10 MFCC as channels over the 49 time steps). Small (B×R config, ~few×10 k params).

- [ ] failing test `test_matchboxnet_builds` (output shape + params < 150 000); implement MatchboxNet (arXiv:2004.08531): prologue 1-D sep-conv → B blocks of R time-channel-separable sub-blocks with residuals → epilogue → GAP → dense; reshape the (49,10,1) input to (49,10) time×channel. Keep TFLM ops. Commit `feat(bench): MatchboxNet`.

---

### Task 4: Keyword-Transformer (KWT)

**Files:** Create `kws_de/architectures/kwt.py`; Test: extend.

**Interfaces:** `build_kwt(input_shape, n_classes) -> keras.Model` — patch/frame embedding + a few transformer encoder layers (MHSA + MLP) + class token → dense. Small (KWT-1 class, ~few×100 k params; keep < 500 KB INT8).

- [ ] failing test `test_kwt_builds` (output shape + params bound); implement KWT (arXiv:2104.00769): linear projection of each time-frame (or patch) → prepend class token + positional embedding → L transformer blocks → class-token head. Note: attention/softmax must remain TFLM-exportable; if an op is unsupported, record it in the report and mark KWT "reference-only (not device-runnable)" rather than forcing it. Commit `feat(bench): Keyword Transformer`.

---

### Task 5: Benchmark runner + report

**Files:** Create `kws_de/benchmark.py`; Test: `tests/test_benchmark.py`.

**Interfaces:**
- `evaluate_architecture(name, train, val, test, *, epochs, seed) -> dict` — build, train (select on val), INT8-export, and return `{name, isolated_acc, catalog_acc, params, macs, int8_bytes, ops, budget_ok}`. (`catalog_acc` reuses `kws_de.eval` catalog machinery against the trained INT8 model; may be `# pragma: no cover` in the heavy part.)
- `render_table(rows: list[dict]) -> str` — Markdown comparison table (pure, tested).
- `main()` — `kws-benchmark`: run all architectures on `load_split`, write `docs/benchmark.md` + `benchmark.json`.

- [ ] **Step 1: failing test** (pure table render + a tiny-model smoke for one arch)

```python
# tests/test_benchmark.py
from kws_de.benchmark import render_table

def test_render_table_has_all_columns_and_rows():
    rows = [
        {"name": "ds_cnn", "isolated_acc": 0.95, "catalog_acc": 0.69,
         "params": 5351, "macs": 2069984, "int8_bytes": 20216, "budget_ok": True},
        {"name": "bc_resnet", "isolated_acc": 0.96, "catalog_acc": 0.70,
         "params": 12000, "macs": 900000, "int8_bytes": 15000, "budget_ok": True},
    ]
    md = render_table(rows)
    for col in ("Architecture", "Isolated", "Catalog", "Params", "MACs", "INT8", "Budget"):
        assert col in md
    assert "ds_cnn" in md and "bc_resnet" in md
```

- [ ] **Step 2: fail. Step 3: implement** `render_table` (pure) + `evaluate_architecture` + `main`. **Step 4: pass. Step 5: commit** `feat(bench): benchmark runner + Markdown/JSON report`.

---

### Task 6: CI smoke + full run

**Files:** Modify `.github/workflows/ci.yml` (add a job or step: for each architecture, build at `n_classes=len(COMMAND_LABELS)`, INT8-export, assert budgets — no training).

- [ ] **Step 1:** add `tests/test_architectures.py::test_all_architectures_export_int8_and_fit_budget` — loop `ARCHITECTURES`, build, `to_int8_tflite` on random rep data, `check_budgets` (skip/soft-flag KWT if it can't INT8-export, per Task 4). **Step 2:** `uv run pytest -q` green. **Step 3:** ruff clean. **Step 4:** locally run `uv run kws-benchmark` (needs Phase-0 splits) → `docs/benchmark.md` + `benchmark.json`; commit both. **Step 5:** push, PR, CI + CodeRabbit.

---

## Self-Review

**Spec coverage (§5):** 5.1 architectures → Tasks 1–4 (registry + 4 builders, common interface, INT8-exportable). 5.2 harness/metrics → Task 5 (`evaluate_architecture` returns isolated+catalog+on-device; `render_table`/`main` → `docs/benchmark.md` + `benchmark.json`); Task 6 (CI smoke + full run). Val-based selection → Task 5 (`evaluate_architecture(... val ...)`). Reuse of eval/budgets/export → Tasks 1/5. Covered.

**Placeholder scan:** heavy training/catalog paths in `evaluate_architecture`/`main` are `# pragma: no cover` I/O with concrete recipes; pure `render_table` + all builders + the budget smoke are fully implemented + tested. No vague TODOs.

**Type consistency:** every builder → `(input_shape, n_classes) -> keras.Model`; `evaluate_architecture` returns the dict `render_table` consumes; `main` writes both artifacts. `load_split` (Phase-0) → `(X, y, is_tts)` feeds train/val/test. Consistent.

## Out of scope

Phase 2 (streaming transducer) — its own plan, written after this benchmark picks the best encoder. Real-mic eval — HW follow-up.
