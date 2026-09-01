# Design: KWS research plan — reusable dataset, architecture benchmark, streaming transducer

Date: 2026-09-01
Status: approved (brainstorming), pending spec review
Builds on: v1 KWS spec, v2 wake+slot spec. This is the umbrella research plan; each phase gets
its own implementation plan (Phase 0 first).

## 1. Goal & boundary

Turn the ad-hoc v2 model work into a **rigorous, reproducible research artifact**:

1. a **sound, documented, reusable dataset** (frozen speaker-disjoint train/val/test splits,
   a datasheet, deterministic regeneration) that every experiment shares;
2. a **paper-style architecture benchmark** (DS-CNN vs BC-ResNet vs MatchboxNet vs
   Keyword-Transformer) scored on isolated-word accuracy, end-to-end command-catalog accuracy,
   and on-device fit; and
3. a **new solution** — a small **streaming CTC/RNN-T transducer** for connected German commands
   — designed from the benchmark's lessons, within the ESP32-S3 budget.

Every model stays full-INT8 and S3-budget-gated (≤ 500 KB / ≤ 3 M MACs). Out of scope: a
recorded real-speech / real-mic test set (that is the HW/mic follow-up spec), external dataset
publishing, and beating Google-Speech-Commands SOTA — the target is *our* German MCU task.

## 2. Success criteria

1. A frozen dataset with train/val/test splits, `DATASHEET.md`, `manifest.json` (hashes +
   per-word real/TTS counts), and a one-command deterministic rebuild — identical bytes from the
   same seed, verifiable without committing audio.
2. `docs/benchmark.md`: a comparison table over ≥ 4 architectures with isolated-word acc,
   catalog full-intent acc, params, MACs, INT8 size, est. latency — reproducible from
   `kws-benchmark`.
3. A streaming transducer evaluated on the same catalog, compared head-to-head with the
   frame-classifier baseline (current best 0.689), plus its on-device budget report.

## 3. Phasing & methodology

Phase 0 (dataset) is prerequisite; Phase 1 (benchmark) consumes it; Phase 2 (new solution) uses
its data + the benchmark's best encoder. **writing-plans produces the Phase-0 plan first**;
Phases 1–2 get their own plans after Phase 0 lands. Common rules across phases: one fixed seed
for splits, INT8 export + budget gates for every model, results committed as Markdown + a
machine-readable JSON.

## 4. Phase 0 — the reusable dataset

Replaces the current train/test-only, pkl-cache-driven data with a versioned, documented dataset.

### 4.1 Splits
- **Train / val / test**, all **speaker-disjoint** (real words split by `speaker_id`; TTS words
  by `engine:voice:rate` "synthetic speaker"). **Val is new** and mandatory — model selection
  and hyperparameters are chosen on val, never test, so the benchmark is honest.
- Splits are frozen by a single seed and recorded in the manifest; a rebuild reproduces them
  exactly.

### 4.2 Artifacts (committed; audio bytes stay gitignored)
- **`DATASHEET.md`** — Datasheets-for-Datasets (Gebru et al.): motivation, composition,
  provenance (real MSWC vs TTS *per word*), TTS engine/voice inventory, licensing (MSWC CC-BY;
  macOS `say`; Piper voice licenses), split policy, recommended uses, and limitations (e.g.
  "17/23 words are synthetic; not a real-speech benchmark").
- **`manifest.json`** — per-word real/TTS counts, per-split sizes, and content hashes of the
  built feature tensors → the dataset is verifiable and reproducible without shipping audio.
- **`kws_de/dataset.py`** — the single builder: `build(seed) -> {train,val,test}` feature
  tensors + the manifest, deterministic. Exposed as `kws-dataset build`.

### 4.3 Interfaces
- `load_split(name) -> (X, y, is_tts)` — the frozen split all experiments consume.
- The v2 `data.py` fetch/TTS/augment logic is reused; Phase 0 wraps it in the split+datasheet+
  manifest discipline and adds the val split.

## 5. Phase 1 — architecture benchmark

### 5.1 Architectures (`kws_de/architectures/`)
Pluggable builders, one interface: `build(input_shape, n_classes) -> keras.Model`. All
INT8-exportable, all budget-gated.
- **DS-CNN** — baseline (existing).
- **BC-ResNet** — broadcasted residual, SOTA-efficient (Kim et al.).
- **MatchboxNet** — 1-D time-channel-separable (Majumdar & Ginsburg).
- **Keyword-Transformer (KWT)** — self-attention KWS (Berg et al.).

### 5.2 Harness (`kws-benchmark`)
Train each architecture on the **frozen splits** with identical seed/epochs/augmentation; select
on **val**; report on **test**:
- **isolated-word accuracy** (single-clip, literature-comparable),
- **catalog full-intent accuracy** (audio → stream → grammar → intent, our task),
- **on-device**: params, MACs, INT8 size, TFLM-op check, estimated latency.
Output: `docs/benchmark.md` (the "alternative architectures document" + assessment) and a
`benchmark.json` for reproducibility. Deterministic; committed.

## 6. Phase 2 — the new solution: streaming CTC/RNN-T transducer

### 6.1 Rationale
The v2 failures (boundary ghosts, long-word "Kühlschrank", multi-word composition) are the known
limitation of frame-classification + a hand-rolled decoder. The literature's fix for connected
commands is a **streaming sequence model** that transcribes the keyword sequence and learns
alignment natively (He et al. 2017; MFA-KWS 2025).

### 6.2 Design
- **Encoder**: the best small encoder from Phase 1 (candidate: BC-ResNet), made streaming.
- **Head**: **CTC** first (simpler), **RNN-T** as a stretch (implicit LM helps the grammar).
  Token set = blank + the 23 keyword tokens.
- **Decode**: greedy/beam CTC → keyword-event sequence → the existing pure `grammar.parse` →
  `Intent`. The grammar layer is reused unchanged; only the acoustic front is replaced.
- **Targets**: sequence labels — training phrases (`device [zone] action`, incl. the transition
  data) become **token sequences**, not per-window class labels.

### 6.3 Evaluation
Catalog full-intent vs the frame-classifier baseline (0.689) on the same test split, per-slot
and per-device deltas (does it fix Kühlschrank/Heizung?), + INT8 S3 budget report. Contribution:
*frame-classification + grammar vs a streaming transducer on a 23-word German MCU command grammar.*

## 7. Reproducibility / CI

- CI runs the benchmark on a **tiny committed fixture**: every architecture must build, INT8-export,
  and pass budget gates (proves the harness + the model zoo are device-deployable). Full training
  runs are local; `benchmark.json` + `docs/benchmark.md` are committed.
- Determinism: one seed → identical splits and (modulo TF nondeterminism, pinned) comparable runs.

## 8. Algorithmic references

- Zhang et al., *Hello Edge (DS-CNN)* — arXiv:1711.07128
- Kim et al., *Broadcasted Residual Learning (BC-ResNet)* — arXiv:2106.04140
- Majumdar & Ginsburg, *MatchboxNet* — arXiv:2004.08531
- Berg et al., *Keyword Transformer* — arXiv:2104.00769
- He et al., *Streaming Small-Footprint KWS with Seq2Seq* — arXiv:1710.09617
- *MFA-KWS* (CTC-Transducer, 2025) — arXiv:2505.19577
- Graves et al., *CTC* (ICML 2006); Graves, *Sequence Transduction / RNN-T* — arXiv:1211.3711
- *Advances in Small-Footprint KWS: A Comprehensive Review* — arXiv:2506.11169
- Gebru et al., *Datasheets for Datasets* — arXiv:1803.09010
- Rybakov et al., *Streaming KWS on Mobile Devices* — arXiv:2005.06720 (streaming conversion)

## 9. Paper hooks

- **E7** — architecture comparison table (isolated + catalog + on-device) across the model zoo.
- **E8** — streaming transducer vs frame-classifier on the connected-command catalog.

## 10. Non-goals

Real-speech/real-mic test set (HW follow-up). External dataset release. GSC SOTA. Changing the
23-word command vocabulary (frozen by the v2 work). Non-fixed-grammar / free-form NL.

## 11. Decomposition (three implementation plans)

1. **Phase 0 — dataset** (`kws_de/dataset.py`, splits, `DATASHEET.md`, `manifest.json`,
   `kws-dataset`). Prerequisite; planned first.
2. **Phase 1 — benchmark** (`kws_de/architectures/`, `kws-benchmark`, `docs/benchmark.md`).
3. **Phase 2 — streaming transducer** (CTC/RNN-T model, sequence targets, catalog eval).
