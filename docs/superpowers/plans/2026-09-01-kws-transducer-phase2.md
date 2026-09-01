# Streaming CTC Command Recogniser (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The paper's new model — a small **streaming CTC** command recogniser that transcribes the keyword sequence directly (learned alignment/segmentation), decoded into the existing pure grammar, evaluated on the command catalog against the frame-classifier baseline (0.689).

**Architecture:** A streaming encoder (the best device-runnable encoder from the Phase-1 benchmark — default `bc_resnet`) + a CTC head over `blank + the command-keyword tokens`. Training uses **phrase** audio (`device [zone] action`) → MFCC frame sequences → token-sequence targets. Greedy CTC decode → keyword events → `kws_de.grammar.parse` → intent. All INT8, S3-budget-gated.

**Tech Stack:** Python 3.11 (uv), TensorFlow/Keras (`tf.nn.ctc_loss`, `keras.ops`/`ctc_decode`), numpy, pytest, ruff. Reuses `kws_de.{config,features,architectures,grammar,dataset,export,budgets}`.

**Spec:** `docs/superpowers/specs/2026-09-01-kws-research-plan-design.md` (Phase 2, §6).

## Global Constraints

- Python `>=3.11`; `uv`; `ruff` (100, `E,F,I,UP,B`). Suite stays green.
- Token set: `CTC_TOKENS = ["_blank_"] + config.DEVICES + config.ZONES + config.ACTIONS` (blank at index 0). No `_unknown_`/`_silence_` (CTC emits blank for non-words).
- Encoder is a **frame-sequence** model: input `(T, N_MFCC)` variable-length MFCC frames (not a fixed 1 s window), output `(T', n_tokens)` logits. Default encoder = `bc_resnet` adapted to emit per-frame logits (global-pool removed); make the encoder name a parameter.
- Phrase data is generated from the **train split speakers only** (no leakage): concatenate `device [zone] action` word clips (+ gaps + noise) → one utterance → MFCC frame sequence; target = the token-id sequence.
- Decode: greedy CTC (collapse repeats, drop blank) → keyword list → `grammar.parse`. Evaluated on the SAME catalog as the baseline; report full-intent + per-slot vs 0.689.
- INT8 export + budget gate for the encoder (≤ 500 KB / ≤ 3 M MACs; note CTC decode is host/grammar-side). If a CTC op is not TFLM-runnable, record it (like KWT) — honest over forced.
- No data/model bytes committed. Commit trailers as in the repo. Branch → PR → CI + CodeRabbit.

---

### Task 1: Token vocabulary + CTC decode

**Files:** Create `kws_de/ctc.py`; Test: `tests/test_ctc.py`.

**Interfaces:** `CTC_TOKENS: list[str]`; `token_id(t) -> int`; `greedy_decode(logits: np.ndarray) -> list[str]` — argmax per frame, collapse consecutive repeats, drop `_blank_`, map ids→token strings.

- [ ] **Step 1: failing test**

```python
# tests/test_ctc.py
import numpy as np
from kws_de.ctc import CTC_TOKENS, greedy_decode, token_id

def _onehot_seq(ids, n):
    L = np.full((len(ids), n), -9.0); 
    for i, t in enumerate(ids): L[i, t] = 9.0
    return L

def test_greedy_collapses_repeats_and_drops_blank():
    n = len(CTC_TOKENS); b = 0
    licht, an = token_id("Licht"), token_id("an")
    # frames: Licht Licht blank an an  -> ["Licht","an"]
    logits = _onehot_seq([licht, licht, b, an, an], n)
    assert greedy_decode(logits) == ["Licht", "an"]

def test_blank_only_is_empty():
    n = len(CTC_TOKENS)
    assert greedy_decode(_onehot_seq([0, 0, 0], n)) == []
```

- [ ] **Step 2: fail. Step 3: implement** `CTC_TOKENS = ["_blank_"] + config.DEVICES + config.ZONES + config.ACTIONS`; `token_id = CTC_TOKENS.index`; `greedy_decode`: `ids = logits.argmax(-1)`; collapse consecutive equal; drop 0; map to strings. **Step 4: pass. Step 5: commit** `feat(ctc): token vocab + greedy CTC decode`.

---

### Task 2: End-to-end intent from logits (decode → grammar)

**Files:** Modify `kws_de/ctc.py`; Test: extend.

**Interfaces:** `logits_to_intent(logits) -> Intent | Rejection` = `grammar.parse(greedy_decode(logits))`.

- [ ] failing test: `logits_to_intent` on a `Licht Küche an` logit sequence → `Intent("Licht","Küche","an")`; a blank-only sequence → `Rejection`. Implement the one-liner composition. Commit `feat(ctc): logits -> grammar -> intent`.

---

### Task 3: Phrase dataset (sequence targets)

**Files:** Create `kws_de/phrases.py`; Test: `tests/test_phrases.py`.

**Interfaces:** `make_phrase(tokens, word_clips, rng, gap_ms=250) -> np.ndarray` (concatenate the word clips with silence gaps → one waveform); `phrase_features(waveform) -> np.ndarray` (streaming MFCC frames `(T, N_MFCC)` via the same `kws_de.features` front-end, but over the whole utterance, not a 1 s window); `build_phrase_batch(catalog, clips_by_word, rng) -> list[(feat_seq, target_ids)]`.

- [ ] failing test: `make_phrase(["Licht","an"], {...}, rng)` returns a waveform longer than either clip; `phrase_features` returns `(T, N_MFCC)` with `T` growing with duration; a built batch pairs each feature-seq with the right token-id target. Implement (reuse `features.mfcc`'s filterbank but sliding over the full signal — factor a `mfcc_sequence(waveform)` helper if needed). Commit `feat(phrases): phrase synthesis + sequence targets`.

---

### Task 4: Streaming CTC encoder

**Files:** Create `kws_de/architectures/ctc_encoder.py`; Test: `tests/test_architectures.py` (extend).

**Interfaces:** `build_ctc_encoder(n_tokens, encoder="bc_resnet") -> keras.Model` — input `(None, N_MFCC, 1)` variable-T, the chosen benchmark encoder body with the final GAP/classifier replaced by a per-frame `Dense(n_tokens)` (time-distributed) → logits `(None, n_tokens)`. Must accept variable-length T.

- [ ] failing test `test_ctc_encoder_variable_length_logits`: build; feed `(1, 60, N_MFCC, 1)` and `(1, 90, …)` → outputs `(1, T', n_tokens)` with T' tracking T. Implement by taking the encoder's conv stack (stride-1 in time) and a time-distributed dense head. Commit `feat(ctc): streaming CTC encoder (per-frame logits)`.

---

### Task 5: CTC training + catalog eval (the numbers)

**Files:** Create `kws_de/transducer.py`; Test: `tests/test_transducer.py`.

**Interfaces:**
- `ctc_train(batches, n_tokens, *, encoder, epochs, seed) -> keras.Model` — pads sequences, trains with `tf.nn.ctc_loss` (blank index 0). Pure-ish; unit-test on a tiny synthetic separable batch that loss decreases.
- `evaluate_catalog(predict_fn) -> dict` (`# pragma: no cover`) — for each catalog intent, synthesise a phrase (Task 3), get per-frame logits, `logits_to_intent`, score full-intent; reuse the catalog list from `kws_de.eval`.
- `main()` — `kws-transducer`: build phrase batches from the train split, `ctc_train`, INT8-export the encoder + `check_budgets`, run `evaluate_catalog`, write `docs/transducer-report.md` (catalog full-intent + per-slot vs the 0.689 baseline + budget).

- [ ] **Step 1: failing test** — tiny synthetic CTC smoke: two token classes, scripted feature sequences, `ctc_train` for a few epochs → training loss decreases. **Step 2: fail. Step 3: implement** `ctc_train` (Keras model + `tf.nn.ctc_loss`, length-masked) + the eval/main glue. **Step 4: pass. Step 5: commit** `feat(ctc): CTC training + catalog eval`.

- [ ] **Run it:** `uv run kws-transducer` → `docs/transducer-report.md` with the real catalog full-intent number. Compare to the frame-classifier 0.689 (better / worse — report honestly; if the encoder can't INT8-export for streaming CTC, report it and keep the float number).

---

### Task 6: CI + report

**Files:** Modify `.github/workflows/ci.yml` (add `--cov=kws_de.ctc --cov=kws_de.phrases`).

- [ ] `uv run pytest -q` green; ruff clean; commit `docs/transducer-report.md` + the machine JSON; push, PR, CI + CodeRabbit.

---

## Self-Review

**Spec coverage (§6):** 6.1 rationale → whole plan. 6.2 design: encoder → Task 4; CTC head/tokens → Task 1; decode → grammar → Tasks 1–2; sequence targets → Task 3. 6.3 eval → Task 5 (catalog full-intent vs 0.689 + budget). Encoder-as-parameter (best benchmark encoder) → Task 4 `encoder=` arg. Covered.

**Placeholder scan:** `evaluate_catalog`/`main` are `# pragma: no cover` I/O with concrete recipes; `greedy_decode`, `logits_to_intent`, phrase geometry, `ctc_train` smoke are fully implemented + tested. No vague TODOs.

**Type consistency:** encoder logits `(T', n_tokens)` → `greedy_decode` → `list[str]` → `grammar.parse` → `Intent|Rejection`; `CTC_TOKENS` index space shared by `token_id`, encoder head width, and `ctc_train` blank=0. Consistent.

## Out of scope

RNN-T (spec's stretch — CTC first). Real-mic eval (HW follow-up). Beating 0.689 is the aim, not a gate — an honest comparison is the deliverable.
