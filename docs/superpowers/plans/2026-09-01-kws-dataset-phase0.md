# KWS Dataset (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ad-hoc v2 data into a sound, reusable dataset — frozen speaker-disjoint train/val/test splits, a datasheet, a verifiable manifest, and a one-command deterministic rebuild — that every later experiment (benchmark, transducer) shares.

**Architecture:** A thin `kws_de/dataset.py` layer over the existing `kws_de/data.py` fetch/TTS/augment code. It adds a three-way speaker-disjoint split (val is new), writes per-split feature tensors + a `manifest.json` (counts, config, content hashes) + a `DATASHEET.md`, all reproducible from one seed. No audio bytes are committed.

**Tech Stack:** Python 3.11 (uv), numpy, pytest, ruff. Reuses `kws_de.data` (MSWC fetch, TTS fill, `build_dataset`, `split_by_speaker`, `_origin_flags`) and `kws_de.config`.

**Spec:** `docs/superpowers/specs/2026-09-01-kws-research-plan-design.md` (Phase 0, §4).

## Global Constraints

- Python `>=3.11`; `uv`; `ruff` (line-length 100, lint `E,F,I,UP,B`). Keep the full suite green.
- Splits are **speaker-disjoint**: real words split by `speaker_id`; TTS words by their synthetic speaker id (`tts:{engine}:{voice}:{rate}` — already produced by `data.py`). No speaker in two splits.
- Three splits: **train / val / test**. Val exists so model selection never touches test.
- Determinism: one `seed` fixes the speaker→split assignment and the real-clip selection. The `manifest.json` records per-word real/TTS counts, split sizes, the config (labels, MFCC params, seed), and content hashes of the built feature tensors — so any rebuild is verifiable.
- No audio/feature bytes committed (`data/` gitignored). Committed: `kws_de/dataset.py`, tests, `DATASHEET.md`; `manifest.json` is committed (it's small text, the dataset's verifiable fingerprint).
- Reuse `data.py` as-is; do not duplicate its fetch/TTS/augment logic. (Note: `data.py` may be concurrently gaining multi-engine TTS — consume whatever `split_by_speaker`/`build_dataset`/`_origin_flags` expose; do not fork them.)
- Commit trailers: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM`. Branch → PR → CI + CodeRabbit → merge.

---

### Task 1: Three-way speaker-disjoint split

**Files:**
- Modify: `kws_de/data.py` (add `split_three_way` next to `split_by_speaker`)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: the `clips_with_speakers: dict[label -> list[(clip, speaker_id)]]` shape `split_by_speaker` already takes.
- Produces: `split_three_way(clips_with_speakers, rng, val_frac=0.15, test_frac=0.15, *, keep_speaker=False) -> tuple[dict, dict, dict]` returning `(train, val, test)`. Each is `dict[label -> list[clip]]` (or `list[(clip, spk)]` when `keep_speaker=True`). **No speaker appears in more than one split.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py (append)
def test_split_three_way_is_speaker_disjoint_and_covers_all():
    from kws_de.data import split_three_way

    rng = np.random.default_rng(7)
    # 20 speakers x 2 clips; tag each clip with its speaker's int so identity survives
    clips = {
        "Licht": [(np.full(config.CLIP_SAMPLES, s, np.float32), f"spk{s}")
                  for s in range(20) for _ in range(2)],
    }
    train, val, test = split_three_way(clips, rng, val_frac=0.2, test_frac=0.2)
    tr = {float(c[0]) for c in train["Licht"]}
    va = {float(c[0]) for c in val["Licht"]}
    te = {float(c[0]) for c in test["Licht"]}
    assert tr and va and te                      # all three non-empty
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)  # disjoint speakers
    assert tr | va | te == set(float(s) for s in range(20))              # all speakers covered
    assert len(train["Licht"]) + len(val["Licht"]) + len(test["Licht"]) == 40
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_data.py::test_split_three_way_is_speaker_disjoint_and_covers_all -v` → FAIL (`ImportError: split_three_way`).

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/data.py (add near split_by_speaker)
def split_three_way(clips_with_speakers, rng, val_frac=0.15, test_frac=0.15, *, keep_speaker=False):
    """Speaker-disjoint train/val/test split. No speaker appears in more than one split.
    Fractions are of the speaker set (val_frac, test_frac; the rest is train)."""
    train, val, test = {}, {}, {}
    for label, items in clips_with_speakers.items():
        speakers = sorted({spk for _, spk in items})
        order = rng.permutation(len(speakers))
        n = len(speakers)
        n_val = round(n * val_frac) if n else 0
        n_test = round(n * test_frac) if n else 0
        # guarantee non-empty val/test when there are enough speakers
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        val_s = {speakers[i] for i in order[:n_val]}
        test_s = {speakers[i] for i in order[n_val : n_val + n_test]}
        def pick(keep):
            return [(c, s) if keep_speaker else c for c, s in items if (s in keep)]
        val[label] = pick(val_s)
        test[label] = pick(test_s)
        train[label] = pick(set(speakers) - val_s - test_s)
    return train, val, test
```

- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_data.py -q` → PASS (existing + new).

- [ ] **Step 5: Commit** — `git commit -m "feat(dataset): three-way speaker-disjoint split"`

---

### Task 2: Manifest builder

**Files:**
- Create: `kws_de/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces: `build_manifest(splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], *, seed: int, labels: list[str]) -> dict`. `splits` maps `"train"/"val"/"test" -> (X, y, is_tts)`. Returns a JSON-serialisable dict with: `seed`, `labels`, `mfcc` params, per-split `{n, per_label_counts, real, tts}`, and a `hash` per split (sha256 of the X bytes) — the verifiable fingerprint.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import numpy as np
from kws_de import config
from kws_de.manifest import build_manifest

def _split(rng, n):
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % config.NUM_CLASSES).astype(np.int64)
    is_tts = (np.arange(n) % 2 == 0)
    return X, y, is_tts

def test_manifest_shape_counts_and_stable_hash():
    rng = np.random.default_rng(0)
    splits = {"train": _split(rng, 30), "val": _split(rng, 6), "test": _split(rng, 6)}
    m = build_manifest(splits, seed=0, labels=config.LABELS)
    assert m["seed"] == 0 and m["labels"] == config.LABELS
    assert m["splits"]["train"]["n"] == 30
    assert m["splits"]["train"]["tts"] + m["splits"]["train"]["real"] == 30
    # hash is a deterministic function of the X bytes
    assert m["splits"]["train"]["hash"] == build_manifest(splits, seed=0, labels=config.LABELS)["splits"]["train"]["hash"]
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ModuleNotFoundError: kws_de.manifest`).

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/manifest.py
import hashlib
import numpy as np
from kws_de import config

def build_manifest(splits, *, seed, labels):
    out = {
        "seed": seed,
        "labels": list(labels),
        "mfcc": {"n_mfcc": config.N_MFCC, "n_frames": config.N_FRAMES,
                 "win": config.WIN_SAMPLES, "hop": config.HOP_SAMPLES,
                 "n_mels": config.N_MELS, "sample_rate": config.SAMPLE_RATE},
        "splits": {},
    }
    for name, (X, y, is_tts) in splits.items():
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        is_tts = np.asarray(is_tts, bool)
        counts = {labels[i]: int((y == i).sum()) for i in range(len(labels))}
        out["splits"][name] = {
            "n": int(len(y)),
            "real": int((~is_tts).sum()),
            "tts": int(is_tts.sum()),
            "per_label_counts": counts,
            "hash": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        }
    return out
```

- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_manifest.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dataset): manifest builder (counts + content hashes)"`

---

### Task 3: Dataset builder + `load_split` + CLI

**Files:**
- Create: `kws_de/dataset.py`
- Modify: `pyproject.toml` (add `kws-dataset` script)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `kws_de.data` (`split_three_way`, `build_dataset`, `_origin_flags`, the cached raw clips), `kws_de.manifest.build_manifest`, `kws_de.config`.
- Produces:
  - `assemble(clips_ws, noises, rng, labels, commands) -> (X, y, is_tts)` — pure: MFCC features + origin flags for one split's raw `(clip, speaker)` dict (wraps `build_dataset` + `_origin_flags`).
  - `load_split(name) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — loads `data/features_{name}.npz` → `(X, y, is_tts)`.
  - `main()` — `kws-dataset build [--seed N]`: split → assemble per split → write `data/features_{train,val,test}.npz` + `data/manifest.json`. `# pragma: no cover` (I/O; needs the raw-clip cache).

- [ ] **Step 1: Write the failing test** (the pure `assemble` + `load_split` round-trip; no fetch needed)

```python
# tests/test_dataset.py
import numpy as np
from kws_de import config
from kws_de.dataset import assemble, load_split

def _clip(rng):
    return rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)

def test_assemble_returns_features_labels_and_origin():
    rng = np.random.default_rng(0)
    clips_ws = {
        config.COMMANDS[0]: [(_clip(rng), "real1"), (_clip(rng), "tts:say:Anna:180")],
        "_unknown_": [(_clip(rng), "real2")],
    }
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y, is_tts = assemble(clips_ws, noises, rng, labels=config.LABELS, commands=config.COMMANDS)
    assert X.shape[1:] == (config.N_FRAMES, config.N_MFCC)
    assert len(X) == len(y) == len(is_tts)
    assert is_tts.dtype == bool and is_tts.any()  # the tts:* speaker rows are flagged

def test_load_split_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    X = np.zeros((3, config.N_FRAMES, config.N_MFCC), np.float32)
    y = np.array([0, 1, 2], np.int64)
    is_tts = np.array([True, False, True])
    np.savez(tmp_path / "features_val.npz", X=X, y=y, is_tts=is_tts)
    Xl, yl, tl = load_split("val")
    assert Xl.shape == X.shape and list(yl) == [0, 1, 2] and list(tl) == [True, False, True]
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ModuleNotFoundError: kws_de.dataset`).

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/dataset.py
import argparse
import json
import numpy as np
from kws_de import config
from kws_de.data import _origin_flags, build_dataset
from kws_de.manifest import build_manifest

def assemble(clips_ws, noises, rng, labels, commands):
    """Raw (clip, speaker) dict for one split -> (X, y, is_tts). Wraps build_dataset
    (features + labels) and _origin_flags (per-row real/TTS origin, same iteration order)."""
    clips = {lbl: [c for c, _ in items] for lbl, items in clips_ws.items()}
    X, y = build_dataset(clips, noises, rng, labels=labels, commands=commands)
    is_tts = _origin_flags(clips_ws, snrs=(20, 10, 0))
    return X, y, np.asarray(is_tts, bool)

def load_split(name):
    d = np.load(config.DATA_DIR / f"features_{name}.npz")
    return d["X"], d["y"], d["is_tts"]

def main() -> None:  # pragma: no cover - I/O wrapper (needs the raw-clip cache)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # 1. load cached raw (clip, speaker) dict (real MSWC + TTS-filled), see kws_de.data
    # 2. rng = np.random.default_rng(args.seed); split_three_way(..., keep_speaker=True)
    # 3. assemble(...) each split; np.savez data/features_{train,val,test}.npz
    # 4. manifest = build_manifest({...}, seed=args.seed, labels=config.COMMAND_LABELS)
    #    (config.DATA_DIR/"manifest.json").write_text(json.dumps(manifest, indent=2))
    raise NotImplementedError("wire the cached raw clips -> split_three_way -> assemble; see plan")
```

Add to `pyproject.toml` `[project.scripts]`: `kws-dataset = "kws_de.dataset:main"`.

- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_dataset.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dataset): builder + load_split + kws-dataset CLI"`

---

### Task 4: DATASHEET.md

**Files:**
- Create: `docs/DATASHEET.md`
- Test: `tests/test_datasheet.py`

**Interfaces:** none (documentation). The test guards that the required Datasheets-for-Datasets sections are present so it can't silently rot into a stub.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_datasheet.py
from pathlib import Path

def test_datasheet_has_required_sections():
    text = Path("docs/DATASHEET.md").read_text(encoding="utf-8")
    for section in ("## Motivation", "## Composition", "## Collection",
                    "## Provenance", "## Licensing", "## Splits",
                    "## Recommended uses", "## Limitations"):
        assert section in text, f"missing datasheet section: {section}"
```

- [ ] **Step 2: Run to verify it fails** — FAIL (file missing).

- [ ] **Step 3: Write `docs/DATASHEET.md`** with these exact `##` sections, filled from the real facts: Motivation (offline German MCU voice control); Composition (23-class command vocab, real MSWC vs TTS per word — cite `manifest.json`); Collection (MSWC streaming subset + macOS `say`/Piper TTS + ESC-50 noise); Provenance (per-word real/TTS counts live in `manifest.json`); Licensing (MSWC CC-BY-4.0; macOS `say`; Piper voice licenses; ESC-50 CC); Splits (speaker-disjoint train/val/test, seed-frozen, no speaker straddles); Recommended uses (KWS architecture research on-device); Limitations (17/23 words synthetic → NOT a real-speech benchmark; real-mic eval is the HW follow-up).

- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_datasheet.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "docs(dataset): DATASHEET.md (Datasheets-for-Datasets)"`

---

### Task 5: CI + full run

**Files:** Modify `.github/workflows/ci.yml` (add `--cov=kws_de.manifest --cov=kws_de.dataset` to the coverage gate).

- [ ] **Step 1:** `uv run pytest -q` — all green. **Step 2:** exact CI coverage command incl. the new pure modules ≥ 85%. **Step 3:** `uv run ruff check . && uv run ruff format --check .` clean. **Step 4:** locally run `uv run kws-dataset build` once (needs the raw-clip cache) → confirm `data/features_{train,val,test}.npz` + `data/manifest.json` are written and `manifest.json` looks right; commit `data/manifest.json`. **Step 5:** push branch, open PR, wait for CI + CodeRabbit.

---

## Self-Review

**Spec coverage (§4):** 4.1 splits → Task 1 (`split_three_way`, val added, speaker-disjoint). 4.2 artifacts → Task 2 (manifest), Task 3 (builder writes npz + manifest), Task 4 (DATASHEET). 4.3 interfaces → Task 3 (`assemble`, `load_split`, `main`/`kws-dataset`). Determinism → seed threads through `split_three_way` + manifest records it; Task 5 verifies a real build. Reuse of `data.py` → Tasks 1/3 extend, don't fork. Covered.

**Placeholder scan:** `dataset.main` is a `# pragma: no cover` I/O wrapper with a spelled-out numbered recipe (not a vague TODO); every tested function is fully implemented. No hidden placeholders.

**Type consistency:** `split_three_way(..., keep_speaker=True)` yields `(clip, spk)` items → `assemble` consumes `clips_ws` of that shape → `build_dataset`/`_origin_flags` (existing signatures) → `(X, y, is_tts)`; `build_manifest` consumes `{name: (X, y, is_tts)}`; `load_split` returns the same triple. Consistent.

## Out of scope (later phases / plans)

Phase 1 architecture benchmark and Phase 2 streaming transducer (separate plans, per the spec §11). This plan only produces the frozen, documented dataset they consume.
