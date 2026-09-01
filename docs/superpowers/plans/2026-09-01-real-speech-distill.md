# Real Speech + Distillation + INT8 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TTS-only command words with real MSWC-de / self-recorded speech (dataset v3), add KWT→DS-CNN knowledge distillation, and close the INT8 PTQ gap with class-balanced calibration — each benchmarked against the current numbers.

**Architecture:** Two new data loaders (`kws_de/mswc.py` mines the extracted MSWC-de tarball by keyword folder; `kws_de/recordings.py` loads drop-in self-recorded WAVs) return the same `{label: [(clip, speaker_id)]}` dict `kws_de.data` already consumes, so the split/augment/manifest pipeline is untouched. `kws_de/distill.py` precomputes frozen-teacher probabilities and trains the unchanged `build_dscnn` student with a concatenated `[one_hot | teacher_probs]` target and a KL+CE loss via plain `model.fit`. `kws_de/export.balanced_calibration` replaces `X_train[:200]`; `benchmark._evaluate_int8` is extracted so benchmark and distill share one INT8 evaluation that reports float and INT8 accuracy side by side.

**Tech Stack:** Python 3.11, TensorFlow 2.21 / Keras 3.15, numpy, librosa, soundfile, pytest, ruff 0.16.5. Run everything with `uv run` from the repo root.

**Spec:** `docs/superpowers/specs/2026-09-01-real-speech-distill-design.md`

## Global Constraints

- No training data or model binaries committed; `data/` and `models/` stay gitignored (`data/mswc/`, `data/recordings/` included).
- Public repo never names the product/app the vocabulary was derived from, nor machine-local paths (external volumes, symlinks). `Aufstelldach` is described only as "a rare camper-hardware compound".
- Ruff `==0.16.5` (`uv run ruff check . && uv run ruff format --check .`), markdownlint (`npx --yes markdownlint-cli@0.42.0 --config .markdownlint.json`), gitleaks; pytest with `--cov-fail-under=85` over `kws_de.features,budgets,data,grammar,stream,manifest,ctc,phrases`.
- Pure helpers get unit tests; network/audio/training I/O is `# pragma: no cover`.
- Every random choice takes an explicit `seed`; loader output order is deterministic.
- Line length 100; imports sorted (ruff `I`); `from __future__` not used (py311 floor).
- Git hooks are active (`core.hooksPath=.githooks`): pre-commit runs ruff+markdownlint, pre-push runs pytest. A failing hook means fix, not `--no-verify`.
- Commit after every task; never push to `main`; the branch is `feat/real-speech-distill`.

---

## File map

| File | Responsibility |
|---|---|
| `kws_de/mswc.py` (new) | Mine extracted MSWC-de tarball → clips dict. Pure: `_folder_index`, `_pick`; I/O: `_decode`, `mine`. |
| `kws_de/recordings.py` (new) | Load `data/recordings/<word>/*.wav` → clips dict. Pure: `centre`; I/O: `load_recordings`. |
| `kws_de/data.py` (modify) | `main`: `--v3 --mswc-root`; `_fetch_and_cache`: `mswc_root` path uses `mswc.mine` + `load_recordings` instead of streaming. |
| `kws_de/dataset.py` (modify) | `load_split(name, prefix="features")`; `build(seed, cache_name, out_prefix)`; CLI `--cache`, `--prefix`. |
| `kws_de/export.py` (modify) | `balanced_calibration(X, y, per_class, seed)`; `main` uses it. |
| `kws_de/benchmark.py` (modify) | `_evaluate_int8(...)` extracted; `render_table` gains Float column; `evaluate_architecture(..., features="v2")`; CLI `--features`. |
| `kws_de/distill.py` (new) | `soften`, `distill_targets`, `make_distill_loss`, `distill`, `main` (`kws-distill`). |
| `pyproject.toml` (modify) | `kws-distill` script. |
| `README.md`, `docs/DATASHEET.md`, `docs/paper.md`, `docs/paper-notes.md` (modify) | v3 data how-to, recordings how-to, E9/E10 results. |
| `tests/test_mswc.py`, `tests/test_recordings.py`, `tests/test_distill.py` (new); `tests/test_export.py`, `tests/test_dataset.py`, `tests/test_benchmark.py` (modify) | Unit tests. |

---

### Task 1: MSWC tarball miner — pure helpers

**Files:**
- Create: `kws_de/mswc.py`
- Test: `tests/test_mswc.py`

**Interfaces:**
- Produces: `_folder_index(root: Path) -> dict[str, Path]` (lower-cased folder name → path, from `root / "clips"`); `_pick(items: list, n: int, rng: np.random.Generator) -> list` (seeded shuffle, first `n`, input untouched).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mswc.py
from pathlib import Path

import numpy as np

from kws_de.mswc import _folder_index, _pick


def test_folder_index_is_case_insensitive(tmp_path: Path):
    (tmp_path / "clips" / "küche").mkdir(parents=True)
    (tmp_path / "clips" / "Licht").mkdir()
    (tmp_path / "clips" / "not_a_dir.txt").write_text("x")
    idx = _folder_index(tmp_path)
    assert idx["küche"] == tmp_path / "clips" / "küche"
    assert idx["licht"] == tmp_path / "clips" / "Licht"
    assert "not_a_dir.txt" not in idx


def test_pick_is_deterministic_bounded_and_non_mutating():
    items = list(range(10))
    a = _pick(items, 4, np.random.default_rng(3))
    b = _pick(items, 4, np.random.default_rng(3))
    c = _pick(items, 4, np.random.default_rng(4))
    assert a == b and len(a) == 4
    assert a != c
    assert items == list(range(10))
    assert _pick(items, 50, np.random.default_rng(0)) != items  # shuffled, all 10
    assert sorted(_pick(items, 50, np.random.default_rng(0))) == items
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/kws-de && uv run pytest tests/test_mswc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kws_de.mswc'`

- [ ] **Step 3: Write the module with the pure helpers**

```python
# kws_de/mswc.py
"""Mine an extracted MSWC-de tarball (https://mlcommons.org/datasets/multilingual-spoken-words/,
CC-BY 4.0) by keyword folder, instead of streaming the HuggingFace mirror
alphabetically (`kws_de.data._fetch_mswc`), which never reached most of our
command words before its scan cap.

Expected layout under `root` (gitignored, e.g. `data/mswc/de/`):

    clips/<keyword>/<clip>.opus      1 s, 16 kHz mono
    de_splits.csv                    SET,LINK,WORD,VALID,SPEAKER,GENDER

Returns the same `{label: [(np.ndarray float32, speaker_id)]}` dict as
`_fetch_mswc`, so split/augment/manifest code is unchanged.
"""

import csv
import subprocess
from pathlib import Path

import numpy as np

from kws_de import config


def _folder_index(root: Path) -> dict[str, Path]:
    """Lower-cased keyword folder name -> folder path (case-insensitive lookup:
    config spells `Küche`, MSWC folders are lower-case)."""
    clips = Path(root) / "clips"
    return {p.name.lower(): p for p in sorted(clips.iterdir()) if p.is_dir()}


def _pick(items: list, n: int, rng: np.random.Generator) -> list:
    """Seeded shuffle, first `n`. Does not mutate `items`."""
    order = rng.permutation(len(items))
    return [items[i] for i in order[:n]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/src/kws-de && uv run pytest tests/test_mswc.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd ~/src/kws-de && git add kws_de/mswc.py tests/test_mswc.py && git commit -m "feat(mswc): tarball miner pure helpers (folder index, seeded pick)"
```

---

### Task 2: MSWC tarball miner — `mine()`

**Files:**
- Modify: `kws_de/mswc.py`
- Test: `tests/test_mswc.py`

**Interfaces:**
- Consumes: `_folder_index`, `_pick` (Task 1).
- Produces: `mine(root: Path, words: list[str], *, n_per_word: int = 300, n_unknown: int = 2000, unknown_per_word_cap: int = 5, seed: int = 0) -> dict[str, list[tuple[np.ndarray, str]]]`; `_decode(path: Path) -> np.ndarray` (float32, 16 kHz mono); `_valid_speakers(root) -> dict[str, str]` (relative clip path `<keyword>/<file>` → speaker id, VALID rows only).

- [ ] **Step 1: Write the failing test (fixture tree with real WAVs)**

```python
# append to tests/test_mswc.py
import csv

import soundfile as sf

from kws_de import config
from kws_de.mswc import mine


def _write_tree(root: Path, words: dict[str, list[tuple[str, str, bool]]]):
    """words: folder -> [(filename, speaker, valid)]. Writes 0.5 s tone WAVs
    (MSWC ships .opus; `_decode` goes through soundfile either way) and the csv."""
    rows = []
    for folder, files in words.items():
        d = root / "clips" / folder
        d.mkdir(parents=True)
        for fname, spk, valid in files:
            t = np.arange(8000) / config.SAMPLE_RATE
            sf.write(d / fname, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
            rows.append(("TRAIN", f"{folder}/{fname}", folder, "TRUE" if valid else "FALSE", spk, ""))
    with open(root / "de_splits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SET", "LINK", "WORD", "VALID", "SPEAKER", "GENDER"])
        w.writerows(rows)


def test_mine_counts_speakers_validity_and_unknown_cap(tmp_path: Path):
    _write_tree(
        tmp_path,
        {
            "licht": [("a.wav", "s1", True), ("b.wav", "s2", True), ("c.wav", "s3", False)],
            "küche": [("a.wav", "s4", True)],
            "haus": [("a.wav", "s5", True), ("b.wav", "s6", True), ("c.wav", "s7", True)],
            "baum": [("a.wav", "s8", True)],
        },
    )
    clips = mine(tmp_path, ["Licht", "Küche", "Aufstelldach"], n_per_word=5,
                 n_unknown=10, unknown_per_word_cap=2, seed=0)
    assert len(clips["Licht"]) == 2  # invalid row excluded
    assert {spk for _, spk in clips["Licht"]} == {"s1", "s2"}
    assert len(clips["Küche"]) == 1 and clips["Küche"][0][1] == "s4"
    assert clips["Aufstelldach"] == []  # folder absent -> empty, not KeyError
    # _unknown_: haus capped at 2, baum 1; target-word folders never leak in
    assert len(clips["_unknown_"]) == 3
    unk = {spk for _, spk in clips["_unknown_"]}
    assert "s8" in unk and len(unk & {"s5", "s6", "s7"}) == 2
    for clip, _ in clips["Licht"] + clips["_unknown_"]:
        assert clip.dtype == np.float32 and clip.ndim == 1 and clip.shape[0] == 8000


def test_mine_is_deterministic_in_seed(tmp_path: Path):
    _write_tree(tmp_path, {"licht": [(f"{i}.wav", f"s{i}", True) for i in range(6)]})
    a = mine(tmp_path, ["Licht"], n_per_word=3, n_unknown=0, seed=1)
    b = mine(tmp_path, ["Licht"], n_per_word=3, n_unknown=0, seed=1)
    c = mine(tmp_path, ["Licht"], n_per_word=3, n_unknown=0, seed=2)
    assert [s for _, s in a["Licht"]] == [s for _, s in b["Licht"]]
    assert [s for _, s in a["Licht"]] != [s for _, s in c["Licht"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/kws-de && uv run pytest tests/test_mswc.py -v`
Expected: FAIL with `ImportError: cannot import name 'mine'`

- [ ] **Step 3: Implement `_valid_speakers`, `_decode`, `mine`**

```python
# append to kws_de/mswc.py


def _valid_speakers(root: Path) -> dict[str, str]:
    """`<keyword>/<file>` -> speaker id for rows with VALID == TRUE."""
    out: dict[str, str] = {}
    with open(Path(root) / "de_splits.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["VALID"].strip().upper() == "TRUE":
                out[row["LINK"]] = row["SPEAKER"]
    return out


def _decode(path: Path) -> np.ndarray:  # pragma: no cover - audio I/O
    """float32 mono 16 kHz. soundfile handles wav and (libsndfile >= 1.0.29) opus;
    otherwise fall back to ffmpeg."""
    import soundfile as sf

    try:
        sig, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - libsndfile without opus support
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
             "-ar", str(config.SAMPLE_RATE), "-"],
            check=True, capture_output=True,
        ).stdout
        return np.frombuffer(raw, np.float32).copy()
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa

        sig = librosa.resample(sig, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return np.asarray(sig, np.float32)


def mine(
    root: Path,
    words: list[str],
    *,
    n_per_word: int = 300,
    n_unknown: int = 2000,
    unknown_per_word_cap: int = 5,
    seed: int = 0,
) -> dict[str, list[tuple[np.ndarray, str]]]:
    """Per target word: up to `n_per_word` VALID clips (seeded pick). `_unknown_`:
    `n_unknown` VALID clips from keyword folders NOT in `words`, at most
    `unknown_per_word_cap` per keyword, folders visited in seeded-shuffled order."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    index = _folder_index(root)
    speakers = _valid_speakers(root)

    def valid_files(folder: Path) -> list[tuple[Path, str]]:
        out = []
        for p in sorted(folder.iterdir()):
            spk = speakers.get(f"{folder.name}/{p.name}")
            if spk is not None:
                out.append((p, spk))
        return out

    clips: dict = {}
    targets = set()
    for w in words:
        folder = index.get(w.lower())
        targets.add(w.lower())
        files = valid_files(folder) if folder else []
        clips[w] = [(_decode(p), spk) for p, spk in _pick(files, n_per_word, rng)]

    clips["_unknown_"] = []
    others = [k for k in index if k not in targets]
    for k in _pick(others, len(others), rng):
        if len(clips["_unknown_"]) >= n_unknown:
            break
        room = min(unknown_per_word_cap, n_unknown - len(clips["_unknown_"]))
        for p, spk in _pick(valid_files(index[k]), room, rng):
            clips["_unknown_"].append((_decode(p), spk))
    return clips
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/src/kws-de && uv run pytest tests/test_mswc.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
cd ~/src/kws-de && uv run ruff check kws_de/mswc.py tests/test_mswc.py && uv run ruff format kws_de/mswc.py tests/test_mswc.py && git add kws_de/mswc.py tests/test_mswc.py && git commit -m "feat(mswc): mine() — VALID-filtered per-keyword clips + diverse _unknown_ pool"
```

---

### Task 3: Self-recorded clips loader

**Files:**
- Create: `kws_de/recordings.py`
- Test: `tests/test_recordings.py`

**Interfaces:**
- Produces: `centre(sig: np.ndarray, n: int = config.CLIP_SAMPLES) -> np.ndarray` (pure); `load_recordings(root: Path, words: list[str]) -> dict[str, list[tuple[np.ndarray, str]]]` with speaker id `"rec:<speaker>"` parsed from `<speaker>_<n>.<ext>`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_recordings.py
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.recordings import centre, load_recordings


def test_centre_pads_short_symmetrically():
    out = centre(np.ones(100, np.float32), n=1000)
    assert out.shape == (1000,)
    assert out[450:550].sum() == 100 and out[:450].sum() == 0 and out[550:].sum() == 0


def test_centre_crops_long_symmetrically():
    sig = np.arange(1000, dtype=np.float32)
    out = centre(sig, n=100)
    assert out.shape == (100,) and out[0] == 450 and out[-1] == 549


def test_load_recordings_speaker_prefix_and_unknown_folders_ignored(tmp_path: Path):
    d = tmp_path / "Aufstelldach"
    d.mkdir()
    t = np.arange(int(0.4 * 16000)) / 16000
    tone = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    padded = np.concatenate([np.zeros(4000, np.float32), tone, np.zeros(4000, np.float32)])
    sf.write(d / "alice_1.wav", padded, 16000)
    sf.write(d / "bob_1.wav", padded, 16000)
    (tmp_path / "Unrelated").mkdir()
    sf.write(tmp_path / "Unrelated" / "x_1.wav", padded, 16000)

    clips = load_recordings(tmp_path, ["Aufstelldach", "Licht"])
    assert sorted(spk for _, spk in clips["Aufstelldach"]) == ["rec:alice", "rec:bob"]
    assert clips["Licht"] == []
    assert "Unrelated" not in clips
    for clip, _ in clips["Aufstelldach"]:
        assert clip.shape == (config.CLIP_SAMPLES,) and clip.dtype == np.float32
        # trimmed+centred: energy sits in the middle, not at the start
        assert np.abs(clip[:2000]).max() < 1e-3 and np.abs(clip[7000:9000]).max() > 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/kws-de && uv run pytest tests/test_recordings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kws_de.recordings'`

- [ ] **Step 3: Implement**

```python
# kws_de/recordings.py
"""Drop-in self-recorded clips for words no public corpus has (e.g. a rare
camper-hardware compound). Layout, gitignored:

    data/recordings/<word>/<speaker>_<n>.wav|m4a|...

Any sample rate/channels; each file is one utterance. Returns the same
`{label: [(clip, speaker_id)]}` dict as `kws_de.mswc.mine`, with speaker id
`rec:<speaker>` so the speaker-disjoint split holds out whole people and
`_origin_flags` counts them as real (only `tts:` is synthetic).
"""

from pathlib import Path

import numpy as np

from kws_de import config


def centre(sig: np.ndarray, n: int = config.CLIP_SAMPLES) -> np.ndarray:
    """Zero-pad or crop `sig` symmetrically to exactly `n` samples."""
    sig = np.asarray(sig, np.float32).ravel()
    if sig.shape[0] >= n:
        start = (sig.shape[0] - n) // 2
        return sig[start : start + n]
    pad = n - sig.shape[0]
    return np.pad(sig, (pad // 2, pad - pad // 2))


def load_recordings(root: Path, words: list[str]) -> dict[str, list[tuple[np.ndarray, str]]]:
    """For each word in `words`, load `root/<word>/*` (skipped if absent),
    trim leading/trailing silence and centre in a CLIP_SAMPLES window."""
    import librosa

    root = Path(root)
    clips: dict = {}
    for w in words:
        clips[w] = []
        folder = root / w
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.name.startswith(".") or not p.is_file():
                continue
            sig, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)
            trimmed, _ = librosa.effects.trim(sig, top_db=30)
            speaker = p.stem.rsplit("_", 1)[0]
            clips[w].append((centre(trimmed), f"rec:{speaker}"))
    return clips
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/src/kws-de && uv run pytest tests/test_recordings.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint and commit**

```bash
cd ~/src/kws-de && uv run ruff check kws_de/recordings.py tests/test_recordings.py && uv run ruff format kws_de/recordings.py tests/test_recordings.py && git add kws_de/recordings.py tests/test_recordings.py && git commit -m "feat(recordings): drop-in self-recorded clip loader (trim, centre, rec: speaker ids)"
```

---

### Task 4: Wire v3 into `kws-data` and `kws-dataset`

**Files:**
- Modify: `kws_de/data.py:248-305` (`main`, `_fetch_and_cache`)
- Modify: `kws_de/dataset.py:30-79` (`load_split`, `build`, `main`)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `kws_de.mswc.mine`, `kws_de.recordings.load_recordings`.
- Produces: `load_split(name: str, prefix: str = "features") -> (X, y, is_tts)`; `build(seed=0, cache_name="raw_clips_merged.pkl", out_prefix="features") -> manifest dict` writing `data/<out_prefix>_{train,val,test}.npz` and `data/manifest.json` when `out_prefix == "features"` else `data/manifest_<suffix>.json` where suffix is `out_prefix` minus `features_`; CLI `kws-dataset build --seed N --cache raw_clips_v3.pkl --prefix features_v3`; `kws-data --fetch --v3 --mswc-root data/mswc/de`.

- [ ] **Step 1: Write the failing test for `load_split(prefix=)`**

```python
# append to tests/test_dataset.py
import numpy as np

from kws_de import config, dataset


def test_load_split_prefix_selects_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    np.savez(tmp_path / "features_v3_test.npz", X=np.zeros((2, 49, 10), np.float32),
             y=np.array([1, 2]), is_tts=np.array([False, True]))
    X, y, is_tts = dataset.load_split("test", prefix="features_v3")
    assert X.shape == (2, 49, 10) and list(y) == [1, 2] and list(is_tts) == [False, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/src/kws-de && uv run pytest tests/test_dataset.py -v -k prefix`
Expected: FAIL with `TypeError: load_split() got an unexpected keyword argument 'prefix'`

- [ ] **Step 3: Update `kws_de/dataset.py`**

Replace `load_split` and the `build` signature/outputs and `main`:

```python
def load_split(name: str, prefix: str = "features"):
    """Load `data/{prefix}_{name}.npz` -> (X, y, is_tts). prefix "features" is the
    frozen v2 dataset, "features_v3" the real-speech rebuild."""
    d = np.load(config.DATA_DIR / f"{prefix}_{name}.npz")
    return d["X"], d["y"], d["is_tts"]


def build(  # pragma: no cover - I/O
    seed: int = 0, cache_name: str = "raw_clips_merged.pkl", out_prefix: str = "features"
):
```

Inside `build`, change the two output paths:

```python
        np.savez(config.DATA_DIR / f"{out_prefix}_{name}.npz", X=X, y=y, is_tts=is_tts)
```

```python
    suffix = out_prefix.removeprefix("features")
    (config.DATA_DIR / f"manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
```

and `main`:

```python
def main() -> None:  # pragma: no cover - CLI wrapper
    """`kws-dataset build [--seed N] [--cache raw_clips_v3.pkl] [--prefix features_v3]`."""
    ap = argparse.ArgumentParser(prog="kws-dataset")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="raw_clips_merged.pkl", help="raw clip cache under data/")
    ap.add_argument("--prefix", default="features", help="output npz prefix (features_v3 ...)")
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    build(seed=args.seed, cache_name=args.cache, out_prefix=args.prefix)
```

Update the docstring line in `build` that mentions the manifest so it reads "`data/manifest<suffix>.json` (suffix `_v3` for prefix `features_v3`)".

- [ ] **Step 4: Update `kws_de/data.py` — `main` and `_fetch_and_cache`**

In `main`, after the `--v2` argument add:

```python
    ap.add_argument(
        "--v3",
        action="store_true",
        help="v2 vocab, real speech mined from an extracted MSWC-de tarball "
        "(--mswc-root) plus data/recordings/, TTS only as backstop",
    )
    ap.add_argument(
        "--mswc-root",
        default=str(config.DATA_DIR / "mswc" / "de"),
        help="extracted MSWC-de tarball root (contains clips/ and de_splits.csv)",
    )
```

and replace the three `args.v2` derivations:

```python
    v2 = args.v2 or args.v3
    words = command_words() if v2 else None
    cache_name = "raw_clips_v3.pkl" if args.v3 else ("raw_clips_v2.pkl" if v2 else "raw_clips.pkl")
    labels = config.COMMAND_LABELS if v2 else None
    out_prefix = "features_v3" if args.v3 else ("features_v2" if v2 else "features")
    if args.fetch:
        _fetch_and_cache(
            safety_cap=args.safety_cap,
            words=words,
            cache_name=cache_name,
            mswc_root=Path(args.mswc_root) if args.v3 else None,
        )
```

In `_fetch_and_cache`, add the kwarg `mswc_root: Path | None = None` and replace the `else:` branch body of the clips cache miss with:

```python
    else:
        if mswc_root is not None:
            from kws_de.mswc import mine
            from kws_de.recordings import load_recordings

            clips = mine(mswc_root, words, n_per_word=n_per_word, n_unknown=n_unknown)
            for w, items in load_recordings(config.DATA_DIR / "recordings", words).items():
                clips[w].extend(items)
            scanned = "mswc-tarball"
        else:
            clips, scanned = _fetch_mswc(words, n_per_word, n_unknown, safety_cap)
        counts = {c: len(clips[c]) for c in words}
        print(f"[mswc] done: scanned={scanned} counts={counts} unknown={len(clips['_unknown_'])}")
        with open(clips_path, "wb") as fh:
            pickle.dump({"clips": clips, "scanned": scanned}, fh)
```

Extend the `_fetch_and_cache` docstring with one sentence: "With `mswc_root`, mine the extracted tarball (`kws_de.mswc.mine`) and merge `data/recordings/` instead of streaming; `_unknown_` gets `n_unknown` clips either way." When `mswc_root` is set, `main` passes `n_unknown=2000` — add that: in `main`'s call, `n_unknown=2000 if args.v3 else 600`.

- [ ] **Step 5: Run the full suite and lint**

Run: `cd ~/src/kws-de && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass; the new prefix test passes.

- [ ] **Step 6: Commit**

```bash
cd ~/src/kws-de && git add kws_de/data.py kws_de/dataset.py tests/test_dataset.py && git commit -m "feat(data): --v3 fetch from MSWC tarball + recordings; dataset --cache/--prefix, load_split(prefix)"
```

---

### Task 5: Class-balanced INT8 calibration

**Files:**
- Modify: `kws_de/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Produces: `balanced_calibration(X, y, *, per_class: int = 20, seed: int = 0) -> np.ndarray` (rows of X; ≤ per_class per class; every class with ≥1 sample present; deterministic in seed).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_export.py
from kws_de.export import balanced_calibration


def test_balanced_calibration_caps_per_class_and_keeps_every_class():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 3 + [2] * 20)
    X = rng.standard_normal((len(y), 4, 3)).astype(np.float32)
    X[:, 0, 0] = y  # tag each row with its class
    rep = balanced_calibration(X, y, per_class=5, seed=0)
    got = np.bincount(rep[:, 0, 0].astype(int), minlength=3)
    assert got.tolist() == [5, 3, 5]


def test_balanced_calibration_is_deterministic_in_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=200)
    X = rng.standard_normal((200, 2, 2)).astype(np.float32)
    a = balanced_calibration(X, y, per_class=10, seed=7)
    b = balanced_calibration(X, y, per_class=10, seed=7)
    c = balanced_calibration(X, y, per_class=10, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/kws-de && uv run pytest tests/test_export.py -v -k balanced`
Expected: FAIL with `ImportError: cannot import name 'balanced_calibration'`

- [ ] **Step 3: Implement and use in `export.main`**

Add to `kws_de/export.py` after `to_int8_tflite`:

```python
def balanced_calibration(X, y, *, per_class: int = 20, seed: int = 0) -> np.ndarray:
    """Stratified PTQ calibration set: up to `per_class` rows per class (seeded
    pick within class), so the quantizer sees every class's activation range
    instead of whatever `X[:200]` happened to hold (training data is grouped by
    label, so a prefix slice is a handful of classes)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    rows = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rows.extend(rng.permutation(idx)[:per_class].tolist())
    return X[np.asarray(rows, dtype=np.int64)]
```

In `main`, replace

```python
    feats = np.load(config.DATA_DIR / f"{prefix}_train.npz")["X"][:200]
    blob = to_int8_tflite(model, feats)
```

with

```python
    d = np.load(config.DATA_DIR / f"{prefix}_train.npz")
    blob = to_int8_tflite(model, balanced_calibration(d["X"], d["y"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/src/kws-de && uv run pytest tests/test_export.py -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
cd ~/src/kws-de && uv run ruff check kws_de/export.py tests/test_export.py && uv run ruff format kws_de/export.py tests/test_export.py && git add kws_de/export.py tests/test_export.py && git commit -m "feat(export): class-balanced PTQ calibration set replaces X_train[:200]"
```

---

### Task 6: Shared INT8 evaluation in `benchmark`

**Files:**
- Modify: `kws_de/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `balanced_calibration` (Task 5), `load_split(prefix)` (Task 4).
- Produces: `_evaluate_int8(name: str, model, X_train, y_train, X_test, y_test, *, seed: int, calib: np.ndarray) -> dict` with keys `name, float_acc, isolated_acc, catalog_acc, catalog_trials, params, macs, int8_bytes, budget_ok`; `evaluate_architecture(name, epochs=EPOCHS, seed=SEED, features="features") -> dict`; `render_table` shows a `Float` column (`-` if `float_acc` missing); CLI `kws-benchmark --features features_v3`.

- [ ] **Step 1: Write the failing test for the Float column**

```python
# append to tests/test_benchmark.py
def test_render_table_float_column_optional():
    base = {"catalog_acc": 0.5, "params": 1, "macs": 1, "int8_bytes": 1, "budget_ok": True}
    md = render_table([
        {"name": "a", "isolated_acc": 0.90, "float_acc": 0.95, **base},
        {"name": "b", "isolated_acc": 0.80, **base},
    ])
    assert "| Float |" in md
    assert "| 0.950 |" in md and "| - |" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/src/kws-de && uv run pytest tests/test_benchmark.py -v`
Expected: `test_render_table_float_column_optional` FAILS (`"| Float |"` not in md); the others pass.

- [ ] **Step 3: Update `render_table`, extract `_evaluate_int8`, add `features`**

`render_table`:

```python
def render_table(rows: list[dict]) -> str:
    """Pure Markdown comparison table. Float = Keras float32 test accuracy (- if the
    row predates it), Isolated = INT8 test accuracy, so the PTQ gap is a column."""
    header = "| Architecture | Float | Isolated | Catalog | Params | MACs | INT8 | Budget |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        budget = "yes" if r["budget_ok"] else "no"
        flt = f"{r['float_acc']:.3f}" if "float_acc" in r else "-"
        lines.append(
            f"| {r['name']} | {flt} | {r['isolated_acc']:.3f} | {r['catalog_acc']:.3f} | "
            f"{r['params']:,} | {r['macs']:,} | {r['int8_bytes']:,} | {budget} |"
        )
    return "\n".join(lines) + "\n"
```

New function (placed before `evaluate_architecture`), imports `_keras_predict` from `kws_de.eval` and `balanced_calibration` from `kws_de.export`:

```python
def _evaluate_int8(
    name: str, model, X_train, y_train, X_test, y_test, *, seed: int, calib
) -> dict:  # pragma: no cover - tflite + catalog TTS
    """Shared tail of every device-model evaluation: float test accuracy, INT8
    export on `calib`, INT8 isolated + catalog accuracy, on-device cost."""
    float_acc = float((_keras_predict(model, X_test) == y_test).mean())
    tflite_bytes = to_int8_tflite(model, calib)
    isolated_acc = float((_tflite_predict(tflite_bytes, X_test) == y_test).mean())
    catalog = run_catalog_eval(make_command_predict_fn(tflite_bytes), CATALOG_VOICES, seed=seed)
    try:
        check_budgets(tflite_bytes, model)
        budget_ok = True
    except AssertionError:
        budget_ok = False
    return {
        "name": name,
        "float_acc": float_acc,
        "isolated_acc": isolated_acc,
        "catalog_acc": catalog["overall_accuracy"],
        "catalog_trials": catalog["total_trials"],
        "params": int(model.count_params()),
        "macs": estimate_macs(model),
        "int8_bytes": len(tflite_bytes),
        "budget_ok": budget_ok,
    }
```

`evaluate_architecture` becomes:

```python
def evaluate_architecture(
    name: str, epochs: int = EPOCHS, seed: int = SEED, features: str = "features"
) -> dict:
    # pragma: no cover - heavy I/O (training + TTS + tflite)
    """Build `name`, train on `load_split("train", features)` (val-selected best
    epoch), then `_evaluate_int8` with a class-balanced calibration set."""
    import tensorflow as tf

    n_classes = len(config.COMMAND_LABELS)
    input_shape = (config.N_FRAMES, config.N_MFCC, 1)

    X_train, y_train, _ = load_split("train", features)
    X_val, y_val, _ = load_split("val", features)
    X_test, y_test, _ = load_split("test", features)

    tf.keras.utils.set_random_seed(seed)
    model = ARCHITECTURES[name](input_shape, n_classes=n_classes)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = str(Path(td) / "best.weights.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
        )
        model, _history = train(
            X_train,
            y_train,
            epochs=epochs,
            seed=seed,
            num_classes=n_classes,
            model=model,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint],
        )
        model.load_weights(ckpt_path)  # best val_accuracy epoch, not necessarily the last

    calib = balanced_calibration(X_train, y_train, seed=seed)
    return _evaluate_int8(name, model, X_train, y_train, X_test, y_test, seed=seed, calib=calib)
```

`main`: add `argparse` with `--features` (default `"features"`), pass it to `evaluate_architecture`, and mention the prefix in the intro line: `f"(`kws_de.dataset.load_split`, prefix `{args.features}`, seed=0)"`. Also replace `"Isolated** = INT8 test-set word accuracy"` with `"**Float** = Keras float32 test-set word accuracy, **Isolated** = INT8 test-set word accuracy (the gap is the PTQ cost)"`.

- [ ] **Step 4: Run tests, lint**

Run: `cd ~/src/kws-de && uv run pytest tests/test_benchmark.py -q && uv run ruff check kws_de/benchmark.py && uv run ruff format --check kws_de/benchmark.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd ~/src/kws-de && git add kws_de/benchmark.py tests/test_benchmark.py && git commit -m "refactor(benchmark): extract _evaluate_int8 (float+INT8 columns, balanced calib); --features"
```

---

### Task 7: Distillation — pure parts

**Files:**
- Create: `kws_de/distill.py`
- Test: `tests/test_distill.py`

**Interfaces:**
- Produces: `soften(p: np.ndarray, T: float) -> np.ndarray`; `distill_targets(y: np.ndarray, teacher_probs: np.ndarray, n_classes: int) -> np.ndarray` shape `(N, 2*n_classes)`; `make_distill_loss(n_classes: int, T: float, alpha: float)` → Keras-compatible `loss(y_true, y_pred)`; `hard_accuracy(n_classes)` → Keras metric fn named `accuracy`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_distill.py
import numpy as np
import tensorflow as tf

from kws_de.distill import distill_targets, hard_accuracy, make_distill_loss, soften


def test_soften_identity_at_T1_and_flatter_at_higher_T():
    p = np.array([[0.7, 0.2, 0.1]], np.float32)
    assert np.allclose(soften(p, 1.0), p, atol=1e-5)
    hot = soften(p, 4.0)
    assert np.isclose(hot.sum(), 1.0) and hot.max() < p.max()
    assert np.argmax(hot) == 0  # ordering preserved


def test_distill_targets_concatenates_one_hot_and_teacher():
    y = np.array([2, 0])
    t = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], np.float32)
    out = distill_targets(y, t, 3)
    assert out.shape == (2, 6)
    assert out[0, :3].tolist() == [0, 0, 1] and np.allclose(out[:, 3:], t)


def test_loss_alpha1_is_plain_cross_entropy():
    y = np.array([1, 0])
    t = np.array([[0.5, 0.5], [0.5, 0.5]], np.float32)
    pred = tf.constant([[0.2, 0.8], [0.9, 0.1]], tf.float32)
    y_true = tf.constant(distill_targets(y, t, 2))
    loss = make_distill_loss(2, T=4.0, alpha=1.0)(y_true, pred)
    ce = tf.keras.losses.sparse_categorical_crossentropy(tf.constant(y), pred)
    assert np.allclose(loss.numpy(), ce.numpy(), atol=1e-5)


def test_loss_kl_term_is_zero_when_student_matches_teacher():
    y = np.array([0])
    t = np.array([[0.3, 0.7]], np.float32)
    y_true = tf.constant(distill_targets(y, t, 2))
    pred = tf.constant(t)
    loss = make_distill_loss(2, T=2.0, alpha=0.0)(y_true, pred)
    assert abs(float(loss.numpy()[0])) < 1e-5


def test_hard_accuracy_reads_one_hot_half():
    y_true = tf.constant(distill_targets(np.array([1, 0]), np.zeros((2, 2), np.float32), 2))
    pred = tf.constant([[0.1, 0.9], [0.1, 0.9]], tf.float32)
    acc = hard_accuracy(2)(y_true, pred)
    assert np.isclose(float(tf.reduce_mean(acc)), 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/kws-de && uv run pytest tests/test_distill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kws_de.distill'`

- [ ] **Step 3: Implement**

```python
# kws_de/distill.py
"""Knowledge distillation (Hinton et al. 2015): a KWT teacher (accurate, not
device-runnable) -> the unchanged DS-CNN student (deployable).

Both models end in softmax, so "logits" are recovered as log(p): log-softmax
equals the logits up to a per-row constant, and softmax(log p / T) is exact
temperature scaling. The teacher is frozen, so its probabilities are computed
once and carried in the target tensor `[one_hot(y) | teacher_probs]`; the
student then trains with plain `model.fit`. No tfmot, no custom train step.
"""

import numpy as np
import tensorflow as tf

_EPS = 1e-9


def soften(p: np.ndarray, T: float) -> np.ndarray:
    """Temperature-scale a probability row-batch: softmax(log(p) / T)."""
    z = np.log(np.asarray(p, np.float64) + _EPS) / T
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def distill_targets(y: np.ndarray, teacher_probs: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, 2C): hard one-hot label followed by the teacher's (untempered) probs."""
    one_hot = np.eye(n_classes, dtype=np.float32)[np.asarray(y)]
    return np.concatenate([one_hot, np.asarray(teacher_probs, np.float32)], axis=1)


def _tf_soften(p, T):
    return tf.nn.softmax(tf.math.log(p + _EPS) / T, axis=-1)


def make_distill_loss(n_classes: int, T: float, alpha: float):
    """alpha * CE(hard, student) + (1 - alpha) * T^2 * KL(teacher_T || student_T)."""

    def loss(y_true, y_pred):
        hard = y_true[:, :n_classes]
        teacher = y_true[:, n_classes:]
        ce = tf.keras.losses.categorical_crossentropy(hard, y_pred)
        t_soft = _tf_soften(teacher, T)
        s_soft = _tf_soften(y_pred, T)
        kl = tf.reduce_sum(t_soft * (tf.math.log(t_soft + _EPS) - tf.math.log(s_soft + _EPS)), -1)
        return alpha * ce + (1.0 - alpha) * (T**2) * kl

    return loss


def hard_accuracy(n_classes: int):
    """Keras metric on the one-hot half, named `accuracy` so
    `ModelCheckpoint(monitor="val_accuracy")` keeps working."""

    def accuracy(y_true, y_pred):
        return tf.keras.metrics.categorical_accuracy(y_true[:, :n_classes], y_pred)

    return accuracy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/src/kws-de && uv run pytest tests/test_distill.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint and commit**

```bash
cd ~/src/kws-de && uv run ruff check kws_de/distill.py tests/test_distill.py && uv run ruff format kws_de/distill.py tests/test_distill.py && git add kws_de/distill.py tests/test_distill.py && git commit -m "feat(distill): soften, targets, KL+CE loss, hard-label metric (pure, tested)"
```

---

### Task 8: Distillation — `distill()` training and `kws-distill` CLI

**Files:**
- Modify: `kws_de/distill.py`
- Modify: `pyproject.toml:31-39` (scripts)
- Test: `tests/test_distill.py`

**Interfaces:**
- Consumes: Task 7 functions; `kws_de.train.train`; `kws_de.model.build_dscnn`; `kws_de.architectures.ARCHITECTURES["kwt"]`; `benchmark._evaluate_int8`, `benchmark.render_table`, `benchmark.CATALOG_VOICES`; `balanced_calibration`; `load_split(prefix)`.
- Produces: `distill(X, y, teacher, *, epochs, seed, T=4.0, alpha=0.5, validation_data=None, callbacks=None) -> (student, history)`; CLI `kws-distill --features features --epochs 40 --seed 0 --T 4 --alpha 0.5` writing `docs/distill-report.md` + `docs/distill-benchmark.json`.

- [ ] **Step 1: Write the failing toy-distillation test**

```python
# append to tests/test_distill.py
from kws_de import config
from kws_de.distill import distill


def _toy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.1, size=(n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    X[y == 1] += 2.0  # trivially separable
    return X, y


def test_distill_student_learns_separable_toy():
    X, y = _toy()
    # A "teacher" that is just a fixed lookup: returns confident-but-soft probs.
    class Teacher:
        def predict(self, Xc, verbose=0):
            hot = (Xc.reshape(len(Xc), -1).mean(1) > 1.0).astype(np.float32)
            return np.stack([1 - hot, hot], 1) * 0.8 + 0.1

    student, history = distill(X, y, Teacher(), epochs=8, seed=0, num_classes=2)
    assert history["accuracy"][-1] > 0.9
    preds = np.argmax(student.predict(X[..., None], verbose=0), 1)
    assert (preds == y).mean() > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/src/kws-de && uv run pytest tests/test_distill.py -v -k toy`
Expected: FAIL with `ImportError: cannot import name 'distill'`

- [ ] **Step 3: Implement `distill()` and `main()`**

Append to `kws_de/distill.py` (add `import argparse`, `import json`, `import tempfile`, `from pathlib import Path`, `from kws_de import config`, `from kws_de.model import build_dscnn` at the top, keeping ruff's import order):

```python
def distill(
    X,
    y,
    teacher,
    *,
    epochs: int,
    seed: int,
    num_classes: int | None = None,
    T: float = 4.0,
    alpha: float = 0.5,
    validation_data=None,
    callbacks=None,
):
    """Train a fresh `build_dscnn` student on (X, y) against `teacher`'s frozen
    probabilities. Mirrors `kws_de.train.train` (adam, batch 32, inverse-frequency
    balancing) but via `sample_weight`, since a 2-D target rules out `class_weight`."""
    tf.keras.utils.set_random_seed(seed)
    num_classes = num_classes if num_classes is not None else len(config.COMMAND_LABELS)
    Xc = np.asarray(X, np.float32)[..., None]
    y = np.asarray(y)
    targets = distill_targets(y, teacher.predict(Xc, verbose=0), num_classes)
    counts = np.bincount(y, minlength=num_classes)
    w_class = np.where(counts > 0, len(y) / (num_classes * np.maximum(counts, 1)), 0.0)
    sample_weight = w_class[y].astype(np.float32)

    student = build_dscnn(num_classes=num_classes)
    student.compile(
        optimizer="adam",
        loss=make_distill_loss(num_classes, T, alpha),
        metrics=[hard_accuracy(num_classes)],
    )
    if validation_data is not None:
        Xv, yv = validation_data
        Xv = np.asarray(Xv, np.float32)[..., None]
        validation_data = (Xv, distill_targets(yv, teacher.predict(Xv, verbose=0), num_classes))
    h = student.fit(
        Xc,
        targets,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    return student, h.history


def _best_val_checkpoint(fit):  # pragma: no cover - training I/O
    """Run `fit(callbacks)` with a best-val_accuracy ModelCheckpoint, reload it."""
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "best.weights.h5")
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            path, save_best_only=True, save_weights_only=True, monitor="val_accuracy", mode="max"
        )
        model, history = fit([ckpt])
        model.load_weights(path)
    return model, history


def main() -> None:  # pragma: no cover - training + TTS + tflite
    """`kws-distill`: on one split/seed, train KWT teacher, undistilled DS-CNN
    baseline, distilled DS-CNN; INT8-evaluate the two device models with a
    class-balanced calibration set, plus the baseline with the legacy
    `X_train[:200]` calibration so the PTQ recovery is one table row."""
    from kws_de.architectures import ARCHITECTURES
    from kws_de.benchmark import CATALOG_VOICES, _evaluate_int8, render_table
    from kws_de.dataset import load_split
    from kws_de.eval import _keras_predict
    from kws_de.export import balanced_calibration
    from kws_de.train import train

    ap = argparse.ArgumentParser(prog="kws-distill")
    ap.add_argument("--features", default="features", help="npz prefix (features | features_v3)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    n_classes = len(config.COMMAND_LABELS)
    X_train, y_train, _ = load_split("train", args.features)
    X_val, y_val, _ = load_split("val", args.features)
    X_test, y_test, _ = load_split("test", args.features)
    shape = (config.N_FRAMES, config.N_MFCC, 1)

    tf.keras.utils.set_random_seed(args.seed)
    teacher, _ = _best_val_checkpoint(
        lambda cb: train(
            X_train, y_train, epochs=args.epochs, seed=args.seed, num_classes=n_classes,
            model=ARCHITECTURES["kwt"](shape, n_classes=n_classes),
            validation_data=(X_val, y_val), callbacks=cb,
        )
    )
    teacher_acc = float((_keras_predict(teacher, X_test) == y_test).mean())

    baseline, _ = _best_val_checkpoint(
        lambda cb: train(
            X_train, y_train, epochs=args.epochs, seed=args.seed, num_classes=n_classes,
            validation_data=(X_val, y_val), callbacks=cb,
        )
    )
    student, _ = _best_val_checkpoint(
        lambda cb: distill(
            X_train, y_train, teacher, epochs=args.epochs, seed=args.seed,
            num_classes=n_classes, T=args.T, alpha=args.alpha,
            validation_data=(X_val, y_val), callbacks=cb,
        )
    )

    calib = balanced_calibration(X_train, y_train, seed=args.seed)
    common = dict(seed=args.seed)
    rows = [
        _evaluate_int8("ds_cnn (first-200 calib)", baseline, X_train, y_train, X_test, y_test,
                       calib=X_train[:200], **common),
        _evaluate_int8("ds_cnn (balanced calib)", baseline, X_train, y_train, X_test, y_test,
                       calib=calib, **common),
        _evaluate_int8("ds_cnn distilled (balanced calib)", student, X_train, y_train,
                       X_test, y_test, calib=calib, **common),
    ]

    repo_root = config.DATA_DIR.parent
    intro = (
        "# Distillation + INT8 calibration report\n\n"
        f"Dataset prefix `{args.features}`, epochs={args.epochs}, seed={args.seed}, "
        f"T={args.T}, alpha={args.alpha}. Teacher = KWT (reference-only, float): "
        f"test accuracy **{teacher_acc:.3f}**. Student = DS-CNN (unchanged "
        "`build_dscnn`). **Float** = Keras float32 test accuracy, **Isolated** = "
        "INT8 test accuracy, **Catalog** = full-intent catalog accuracy "
        f"({len(CATALOG_VOICES)} voices). Rows 1-2 differ only in the PTQ "
        "calibration set (`X_train[:200]` vs `kws_de.export.balanced_calibration`).\n\n"
    )
    (repo_root / "docs" / "distill-report.md").write_text(intro + render_table(rows))
    (repo_root / "docs" / "distill-benchmark.json").write_text(
        json.dumps({"teacher_acc": teacher_acc, "rows": rows}, indent=2, ensure_ascii=False) + "\n"
    )
    print("wrote docs/distill-report.md, docs/distill-benchmark.json")


if __name__ == "__main__":  # pragma: no cover
    main()
```

Add to `pyproject.toml` `[project.scripts]`: `kws-distill = "kws_de.distill:main"`.

Add `docs/distill-report.md` and `docs/distill-benchmark.json` to `.gitignore` only if `docs/transducer-report.md` is there too (check `git check-ignore docs/transducer-report.md`); otherwise leave them untracked like the transducer outputs.

- [ ] **Step 4: Run tests, lint, reinstall scripts**

Run: `cd ~/src/kws-de && uv sync && uv run pytest tests/test_distill.py -q && uv run ruff check kws_de/distill.py && uv run ruff format --check kws_de/distill.py && uv run kws-distill --help`
Expected: tests pass; help text prints.

- [ ] **Step 5: Commit**

```bash
cd ~/src/kws-de && git add kws_de/distill.py tests/test_distill.py pyproject.toml uv.lock && git commit -m "feat(distill): distill() trainer + kws-distill report (teacher, baseline, distilled, calib A/B)"
```

---

### Task 9: Run E9/E10 on the frozen v2 dataset; docs

**Files:**
- Modify: `README.md` (Data + Quick start), `docs/DATASHEET.md` (Collection), `docs/paper.md`, `docs/paper-notes.md`
- Modify: `docs/superpowers/specs/2026-09-01-real-speech-distill-design.md` §3.4 (DATASHEET is hand-written, not generated)

**Interfaces:**
- Consumes: `kws-distill` (Task 8). The v3 data run needs the 18 GB tarball + recordings the user provides; this task runs on the existing `features` (v2) split so E9/E10 land now and are re-run on v3 later.

- [ ] **Step 1: Run the distillation/calibration experiment on v2**

Run: `cd ~/src/kws-de && uv run kws-distill --features features --epochs 40 --seed 0 2>&1 | tail -5 && cat docs/distill-report.md`
Expected: report with teacher accuracy and 3 rows. Takes a while (KWT + 2×DS-CNN training + catalog TTS).

- [ ] **Step 2: Write results into the paper**

In `docs/paper.md`, after the E8 section, add `### E9 — Knowledge distillation (KWT → DS-CNN)` and `### E10 — INT8 calibration` with the table from `docs/distill-report.md` verbatim and one honest paragraph each: did distillation beat the baseline on Isolated/Catalog; how much of the 1.63 % PTQ gap balanced calibration recovered; the decision on QAT per the spec gate (> 1 % residual → next spec, else closed). Update the conclusion bullets accordingly. Reference `docs/superpowers/specs/2026-09-01-real-speech-distill-design.md`.

In `docs/paper-notes.md`, add `### E9/E10 — distillation + balanced calibration (feat/real-speech-distill)` with date, command line, the numbers, and the QAT decision.

- [ ] **Step 3: README + DATASHEET + spec fix**

README `## Data`: add the v3 path:

```markdown
**v3 (real speech):** download the MSWC German audio + splits tarballs
(<https://mlcommons.org/datasets/multilingual-spoken-words/>, CC-BY 4.0,
~18 GB) and extract to `data/mswc/de/` so it contains `clips/` and
`de_splits.csv`. Words no public corpus has (e.g. `Aufstelldach`, a rare
camper-hardware compound) are self-recorded: one word per file,
`data/recordings/<word>/<speaker>_<n>.wav` (phone voice memo is fine, any
sample rate; 5–10 speakers × ~10 takes, quiet and in-vehicle). Then:

    uv run kws-data --fetch --v3 --mswc-root data/mswc/de
    uv run kws-dataset build --seed 0 --cache raw_clips_v3.pkl --prefix features_v3
    uv run kws-benchmark --features features_v3
    uv run kws-distill --features features_v3
```

`docs/DATASHEET.md` `## Collection`: one paragraph describing the v3 path (tarball mining by keyword folder with VALID filter, self-recordings with `rec:` speaker ids, TTS as backstop only), with the note that the v3 real/TTS-per-word table is written here once the v3 build has run.

Spec §3.4: replace "generated by the existing datasheet code path" with "hand-written from `data/manifest_v3.json`".

- [ ] **Step 4: Lint docs and run the suite**

Run: `cd ~/src/kws-de && npx --yes markdownlint-cli@0.42.0 --config .markdownlint.json README.md docs/paper.md docs/paper-notes.md docs/DATASHEET.md && uv run pytest -q`
Expected: no lint output; all tests pass (`test_datasheet_has_required_sections` still green).

- [ ] **Step 5: Commit**

```bash
cd ~/src/kws-de && git add README.md docs/DATASHEET.md docs/paper.md docs/paper-notes.md docs/superpowers/specs/2026-09-01-real-speech-distill-design.md && git commit -m "docs: E9 distillation + E10 calibration results on v2; v3 real-speech data how-to"
```

---

## Self-review

- **Spec coverage:** §3.1 → Tasks 1, 2, 4; §3.2 → Tasks 3, 4, 9 (README); §3.3 → explicitly not built; §3.4 → Task 4 (+ Task 9 DATASHEET note); §4 → Tasks 6, 7, 8; §5 → Tasks 5, 6, 8 (calib A/B row); §6 constraints → Global Constraints; §7 tests → each task's tests; §8 → Task 9.
- **Placeholders:** none; every code step has full code. The v3 *run* (tarball + recordings) is user-provided input, stated as such.
- **Type consistency:** `mine(...)`/`load_recordings(...)` both return `dict[str, list[tuple[np.ndarray, str]]]`; `load_split(name, prefix)` positional prefix used in benchmark/distill; `_evaluate_int8(name, model, X_train, y_train, X_test, y_test, *, seed, calib)` matches its three call sites; `distill(..., num_classes=...)` used by the toy test and `main`.
