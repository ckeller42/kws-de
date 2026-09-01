# kws-de Model Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the German keyword-spotting model pipeline — data → MFCC → DS-CNN → INT8 export — with Mac + CI tests, on-device resource-budget gates, and a performance-evaluation report.

**Architecture:** A `uv`/Keras Python package. Pure, unit-testable stages (config, feature extraction, budgets, metrics) are separated from heavy I/O (dataset download, full training). CI runs the fast/pure tests plus INT8-export and budget gates against committed fixtures; full training runs locally and produces the committed model artifact.

**Tech Stack:** Python 3.11, TensorFlow/Keras (tf.lite for INT8), librosa (MFCC), HuggingFace `datasets` (MSWC pull), numpy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-31-kws-de-design.md`

## Global Constraints

- Python `>=3.11`; deps managed with `uv`; lint/format with `ruff` (line-length 100, lint `E,F,I,UP,B`).
- Commands (spoken, order fixed): `Licht, Kühlschrank, Camping, Heizung, Wasser`; label set = commands + `_unknown_` + `_silence_` (7 classes).
- Audio: 16 kHz mono, 1000 ms clips. MFCC: 30 ms window (480 samples), 20 ms hop (320 samples), 40 mel bins, 10 cepstra → feature shape `(49, 10)`.
- Exported model: full-INT8 (int8 input+output), all ops TFLM-supported. Budgets: model ≤ 500 000 bytes, tensor-arena ≤ 300 000 bytes, MACs/inference ≤ 3 000 000, estimated latency < 30 ms.
- Training data and model binaries are gitignored (`data/`, `models/`, `*.npy`, `*.tflite` except `tests/fixtures/**`). Data-fetch scripts are versioned; downloaded bytes are not.
- No machine-specific paths in code or docs (data lives under the repo's gitignored `data/`/`models/`; where those are stored is a local detail, not the project's concern).
- Commit style: end messages with the Co-Authored-By + Claude-Session trailers. Work on a branch → PR → CI + CodeRabbit → merge; never commit to `main` directly.

---

### Task 1: Package skeleton + config

**Files:**
- Create: `kws_de/__init__.py`
- Create: `kws_de/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `kws_de.config` with constants `SAMPLE_RATE=16000`, `CLIP_MS=1000`, `CLIP_SAMPLES=16000`, `WIN_SAMPLES=480`, `HOP_SAMPLES=320`, `N_MELS=40`, `N_MFCC=10`, `N_FRAMES=49`, `COMMANDS: list[str]`, `LABELS: list[str]`, `NUM_CLASSES=7`; budget constants `MAX_MODEL_BYTES`, `MAX_ARENA_BYTES`, `MAX_MACS`, `MAX_LATENCY_MS`; `DATA_DIR`, `MODELS_DIR` (`pathlib.Path`, relative to the repo root); `label_index(label: str) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from kws_de import config

def test_labels_are_commands_plus_aux():
    assert config.COMMANDS == ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
    assert config.LABELS == config.COMMANDS + ["_unknown_", "_silence_"]
    assert config.NUM_CLASSES == 7

def test_label_index_roundtrip():
    for i, lbl in enumerate(config.LABELS):
        assert config.label_index(lbl) == i

def test_frame_count_matches_audio_geometry():
    # (CLIP_SAMPLES - WIN) // HOP + 1
    expected = (config.CLIP_SAMPLES - config.WIN_SAMPLES) // config.HOP_SAMPLES + 1
    assert config.N_FRAMES == expected

def test_data_dirs_are_under_repo_root():
    # Paths are repo-relative — no machine-specific absolute locations.
    assert config.DATA_DIR.name == "data"
    assert config.MODELS_DIR.name == "models"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: kws_de.config`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/__init__.py
"""German keyword-spotting model pipeline for ESP32-S3."""

# kws_de/config.py
from pathlib import Path

SAMPLE_RATE = 16000
CLIP_MS = 1000
CLIP_SAMPLES = SAMPLE_RATE * CLIP_MS // 1000  # 16000
WIN_SAMPLES = 480   # 30 ms
HOP_SAMPLES = 320   # 20 ms
N_MELS = 40
N_MFCC = 10
N_FRAMES = (CLIP_SAMPLES - WIN_SAMPLES) // HOP_SAMPLES + 1  # 49

COMMANDS = ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
LABELS = COMMANDS + ["_unknown_", "_silence_"]
NUM_CLASSES = len(LABELS)  # 7

# On-device resource budgets (see spec Global Constraints).
MAX_MODEL_BYTES = 500_000
MAX_ARENA_BYTES = 300_000
MAX_MACS = 3_000_000
MAX_LATENCY_MS = 30

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"      # gitignored; where it physically lives is a local detail
MODELS_DIR = _REPO_ROOT / "models"  # gitignored

def label_index(label: str) -> int:
    return LABELS.index(label)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add kws_de/__init__.py kws_de/config.py tests/test_config.py
git commit -m "feat: package skeleton + config (labels, audio geometry, budgets)"
```

---

### Task 2: MFCC feature front-end + golden vectors

**Files:**
- Create: `kws_de/features.py`
- Create: `tests/fixtures/mfcc_golden.npz` (generated in Step 3)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `kws_de.config` (audio + MFCC params).
- Produces: `mfcc(samples: np.ndarray) -> np.ndarray` returning float32 shape `(N_FRAMES, N_MFCC)`; input is 1-D float32 in [-1, 1], padded/truncated to `CLIP_SAMPLES`. Deterministic — same input always yields identical output (the host↔device golden anchor).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
from kws_de import config
from kws_de.features import mfcc

def test_shape_and_dtype():
    x = np.zeros(config.CLIP_SAMPLES, dtype=np.float32)
    out = mfcc(x)
    assert out.shape == (config.N_FRAMES, config.N_MFCC)
    assert out.dtype == np.float32

def test_pads_and_truncates():
    short = np.ones(1000, dtype=np.float32)
    long = np.ones(config.CLIP_SAMPLES * 2, dtype=np.float32)
    assert mfcc(short).shape == (config.N_FRAMES, config.N_MFCC)
    assert mfcc(long).shape == (config.N_FRAMES, config.N_MFCC)

def test_deterministic_against_golden():
    # A fixed 440 Hz tone must always produce the same coefficients (host/device anchor).
    t = np.arange(config.CLIP_SAMPLES) / config.SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = mfcc(tone)
    golden = np.load("tests/fixtures/mfcc_golden.npz")["mfcc"]
    np.testing.assert_allclose(out, golden, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.features` (and missing fixture).

- [ ] **Step 3: Write minimal implementation + generate the golden fixture**

```python
# kws_de/features.py
import librosa
import numpy as np
from kws_de import config

def _fit_length(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).ravel()
    if x.shape[0] < config.CLIP_SAMPLES:
        x = np.pad(x, (0, config.CLIP_SAMPLES - x.shape[0]))
    return x[: config.CLIP_SAMPLES]

def mfcc(samples: np.ndarray) -> np.ndarray:
    x = _fit_length(samples)
    m = librosa.feature.mfcc(
        y=x, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.WIN_SAMPLES, hop_length=config.HOP_SAMPLES,
        n_mels=config.N_MELS, center=False,
    )  # shape (N_MFCC, frames)
    out = m.T.astype(np.float32)  # (frames, N_MFCC)
    return out[: config.N_FRAMES]
```

Generate the golden fixture once (this defines the contract the device MFCC must match):

```bash
uv run python -c "
import numpy as np; from kws_de import config; from kws_de.features import mfcc
t = np.arange(config.CLIP_SAMPLES)/config.SAMPLE_RATE
tone = (0.5*np.sin(2*np.pi*440*t)).astype(np.float32)
np.savez('tests/fixtures/mfcc_golden.npz', mfcc=mfcc(tone))
print('wrote golden', mfcc(tone).shape)
"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add kws_de/features.py tests/test_features.py tests/fixtures/mfcc_golden.npz
git commit -m "feat: MFCC front-end + golden-vector fixture (host/device anchor)"
```

---

### Task 3: Noise augmentation (SNR mixing)

**Files:**
- Create: `kws_de/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: `kws_de.config`.
- Produces: `mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray` — returns `signal` with `noise` added so the measured SNR equals `snr_db` (noise tiled/cropped to length). `measure_snr(signal, noisy) -> float` helper returns dB.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
from kws_de.augment import mix_at_snr, measure_snr

def test_mix_hits_target_snr():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000).astype(np.float32)
    noise = rng.standard_normal(4000).astype(np.float32)  # shorter -> must tile
    for target in (20.0, 10.0, 0.0):
        noisy = mix_at_snr(sig, noise, target, rng)
        assert noisy.shape == sig.shape
        assert abs(measure_snr(sig, noisy) - target) < 0.5

def test_zero_signal_is_safe():
    rng = np.random.default_rng(1)
    sig = np.zeros(16000, dtype=np.float32)
    noise = rng.standard_normal(16000).astype(np.float32)
    out = mix_at_snr(sig, noise, 10.0, rng)
    assert out.shape == sig.shape and np.all(np.isfinite(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_augment.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.augment`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/augment.py
import numpy as np

def _tile_to(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if noise.shape[0] < n:
        reps = int(np.ceil(n / noise.shape[0]))
        noise = np.tile(noise, reps)
    start = 0 if noise.shape[0] == n else int(rng.integers(0, noise.shape[0] - n + 1))
    return noise[start : start + n]

def mix_at_snr(signal, noise, snr_db, rng):
    sig = np.asarray(signal, dtype=np.float32)
    nz = _tile_to(np.asarray(noise, dtype=np.float32), sig.shape[0], rng)
    p_sig = float(np.mean(sig**2))
    p_nz = float(np.mean(nz**2)) or 1e-12
    if p_sig <= 1e-12:  # silence: return scaled noise at a fixed level
        return (nz / np.sqrt(p_nz) * 0.01).astype(np.float32)
    gain = np.sqrt(p_sig / (p_nz * (10 ** (snr_db / 10))))
    return (sig + gain * nz).astype(np.float32)

def measure_snr(signal, noisy):
    sig = np.asarray(signal, dtype=np.float32)
    nz = np.asarray(noisy, dtype=np.float32) - sig
    p_sig = float(np.mean(sig**2)); p_nz = float(np.mean(nz**2)) or 1e-12
    return 10 * np.log10(p_sig / p_nz)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_augment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add kws_de/augment.py tests/test_augment.py
git commit -m "feat: SNR-targeted noise mixing for augmentation"
```

---

### Task 4: Dataset assembly + MSWC/ESC-50 fetch

**Files:**
- Create: `kws_de/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `kws_de.config`, `kws_de.features.mfcc`, `kws_de.augment`.
- Produces:
  - `build_dataset(clips: dict[str, list[np.ndarray]], noises: list[np.ndarray], rng, snrs=(20,10,0)) -> tuple[np.ndarray, np.ndarray]` — pure: turns per-label raw clips into `(X, y)` where `X` is float32 `(N, N_FRAMES, N_MFCC)` and `y` is int labels; commands get noise-augmented copies at each SNR, `_silence_` from noise-only, `_unknown_` from clips under key `_unknown_`.
  - `main()` — CLI entry (`kws-data --fetch`): downloads the MSWC German per-keyword subset (HF `datasets` streaming) + ESC-50, caches features to `DATA_DIR`. Thin I/O wrapper, not unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import numpy as np
from kws_de import config
from kws_de.data import build_dataset

def _clip(rng): return rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)

def test_build_dataset_shapes_and_labels():
    rng = np.random.default_rng(0)
    clips = {c: [_clip(rng) for _ in range(3)] for c in config.COMMANDS}
    clips["_unknown_"] = [_clip(rng) for _ in range(4)]
    noises = [rng.standard_normal(8000).astype(np.float32) for _ in range(2)]
    X, y = build_dataset(clips, noises, rng, snrs=(20, 0))
    assert X.ndim == 3 and X.shape[1:] == (config.N_FRAMES, config.N_MFCC)
    assert X.shape[0] == y.shape[0]
    assert set(np.unique(y)).issubset(set(range(config.NUM_CLASSES)))
    # _silence_ class must be present (built from noise)
    assert config.label_index("_silence_") in set(y.tolist())

def test_commands_are_augmented_per_snr():
    rng = np.random.default_rng(1)
    clips = {c: [_clip(rng)] for c in config.COMMANDS}
    clips["_unknown_"] = [_clip(rng)]
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X2, y2 = build_dataset(clips, noises, rng, snrs=(20, 10))
    X1, y1 = build_dataset(clips, noises, rng, snrs=(20,))
    licht = config.label_index("Licht")
    assert (y2 == licht).sum() == 2 * (y1 == licht).sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.data`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/data.py
import argparse
import numpy as np
from kws_de import config
from kws_de.augment import mix_at_snr
from kws_de.features import mfcc

def build_dataset(clips, noises, rng, snrs=(20, 10, 0)):
    X, y = [], []
    def add(sig, label):
        X.append(mfcc(sig)); y.append(config.label_index(label))
    for cmd in config.COMMANDS:
        for clip in clips.get(cmd, []):
            add(clip, cmd)  # clean
            for snr in snrs:
                noise = noises[int(rng.integers(0, len(noises)))]
                add(mix_at_snr(clip, noise, snr, rng), cmd)
    for clip in clips.get("_unknown_", []):
        add(clip, "_unknown_")
    n_sil = max(1, len(clips.get("_unknown_", [])))
    for _ in range(n_sil):
        noise = noises[int(rng.integers(0, len(noises)))]
        sil = mix_at_snr(np.zeros(config.CLIP_SAMPLES, np.float32), noise, 0.0, rng)
        add(sil, "_silence_")
    return np.asarray(X, np.float32), np.asarray(y, np.int64)

def main() -> None:  # pragma: no cover - thin I/O wrapper (manual/integration)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        _fetch_and_cache()

def _fetch_and_cache() -> None:  # pragma: no cover
    # MSWC German per-keyword subset via HF datasets streaming + ESC-50 noise.
    # from datasets import load_dataset
    #   ds = load_dataset("MLCommons/ml_spoken_words", "de_wav", streaming=True, split="train")
    #   keep clips whose "keyword" matches our commands (+ a sample for _unknown_)
    # Cache raw clips and/or extracted features under config.DATA_DIR.
    raise NotImplementedError("wire MSWC + ESC-50 fetch; see spec §4")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add kws_de/data.py tests/test_data.py
git commit -m "feat: dataset assembly (augmented commands + silence/unknown)"
```

---

### Task 5: DS-CNN model

**Files:**
- Create: `kws_de/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `kws_de.config`.
- Produces: `build_dscnn() -> tf.keras.Model` — input `(N_FRAMES, N_MFCC, 1)`, output `NUM_CLASSES` softmax; depthwise-separable conv stack; ≤ 40 000 params.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
from kws_de import config
from kws_de.model import build_dscnn

def test_output_shape_and_params():
    m = build_dscnn()
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 40_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.model`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/model.py
import tensorflow as tf
from kws_de import config

def build_dscnn() -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input((config.N_FRAMES, config.N_MFCC, 1))
    x = L.Conv2D(32, (3, 3), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    for _ in range(3):
        x = L.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x); x = L.ReLU()(x)
        x = L.Conv2D(32, (1, 1), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="dscnn_kws_de")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kws_de/model.py tests/test_model.py
git commit -m "feat: small DS-CNN model (<40k params)"
```

---

### Task 6: Training + smoke convergence

**Files:**
- Create: `kws_de/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `kws_de.model.build_dscnn`, `kws_de.config`.
- Produces: `train(X, y, epochs=..., seed=0) -> tuple[tf.keras.Model, dict]` returning the fitted model + history dict; reshapes `X` to `(...,1)`, compiles sparse-categorical. `main()` — CLI (`kws-train`) loads cached features, trains, saves `models/kws.keras`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
from kws_de import config
from kws_de.train import train

def test_smoke_overfits_tiny_separable_data():
    rng = np.random.default_rng(0)
    # Two easily-separable clusters mapped to 2 classes -> accuracy must beat chance.
    n = 60
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % config.NUM_CLASSES).astype(np.int64)
    X += y[:, None, None]  # inject class-dependent shift so it's learnable
    model, hist = train(X, y, epochs=8, seed=0)
    assert hist["accuracy"][-1] > hist["accuracy"][0]  # learning happened
    assert hist["accuracy"][-1] > 1.5 / config.NUM_CLASSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.train`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/train.py
import argparse
import numpy as np
import tensorflow as tf
from kws_de import config
from kws_de.model import build_dscnn

def train(X, y, epochs=30, seed=0):
    tf.keras.utils.set_random_seed(seed)
    X = np.asarray(X, np.float32)[..., None]
    model = build_dscnn()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(X, np.asarray(y), epochs=epochs, batch_size=32, verbose=0)
    return model, h.history

def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(config.DATA_DIR / "features.npz")
    model, _ = train(X["X"], X["y"], epochs=args.epochs)
    model.save(config.MODELS_DIR / "kws.keras")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kws_de/train.py tests/test_train.py
git commit -m "feat: training loop + smoke convergence test"
```

---

### Task 7: INT8 export → tflite + C array + metadata

**Files:**
- Create: `kws_de/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `kws_de.config`.
- Produces:
  - `to_int8_tflite(model, rep_samples: np.ndarray) -> bytes` — full-INT8 conversion (int8 in/out) using `rep_samples` `(M, N_FRAMES, N_MFCC)` as representative data.
  - `write_c_array(tflite: bytes, path) -> None` — emits `model_data.h` (`const unsigned char g_model[]` + length).
  - `write_metadata(path) -> None` — labels + MFCC params + budgets as JSON.
  - `main()` — CLI (`kws-export`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
import numpy as np
import tensorflow as tf
from kws_de import config
from kws_de.model import build_dscnn
from kws_de.export import to_int8_tflite, write_c_array

def test_export_is_full_int8_and_runs(tmp_path):
    rng = np.random.default_rng(0)
    model = build_dscnn()
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(model, rep)
    itp = tf.lite.Interpreter(model_content=blob); itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8
    x = rng.integers(-128, 127, size=inp["shape"], dtype=np.int8)
    itp.set_tensor(inp["index"], x); itp.invoke()
    assert itp.get_tensor(out["index"]).shape[-1] == config.NUM_CLASSES

def test_c_array_header(tmp_path):
    p = tmp_path / "model_data.h"
    write_c_array(b"\x01\x02\x03", p)
    txt = p.read_text()
    assert "g_model[]" in txt and "g_model_len = 3" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.export`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/export.py
import argparse
import json
import numpy as np
import tensorflow as tf
from kws_de import config

def to_int8_tflite(model, rep_samples) -> bytes:
    rep = np.asarray(rep_samples, np.float32)[..., None]
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

def write_c_array(tflite: bytes, path) -> None:
    body = ", ".join(str(b) for b in tflite)
    with open(path, "w") as fh:
        fh.write("// Auto-generated. Do not edit.\n")
        fh.write(f"const unsigned char g_model[] = {{{body}}};\n")
        fh.write(f"const unsigned int g_model_len = {len(tflite)};\n")

def write_metadata(path) -> None:
    meta = {"labels": config.LABELS,
            "mfcc": {"n_mfcc": config.N_MFCC, "n_frames": config.N_FRAMES,
                     "win": config.WIN_SAMPLES, "hop": config.HOP_SAMPLES,
                     "n_mels": config.N_MELS, "sample_rate": config.SAMPLE_RATE},
            "budgets": {"model_bytes": config.MAX_MODEL_BYTES,
                        "arena_bytes": config.MAX_ARENA_BYTES, "macs": config.MAX_MACS}}
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(config.MODELS_DIR))
    args = ap.parse_args()
    import pathlib
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(config.MODELS_DIR / "kws.keras")
    feats = np.load(config.DATA_DIR / "features.npz")["X"][:100]
    blob = to_int8_tflite(model, feats)
    (out / "model.tflite").write_bytes(blob)
    write_c_array(blob, out / "model_data.h")
    write_metadata(out / "metadata.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add kws_de/export.py tests/test_export.py
git commit -m "feat: INT8 tflite export + C-array header + metadata"
```

---

### Task 8: On-device budget gates

**Files:**
- Create: `kws_de/budgets.py`
- Test: `tests/test_budgets.py`

**Interfaces:**
- Consumes: `kws_de.config`.
- Produces:
  - `estimate_macs(model) -> int` — sum of MACs over Conv2D/DepthwiseConv2D/Dense.
  - `tflite_op_types(tflite: bytes) -> set[str]` — op names via a temp interpreter.
  - `is_full_int8(tflite: bytes) -> bool` — input+output int8.
  - `check_budgets(tflite: bytes, model) -> dict` — returns `{model_bytes, macs, int8, ops}` and raises `AssertionError` if any budget is exceeded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budgets.py
import numpy as np
from kws_de import config
from kws_de.model import build_dscnn
from kws_de.export import to_int8_tflite
from kws_de.budgets import estimate_macs, is_full_int8, check_budgets

def test_macs_and_budgets_pass_for_small_model():
    m = build_dscnn()
    assert 0 < estimate_macs(m) <= config.MAX_MACS
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)
    assert is_full_int8(blob)
    report = check_budgets(blob, m)
    assert report["model_bytes"] <= config.MAX_MODEL_BYTES
    assert report["int8"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budgets.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.budgets`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/budgets.py
import numpy as np
import tensorflow as tf
from kws_de import config

def estimate_macs(model) -> int:
    total = 0
    for layer in model.layers:
        cfg = getattr(layer, "output_shape", None)
        if isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, cout = layer.output_shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            cin = layer.input_shape[-1]
            total += h * w * cout * cin * k
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            _, h, w, c = layer.output_shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            total += h * w * c * k
        elif isinstance(layer, tf.keras.layers.Dense):
            total += layer.input_shape[-1] * layer.units
    return int(total)

def _interp(tflite: bytes):
    itp = tf.lite.Interpreter(model_content=tflite); itp.allocate_tensors(); return itp

def tflite_op_types(tflite: bytes) -> set:
    itp = _interp(tflite)
    return {d["op_name"] for d in itp._get_ops_details()}  # noqa: SLF001

def is_full_int8(tflite: bytes) -> bool:
    itp = _interp(tflite)
    return (itp.get_input_details()[0]["dtype"] == np.int8
            and itp.get_output_details()[0]["dtype"] == np.int8)

def check_budgets(tflite: bytes, model) -> dict:
    report = {"model_bytes": len(tflite), "macs": estimate_macs(model),
              "int8": is_full_int8(tflite), "ops": sorted(tflite_op_types(tflite))}
    assert report["model_bytes"] <= config.MAX_MODEL_BYTES, "model too large"
    assert report["macs"] <= config.MAX_MACS, "MAC budget exceeded"
    assert report["int8"], "model is not full-INT8"
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_budgets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kws_de/budgets.py tests/test_budgets.py
git commit -m "feat: on-device budget gates (size/MACs/INT8/ops)"
```

---

### Task 9: Evaluation metrics + report

**Files:**
- Create: `kws_de/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `kws_de.config`.
- Produces:
  - `metrics(y_true, y_pred) -> dict` — accuracy, per-class accuracy, `_unknown_` false-accept rate (fraction of true-`_unknown_` predicted as a command), confusion matrix (`np.ndarray`).
  - `snr_sweep(eval_fn, snrs) -> dict[float, float]` — maps SNR → accuracy using a caller-supplied `eval_fn(snr) -> float`.
  - `render_report(results: dict) -> str` — Markdown.
  - `main()` — CLI (`kws-eval`) writing `docs/eval-report.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
import numpy as np
from kws_de import config
from kws_de.eval import metrics, snr_sweep, render_report

def test_metrics_perfect_and_unknown_fa():
    y = np.array([config.label_index(l) for l in
                  ["Licht", "Wasser", "_unknown_", "_silence_"]])
    perfect = metrics(y, y.copy())
    assert perfect["accuracy"] == 1.0
    assert perfect["unknown_false_accept"] == 0.0
    # unknown predicted as a command -> false accept = 1.0
    yp = y.copy(); yp[2] = config.label_index("Licht")
    assert metrics(y, yp)["unknown_false_accept"] == 1.0

def test_snr_sweep_and_report():
    sweep = snr_sweep(lambda s: 0.9 if s >= 10 else 0.5, [20, 10, 0])
    assert sweep[20] == 0.9 and sweep[0] == 0.5
    md = render_report({"accuracy": 0.9, "snr_sweep": sweep})
    assert "Accuracy" in md and "SNR" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: kws_de.eval`.

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/eval.py
import argparse
import numpy as np
from kws_de import config

def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = config.NUM_CLASSES
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    per_class = {config.LABELS[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0)
                 for i in range(n)}
    unk = config.label_index("_unknown_")
    cmd_idx = {config.label_index(c) for c in config.COMMANDS}
    unk_mask = y_true == unk
    fa = (float(np.mean([p in cmd_idx for p in y_pred[unk_mask]]))
          if unk_mask.any() else 0.0)
    return {"accuracy": acc, "per_class": per_class,
            "unknown_false_accept": fa, "confusion": cm}

def snr_sweep(eval_fn, snrs) -> dict:
    return {float(s): float(eval_fn(s)) for s in snrs}

def render_report(results: dict) -> str:
    lines = ["# kws-de Evaluation Report", "",
             f"**Accuracy:** {results.get('accuracy', 0):.3f}", "", "## SNR sweep", "",
             "| SNR (dB) | Accuracy |", "|---|---|"]
    for snr, acc in sorted(results.get("snr_sweep", {}).items(), reverse=True):
        lines.append(f"| {snr:.0f} | {acc:.3f} |")
    return "\n".join(lines) + "\n"

def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/eval-report.md")
    args = ap.parse_args()
    # load model + held-out features, compute metrics + SNR sweep, then:
    # open(args.out, "w").write(render_report(results))
    raise NotImplementedError("wire held-out eval; see spec §9")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add kws_de/eval.py tests/test_eval.py
git commit -m "feat: eval metrics (accuracy, unknown-FA, SNR sweep) + report"
```

---

### Task 10: CI wiring — coverage gate on pure modules

**Files:**
- Modify: `.github/workflows/ci.yml` (already lists `kws_de.features`, `kws_de.budgets`, `kws_de.data` in the coverage gate — verify it stays green with the real modules)
- Test: the full suite runs in CI.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: green CI (test + markdownlint + gitleaks).

- [ ] **Step 1: Run the full suite locally**

Run: `uv run pytest -q`
Expected: PASS (all tasks' tests).

- [ ] **Step 2: Run the exact CI coverage command locally**

Run:
```bash
uv run pytest -q --cov=kws_de.features --cov=kws_de.budgets --cov=kws_de.data --cov-fail-under=85
```
Expected: PASS with coverage ≥ 85% on those three modules. If a module is under 85%, add a focused unit test for the uncovered pure branch (not the `# pragma: no cover` I/O wrappers).

- [ ] **Step 3: Lint + format check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean (fix with `uv run ruff format .` if needed).

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin <branch>
gh pr create --fill
```
Expected: CI (test, markdownlint, gitleaks) green; wait for CodeRabbit; address feedback; then merge.

- [ ] **Step 5: Commit any CI fixes**

```bash
git add -A && git commit -m "ci: keep coverage gate green on real modules"
```

---

## Self-Review

**Spec coverage:**
- §2 units → Tasks 1–9 (config, features, augment, data, model, train, export, budgets, eval). ✅
- §3 command set → Task 1 config `COMMANDS`/`LABELS`. ✅
- §4 data (recipe versioned, MSWC subset) → Task 4 (`build_dataset` pure + `_fetch_and_cache` I/O). ✅
- §5 model (DS-CNN) → Task 5. ✅
- §6 export (INT8 + C array + metadata) → Task 7. ✅
- §7 testing + budget gates + golden vectors → Tasks 2 (golden), 8 (budgets), 10 (CI). ✅
- §9 eval (accuracy, unknown-FA, SNR sweep, report) → Task 9. ✅
- §8 firmware demo, §13 Sphinx docs → **separate plans** (independent subsystems; noted below). Intentional gap.
- Budget: tensor-arena and latency budgets — `check_budgets` covers size/MACs/INT8/ops; arena + latency are estimated on-device (firmware plan) since TFLM arena sizing needs the micro interpreter. Noted as a known limitation carried to the firmware plan.

**Placeholder scan:** `_fetch_and_cache`, `train.main`, `export.main`, `eval.main` raise `NotImplementedError`/are `# pragma: no cover` I/O wrappers by design (integration glue, not unit logic) — their pure cores are fully implemented and tested. No hidden TODOs in tested code.

**Type consistency:** `mfcc` returns `(N_FRAMES, N_MFCC)`; `build_dataset` stacks to `(N, N_FRAMES, N_MFCC)`; `train`/`export` add the trailing channel `(...,1)`. `to_int8_tflite(model, rep_samples)` and `check_budgets(tflite, model)` signatures match their tests. Labels/indices flow through `config.label_index` everywhere. Consistent.

## Out of scope (follow-on plans)

- **Firmware demo** (`firmware/`, ESP-IDF C: CoreS3 BSP mic → AFE → MFCC → TFLM+ESP-NN → LVGL label). Built/flashed on buspi; consumes `model_data.h` + `metadata.json` from Task 7. Its own plan — includes the on-device tensor-arena + latency measurement that closes the two budgets this plan estimates only.
- **Sphinx + sphinx-likec4 docs** (`docs/conf.py`, `docs/likec4/*.c4`, algorithms prose). Its own plan (node ≥ 20 build).
