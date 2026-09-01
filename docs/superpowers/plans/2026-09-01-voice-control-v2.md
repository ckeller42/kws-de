# Voice Control v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add wake-word ("Hey Bus") + streaming slot commands (`<device> [zone] <action>`) on top of the v1 KWS: a tiny always-on wake model, a streaming keyword detector, a pure grammar slot-filler, retrained command recogniser, INT8 exports, and an eval report (wake false-accepts/hour + full-intent accuracy).

**Architecture:** Two stages. A tiny always-on wake model gates a streaming command recogniser; a ring-buffer streaming detector turns posteriors into discrete keyword events; a pure grammar composes the events into a validated intent. Reuses v1's `config`/`features`/`augment`/`model`/`budgets`/`data` code.

**Tech Stack:** Python 3.11, TensorFlow/Keras (tf.lite INT8), librosa, numpy, pytest, ruff — same as v1. **Wake stage reuses [microWakeWord](https://github.com/kahrendt/microWakeWord)** (+ Piper TTS for its "Hey Bus" training samples) instead of a from-scratch model — see spec §12 (prior art & reuse decisions).

**Spec:** `docs/superpowers/specs/2026-09-01-voice-control-v2-design.md` — read §12 for why the wake stage is microWakeWord and the command→intent stage is ours.

## Global Constraints

- Python `>=3.11`; `uv`; `ruff` (line-length 100, lint `E,F,I,UP,B`). Keep the full suite green.
- Reuse v1 audio/MFCC constants (16 kHz, 1000 ms, 30/20 ms MFCC → `(49, 10)`); the streaming path uses the SAME `kws_de.features.mfcc` (host↔device golden-vector contract).
- v2 vocab (GROUNDED in the real controllable functions): wake `Hey Bus`; devices `Licht, Kühlschrank, Heizung, Aufstelldach, Campingmodus, USB, Wasser, Energie`; light zones `Küche, Dach, Außen, Lesen` (apply to `Licht` only); actions `an, aus, auf, zu, heller, dunkler, wärmer, kälter, leise, Eco, Max, Normal`. Command label set = devices + zones + actions + `_unknown_` + `_silence_` (26 classes). Wake label set = `wake` + `_not_`. `Kühlschrank` is the spoken word for the fridge/cooler function (better real MSWC coverage than the app's compound term).
- Per-device allowed actions are FIXED by `DEVICE_ACTIONS` (Task 1); only `ZONED_DEVICES = ["Licht"]` take a zone. The grammar (Task 2) is device-specific: it rejects actions not valid for the device (e.g. "Aufstelldach an") and zones on non-zoned devices (e.g. "Heizung Küche").
- v1 config constants (`COMMANDS`, `LABELS`, `NUM_CLASSES=7`) MUST stay intact and v1 tests stay green — v2 vocab is ADDITIVE (new constants), not a rewrite.
- Grammar order: `device → zone? → action`. Bare `device action` (no zone) is valid. Missing device or action, out-of-order, duplicate slots, action-invalid-for-device, or zone-on-non-zoned-device → rejection.
- Exports full-INT8; budgets (per v1): command model ≤ 500 000 B / ≤ 3 000 000 MACs; wake model held tighter (≤ 150 000 B, ≤ 1 000 000 MACs) as it is always-on.
- The wake stage is **microWakeWord** — a proven S3-native streaming wake library (spec §12). We add it as a dependency and use its trainer (Piper TTS) for "Hey Bus"; we do NOT hand-roll a wake model. Its output is a TFLite-Micro streaming model we budget-check like any other. The command→intent stage stays ours (no open MCU equivalent exists).
- No training data / model binaries committed (gitignored). No machine-specific paths. No third-party-app or decompile provenance anywhere in the repo (public repo).
- Commit trailers: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM`. Work on a branch → PR → CI + CodeRabbit → merge.

---

### Task 1: v2 vocab config (additive)

**Files:**
- Modify: `kws_de/config.py`
- Test: `tests/test_config_v2.py`

**Interfaces:**
- Produces (added to `kws_de.config`): `WAKE_WORD="Hey Bus"`, `WAKE_LABELS=["wake","_not_"]`, `DEVICES`, `ZONES`, `ACTIONS` (lists), `COMMAND_LABELS = DEVICES + ZONES + ACTIONS + ["_unknown_","_silence_"]`, `command_index(label)->int`. Existing v1 constants unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_v2.py
from kws_de import config

def test_v2_vocab():
    assert config.WAKE_WORD == "Hey Bus"
    assert config.WAKE_LABELS == ["wake", "_not_"]
    assert config.DEVICES == ["Licht", "Kühlschrank", "Heizung", "Aufstelldach",
                              "Campingmodus", "USB", "Wasser", "Energie"]
    assert config.ZONES == ["Küche", "Dach", "Außen", "Lesen"]
    assert "auf" in config.ACTIONS and "heller" in config.ACTIONS and "Eco" in config.ACTIONS
    assert config.ZONED_DEVICES == ["Licht"]
    assert config.COMMAND_LABELS == (
        config.DEVICES + config.ZONES + config.ACTIONS + ["_unknown_", "_silence_"]
    )

def test_device_actions_cover_all_devices_and_use_valid_actions():
    assert set(config.DEVICE_ACTIONS) == set(config.DEVICES)
    for dev, acts in config.DEVICE_ACTIONS.items():
        assert acts and all(a in config.ACTIONS for a in acts)

def test_command_index_roundtrip():
    for i, lbl in enumerate(config.COMMAND_LABELS):
        assert config.command_index(lbl) == i

def test_v1_constants_unchanged():
    assert config.COMMANDS == ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
    assert config.NUM_CLASSES == 7
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_config_v2.py -v` → FAIL (`AttributeError: WAKE_WORD`).

- [ ] **Step 3: Write minimal implementation** (append to `kws_de/config.py`)

```python
# --- v2: wake word + slot commands (additive; v1 constants above untouched) ---
WAKE_WORD = "Hey Bus"
WAKE_LABELS = ["wake", "_not_"]
DEVICES = ["Licht", "Kühlschrank", "Heizung", "Aufstelldach",
           "Campingmodus", "USB", "Wasser", "Energie"]
ZONES = ["Küche", "Dach", "Außen", "Lesen"]   # light zones — apply to Licht only
ACTIONS = ["an", "aus", "auf", "zu", "heller", "dunkler",
           "wärmer", "kälter", "leise", "Eco", "Max", "Normal"]
ZONED_DEVICES = ["Licht"]
# Per-device allowed actions — grounded in the real controllable functions.
DEVICE_ACTIONS = {
    "Licht": ["an", "aus", "heller", "dunkler"],
    "Kühlschrank": ["an", "aus", "leise"],
    "Heizung": ["an", "aus", "wärmer", "kälter"],
    "Aufstelldach": ["auf", "zu"],
    "Campingmodus": ["an", "aus"],
    "USB": ["an", "aus"],
    "Wasser": ["an", "aus"],
    "Energie": ["Eco", "Max", "Normal"],
}
COMMAND_LABELS = DEVICES + ZONES + ACTIONS + ["_unknown_", "_silence_"]

def command_index(label: str) -> int:
    return COMMAND_LABELS.index(label)
```

- [ ] **Step 4: Run test to verify it passes** — `uv run pytest tests/test_config_v2.py tests/test_config.py -v` → PASS (v1 config test still green).

- [ ] **Step 5: Commit** — `git commit -m "feat(v2): additive wake + slot-command vocab in config"`

---

### Task 2: Grammar / slot-filler (pure)

**Files:**
- Create: `kws_de/grammar.py`
- Test: `tests/test_grammar.py`

**Interfaces:**
- Consumes: `kws_de.config` (DEVICES/ZONES/ACTIONS).
- Produces: `@dataclass Intent(device: str, zone: str | None, action: str)`; `@dataclass Rejection(reason: str)`; `parse(events: list[str]) -> Intent | Rejection`. `events` is the ordered list of keyword labels emitted by the streaming detector; `_unknown_`/`_silence_` events are ignored.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grammar.py
from kws_de.grammar import parse, Intent, Rejection

def test_device_zone_action():
    assert parse(["Licht", "Küche", "an"]) == Intent("Licht", "Küche", "an")

def test_device_action_no_zone():
    assert parse(["Licht", "an"]) == Intent("Licht", None, "an")

def test_ignores_unknown_and_silence():
    assert parse(["_silence_", "Licht", "_unknown_", "an"]) == Intent("Licht", None, "an")

def test_missing_action_rejected():
    assert isinstance(parse(["Licht", "Küche"]), Rejection)

def test_missing_device_rejected():
    assert isinstance(parse(["Küche", "an"]), Rejection)

def test_out_of_order_rejected():
    assert isinstance(parse(["an", "Licht"]), Rejection)

def test_duplicate_slot_rejected():
    assert isinstance(parse(["Licht", "Heizung", "an"]), Rejection)

def test_action_invalid_for_device_rejected():
    assert isinstance(parse(["Aufstelldach", "an"]), Rejection)   # roof has no an/aus

def test_roof_auf_valid():
    assert parse(["Aufstelldach", "auf"]) == Intent("Aufstelldach", None, "auf")

def test_zone_on_non_zoned_device_rejected():
    assert isinstance(parse(["Heizung", "Küche", "an"]), Rejection)

def test_light_zone_brightness_valid():
    assert parse(["Licht", "Küche", "heller"]) == Intent("Licht", "Küche", "heller")

def test_energy_mode_valid():
    assert parse(["Energie", "Eco"]) == Intent("Energie", None, "Eco")
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_grammar.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/grammar.py
from dataclasses import dataclass
from kws_de import config

@dataclass(frozen=True)
class Intent:
    device: str
    zone: str | None
    action: str

@dataclass(frozen=True)
class Rejection:
    reason: str

def parse(events: list[str]) -> Intent | Rejection:
    toks = [e for e in events if e not in ("_unknown_", "_silence_")]
    device = zone = action = None
    for t in toks:
        if t in config.DEVICES:
            if device is not None:
                return Rejection(f"duplicate device: {t}")
            if zone is not None or action is not None:
                return Rejection("device out of order")
            device = t
        elif t in config.ZONES:
            if zone is not None:
                return Rejection(f"duplicate zone: {t}")
            if device is None or action is not None:
                return Rejection("zone out of order")
            zone = t
        elif t in config.ACTIONS:
            if action is not None:
                return Rejection(f"duplicate action: {t}")
            action = t
        else:
            return Rejection(f"unknown token: {t}")
    if device is None:
        return Rejection("missing device")
    if action is None:
        return Rejection("missing action")
    if zone is not None and device not in config.ZONED_DEVICES:
        return Rejection(f"{device} takes no zone")
    if action not in config.DEVICE_ACTIONS[device]:
        return Rejection(f"{action} invalid for {device}")
    return Intent(device, zone, action)
```

- [ ] **Step 4: Run test to verify it passes** — `uv run pytest tests/test_grammar.py -v` → PASS (7 tests).

- [ ] **Step 5: Commit** — `git commit -m "feat(v2): pure grammar slot-filler (device [zone] action -> intent)"`

---

### Task 3: Streaming keyword detector

**Files:**
- Create: `kws_de/stream.py`
- Test: `tests/test_stream.py`

**Interfaces:**
- Produces: `KeywordStream(predict_fn, labels, smooth_win=3, threshold=0.6, refractory=5)` where `predict_fn(feat_window) -> np.ndarray` returns per-label posteriors. Method `push(posterior: np.ndarray) -> list[str]` returns keyword events fired this step (0 or 1), applying trailing-average smoothing over `smooth_win`, a `threshold`, and a `refractory` debounce (no re-fire within `refractory` steps). `_silence_` never fires as an event. `reset()` clears state.

*(Testing note: the detector is fed posteriors directly, so it is unit-testable with scripted posterior arrays and no real model.)*

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream.py
import numpy as np
from kws_de.stream import KeywordStream

LABELS = ["Licht", "an", "_silence_"]

def _one_hot(i, n=3, p=0.9):
    v = np.full(n, (1 - p) / (n - 1)); v[i] = p; return v

def test_fires_once_per_sustained_word():
    ks = KeywordStream(None, LABELS, smooth_win=3, threshold=0.6, refractory=4)
    events = []
    for _ in range(6):                # 'Licht' sustained
        events += ks.push(_one_hot(0))
    assert events == ["Licht"]        # exactly one event despite 6 frames

def test_refractory_blocks_immediate_refire_then_allows_new_word():
    ks = KeywordStream(None, LABELS, smooth_win=2, threshold=0.6, refractory=3)
    seq = [0, 0, 0, 1, 1, 1]          # Licht then an
    out = []
    for i in seq:
        out += ks.push(_one_hot(i))
    assert out == ["Licht", "an"]

def test_silence_never_fires():
    ks = KeywordStream(None, LABELS, smooth_win=1, threshold=0.5, refractory=1)
    out = []
    for _ in range(5):
        out += ks.push(_one_hot(2))   # _silence_
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_stream.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/stream.py
from collections import deque
import numpy as np

class KeywordStream:
    def __init__(self, predict_fn, labels, smooth_win=3, threshold=0.6, refractory=5):
        self.predict_fn = predict_fn
        self.labels = list(labels)
        self.smooth_win = smooth_win
        self.threshold = threshold
        self.refractory = refractory
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=self.smooth_win)
        self._cooldown = 0
        self._last = None

    def push(self, posterior) -> list:
        self._hist.append(np.asarray(posterior, dtype=np.float64))
        if self._cooldown > 0:
            self._cooldown -= 1
        smoothed = np.mean(self._hist, axis=0)
        idx = int(np.argmax(smoothed))
        label = self.labels[idx]
        if (
            smoothed[idx] >= self.threshold
            and label != "_silence_"
            and self._cooldown == 0
        ):
            self._cooldown = self.refractory
            self._last = label
            return [label]
        return []
```

- [ ] **Step 4: Run test to verify it passes** — `uv run pytest tests/test_stream.py -v` → PASS (3 tests).

- [ ] **Step 5: Commit** — `git commit -m "feat(v2): streaming keyword detector (smoothing + threshold + debounce)"`

---

### Task 4: Expanded-vocab command dataset

**Files:**
- Modify: `kws_de/data.py` (generalise the fetch/build to a caller-supplied label set)
- Test: `tests/test_data_v2.py`

**Interfaces:**
- Produces: `build_dataset` (v1) reused, driven by `config.COMMAND_LABELS`. Add `command_words() -> list[str]` returning `DEVICES + ZONES + ACTIONS` (the words needing clips). `_fetch_and_cache` extended to pull those words from MSWC (streaming + early-stop, safety cap) and TTS-fill the thin ones — same code path as v1, parameterised by the word list instead of `COMMANDS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_v2.py
import numpy as np
from kws_de import config
from kws_de.data import build_dataset, command_words

def test_command_words_are_slot_vocab():
    assert command_words() == config.DEVICES + config.ZONES + config.ACTIONS

def test_build_dataset_over_command_labels():
    rng = np.random.default_rng(0)
    clips = {w: [rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)]
             for w in command_words()}
    clips["_unknown_"] = [rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)]
    noises = [rng.standard_normal(8000).astype(np.float32)]
    X, y = build_dataset(clips, noises, rng, snrs=(20,), labels=config.COMMAND_LABELS)
    assert X.shape[1:] == (config.N_FRAMES, config.N_MFCC)
    assert set(np.unique(y)).issubset(set(range(len(config.COMMAND_LABELS))))
```

*(If v1's `build_dataset` hard-codes `COMMANDS`/`LABELS`, add an optional `labels`/`commands` parameter defaulting to the v1 values so v1 callers and tests are unchanged.)*

- [ ] **Step 2: Run to verify it fails**, **Step 3: implement** (parameterise `build_dataset` with `labels`/`commands`; add `command_words`; extend `_fetch_and_cache` to loop `command_words()`), **Step 4: run** `uv run pytest tests/test_data_v2.py tests/test_data.py -v` → all pass (v1 unchanged), **Step 5: commit** `feat(v2): dataset over the expanded slot vocab`.

---

### Task 5: Wake word via microWakeWord + detector wrapper

Adopt **microWakeWord** for the wake model (spec §12) — do NOT hand-roll one. microWakeWord is a
training-time dependency that produces a streaming TFLite-Micro model for "Hey Bus"; at runtime we
only load that model and wrap it with a cutoff + debounce. The unit-testable part is that wrapper.

**Files:**
- Create: `kws_de/wake.py`
- Test: `tests/test_wake.py`
- Modify: `pyproject.toml` (add microWakeWord as a `train` extra — training-time only)

**Interfaces:**
- Produces:
  - `WakeDetector(cutoff=0.8, refractory=20)` — `.push(prob: float) -> bool`: microWakeWord emits a per-step wake probability; the detector fires once when `prob >= cutoff`, then suppresses for `refractory` steps (debounce). Pure — unit-tested with scripted probabilities, no model needed.
  - `load_wake_tflite(path) -> Callable[[np.ndarray], float]` — loads the microWakeWord streaming TFLite model and returns a `predict_fn(features) -> wake_prob`. (`# pragma: no cover` — needs the artifact.)
  - `train_hey_bus(out_dir) -> Path` — `# pragma: no cover` wrapper invoking microWakeWord's trainer (Piper TTS "Hey Bus" positives + its negative/ambient sets) to produce `hey_bus.tflite`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wake.py
from kws_de.wake import WakeDetector

def test_fires_once_on_sustained_wake():
    d = WakeDetector(cutoff=0.8, refractory=5)
    fired = [d.push(0.95) for _ in range(6)]
    assert fired.count(True) == 1          # one wake despite sustained high prob

def test_no_fire_below_cutoff():
    d = WakeDetector(cutoff=0.8, refractory=5)
    assert not any(d.push(0.5) for _ in range(6))

def test_refractory_then_new_wake():
    d = WakeDetector(cutoff=0.8, refractory=3)
    seq = [0.95, 0.1, 0.1, 0.1, 0.95]      # wake, gap past refractory, wake
    assert [d.push(p) for p in seq].count(True) == 2
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_wake.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# kws_de/wake.py
from pathlib import Path

class WakeDetector:
    def __init__(self, cutoff: float = 0.8, refractory: int = 20):
        self.cutoff = cutoff
        self.refractory = refractory
        self._cooldown = 0

    def push(self, prob: float) -> bool:
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if prob >= self.cutoff:
            self._cooldown = self.refractory
            return True
        return False

def load_wake_tflite(path):  # pragma: no cover - needs the trained artifact
    import numpy as np, tensorflow as tf
    itp = tf.lite.Interpreter(model_path=str(path)); itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    def predict_fn(features):
        itp.set_tensor(inp["index"], np.asarray(features, inp["dtype"]).reshape(inp["shape"]))
        itp.invoke()
        return float(itp.get_tensor(out["index"]).ravel()[-1])
    return predict_fn

def train_hey_bus(out_dir) -> Path:  # pragma: no cover - invokes microWakeWord trainer
    # Use microWakeWord's training framework: generate "Hey Bus" positives via Piper TTS,
    # combine with its ambient/negative sets, train the streaming model, export TFLite-Micro.
    # See https://github.com/kahrendt/microWakeWord (+ microwakeword-trainer). Produces
    # hey_bus.tflite in out_dir.
    raise NotImplementedError("wire microWakeWord trainer; see spec §12")
```

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
train = ["microwakeword"]   # training-time only; runtime just loads the tflite
```

- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_wake.py -v` → PASS (3 tests).

- [ ] **Step 5: Commit** — `git commit -m "feat(v2): wake stage via microWakeWord + cutoff/debounce detector"`

---

### Task 6: Retrain command recogniser + streaming/intent eval

**Files:**
- Modify: `kws_de/train.py` (accept a label set), `kws_de/eval.py` (streaming + intent metrics)
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `intent_accuracy(true_intents, pred_intents) -> float` (fraction where device+zone+action all match); `evaluate_streaming(predict_fn, clips_sequences) -> dict` glue (thin, I/O — `# pragma: no cover`). `train(X, y, ...)` reused with the command label count.

- [ ] **Step 1: failing test**

```python
# tests/test_eval_v2.py
from kws_de.eval import intent_accuracy
from kws_de.grammar import Intent

def test_intent_accuracy_all_slots_must_match():
    t = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "aus")]
    p = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "an")]  # 2nd action wrong
    assert intent_accuracy(t, p) == 0.5
```

- [ ] **Step 2: fail**, **Step 3: implement** `intent_accuracy` (pure) + wire streaming eval that runs `KeywordStream` over held-out command utterances, `grammar.parse` the events, and compares intents, **Step 4: pass**, **Step 5: commit** `feat(v2): intent accuracy + streaming eval`.

---

### Task 7: INT8 export + budgets for both models

**Files:**
- Modify: `kws_de/export.py`, `kws_de/budgets.py`
- Test: `tests/test_export_v2.py`

**Interfaces:**
- Produces: reuse `to_int8_tflite` for the COMMAND model (ours). `check_wake_budgets(tflite: bytes) -> dict` — budget-checks the microWakeWord wake TFLite from its **bytes only** (size ≤ 150 000 B, full-INT8, TFLM-supported ops); MACs are not asserted here since the wake model is external (no Keras graph). The command model still uses v1's `check_budgets(tflite, keras_model)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_v2.py
import numpy as np
from kws_de import config
from kws_de.model import build_dscnn          # stand-in small model; real wake tflite comes
from kws_de.export import to_int8_tflite       # from microWakeWord (spec §12)
from kws_de.budgets import check_wake_budgets

def test_wake_budget_checks_tflite_bytes_only():
    m = build_dscnn()
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)              # a small INT8 tflite ~ wake-model size class
    r = check_wake_budgets(blob)
    assert r["model_bytes"] <= 150_000 and r["int8"] is True
```

- [ ] **Step 2: fail**, **Step 3: implement** `check_wake_budgets(tflite)` (reuse `is_full_int8`/`tflite_op_types`; assert size ≤ 150 000 and int8), **Step 4: pass**, **Step 5: commit** `feat(v2): command INT8 export + byte-only wake budget check`.

---

### Task 8: Real run + eval report (numbers)

**Files:**
- Modify: `kws_de/wake.py::main`, `kws_de/eval.py::main`
- Create (generated, gitignored inputs): `docs/eval-report-v2.md`

**Interfaces:** I/O wrappers (`# pragma: no cover`). This task RUNS the pipeline on real + TTS data and writes the report.

- [ ] **Step 1:** Fetch/TTS the full grounded vocab: reuse v1 cached clips for `Licht/Kühlschrank/Heizung/Wasser`; fetch MSWC where available for the new words (`an, aus, auf, zu, heller, dunkler, wärmer, kälter, leise, Eco, Max, Normal, Küche, Dach, Außen, Lesen` + devices `Aufstelldach, Campingmodus, USB, Energie`) and TTS-fill the rest (macOS `say`, German voices). Report per-word real/TTS counts honestly.
- [ ] **Step 2:** Train the command recogniser on `config.COMMAND_LABELS` (26 classes).
- [ ] **Step 3 (local wake training):** produce `hey_bus.tflite` via `wake.train_hey_bus`, which drives **microWakeWord in the local 3.10 venv** (proven working at `/Volumes/External/Users/ext_ckeller/mww-train/.venv`): Piper generates "Hey Bus" positives, microWakeWord trains on CPU/Metal. Formalise the env into a committed `train/mww/` setup (README + a `uv`-managed 3.10 requirements + an invocation script); gitignore the venv + datasets. If training is too slow to finish in-run, train a reduced-sample model and note it.
- [ ] **Step 4 (export + budgets):** command model → INT8 + `check_budgets`; `check_wake_budgets` on `hey_bus.tflite`.
- [ ] **Step 5 (THE CATALOG — end-to-end):** build the **command catalog** = every valid intent, enumerated via `config.DEVICE_ACTIONS` (and zones for `Licht` only) — e.g. "Licht Küche an", "Licht heller", "Heizung wärmer", "Aufstelldach auf", "Energie Eco", "Kühlschrank leise", … For each catalog entry, synthesize utterances (macOS `say`, several voices), run the **full pipeline** (audio → MFCC → `KeywordStream` → `grammar.parse` → `Intent`), and compute **full-intent accuracy per catalog entry + overall**. Also: wake false-accepts/hour + detection rate; per-slot accuracy; SNR sweep. Write `docs/eval-report-v2.md` with the catalog table (per-command intent accuracy), the wake FA/hour headline, honest real/TTS provenance, and budgets.
- [ ] **Step 6:** Keep suite green + ruff clean; commit code + `train/mww/` setup + `docs/eval-report-v2.md` (no data/models/venvs). `feat(v2): real run + full command-catalog eval + local microWakeWord training`.

---

### Task 9: CI + coverage

**Files:** Modify `.github/workflows/ci.yml` (add `--cov=kws_de.grammar --cov=kws_de.stream`).

- [ ] **Step 1:** `uv run pytest -q` all green. **Step 2:** exact CI coverage command incl. new pure modules ≥ 85%. **Step 3:** ruff clean. **Step 4:** push branch, open PR, wait for CI + CodeRabbit. **Step 5:** commit any CI fixes.

---

## Self-Review

**Spec coverage:** §2 two-stage → Tasks 5/6/3; §3 vocab → Task 1; §4 components (`wake`,`stream`,`grammar`,export) → Tasks 2,3,5,7; §5 streaming algo → Task 3; §6 grammar → Task 2; §7 eval (wake FA/hr, intent accuracy, SNR) → Tasks 6,8; §8 algorithms → cited in spec, honored by reuse of v1 MFCC/streaming; §10 docs → separate (noted). §firmware → out of scope (separate plan). All covered.

**Placeholder scan:** integration/I/O wrappers (`_fetch_and_cache` extension, `train.main`, `wake.main`, `eval.main`, `evaluate_streaming`) are `# pragma: no cover` by design; the pure logic (config, grammar, stream, intent_accuracy, budgets) is fully implemented + tested. No hidden TODOs in tested code.

**Type consistency:** `KeywordStream.push` returns `list[str]` of labels → `grammar.parse(list[str])` → `Intent|Rejection`; `intent_accuracy(list[Intent], list[Intent])`. `build_dataset(..., labels=COMMAND_LABELS)` additive param keeps v1 callers valid. Consistent.

## Out of scope (follow-on plans)

- ESP-IDF firmware: always-on wake → command window → streaming infer → grammar → camper BLE control write, on the CoreS3 (consumes both `model_data.h` files + metadata).
- Sphinx + sphinx-likec4 docs: two-stage pipeline + dynamic views.
