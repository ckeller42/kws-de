# CoreS3 Dual-Mode Firmware (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One ESP-IDF firmware for M5Stack CoreS3 with a guided speech recorder (words / sentences / negatives → WAV on flash → USB drive) and a minimal on-device keyword recogniser fed by `kws-export`, all built and unit-tested in CI.

**Architecture:** Always-on audio task fills a 16 kHz int16 ring buffer; exactly one consumer (record or recognise) drains it; LVGL UI task drives a 320×240 touch screen. Storage is a wear-levelled FAT partition mounted at `/rec`, alternatively exported as a TinyUSB mass-storage device. Pure-C units (`mfcc`, `stream`, `wav`, `prompts`) have zero IDF dependencies and are tested on the host with plain `cc`; Python generates every constant table so C never re-derives librosa math.

**Tech Stack:** ESP-IDF (tag pinned by Task 1, expected v5.5.x), esp-bsp `m5stack_core_s3`, LVGL 9 via BSP, esp-sr AFE (VAD), esp-tflite-micro (ESP-NN), esp_tinyusb MSC, FATFS + wear levelling, NVS. Host: plain C11 + `make`. Python side: `kws_de` (numpy/librosa/scipy, TF only for the model export).

**Spec:** `docs/superpowers/specs/2026-09-01-cores3-firmware-design.md`

## Global Constraints

- Public repo: no speaker names, no vehicle/brand or decompiled-app provenance, no machine-specific paths (`/Volumes/External/...`, home dirs) in any committed file. Speaker ids are `spkNN` only.
- Never commit `data/`, `models/`, `firmware/build/`, `firmware/managed_components/`, `firmware/sdkconfig`, `firmware/dependencies.lock` (all gitignored already). `firmware/main/gen/*.h` ARE committed.
- IDF version is pinned in exactly three places and must agree: `firmware/README.md`, `firmware/CMakeLists.txt` (`IDF_VERSION` check), `.github/workflows/firmware.yml` (`esp_idf_version`). Task 1 chooses the tag; every later task uses `IDF_TAG` to mean that value.
- Build system is `idf.py` (CMake). Our own CMake is one `project()` plus one `idf_component_register()` per component.
- Local firmware builds on the Mac: `docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG idf.py build` (no local IDF install). Host C tests: `make -C firmware/test` (needs only `cc`).
- Python: `uv run ...` for everything; ruff (`ruff check . && ruff format --check .`) and pytest must pass; markdownlint config `.markdownlint.json` (docs/superpowers is ignored, `firmware/README.md` is NOT).
- Pre-commit hook runs ruff + markdownlint; pre-push runs pytest. Never bypass with `--no-verify`.
- TFLM op set is exactly `CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX, RESHAPE, ADD` (gated in `tests/test_transducer.py`).
- MFCC parameters are the single source `kws_de.config`: `SAMPLE_RATE=16000, WIN_SAMPLES=480, HOP_SAMPLES=320, N_MELS=40, N_MFCC=10, N_FRAMES=49`. C reads them from `features_config.h`, never literal.
- Recogniser model = the v2 command model (`models/command.tflite`, labels `config.COMMAND_LABELS`, 26 classes, input int8 `[1,49,10,1]`).
- Detector parameters (from `kws_de.eval.run_catalog_eval` defaults): `smooth_win=3, threshold=0.5, min_consecutive=2, gap_steps=2`.
- Recording caps: pre-roll 300 ms, trailing silence 500 ms, word cap 4000 ms, sentence/negative cap 6000 ms, no-speech timeout 8000 ms, auto-advance hold 700 ms, flash-full threshold 200 KB free.
- Commit messages: conventional prefix (`feat:`, `fix:`, `docs:`, `ci:`, `test:`), no AI co-author trailers beyond what the repo hooks add.

## File map

| Path | Owner task | Responsibility |
|---|---|---|
| `firmware/CMakeLists.txt`, `sdkconfig.defaults`, `partitions.csv`, `idf_component.yml`, `main/CMakeLists.txt`, `main/main.c` | 1 | project skeleton, IDF pin, boots to a splash screen |
| `.github/workflows/firmware.yml` | 1 (build), 2 (gen-fresh), 3 (host-test), 6 (shellcheck) | firmware CI |
| `firmware/README.md` | 1, 8 | IDF pin, build/flash/pull how-to, manual checklist |
| `kws_de/firmware_gen.py`, `tests/test_firmware_gen.py` | 2 | TF-free generator: `labels.h`, `prompts.h`, `features_config.h`, `test_vectors.h` |
| `kws_de/config.py` (`NEGATIVE_PROMPTS`) | 2 | negative prompt set |
| `firmware/main/gen/*.h` | 2 (config-derived), 4 (model-derived) | committed generated headers |
| `firmware/main/{mfcc,stream,wav,prompts}.{c,h}`, `firmware/test/{Makefile,test_*.c}` | 3 | pure C units + host tests |
| `kws_de/export.py` (`--firmware`), `tests/test_export_firmware.py` | 4 | `model_data.h`, `model_config.h` |
| `firmware/main/{audio,storage,record}.{c,h}`, `main/ui/ui_record.c`, `main/ui/ui.h` | 5 | audio ring buffer, FAT mount, recorder + screen |
| `firmware/main/usb_drive.{c,h}`, `main/ui/ui_usb.c`, `scripts/pull-recordings.sh`, `tests/test_pull_recordings.py` | 6 | USB MSC + host pull script |
| `firmware/main/recognise.{c,cc,h}`, `main/ui/ui_recognise.c` | 7 | TFLM inference + screen + log |
| `scripts/flash.sh`, `firmware/README.md` | 8 | flash from any host, checklist, Pi notes |

Execution order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Task 4 needs a trained `models/command.keras` locally (exists on the primary machine); if it is missing, the implementer reports BLOCKED rather than committing a placeholder blob.

---

### Task 1: IDF version spike, project skeleton, CI build job

**Files:**
- Create: `firmware/CMakeLists.txt`, `firmware/sdkconfig.defaults`, `firmware/partitions.csv`, `firmware/main/CMakeLists.txt`, `firmware/main/idf_component.yml`, `firmware/main/main.c`, `firmware/README.md`, `.github/workflows/firmware.yml`
- Modify: `.gitignore` (already has the firmware block — verify, do not duplicate)

**Interfaces:**
- Produces: `IDF_TAG` (the pinned tag string, e.g. `v5.5.1`), the `app_main` skeleton that later tasks extend, the `storage` partition name (`storage`, type `data`, subtype `fat`).

- [ ] **Step 1: Spike — does the component set build on IDF v6.2?**

Run (Mac, Docker; ~10 min first pull):

```bash
mkdir -p firmware/main && cd firmware
cat > main/idf_component.yml <<'EOF'
dependencies:
  idf: ">=5.5"
  espressif/m5stack_core_s3: "^2"
  espressif/esp-sr: "^2"
  espressif/esp-tflite-micro: "^1"
  espressif/esp_tinyusb: "^1"
EOF
cat > CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(kws_de_fw)
EOF
cat > main/CMakeLists.txt <<'EOF'
idf_component_register(SRCS "main.c" INCLUDE_DIRS "." "gen")
EOF
printf '#include <stdio.h>\nvoid app_main(void){printf("kws-de fw\\n");}\n' > main/main.c
cd .. && docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:v6.2 \
  bash -c 'idf.py set-target esp32s3 && idf.py build' 2>&1 | tail -30
```

If the build succeeds, `IDF_TAG=v6.2`. If any managed component refuses (`idf` version constraint or compile error), retry with `espressif/idf:v5.5.1` (then `v5.4.2`); the first tag that builds is `IDF_TAG`. Record the outcome and the tag in the report. Expected: v5.5.x builds; v6.2 likely fails on esp-sr or esp-tflite-micro.

- [ ] **Step 2: Write the real project files**

`firmware/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# Pinned in firmware/README.md and .github/workflows/firmware.yml too — keep all three equal.
set(KWS_DE_IDF_PIN "IDF_TAG")
if(NOT "$ENV{IDF_VERSION}" STREQUAL "" AND NOT "$ENV{IDF_VERSION}" MATCHES "^${KWS_DE_IDF_PIN}")
  message(WARNING "kws-de firmware is pinned to ESP-IDF ${KWS_DE_IDF_PIN}, building with $ENV{IDF_VERSION}")
endif()
project(kws_de_fw)
```

(`IDF_VERSION` is exported by `export.sh`/the docker image; the check is a warning, not an error, so a deliberate local experiment still builds. CI enforces the pin by construction.)

`firmware/partitions.csv`:

```csv
# Name,   Type, SubType, Offset,  Size
nvs,      data, nvs,     0x9000,  0x6000
phy_init, data, phy,     0xf000,  0x1000
factory,  app,  factory, 0x10000, 3M
storage,  data, fat,     ,        10M
```

`firmware/sdkconfig.defaults`:

```ini
CONFIG_IDF_TARGET="esp32s3"
CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y
CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y
CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=4096
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_FATFS_LFN_HEAP=y
CONFIG_FATFS_MAX_LFN=64
CONFIG_WL_SECTOR_SIZE_4096=y
CONFIG_TINYUSB_MSC_ENABLED=y
CONFIG_ESP_CONSOLE_UART_DEFAULT=y
CONFIG_LV_COLOR_DEPTH_16=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192
```

`firmware/main/main.c` (splash only; Task 5 adds the state machine):

```c
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "esp_log.h"
#include "lvgl.h"

static const char *TAG = "main";

void app_main(void)
{
    bsp_i2c_init();
    lv_display_t *disp = bsp_display_start();
    bsp_display_backlight_on();
    (void)disp;

    bsp_display_lock(0);
    lv_obj_t *label = lv_label_create(lv_screen_active());
    lv_label_set_text(label, "kws-de firmware");
    lv_obj_center(label);
    bsp_display_unlock();

    ESP_LOGI(TAG, "boot ok, free heap %lu", (unsigned long)esp_get_free_heap_size());
}
```

`firmware/main/CMakeLists.txt`:

```cmake
idf_component_register(SRCS "main.c" INCLUDE_DIRS "." "gen")
```

Create an empty `firmware/main/gen/.gitkeep` so the include dir exists before Task 2.

- [ ] **Step 3: Build with the pinned tag**

```bash
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG \
  bash -c 'idf.py set-target esp32s3 && idf.py build && ls -l build/*.bin'
```

Expected: `build/kws_de_fw.bin` exists, `build/partition_table/partition-table.bin` exists, no warnings about `partitions.csv`.

- [ ] **Step 4: CI workflow (build job only; later tasks append jobs)**

`.github/workflows/firmware.yml`:

```yaml
name: firmware
on:
  push: {branches: [main]}
  pull_request:
permissions: {contents: read}
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - name: idf.py build (pinned IDF)
        uses: espressif/esp-idf-ci-action@v1
        with:
          esp_idf_version: IDF_TAG
          target: esp32s3
          path: firmware
      - name: merge single flashable image
        run: |
          pip install esptool
          cd firmware/build
          esptool.py --chip esp32s3 merge_bin -o "kws-de-fw-${GITHUB_SHA::7}.bin" @flash_args
          ls -l kws-de-fw-*.bin
      - uses: actions/upload-artifact@v4
        with:
          name: kws-de-fw-${{ github.sha }}
          path: firmware/build/kws-de-fw-*.bin
          retention-days: 30
```

Pin `espressif/esp-idf-ci-action` and `actions/upload-artifact` to full commit SHAs the same way `ci.yml` pins `actions/checkout` (look them up with `gh api repos/espressif/esp-idf-ci-action/git/ref/tags/v1` and `gh api repos/actions/upload-artifact/git/ref/tags/v4`, write `@<sha> # v1`).

- [ ] **Step 5: `firmware/README.md` (first version)**

```markdown
# kws-de firmware (M5Stack CoreS3)

Dual-mode firmware: guided speech recorder (words / sentences / negatives → WAV on flash → USB drive) and an on-device keyword recogniser fed by `kws-export`.

## ESP-IDF version

Pinned to **IDF_TAG**. The same string lives in `CMakeLists.txt` and `.github/workflows/firmware.yml`; change all three together.

## Build (no local IDF needed)

```bash
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG \
  bash -c 'idf.py set-target esp32s3 && idf.py build'
```

With a native IDF install (e.g. on the Pi): `cd firmware && idf.py set-target esp32s3 && idf.py build`.

## Host unit tests

```bash
make -C firmware/test
```

Pure-C units only (`mfcc`, `stream`, `wav`, `prompts`); needs `cc` and nothing else.

## Flash

See `scripts/flash.sh` (any host with `esptool.py`, no IDF).
```

- [ ] **Step 6: Commit**

```bash
git add firmware .github/workflows/firmware.yml
git commit -m "feat(firmware): CoreS3 project skeleton, IDF pin IDF_TAG, CI build job"
```

Replace every literal `IDF_TAG` in the committed files with the tag from Step 1 before committing (grep: `grep -rn IDF_TAG firmware .github` must return nothing).

---

### Task 2: `kws-fwgen` — config-derived headers + `NEGATIVE_PROMPTS` + gen-fresh CI

**Files:**
- Create: `kws_de/firmware_gen.py`, `tests/test_firmware_gen.py`, `firmware/main/gen/labels.h`, `firmware/main/gen/prompts.h`, `firmware/main/gen/features_config.h`, `firmware/main/gen/test_vectors.h`
- Modify: `kws_de/config.py` (append `NEGATIVE_PROMPTS`), `pyproject.toml` (`[project.scripts]` add `kws-fwgen`), `.github/workflows/firmware.yml` (add `gen-fresh` job)

**Interfaces:**
- Consumes: `config.COMMAND_LABELS`, `config.DEVICES/ZONES/ACTIONS`, `kws_de.eval.build_catalog()` (→ `Intent(device, zone, action)`), `librosa.filters.mel`, `scipy.signal.get_window`, `scipy.fft.dct`, `kws_de.features.mfcc`.
- Produces (C side, all `static const`, names exact):
  - `labels.h`: `#define KWS_NUM_LABELS 26`, `static const char *const KWS_LABELS[KWS_NUM_LABELS]`, `#define KWS_SILENCE_INDEX <i>`, `#define KWS_UNKNOWN_INDEX <i>`.
  - `prompts.h`: `#define KWS_NUM_WORD_PROMPTS`, `KWS_WORD_PROMPTS[]` (display text = label, e.g. "Kühlschrank"), `KWS_WORD_SLUGS[]` (e.g. "kuehlschrank"); same triple for `SENTENCE` and `NEG`. Sentence display text = `"<device> [<zone>] <action>"` joined by spaces; slug = same joined by `-`.
  - `features_config.h`: `KWS_SAMPLE_RATE, KWS_WIN, KWS_HOP, KWS_N_MELS, KWS_N_MFCC, KWS_N_FRAMES, KWS_N_BINS (=241)`, `KWS_TOP_DB (80.0f)`, `KWS_AMIN (1e-10f)`, `static const float KWS_WINDOW[KWS_WIN]`, `KWS_MEL[KWS_N_MELS][KWS_N_BINS]`, `KWS_DCT[KWS_N_MFCC][KWS_N_MELS]`, detector params `KWS_SMOOTH_WIN 3, KWS_THRESHOLD 0.5f, KWS_MIN_CONSECUTIVE 2, KWS_GAP_STEPS 2`.
  - `test_vectors.h`: `static const int16_t TV_PCM[16000]`, `static const float TV_MFCC[49][10]`.
- CLI: `kws-fwgen [--out firmware/main/gen]` writes the four files deterministically.

- [ ] **Step 1: Add `NEGATIVE_PROMPTS` to `kws_de/config.py`** (append at end of file)

```python
# Guided-recorder "negative" prompts: everyday German sentences that contain
# none of the command vocabulary. Used only for on-device recording; the
# recordings feed false-accept evaluation later.
NEGATIVE_PROMPTS = [
    "wie spät ist es",
    "wo sind wir gerade",
    "hast du den Schlüssel gesehen",
    "morgen wird es regnen",
    "ich habe Hunger",
    "wann fahren wir los",
    "das war ein schöner Tag",
    "kannst du mir helfen",
    "der Kaffee ist fertig",
    "wir brauchen noch Brot",
    "ich gehe kurz raus",
    "mach die Musik leiser",
    "wie weit ist es noch",
    "das Wetter ist super",
    "ich bin müde",
    "hast du gut geschlafen",
    "wir sind gleich da",
    "gib mir bitte das Handtuch",
    "die Kinder schlafen schon",
    "was gibt es zu essen",
]
```

- [ ] **Step 2: Write the failing tests** — `tests/test_firmware_gen.py`

```python
import re

import numpy as np

from kws_de import config, features, firmware_gen


def test_negative_prompts_contain_no_command_words():
    vocab = {w.lower() for w in config.DEVICES + config.ZONES + config.ACTIONS}
    assert len(config.NEGATIVE_PROMPTS) >= 15
    for p in config.NEGATIVE_PROMPTS:
        assert not (set(p.lower().split()) & vocab), p


def test_slug_is_ascii_and_stable():
    assert firmware_gen.slug("Kühlschrank") == "kuehlschrank"
    assert firmware_gen.slug("Licht Außen an") == "licht-aussen-an"
    assert firmware_gen.slug("wie spät ist es") == "wie-spaet-ist-es"
    assert re.fullmatch(r"[a-z0-9-]+", firmware_gen.slug("Straße  weiß")) 


def test_prompt_sets_cover_labels_and_catalog():
    words, sentences, negs = firmware_gen.prompt_sets()
    assert [w for w, _ in words] == [l for l in config.COMMAND_LABELS if not l.startswith("_")]
    assert len(sentences) == len(firmware_gen.build_catalog())
    assert len(negs) == len(config.NEGATIVE_PROMPTS)
    assert len({s for _, s in words + sentences + negs}) == len(words + sentences + negs)


def test_c_tables_reproduce_librosa_mfcc():
    win, mel, dct = firmware_gen.mfcc_tables()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32) * 0.1
    ref = features.mfcc(x)
    got = firmware_gen.mfcc_reference(x, win, mel, dct)
    assert np.allclose(got, ref, atol=1e-3)


def test_generate_is_deterministic_and_complete(tmp_path):
    firmware_gen.generate(tmp_path)
    firmware_gen.generate(tmp_path / "again")
    for name in ("labels.h", "prompts.h", "features_config.h", "test_vectors.h"):
        a = (tmp_path / name).read_text()
        assert a == (tmp_path / "again" / name).read_text()
    labels = (tmp_path / "labels.h").read_text()
    assert "#define KWS_NUM_LABELS 26" in labels
    assert f"#define KWS_SILENCE_INDEX {config.COMMAND_LABELS.index('_silence_')}" in labels
    fc = (tmp_path / "features_config.h").read_text()
    assert "#define KWS_N_BINS 241" in fc and "KWS_MEL[40][241]" in fc
    tv = (tmp_path / "test_vectors.h").read_text()
    assert "TV_PCM[16000]" in tv and "TV_MFCC[49][10]" in tv
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_firmware_gen.py -q`
Expected: FAIL with `ImportError: cannot import name 'firmware_gen'` (and `NEGATIVE_PROMPTS` test passes already after Step 1).

- [ ] **Step 4: Implement `kws_de/firmware_gen.py`**

```python
"""TF-free generator for the firmware's config-derived headers.

Everything C must agree with Python on — labels, prompts, MFCC constant
tables, a golden MFCC vector — is emitted here as `static const` data so
`firmware/main/mfcc.c` never re-derives librosa math. `kws-export --firmware`
(model_data.h, model_config.h) is the model-derived half."""

import argparse
import pathlib

import numpy as np
import scipy.fft
import scipy.signal
from librosa.filters import mel as mel_filter

from kws_de import config, features
from kws_de.eval import build_catalog

TOP_DB = 80.0
AMIN = 1e-10
DETECTOR = {"smooth_win": 3, "threshold": 0.5, "min_consecutive": 2, "gap_steps": 2}
_TRANS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def slug(text: str) -> str:
    s = text.lower().translate(_TRANS)
    return "-".join(w for w in "".join(c if c.isalnum() else " " for c in s).split())


def prompt_sets() -> tuple[list, list, list]:
    """(display, slug) pairs for words, sentences, negatives — in canonical
    (unshuffled) order; the device shuffles with its on-screen seed."""
    words = [(l, slug(l)) for l in config.COMMAND_LABELS if not l.startswith("_")]
    sentences = []
    for it in build_catalog():
        text = " ".join(t for t in (it.device, it.zone, it.action) if t)
        sentences.append((text, slug(text)))
    negs = [(p, slug(p)) for p in config.NEGATIVE_PROMPTS]
    return words, sentences, negs


def mfcc_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Hann window, Slaney mel filterbank, ortho DCT-II rows — the
    three constant matrices librosa.feature.mfcc uses with our config."""
    win = scipy.signal.get_window("hann", config.WIN_SAMPLES, fftbins=True).astype(np.float32)
    mel = mel_filter(sr=config.SAMPLE_RATE, n_fft=config.WIN_SAMPLES, n_mels=config.N_MELS)
    dct = scipy.fft.dct(np.eye(config.N_MELS), type=2, norm="ortho", axis=0)[: config.N_MFCC]
    return win, mel.astype(np.float32), dct.astype(np.float32)


def mfcc_reference(x, win, mel, dct) -> np.ndarray:
    """Pure-numpy mirror of the C pipeline (what mfcc.c must reproduce)."""
    x = features._fit_length(x)
    n = config.N_FRAMES
    frames = np.stack([x[i * config.HOP_SAMPLES : i * config.HOP_SAMPLES + config.WIN_SAMPLES] for i in range(n)])
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2  # (n, 241)
    logmel = 10.0 * np.log10(np.maximum(AMIN, spec @ mel.T))  # (n, 40)
    logmel = np.maximum(logmel, logmel.max() - TOP_DB)
    return (logmel @ dct.T).astype(np.float32)  # (n, 10)


def _c_float_rows(name, arr) -> str:
    arr = np.atleast_2d(arr)
    rows = ",\n".join("  {" + ", ".join(f"{v:.8e}f" for v in row) + "}" for row in arr)
    dims = "".join(f"[{d}]" for d in arr.shape) if arr.ndim > 1 else f"[{arr.shape[0]}]"
    if arr.ndim == 1 or arr.shape[0] == 1 and name == "KWS_WINDOW":
        flat = ", ".join(f"{v:.8e}f" for v in np.ravel(arr))
        return f"static const float {name}[{arr.size}] = {{{flat}}};\n"
    return f"static const float {name}{dims} = {{\n{rows}\n}};\n"


def _c_strings(name, items) -> str:
    body = ",\n".join(f'  "{s}"' for s in items)
    return f"static const char *const {name}[{len(items)}] = {{\n{body}\n}};\n"


def generate(out) -> None:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hdr = "/* generated by kws-fwgen — do not edit */\n#pragma once\n"

    labels = config.COMMAND_LABELS
    (out / "labels.h").write_text(
        hdr
        + f"#define KWS_NUM_LABELS {len(labels)}\n"
        + f"#define KWS_SILENCE_INDEX {labels.index('_silence_')}\n"
        + f"#define KWS_UNKNOWN_INDEX {labels.index('_unknown_')}\n"
        + _c_strings("KWS_LABELS", labels)
    )

    words, sentences, negs = prompt_sets()
    p = hdr
    for tag, items in (("WORD", words), ("SENTENCE", sentences), ("NEG", negs)):
        p += f"#define KWS_NUM_{tag}_PROMPTS {len(items)}\n"
        p += _c_strings(f"KWS_{tag}_PROMPTS", [d for d, _ in items])
        p += _c_strings(f"KWS_{tag}_SLUGS", [s for _, s in items])
    (out / "prompts.h").write_text(p)

    win, mel, dct = mfcc_tables()
    fc = hdr
    fc += f"#define KWS_SAMPLE_RATE {config.SAMPLE_RATE}\n#define KWS_WIN {config.WIN_SAMPLES}\n"
    fc += f"#define KWS_HOP {config.HOP_SAMPLES}\n#define KWS_N_MELS {config.N_MELS}\n"
    fc += f"#define KWS_N_MFCC {config.N_MFCC}\n#define KWS_N_FRAMES {config.N_FRAMES}\n"
    fc += f"#define KWS_N_BINS {config.WIN_SAMPLES // 2 + 1}\n"
    fc += f"#define KWS_TOP_DB {TOP_DB}f\n#define KWS_AMIN {AMIN}f\n"
    for k, v in DETECTOR.items():
        fc += f"#define KWS_{k.upper()} {v}{'f' if isinstance(v, float) else ''}\n"
    fc += _c_float_rows("KWS_WINDOW", win) + _c_float_rows("KWS_MEL", mel) + _c_float_rows("KWS_DCT", dct)
    (out / "features_config.h").write_text(fc)

    rng = np.random.default_rng(0)
    t = np.arange(config.CLIP_SAMPLES) / config.SAMPLE_RATE
    sig = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1300 * t) + 0.05 * rng.standard_normal(t.size)
    sig[: config.SAMPLE_RATE // 4] = 0.0  # a quarter second of digital silence exercises the top_db clamp
    pcm = np.clip(np.round(sig * 32767), -32768, 32767).astype(np.int16)
    ref = mfcc_reference(pcm.astype(np.float32) / 32768.0, win, mel, dct)
    tv = hdr + "#include <stdint.h>\n"
    tv += f"static const int16_t TV_PCM[{pcm.size}] = {{{', '.join(map(str, pcm.tolist()))}}};\n"
    tv += _c_float_rows("TV_MFCC", ref)
    (out / "test_vectors.h").write_text(tv)


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="firmware/main/gen")
    generate(ap.parse_args().out)
```

Add to `pyproject.toml` `[project.scripts]`: `kws-fwgen = "kws_de.firmware_gen:main"`, then `uv sync --extra dev --extra tts` (mirror whatever extras the lockfile currently carries — see `uv.lock` drift note in the distill ledger; commit `uv.lock` if it changes).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_firmware_gen.py tests/test_config_v2.py -q`
Expected: all PASS. If `test_c_tables_reproduce_librosa_mfcc` fails, first check that `librosa.feature.mfcc` still uses `power=2.0`, `top_db=80`, `norm="ortho"` (it does in librosa 0.11); the reference must match librosa to 1e-3, do not loosen the tolerance.

- [ ] **Step 6: Generate the committed headers + CI job**

```bash
uv run kws-fwgen && rm -f firmware/main/gen/.gitkeep && ls -l firmware/main/gen
```

Append to `.github/workflows/firmware.yml`:

```yaml
  gen-fresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - uses: actions/setup-python@f677139bbe7f9c59b41e40162b753c062f5d49a3 # v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - name: committed config-derived headers match kws-fwgen
        run: |
          kws-fwgen --out /tmp/gen
          for f in labels.h prompts.h features_config.h test_vectors.h; do
            diff -q "/tmp/gen/$f" "firmware/main/gen/$f"
          done
```

- [ ] **Step 7: Commit**

```bash
git add kws_de/firmware_gen.py kws_de/config.py tests/test_firmware_gen.py pyproject.toml uv.lock \
  firmware/main/gen/labels.h firmware/main/gen/prompts.h firmware/main/gen/features_config.h \
  firmware/main/gen/test_vectors.h .github/workflows/firmware.yml
git rm -q --cached firmware/main/gen/.gitkeep 2>/dev/null || true
git commit -m "feat: kws-fwgen — config-derived firmware headers, NEGATIVE_PROMPTS, gen-fresh CI"
```

---

### Task 3: Pure-C units (`mfcc`, `stream`, `wav`, `prompts`) + host tests + CI host-test job

**Files:**
- Create: `firmware/main/mfcc.c`, `firmware/main/mfcc.h`, `firmware/main/stream.c`, `firmware/main/stream.h`, `firmware/main/wav.c`, `firmware/main/wav.h`, `firmware/main/prompts.c`, `firmware/main/prompts.h`, `firmware/test/Makefile`, `firmware/test/test_mfcc.c`, `firmware/test/test_stream.c`, `firmware/test/test_wav.c`, `firmware/test/test_prompts.c`
- Modify: `firmware/main/CMakeLists.txt` (add the four `.c` to `SRCS`), `.github/workflows/firmware.yml` (add `host-test` job)

**Interfaces:**
- Consumes: `gen/features_config.h`, `gen/labels.h`, `gen/prompts.h`, `gen/test_vectors.h` (Task 2 names).
- Produces (C API, exact):

```c
/* mfcc.h — sliding-window MFCC front end. No IDF dependencies. */
#include <stdint.h>
#include "features_config.h"
typedef struct {
    float logmel[KWS_N_FRAMES][KWS_N_MELS]; /* ring of per-frame log-mel rows */
    int   head;                              /* index of the OLDEST row */
    int   count;                             /* rows filled (<= KWS_N_FRAMES) */
} mfcc_state_t;
void mfcc_init(mfcc_state_t *s);
/* Push exactly KWS_WIN samples (one analysis window, caller advances by KWS_HOP). */
void mfcc_push_frame(mfcc_state_t *s, const int16_t pcm[KWS_WIN]);
/* Apply librosa's top_db clamp over the whole window and the DCT: out is [KWS_N_FRAMES][KWS_N_MFCC],
   oldest frame first. Rows not yet filled are computed from zeros (matches Python zero-padding). */
void mfcc_finish(const mfcc_state_t *s, float out[KWS_N_FRAMES][KWS_N_MFCC]);
/* One-shot: whole 1 s clip (KWS_N_FRAMES*KWS_HOP + KWS_WIN - KWS_HOP samples = 16000). */
void mfcc_compute(const int16_t *pcm, float out[KWS_N_FRAMES][KWS_N_MFCC]);
/* TFLite int8 quantisation: q = round(x/scale) + zero_point, clamped. */
void mfcc_quantize(const float in[KWS_N_FRAMES][KWS_N_MFCC], int8_t *out, float scale, int zero_point);

/* stream.h — port of kws_de.stream.KeywordStream */
#include "labels.h"
typedef struct {
    float hist[KWS_SMOOTH_WIN][KWS_NUM_LABELS];
    int   hist_len, hist_pos;
    int   run_label;            /* -1 = None */
    int   run_len;
    int   run_fired;
    int   last_fired_label;     /* -1 = None */
    int   gap_since_last_fired;
} stream_t;
void stream_reset(stream_t *s);
/* Returns the fired label index, or -1. */
int  stream_push(stream_t *s, const float posterior[KWS_NUM_LABELS]);

/* wav.h */
#define WAV_HEADER_BYTES 44
void wav_write_header(uint8_t out[WAV_HEADER_BYTES], uint32_t n_samples, uint32_t sample_rate);

/* prompts.h (our module; includes gen/prompts.h) */
typedef enum { PROMPT_WORDS = 0, PROMPT_SENTENCES = 1, PROMPT_NEGS = 2 } prompt_set_t;
typedef struct { prompt_set_t set; uint32_t seed; int order[64]; int count; int index; } prompt_session_t;
/* Fisher–Yates with a 32-bit xorshift seeded from `seed`; same seed → same order. */
void        prompt_session_init(prompt_session_t *p, prompt_set_t set, uint32_t seed);
const char *prompt_text(const prompt_session_t *p);   /* display text for order[index] */
const char *prompt_slug(const prompt_session_t *p);
int         prompt_advance(prompt_session_t *p);      /* returns 0 when the set is exhausted */
uint32_t    prompt_cap_ms(prompt_set_t set);          /* 4000 words, 6000 otherwise */
```

- [ ] **Step 1: Makefile and the four failing tests**

`firmware/test/Makefile`:

```make
CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../main -I../main/gen
TESTS := test_mfcc test_stream test_wav test_prompts

all: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done
	@echo "host tests OK"

test_mfcc: test_mfcc.c ../main/mfcc.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_stream: test_stream.c ../main/stream.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_wav: test_wav.c ../main/wav.c
	$(CC) $(CFLAGS) -o $@ $^
test_prompts: test_prompts.c ../main/prompts.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TESTS)
.PHONY: all clean
```

`firmware/test/test_mfcc.c`:

```c
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "mfcc.h"
#include "test_vectors.h"

int main(void)
{
    static float out[KWS_N_FRAMES][KWS_N_MFCC];
    mfcc_compute(TV_PCM, out);
    float worst = 0.f;
    for (int t = 0; t < KWS_N_FRAMES; t++)
        for (int c = 0; c < KWS_N_MFCC; c++) {
            float d = fabsf(out[t][c] - TV_MFCC[t][c]);
            if (d > worst) worst = d;
        }
    printf("mfcc max abs err %g\n", worst);
    assert(worst < 1e-2f);           /* float vs float64 numpy; int8 step is ~3.0 */

    /* Streaming path must equal the one-shot path. */
    mfcc_state_t s;
    mfcc_init(&s);
    for (int t = 0; t < KWS_N_FRAMES; t++) mfcc_push_frame(&s, TV_PCM + t * KWS_HOP);
    static float out2[KWS_N_FRAMES][KWS_N_MFCC];
    mfcc_finish(&s, out2);
    assert(memcmp(out, out2, sizeof out) == 0);

    /* Quantisation: scale 3.0, zp 80 → clamps and rounds. */
    float q_in[KWS_N_FRAMES][KWS_N_MFCC] = {{0}};
    q_in[0][0] = 4.4f; q_in[0][1] = -1000.f; q_in[0][2] = 1000.f;
    int8_t q[KWS_N_FRAMES * KWS_N_MFCC];
    mfcc_quantize(q_in, q, 3.0f, 80);
    assert(q[0] == 81 && q[1] == -128 && q[2] == 127 && q[3] == 80);
    puts("test_mfcc OK");
    return 0;
}
```

`firmware/test/test_stream.c` (vectors are `tests/test_stream.py` translated; label 0 = Licht, 1 = an, 2 = `_silence_` there — here we use real indices from `labels.h`):

```c
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "stream.h"

static int LICHT, AN, SIL;

static void one_hot(float *v, int i, float p)
{
    for (int k = 0; k < KWS_NUM_LABELS; k++) v[k] = (1.f - p) / (KWS_NUM_LABELS - 1);
    v[i] = p;
}

/* push a sequence, collect fired labels */
static int run(stream_t *s, const int *seq, int n, int *events)
{
    int e = 0;
    float v[KWS_NUM_LABELS];
    for (int i = 0; i < n; i++) {
        one_hot(v, seq[i], 0.9f);
        int r = stream_push(s, v);
        if (r >= 0) events[e++] = r;
    }
    return e;
}

static int find(const char *name)
{
    for (int i = 0; i < KWS_NUM_LABELS; i++) if (!strcmp(KWS_LABELS[i], name)) return i;
    return -1;
}

int main(void)
{
    LICHT = find("Licht"); AN = find("an"); SIL = KWS_SILENCE_INDEX;
    assert(LICHT >= 0 && AN >= 0);
    stream_t s; int ev[16]; int n;

    /* fires once per sustained word (smooth_win=3 → first candidate needs the mean to cross) */
    stream_reset(&s);
    int a[] = {LICHT, LICHT, LICHT, LICHT, LICHT, LICHT};
    n = run(&s, a, 6, ev); assert(n == 1 && ev[0] == LICHT);

    /* two words back to back, no swallowing */
    stream_reset(&s);
    int b[] = {LICHT, LICHT, LICHT, LICHT, AN, AN, AN, AN, AN};
    n = run(&s, b, 9, ev); assert(n == 2 && ev[0] == LICHT && ev[1] == AN);

    /* same word twice with a silence gap >= gap_steps between runs */
    stream_reset(&s);
    int c[] = {LICHT, LICHT, LICHT, LICHT, SIL, SIL, SIL, SIL, SIL, LICHT, LICHT, LICHT, LICHT};
    n = run(&s, c, 13, ev); assert(n == 2 && ev[0] == LICHT && ev[1] == LICHT);

    /* silence never fires */
    stream_reset(&s);
    int d[] = {SIL, SIL, SIL, SIL, SIL};
    n = run(&s, d, 5, ev); assert(n == 0);

    /* below threshold never fires */
    stream_reset(&s);
    float weak[KWS_NUM_LABELS]; n = 0;
    for (int i = 0; i < 6; i++) { one_hot(weak, LICHT, 0.3f); if (stream_push(&s, weak) >= 0) n++; }
    assert(n == 0);
    puts("test_stream OK");
    return 0;
}
```

(The Python tests use `smooth_win=1` for the back-to-back cases; with the firmware's `smooth_win=3` the trailing-mean needs one extra step per transition, hence the longer runs above. The implementer must verify each expected event list by running the same sequences through `KeywordStream(None, config.COMMAND_LABELS, smooth_win=3, threshold=0.5, min_consecutive=2, gap_steps=2)` in `uv run python` and adjust the C expectations to whatever Python returns — Python is the oracle.)

`firmware/test/test_wav.c`:

```c
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "wav.h"

static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t *p) { return p[0] | p[1] << 8; }

int main(void)
{
    uint8_t h[WAV_HEADER_BYTES];
    wav_write_header(h, 16000, 16000);
    assert(!memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WAVEfmt ", 8) && !memcmp(h + 36, "data", 4));
    assert(le32(h + 4) == 36 + 32000);
    assert(le32(h + 16) == 16 && le16(h + 20) == 1 && le16(h + 22) == 1);
    assert(le32(h + 24) == 16000 && le32(h + 28) == 32000 && le16(h + 32) == 2 && le16(h + 34) == 16);
    assert(le32(h + 40) == 32000);
    wav_write_header(h, 0, 16000);   assert(le32(h + 40) == 0 && le32(h + 4) == 36);
    wav_write_header(h, 96000, 16000); assert(le32(h + 40) == 192000);
    puts("test_wav OK");
    return 0;
}
```

`firmware/test/test_prompts.c`:

```c
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "prompts.h"

int main(void)
{
    prompt_session_t a, b;
    prompt_session_init(&a, PROMPT_WORDS, 17);
    prompt_session_init(&b, PROMPT_WORDS, 17);
    assert(a.count == KWS_NUM_WORD_PROMPTS && a.count == 24);
    assert(!memcmp(a.order, b.order, sizeof(int) * a.count));
    prompt_session_init(&b, PROMPT_WORDS, 18);
    assert(memcmp(a.order, b.order, sizeof(int) * a.count) != 0);

    /* every index appears exactly once */
    int seen[64] = {0};
    for (int i = 0; i < a.count; i++) seen[a.order[i]]++;
    for (int i = 0; i < a.count; i++) assert(seen[i] == 1);

    /* walk to the end */
    int steps = 1;
    while (prompt_advance(&a)) steps++;
    assert(steps == a.count);

    prompt_session_init(&a, PROMPT_SENTENCES, 1); assert(a.count == KWS_NUM_SENTENCE_PROMPTS);
    prompt_session_init(&a, PROMPT_NEGS, 1);      assert(a.count == KWS_NUM_NEG_PROMPTS);
    assert(strlen(prompt_text(&a)) > 0 && strlen(prompt_slug(&a)) > 0);
    assert(prompt_cap_ms(PROMPT_WORDS) == 4000 && prompt_cap_ms(PROMPT_NEGS) == 6000);
    puts("test_prompts OK");
    return 0;
}
```

- [ ] **Step 2: Run to verify failure**

Run: `make -C firmware/test`
Expected: compile errors `mfcc.h: No such file or directory` (and the rest).

- [ ] **Step 3: Implement `mfcc.c`/`mfcc.h`**

`firmware/main/mfcc.h`: the declarations from the Interfaces block (verbatim), wrapped in `#pragma once`.

`firmware/main/mfcc.c`:

```c
#include "mfcc.h"
#include <math.h>
#include <string.h>

/* 480 is 2^5*3*5 — not a power of two — so we do a plain DFT with a twiddle
   table. ~116k MACs per frame; a 100 ms inference only adds 5 frames, so this
   stays far below 1 ms of CPU on the S3. ponytail: swap for esp-dsp mixed
   radix if the front end ever shows up in the profile. */
static float s_cos[KWS_WIN], s_sin[KWS_WIN];
static int s_twiddle_ready;

static void twiddle_init(void)
{
    if (s_twiddle_ready) return;
    for (int n = 0; n < KWS_WIN; n++) {
        s_cos[n] = cosf(2.f * (float)M_PI * n / KWS_WIN);
        s_sin[n] = sinf(2.f * (float)M_PI * n / KWS_WIN);
    }
    s_twiddle_ready = 1;
}

static void frame_logmel(const int16_t pcm[KWS_WIN], float logmel[KWS_N_MELS])
{
    float x[KWS_WIN], power[KWS_N_BINS];
    for (int n = 0; n < KWS_WIN; n++) x[n] = (pcm[n] / 32768.f) * KWS_WINDOW[n];
    for (int k = 0; k < KWS_N_BINS; k++) {
        float re = 0.f, im = 0.f;
        for (int n = 0; n < KWS_WIN; n++) {
            int idx = (k * n) % KWS_WIN;
            re += x[n] * s_cos[idx];
            im -= x[n] * s_sin[idx];
        }
        power[k] = re * re + im * im;
    }
    for (int m = 0; m < KWS_N_MELS; m++) {
        float acc = 0.f;
        for (int k = 0; k < KWS_N_BINS; k++) acc += KWS_MEL[m][k] * power[k];
        logmel[m] = 10.f * log10f(acc > KWS_AMIN ? acc : KWS_AMIN);
    }
}

void mfcc_init(mfcc_state_t *s)
{
    twiddle_init();
    memset(s, 0, sizeof *s);
}

void mfcc_push_frame(mfcc_state_t *s, const int16_t pcm[KWS_WIN])
{
    int slot = (s->head + s->count) % KWS_N_FRAMES;
    if (s->count == KWS_N_FRAMES) { slot = s->head; s->head = (s->head + 1) % KWS_N_FRAMES; }
    else s->count++;
    frame_logmel(pcm, s->logmel[slot]);
}

void mfcc_finish(const mfcc_state_t *s, float out[KWS_N_FRAMES][KWS_N_MFCC])
{
    static float zero_row[KWS_N_MELS];
    static int zero_ready;
    if (!zero_ready) {               /* log-mel of an all-zero frame = 10*log10(AMIN) */
        for (int m = 0; m < KWS_N_MELS; m++) zero_row[m] = 10.f * log10f(KWS_AMIN);
        zero_ready = 1;
    }
    const float *rows[KWS_N_FRAMES];
    float peak = -1e30f;
    for (int t = 0; t < KWS_N_FRAMES; t++) {
        rows[t] = t < s->count ? s->logmel[(s->head + t) % KWS_N_FRAMES] : zero_row;
        for (int m = 0; m < KWS_N_MELS; m++) if (rows[t][m] > peak) peak = rows[t][m];
    }
    float floor_db = peak - KWS_TOP_DB;
    for (int t = 0; t < KWS_N_FRAMES; t++)
        for (int c = 0; c < KWS_N_MFCC; c++) {
            float acc = 0.f;
            for (int m = 0; m < KWS_N_MELS; m++) {
                float v = rows[t][m] < floor_db ? floor_db : rows[t][m];
                acc += KWS_DCT[c][m] * v;
            }
            out[t][c] = acc;
        }
}

void mfcc_compute(const int16_t *pcm, float out[KWS_N_FRAMES][KWS_N_MFCC])
{
    mfcc_state_t s;
    mfcc_init(&s);
    for (int t = 0; t < KWS_N_FRAMES; t++) mfcc_push_frame(&s, pcm + t * KWS_HOP);
    mfcc_finish(&s, out);
}

void mfcc_quantize(const float in[KWS_N_FRAMES][KWS_N_MFCC], int8_t *out, float scale, int zero_point)
{
    for (int t = 0; t < KWS_N_FRAMES; t++)
        for (int c = 0; c < KWS_N_MFCC; c++) {
            long q = lroundf(in[t][c] / scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            *out++ = (int8_t)q;
        }
}
```

`M_PI` needs `#define _USE_MATH_DEFINES`/`-D_GNU_SOURCE` on some libcs; if the host build complains, define `KWS_PI 3.14159265358979f` locally instead of `M_PI`.

- [ ] **Step 4: Implement `stream.c`/`stream.h`** (line-by-line port of `KeywordStream.push`)

```c
#include "stream.h"
#include <string.h>

void stream_reset(stream_t *s)
{
    memset(s, 0, sizeof *s);
    s->run_label = -1;
    s->last_fired_label = -1;
}

int stream_push(stream_t *s, const float posterior[KWS_NUM_LABELS])
{
    memcpy(s->hist[s->hist_pos], posterior, sizeof(float) * KWS_NUM_LABELS);
    s->hist_pos = (s->hist_pos + 1) % KWS_SMOOTH_WIN;
    if (s->hist_len < KWS_SMOOTH_WIN) s->hist_len++;

    float best = -1.f; int idx = 0;
    for (int k = 0; k < KWS_NUM_LABELS; k++) {
        float m = 0.f;
        for (int h = 0; h < s->hist_len; h++) m += s->hist[h][k];
        m /= s->hist_len;
        if (m > best) { best = m; idx = k; }
    }
    int candidate = (best >= KWS_THRESHOLD && idx != KWS_SILENCE_INDEX) ? idx : -1;

    if (candidate == s->run_label) s->run_len++;
    else { s->run_label = candidate; s->run_len = 1; s->run_fired = 0; }

    int fired = -1;
    if (candidate >= 0 && s->run_len >= KWS_MIN_CONSECUTIVE && !s->run_fired) {
        int gap_ok = candidate != s->last_fired_label || s->gap_since_last_fired >= KWS_GAP_STEPS;
        if (gap_ok) {
            fired = candidate;
            s->run_fired = 1;
            s->last_fired_label = candidate;
            s->gap_since_last_fired = 0;
        }
    }
    if (s->last_fired_label >= 0 && candidate != s->last_fired_label) s->gap_since_last_fired++;
    return fired;
}
```

(`np.argmax` returns the FIRST max on ties; `m > best` with `best=-1` preserves that.)

- [ ] **Step 5: Implement `wav.c`/`wav.h`**

```c
#include "wav.h"
#include <string.h>

static void put32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void put16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }

void wav_write_header(uint8_t out[WAV_HEADER_BYTES], uint32_t n_samples, uint32_t sample_rate)
{
    uint32_t data_bytes = n_samples * 2;   /* mono int16 */
    memcpy(out, "RIFF", 4);      put32(out + 4, 36 + data_bytes);
    memcpy(out + 8, "WAVEfmt ", 8);
    put32(out + 16, 16);         put16(out + 20, 1); put16(out + 22, 1);
    put32(out + 24, sample_rate); put32(out + 28, sample_rate * 2);
    put16(out + 32, 2);          put16(out + 34, 16);
    memcpy(out + 36, "data", 4); put32(out + 40, data_bytes);
}
```

- [ ] **Step 6: Implement `prompts.c`/`prompts.h`**

`firmware/main/prompts.h` includes `<stdint.h>` and `"gen/prompts.h"` — **rename the generated file's include path collision**: our module header is `firmware/main/prompts.h`, the generated one is `firmware/main/gen/prompts.h`; because both dirs are on the include path, include the generated one as `#include "gen/prompts.h"` and make `firmware/main/CMakeLists.txt` `INCLUDE_DIRS "."` only (drop `"gen"`), so every generated header is referenced as `gen/<name>.h`. Update `mfcc.h` (`gen/features_config.h`), `stream.h` (`gen/labels.h`), the tests (`gen/test_vectors.h`) and the Makefile (`-I../main` only) accordingly.

```c
#include "prompts.h"
#include <string.h>

static uint32_t xorshift(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static void set_tables(prompt_set_t set, const char *const **text, const char *const **slug, int *n)
{
    switch (set) {
    case PROMPT_SENTENCES: *text = KWS_SENTENCE_PROMPTS; *slug = KWS_SENTENCE_SLUGS; *n = KWS_NUM_SENTENCE_PROMPTS; break;
    case PROMPT_NEGS:      *text = KWS_NEG_PROMPTS;      *slug = KWS_NEG_SLUGS;      *n = KWS_NUM_NEG_PROMPTS;      break;
    default:               *text = KWS_WORD_PROMPTS;     *slug = KWS_WORD_SLUGS;     *n = KWS_NUM_WORD_PROMPTS;     break;
    }
}

void prompt_session_init(prompt_session_t *p, prompt_set_t set, uint32_t seed)
{
    const char *const *t, *const *s; int n;
    set_tables(set, &t, &s, &n);
    p->set = set; p->seed = seed; p->count = n; p->index = 0;
    for (int i = 0; i < n; i++) p->order[i] = i;
    uint32_t rng = seed ? seed : 0x9E3779B9u;
    for (int i = n - 1; i > 0; i--) {       /* Fisher–Yates */
        int j = (int)(xorshift(&rng) % (uint32_t)(i + 1));
        int tmp = p->order[i]; p->order[i] = p->order[j]; p->order[j] = tmp;
    }
}

const char *prompt_text(const prompt_session_t *p)
{
    const char *const *t, *const *s; int n; set_tables(p->set, &t, &s, &n);
    return t[p->order[p->index]];
}

const char *prompt_slug(const prompt_session_t *p)
{
    const char *const *t, *const *s; int n; set_tables(p->set, &t, &s, &n);
    return s[p->order[p->index]];
}

int prompt_advance(prompt_session_t *p)
{
    if (p->index + 1 >= p->count) return 0;
    p->index++;
    return 1;
}

uint32_t prompt_cap_ms(prompt_set_t set) { return set == PROMPT_WORDS ? 4000 : 6000; }
```

`order[64]` is enough: 24 words, ≤ 49 sentences, 20 negatives. Add `_Static_assert(KWS_NUM_SENTENCE_PROMPTS <= 64, "grow prompt_session_t.order")` in `prompts.c`.

- [ ] **Step 7: Run host tests**

Run: `make -C firmware/test`
Expected: four `... OK` lines and `host tests OK`. Then run the firmware build (Task 1 docker command) to confirm the new sources compile for xtensa too (`-Werror` is not on there, but `printf("%g")` etc. must not warn).

- [ ] **Step 8: CI job**

Append to `.github/workflows/firmware.yml`:

```yaml
  host-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - run: make -C firmware/test
```

- [ ] **Step 9: Commit**

```bash
git add firmware/main/{mfcc,stream,wav,prompts}.{c,h} firmware/main/CMakeLists.txt firmware/test .github/workflows/firmware.yml
git commit -m "feat(firmware): pure-C mfcc/stream/wav/prompts units with host tests; CI host-test job"
```

---

### Task 4: `kws-export --v2 --firmware` — model-derived headers

**Files:**
- Modify: `kws_de/export.py` (`main`, new `write_model_config`)
- Create: `tests/test_export_firmware.py`, `firmware/main/gen/model_data.h`, `firmware/main/gen/model_config.h`

**Interfaces:**
- Consumes: `to_int8_tflite`, `balanced_calibration`, `write_c_array` (existing), `tf.lite.Interpreter` input/output details.
- Produces: `gen/model_data.h` (`g_model[]`, `g_model_len` — existing `write_c_array` format), `gen/model_config.h`:

```c
#define KWS_MODEL_INPUT_SCALE 3.0357947e+00f
#define KWS_MODEL_INPUT_ZERO_POINT 80
#define KWS_MODEL_OUTPUT_SCALE 3.90625e-03f
#define KWS_MODEL_OUTPUT_ZERO_POINT -128
#define KWS_MODEL_ARENA_BYTES 120000   /* measured peak * 1.2, rounded up to 4 KB */
#define KWS_MODEL_NUM_CLASSES 26
```

- [ ] **Step 1: Failing test** — `tests/test_export_firmware.py`

```python
import numpy as np

from kws_de import config
from kws_de.export import balanced_calibration, to_int8_tflite, write_model_config
from kws_de.model import build_dscnn


def test_write_model_config_reports_quant_and_arena(tmp_path):
    rng = np.random.default_rng(0)
    model = build_dscnn(num_classes=len(config.COMMAND_LABELS))
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(model, rep)
    p = tmp_path / "model_config.h"
    info = write_model_config(blob, p)
    txt = p.read_text()
    assert "#define KWS_MODEL_INPUT_SCALE" in txt and "#define KWS_MODEL_INPUT_ZERO_POINT" in txt
    assert f"#define KWS_MODEL_NUM_CLASSES {len(config.COMMAND_LABELS)}" in txt
    assert info["arena_bytes"] % 4096 == 0 and info["arena_bytes"] > 0
    assert f"#define KWS_MODEL_ARENA_BYTES {info['arena_bytes']}" in txt
```

Check `build_dscnn`'s signature first (`grep -n "def build_dscnn" kws_de/model.py`); if it takes `n_classes` or reads `config.NUM_CLASSES`, adapt the call the way `tests/test_export_v2.py` does.

- [ ] **Step 2: Run** — `uv run pytest tests/test_export_firmware.py -q` → FAIL `ImportError: write_model_config`.

- [ ] **Step 3: Implement in `kws_de/export.py`**

```python
def write_model_config(tflite: bytes, path) -> dict:
    """Quantisation params + a TFLM arena estimate for firmware/main/gen/model_config.h.
    The arena is the desktop interpreter's tensor-memory peak (a proxy for
    TFLM's planner) x1.2, rounded up to 4 KB; the device logs the real
    `arena_used_bytes()` so the constant can be tightened later."""
    itp = tf.lite.Interpreter(model_content=tflite)
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    tensor_bytes = sum(
        int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        for t in itp.get_tensor_details()
        if len(t["shape"]) and t["dtype"] == np.int8
    )
    arena = -(-int(tensor_bytes * 1.2) // 4096) * 4096
    lines = [
        "/* generated by kws-export --firmware — do not edit */",
        "#pragma once",
        f"#define KWS_MODEL_INPUT_SCALE {inp['quantization'][0]:.8e}f",
        f"#define KWS_MODEL_INPUT_ZERO_POINT {int(inp['quantization'][1])}",
        f"#define KWS_MODEL_OUTPUT_SCALE {out['quantization'][0]:.8e}f",
        f"#define KWS_MODEL_OUTPUT_ZERO_POINT {int(out['quantization'][1])}",
        f"#define KWS_MODEL_ARENA_BYTES {arena}",
        f"#define KWS_MODEL_NUM_CLASSES {int(out['shape'][-1])}",
    ]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")
    return {"arena_bytes": arena, "input_scale": float(inp["quantization"][0])}
```

In `main()` add `ap.add_argument("--firmware", action="store_true", help="also write firmware/main/gen/{model_data,model_config}.h (implies --v2)")`; after parsing, `if args.firmware: args.v2 = True`; after the existing writes:

```python
    if args.firmware:
        gen = pathlib.Path("firmware/main/gen")
        gen.mkdir(parents=True, exist_ok=True)
        write_c_array(blob, gen / "model_data.h")
        write_model_config(blob, gen / "model_config.h")
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_export_firmware.py tests/test_export.py tests/test_export_v2.py -q` → PASS.

- [ ] **Step 5: Generate real headers from the distilled model and build**

```bash
uv run kws-export --firmware && ls -l firmware/main/gen/model_*.h && head -3 firmware/main/gen/model_config.h
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG idf.py build
```

If `models/command.keras` is missing → report BLOCKED (do not fabricate a blob). Expected `g_model_len` ≈ 20 000–21 000 (the distilled DS-CNN is 20 272 bytes per `docs/distill-report.md`).

- [ ] **Step 6: Commit**

```bash
git add kws_de/export.py tests/test_export_firmware.py firmware/main/gen/model_data.h firmware/main/gen/model_config.h
git commit -m "feat: kws-export --firmware writes model_data.h + model_config.h; commit distilled v2 model headers"
```

---

### Task 5: Audio ring buffer, storage mount, energy VAD, guided recorder + LVGL screen

**Ruling (deviation from spec §2, recorded here for the reviewer):** the spec names ESP-SR AFE VAD mode 3. esp-sr v2's VAD needs its model files in a dedicated `model` partition plus the AFE pipeline; for a prompted recorder where the speaker is looking at the screen, an energy VAD (RMS over 20 ms frames against an adaptive noise floor) is sufficient, has no partition/flash cost, is host-testable, and removes esp-sr from the build entirely. Cost if wrong: worse end-pointing in noisy rooms; upgrade path is swapping `vad.c` for the AFE fetch loop. Drop `espressif/esp-sr` from `idf_component.yml` in this task.

**Files:**
- Create: `firmware/main/audio.c/.h`, `firmware/main/storage.c/.h`, `firmware/main/vad.c/.h`, `firmware/main/record.c/.h`, `firmware/main/ui/ui.h`, `firmware/main/ui/ui_record.c`, `firmware/test/test_vad.c`
- Modify: `firmware/main/main.c`, `firmware/main/CMakeLists.txt`, `firmware/main/idf_component.yml`, `firmware/test/Makefile`

**Interfaces:**

```c
/* audio.h — always-on capture into a ring buffer */
#define AUDIO_RING_SAMPLES (KWS_SAMPLE_RATE * 10)          /* 10 s */
void     audio_start(void);                                  /* codec init + task, never returns error silently: abort() on failure */
uint32_t audio_write_pos(void);                              /* monotonically increasing sample counter */
/* Copy `n` samples ending at absolute position `end` (end - n must be inside the ring). */
void     audio_read(uint32_t end, int16_t *dst, uint32_t n);

/* storage.h */
esp_err_t storage_mount(void);      /* wear-levelled FAT "storage" partition at /rec */
esp_err_t storage_unmount(void);
uint64_t  storage_free_bytes(void);
#define STORAGE_MIN_FREE_BYTES (200 * 1024)

/* vad.h — pure C, host-testable */
typedef struct { float noise; int speech_frames; int silence_frames; int in_speech; } vad_t;
void vad_reset(vad_t *v);
/* One 20 ms frame (KWS_HOP samples). Returns 1 while speech is active. Speech opens at
   rms > max(noise*4, 300) for 2 consecutive frames; noise floor tracks rms exponentially (alpha 0.05)
   only while not in speech. Closes after VAD_TRAILING_FRAMES (=25 → 500 ms) below threshold. */
int  vad_push(vad_t *v, const int16_t *frame, int n);
#define VAD_TRAILING_FRAMES 25

/* record.h */
typedef enum { REC_CMD_REDO, REC_CMD_SKIP, REC_CMD_NEXT, REC_CMD_NEW_SPEAKER, REC_CMD_SET_WORDS, REC_CMD_SET_SENTENCES, REC_CMD_SET_NEGS, REC_CMD_PAUSE, REC_CMD_RESUME } record_cmd_t;
typedef enum { REC_IDLE, REC_LISTENING, REC_CAPTURING, REC_SAVED, REC_CLIPPED, REC_TIMEOUT, REC_FULL, REC_DONE } record_phase_t;
typedef struct {
    record_phase_t phase;
    prompt_set_t set; uint32_t seed; int index, count;
    char prompt[96]; char speaker[8];      /* "spk03" */
    float level_dbfs;                      /* for the bar, updated every 100 ms */
} record_status_t;
void record_start(void);                        /* creates the task (starts paused) */
void record_post(record_cmd_t cmd);             /* from UI callbacks */
void record_get_status(record_status_t *out);   /* copy under mutex */

/* ui/ui.h */
void ui_init(void);                                       /* after bsp_display_start */
void ui_show_record(void);
void ui_record_refresh(const record_status_t *st);        /* called from record task, takes bsp_display_lock */
void ui_show_usb(void);                                   /* Task 6 */
void ui_show_recognise(void);                             /* Task 7 */
typedef enum { UI_MODE_RECORD, UI_MODE_USB, UI_MODE_RECOGNISE } ui_mode_t;
void app_set_mode(ui_mode_t m);                           /* in main.c; the only place that suspends/resumes consumers */
```

- [ ] **Step 1: VAD test first** — `firmware/test/test_vad.c` (add `test_vad: test_vad.c ../main/vad.c` with `-lm` to the Makefile and to `TESTS`)

```c
#include <assert.h>
#include <stdio.h>
#include "vad.h"
#include "gen/features_config.h"

static void fill(int16_t *f, int n, int amp) { for (int i = 0; i < n; i++) f[i] = (i & 1) ? amp : -amp; }

int main(void)
{
    vad_t v; vad_reset(&v);
    int16_t f[KWS_HOP];
    fill(f, KWS_HOP, 50);                      /* quiet room */
    for (int i = 0; i < 50; i++) assert(vad_push(&v, f, KWS_HOP) == 0);
    fill(f, KWS_HOP, 4000);                    /* speech */
    assert(vad_push(&v, f, KWS_HOP) == 0);     /* needs 2 consecutive frames */
    assert(vad_push(&v, f, KWS_HOP) == 1);
    for (int i = 0; i < 20; i++) assert(vad_push(&v, f, KWS_HOP) == 1);
    fill(f, KWS_HOP, 50);
    for (int i = 0; i < VAD_TRAILING_FRAMES - 1; i++) assert(vad_push(&v, f, KWS_HOP) == 1);
    assert(vad_push(&v, f, KWS_HOP) == 0);     /* closes exactly after 500 ms of silence */
    puts("test_vad OK");
    return 0;
}
```

- [ ] **Step 2: `vad.c`**

```c
#include "vad.h"
#include <math.h>

void vad_reset(vad_t *v) { v->noise = 300.f; v->speech_frames = v->silence_frames = v->in_speech = 0; }

int vad_push(vad_t *v, const int16_t *frame, int n)
{
    double acc = 0;
    for (int i = 0; i < n; i++) acc += (double)frame[i] * frame[i];
    float rms = sqrtf((float)(acc / n));
    float thr = v->noise * 4.f > 300.f ? v->noise * 4.f : 300.f;
    if (!v->in_speech) v->noise += 0.05f * (rms - v->noise);
    if (rms > thr) { v->speech_frames++; v->silence_frames = 0; }
    else { v->silence_frames++; v->speech_frames = 0; }
    if (!v->in_speech && v->speech_frames >= 2) v->in_speech = 1;
    if (v->in_speech && v->silence_frames >= VAD_TRAILING_FRAMES) { v->in_speech = 0; v->speech_frames = 0; }
    return v->in_speech;
}
```

Run `make -C firmware/test` → `test_vad OK`.

- [ ] **Step 3: `audio.c`** (BSP codec → ring buffer)

```c
#include "audio.h"
#include <string.h>
#include "bsp/esp-bsp.h"
#include "esp_codec_dev.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "audio";
static int16_t *s_ring;                 /* PSRAM, AUDIO_RING_SAMPLES */
static volatile uint32_t s_pos;         /* absolute sample count written */
static esp_codec_dev_handle_t s_mic;

static void audio_task(void *arg)
{
    int16_t chunk[KWS_HOP * 2];         /* stereo read, 20 ms */
    for (;;) {
        ESP_ERROR_CHECK(esp_codec_dev_read(s_mic, chunk, sizeof chunk));
        uint32_t pos = s_pos;
        for (int i = 0; i < KWS_HOP; i++) s_ring[(pos + i) % AUDIO_RING_SAMPLES] = chunk[2 * i];  /* left mic */
        s_pos = pos + KWS_HOP;
    }
}

void audio_start(void)
{
    s_ring = heap_caps_calloc(AUDIO_RING_SAMPLES, sizeof(int16_t), MALLOC_CAP_SPIRAM);
    assert(s_ring);
    s_mic = bsp_audio_codec_microphone_init();
    assert(s_mic);
    esp_codec_dev_sample_info_t fs = {.bits_per_sample = 16, .channel = 2, .sample_rate = KWS_SAMPLE_RATE};
    ESP_ERROR_CHECK(esp_codec_dev_open(s_mic, &fs));
    ESP_ERROR_CHECK(esp_codec_dev_set_in_gain(s_mic, 30.0));
    xTaskCreatePinnedToCore(audio_task, "audio", 4096, NULL, 10, NULL, 1);
    ESP_LOGI(TAG, "capture running");
}

uint32_t audio_write_pos(void) { return s_pos; }

void audio_read(uint32_t end, int16_t *dst, uint32_t n)
{
    uint32_t start = end - n;
    for (uint32_t i = 0; i < n; i++) dst[i] = s_ring[(start + i) % AUDIO_RING_SAMPLES];
}
```

The CoreS3 BSP's microphone path is stereo (ES7210, two mics); if `esp_codec_dev_open` rejects `channel = 2`, check `managed_components/espressif__m5stack_core_s3/` examples for the accepted config and use that — never silence the error.

- [ ] **Step 4: `storage.c`**

```c
#include "storage.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "wear_levelling.h"

static const char *TAG = "storage";
static wl_handle_t s_wl = WL_INVALID_HANDLE;

esp_err_t storage_mount(void)
{
    esp_vfs_fat_mount_config_t cfg = {.max_files = 4, .format_if_mount_failed = true, .allocation_unit_size = 4096};
    esp_err_t err = esp_vfs_fat_spiflash_mount_rw_wl("/rec", "storage", &cfg, &s_wl);
    ESP_LOGI(TAG, "mount /rec: %s, free %llu KB", esp_err_to_name(err), storage_free_bytes() / 1024);
    return err;
}

esp_err_t storage_unmount(void)
{
    esp_err_t err = esp_vfs_fat_spiflash_unmount_rw_wl("/rec", s_wl);
    s_wl = WL_INVALID_HANDLE;
    return err;
}

wl_handle_t storage_wl_handle(void) { return s_wl; }   /* Task 6 needs it for MSC; add to storage.h */

uint64_t storage_free_bytes(void)
{
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info("/rec", &total, &free_b) != ESP_OK) return 0;
    return free_b;
}
```

- [ ] **Step 5: `record.c`** — the state machine

```c
#include "record.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "audio.h"
#include "storage.h"
#include "vad.h"
#include "wav.h"
#include "prompts.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "ui/ui.h"

static const char *TAG = "record";
#define PREROLL_SAMPLES (KWS_SAMPLE_RATE * 300 / 1000)
#define NO_SPEECH_MS 8000
#define HOLD_MS 700

static QueueHandle_t s_cmds;
static SemaphoreHandle_t s_lock;
static record_status_t s_st;
static prompt_session_t s_prompts;
static uint32_t s_speaker;
static int s_paused = 1;
static int16_t *s_take;                       /* PSRAM, 6 s + pre-roll */
#define TAKE_MAX (KWS_SAMPLE_RATE * 6 + PREROLL_SAMPLES)

static void status_set(record_phase_t ph)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_st.phase = ph;
    s_st.set = s_prompts.set; s_st.seed = s_prompts.seed;
    s_st.index = s_prompts.index; s_st.count = s_prompts.count;
    strlcpy(s_st.prompt, prompt_text(&s_prompts), sizeof s_st.prompt);
    snprintf(s_st.speaker, sizeof s_st.speaker, "spk%02lu", (unsigned long)s_speaker);
    record_status_t copy = s_st;
    xSemaphoreGive(s_lock);
    ui_record_refresh(&copy);
}

static void nvs_load(void)
{
    nvs_handle_t h;
    ESP_ERROR_CHECK(nvs_open("kws", NVS_READWRITE, &h));
    if (nvs_get_u32(h, "speaker", &s_speaker) != ESP_OK) { s_speaker = 1; nvs_set_u32(h, "speaker", 1); }
    nvs_close(h);
}

static void nvs_bump_speaker(void)
{
    nvs_handle_t h;
    ESP_ERROR_CHECK(nvs_open("kws", NVS_READWRITE, &h));
    s_speaker++;
    nvs_set_u32(h, "speaker", s_speaker);
    nvs_commit(h);
    nvs_close(h);
}

/* /rec/spk03/licht/001.wav | /rec/spk03/_phrase_/licht-hinten-an_001.wav | /rec/spk03/_neg_/... */
static int next_path(char *out, size_t n)
{
    char dir[64];
    const char *sub = s_prompts.set == PROMPT_WORDS ? prompt_slug(&s_prompts)
                    : s_prompts.set == PROMPT_SENTENCES ? "_phrase_" : "_neg_";
    snprintf(dir, sizeof dir, "/rec/%s", s_st.speaker);            mkdir(dir, 0777);
    snprintf(dir, sizeof dir, "/rec/%s/%s", s_st.speaker, sub);    mkdir(dir, 0777);
    for (int i = 1; i < 1000; i++) {
        struct stat st;
        if (s_prompts.set == PROMPT_WORDS) snprintf(out, n, "%s/%03d.wav", dir, i);
        else snprintf(out, n, "%s/%s_%03d.wav", dir, prompt_slug(&s_prompts), i);
        if (stat(out, &st) != 0) return 0;
    }
    return -1;
}

static void append_session_csv(const char *path, uint32_t ms, float peak_dbfs)
{
    char csv[64];
    snprintf(csv, sizeof csv, "/rec/%s/session.csv", s_st.speaker);
    struct stat st; int fresh = stat(csv, &st) != 0;
    FILE *f = fopen(csv, "a");
    if (!f) { ESP_LOGE(TAG, "csv open failed"); return; }
    if (fresh) fputs("prompt,file,ms,peak_dbfs,set,seed,ts\n", f);
    static const char *setname[] = {"words", "sentences", "negatives"};
    fprintf(f, "\"%s\",%s,%lu,%.1f,%s,%lu,%lld\n", prompt_text(&s_prompts), path + 5 /* strip /rec/ */,
            (unsigned long)ms, peak_dbfs, setname[s_prompts.set], (unsigned long)s_prompts.seed,
            esp_timer_get_time() / 1000);
    fclose(f);
}

static int save_take(uint32_t n_samples, float peak_dbfs)
{
    char path[128];
    if (next_path(path, sizeof path) != 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { ESP_LOGE(TAG, "open %s failed", path); return -1; }
    uint8_t hdr[WAV_HEADER_BYTES];
    wav_write_header(hdr, n_samples, KWS_SAMPLE_RATE);
    fwrite(hdr, 1, sizeof hdr, f);
    fwrite(s_take, sizeof(int16_t), n_samples, f);
    fclose(f);
    append_session_csv(path, n_samples * 1000 / KWS_SAMPLE_RATE, peak_dbfs);
    ESP_LOGI(TAG, "saved %s (%lu samples)", path, (unsigned long)n_samples);
    return 0;
}

/* Returns: 0 saved, 1 redo (clipped/timeout/full), -1 command interrupted (cmd in *cmd) */
static int capture_one(record_cmd_t *cmd)
{
    if (storage_free_bytes() < STORAGE_MIN_FREE_BYTES) { status_set(REC_FULL); return 1; }
    vad_t vad; vad_reset(&vad);
    int16_t frame[KWS_HOP];
    uint32_t cap = prompt_cap_ms(s_prompts.set) * (KWS_SAMPLE_RATE / 1000);
    uint32_t cursor = audio_write_pos();
    uint32_t speech_start = 0, n = 0, idle_frames = 0;
    int peak = 0, capturing = 0;
    status_set(REC_LISTENING);
    for (;;) {
        if (xQueueReceive(s_cmds, cmd, 0) == pdTRUE) return -1;
        while (audio_write_pos() < cursor + KWS_HOP) vTaskDelay(pdMS_TO_TICKS(5));
        audio_read(cursor + KWS_HOP, frame, KWS_HOP);
        cursor += KWS_HOP;
        int active = vad_push(&vad, frame, KWS_HOP);
        for (int i = 0; i < KWS_HOP; i++) { int a = frame[i] < 0 ? -frame[i] : frame[i]; if (a > peak) peak = a; }
        if ((cursor / KWS_HOP) % 5 == 0) {        /* level bar every 100 ms */
            xSemaphoreTake(s_lock, portMAX_DELAY);
            s_st.level_dbfs = 20.f * log10f((vad.noise > 1 ? vad.noise : 1) / 32768.f);
            xSemaphoreGive(s_lock);
            ui_record_refresh(&s_st);
        }
        if (!capturing) {
            if (active) {
                capturing = 1; peak = 0;
                speech_start = cursor - KWS_HOP - PREROLL_SAMPLES;
                n = PREROLL_SAMPLES + KWS_HOP;
                audio_read(cursor, s_take, n);
                status_set(REC_CAPTURING);
            } else if (++idle_frames * 20 >= NO_SPEECH_MS) { status_set(REC_TIMEOUT); return 1; }
            continue;
        }
        memcpy(s_take + n, frame, sizeof frame); n += KWS_HOP;
        if (peak >= 32767) { status_set(REC_CLIPPED); return 1; }
        if (!active || n >= cap + PREROLL_SAMPLES) break;
    }
    (void)speech_start;
    float peak_dbfs = 20.f * log10f((peak > 0 ? peak : 1) / 32768.f);
    if (save_take(n, peak_dbfs) != 0) { status_set(REC_FULL); return 1; }
    status_set(REC_SAVED);
    vTaskDelay(pdMS_TO_TICKS(HOLD_MS));
    return 0;
}

static void record_task(void *arg)
{
    record_cmd_t cmd;
    for (;;) {
        if (s_paused) { xQueueReceive(s_cmds, &cmd, portMAX_DELAY); }
        else {
            int r = capture_one(&cmd);
            if (r == 0 || r == 1) {
                if (r == 0 && !prompt_advance(&s_prompts)) { status_set(REC_DONE); s_paused = 1; }
                if (r == 1) vTaskDelay(pdMS_TO_TICKS(HOLD_MS));
                continue;
            }
        }
        switch (cmd) {                                    /* r == -1 or woken while paused */
        case REC_CMD_PAUSE:  s_paused = 1; status_set(REC_IDLE); break;
        case REC_CMD_RESUME: s_paused = 0; break;
        case REC_CMD_REDO:   break;                       /* loop re-captures the same prompt */
        case REC_CMD_SKIP:
        case REC_CMD_NEXT:   if (!prompt_advance(&s_prompts)) { status_set(REC_DONE); s_paused = 1; } break;
        case REC_CMD_NEW_SPEAKER: nvs_bump_speaker(); prompt_session_init(&s_prompts, s_prompts.set, s_prompts.seed + 1); status_set(REC_IDLE); break;
        case REC_CMD_SET_WORDS:     prompt_session_init(&s_prompts, PROMPT_WORDS,     (uint32_t)esp_timer_get_time()); status_set(REC_IDLE); break;
        case REC_CMD_SET_SENTENCES: prompt_session_init(&s_prompts, PROMPT_SENTENCES, (uint32_t)esp_timer_get_time()); status_set(REC_IDLE); break;
        case REC_CMD_SET_NEGS:      prompt_session_init(&s_prompts, PROMPT_NEGS,      (uint32_t)esp_timer_get_time()); status_set(REC_IDLE); break;
        }
    }
}

void record_start(void)
{
    s_cmds = xQueueCreate(8, sizeof(record_cmd_t));
    s_lock = xSemaphoreCreateMutex();
    s_take = heap_caps_malloc(TAKE_MAX * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    assert(s_take);
    nvs_load();
    prompt_session_init(&s_prompts, PROMPT_WORDS, (uint32_t)esp_timer_get_time() | 1);
    snprintf(s_st.speaker, sizeof s_st.speaker, "spk%02lu", (unsigned long)s_speaker);
    xTaskCreatePinnedToCore(record_task, "record", 8192, NULL, 5, NULL, 0);
    status_set(REC_IDLE);
}

void record_post(record_cmd_t cmd) { xQueueSend(s_cmds, &cmd, 0); }

void record_get_status(record_status_t *out)
{
    xSemaphoreTake(s_lock, portMAX_DELAY); *out = s_st; xSemaphoreGive(s_lock);
}
```

Seed display: the seed is `esp_timer_get_time()` truncated — show it as `seed %lu` on screen and in `session.csv` (column `seed`) so a session's order is reproducible with `prompt_session_init`.

- [ ] **Step 6: `ui/ui.h` + `ui/ui_record.c`** (LVGL 9)

```c
#include "ui.h"
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "lvgl.h"
#include "record.h"

static lv_obj_t *scr, *l_set, *l_counter, *l_speaker, *l_prompt, *bar, *l_phase, *b_next;

static void on_cmd(lv_event_t *e) { record_post((record_cmd_t)(intptr_t)lv_event_get_user_data(e)); }
static void on_mode_recognise(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECOGNISE); }
static void on_mode_usb(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_USB); }

static lv_obj_t *button(lv_obj_t *parent, const char *txt, lv_event_cb_t cb, void *ud, int x, int y, int w)
{
    lv_obj_t *b = lv_button_create(parent);
    lv_obj_set_size(b, w, 40);
    lv_obj_set_pos(b, x, y);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, ud);
    lv_obj_t *l = lv_label_create(b);
    lv_label_set_text(l, txt);
    lv_obj_center(l);
    return b;
}

void ui_show_record(void)
{
    bsp_display_lock(0);
    scr = lv_obj_create(NULL);
    l_set = lv_label_create(scr);     lv_obj_set_pos(l_set, 8, 6);
    l_counter = lv_label_create(scr); lv_obj_set_pos(l_counter, 200, 6);
    l_speaker = lv_label_create(scr); lv_obj_set_pos(l_speaker, 264, 6);
    l_prompt = lv_label_create(scr);  lv_obj_set_width(l_prompt, 304); lv_obj_set_pos(l_prompt, 8, 50);
    lv_obj_set_style_text_font(l_prompt, &lv_font_montserrat_28, 0);
    lv_label_set_long_mode(l_prompt, LV_LABEL_LONG_WRAP);
    lv_obj_set_style_text_align(l_prompt, LV_TEXT_ALIGN_CENTER, 0);
    bar = lv_bar_create(scr); lv_obj_set_size(bar, 200, 14); lv_obj_set_pos(bar, 60, 128);
    lv_bar_set_range(bar, -60, 0);
    l_phase = lv_label_create(scr);   lv_obj_set_pos(l_phase, 8, 148);
    button(scr, "Redo", on_cmd, (void *)REC_CMD_REDO, 8, 176, 70);
    button(scr, "Skip", on_cmd, (void *)REC_CMD_SKIP, 92, 176, 70);
    b_next = button(scr, "Next", on_cmd, (void *)REC_CMD_NEXT, 236, 176, 76);
    button(scr, "Recog", on_mode_recognise, NULL, 8, 222, 60);
    button(scr, "USB", on_mode_usb, NULL, 76, 222, 50);
    button(scr, "+Spk", on_cmd, (void *)REC_CMD_NEW_SPEAKER, 134, 222, 56);
    button(scr, "W", on_cmd, (void *)REC_CMD_SET_WORDS, 198, 222, 34);
    button(scr, "S", on_cmd, (void *)REC_CMD_SET_SENTENCES, 238, 222, 34);
    button(scr, "N", on_cmd, (void *)REC_CMD_SET_NEGS, 278, 222, 34);
    lv_screen_load(scr);
    bsp_display_unlock();
}

void ui_record_refresh(const record_status_t *st)
{
    static const char *setname[] = {"words", "sentences", "negatives"};
    static const char *phase[] = {"paused", "listening...", "recording", "saved", "CLIPPED - redo", "no speech - redo", "flash full", "set complete"};
    char buf[48];
    if (!bsp_display_lock(50)) return;          /* skip a frame rather than block the recorder */
    snprintf(buf, sizeof buf, "%s · seed %lu", setname[st->set], (unsigned long)st->seed); lv_label_set_text(l_set, buf);
    snprintf(buf, sizeof buf, "%d/%d", st->index + 1, st->count);                        lv_label_set_text(l_counter, buf);
    lv_label_set_text(l_speaker, st->speaker);
    lv_label_set_text(l_prompt, st->prompt);
    lv_label_set_text(l_phase, phase[st->phase]);
    lv_bar_set_value(bar, (int)st->level_dbfs, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(bar, st->phase == REC_CLIPPED ? lv_palette_main(LV_PALETTE_RED) : lv_palette_main(LV_PALETTE_GREEN), LV_PART_INDICATOR);
    bsp_display_unlock();
}
```

Screen is 320×240; the bottom row of six small buttons replaces the spec's mockup row (`[⇄ Recognise] [USB] [+ Speaker]`) plus the three set selectors — the mockup had no set selector, which is needed to reach sentences/negatives. `lv_font_montserrat_28` must be enabled: add `CONFIG_LV_FONT_MONTSERRAT_28=y` to `sdkconfig.defaults` (umlauts render because Montserrat covers Latin-1 in LVGL's built-in ranges; if `ü` shows as a box, fall back to `lv_font_montserrat_24` which is enabled by default in the BSP config — verify on device).

- [ ] **Step 7: `main.c` state machine**

```c
#include "bsp/esp-bsp.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "audio.h"
#include "record.h"
#include "storage.h"
#include "ui/ui.h"

static const char *TAG = "main";
static ui_mode_t s_mode = UI_MODE_RECORD;

void app_set_mode(ui_mode_t m)
{
    if (m == s_mode) return;
    ESP_LOGI(TAG, "mode %d -> %d", s_mode, m);
    if (s_mode == UI_MODE_RECORD) record_post(REC_CMD_PAUSE);
    /* Tasks 6/7 add: usb_drive_enter/exit, recognise_pause/resume */
    s_mode = m;
    if (m == UI_MODE_RECORD) { ui_show_record(); record_post(REC_CMD_RESUME); }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    bsp_i2c_init();
    bsp_display_start();
    bsp_display_backlight_on();
    ESP_ERROR_CHECK(storage_mount());
    audio_start();
    ui_show_record();
    record_start();
    record_post(REC_CMD_RESUME);
}
```

Update `firmware/main/CMakeLists.txt`:

```cmake
idf_component_register(
  SRCS "main.c" "audio.c" "storage.c" "vad.c" "record.c" "mfcc.c" "stream.c" "wav.c" "prompts.c" "ui/ui_record.c"
  INCLUDE_DIRS "."
  REQUIRES fatfs wear_levelling nvs_flash esp_timer)
```

Remove `espressif/esp-sr` from `idf_component.yml`.

- [ ] **Step 8: Build + host tests + on-device smoke (if a CoreS3 is attached)**

```bash
make -C firmware/test
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG idf.py build
```

Expected: both clean. With hardware: flash (Task 8's script or `idf.py flash` inside the container with `--device /dev/cu.usbmodem*` passed to docker), speak a word, confirm `/rec/spk01/<slug>/001.wav` appears via `idf.py monitor` log line `saved ...`. Without hardware: state so in the report; the reviewer treats the build + host tests as the gate.

- [ ] **Step 9: Commit**

```bash
git add firmware/main firmware/test
git commit -m "feat(firmware): audio ring buffer, FAT storage, energy VAD, guided recorder with LVGL screen"
```

---

### Task 6: USB mass-storage mode + `scripts/pull-recordings.sh`

**Files:**
- Create: `firmware/main/usb_drive.c/.h`, `firmware/main/ui/ui_usb.c`, `scripts/pull-recordings.sh`, `tests/test_pull_recordings.py`
- Modify: `firmware/main/main.c` (`app_set_mode`), `firmware/main/storage.c/.h` (mount goes through esp_tinyusb so MSC and FAT share the wear-levelling handle), `firmware/main/CMakeLists.txt`, `.github/workflows/firmware.yml` (shellcheck step in `host-test`)

**Interfaces:**
- Consumes: `storage_wl_handle()` (Task 5), `ui.h` `app_set_mode`, `record_post(REC_CMD_PAUSE/RESUME)`.
- Produces:

```c
/* usb_drive.h */
esp_err_t usb_drive_enter(void);   /* unmount /rec from the app, expose partition as MSC "KWSREC" */
esp_err_t usb_drive_exit(void);    /* stop USB, remount /rec */
```

- [ ] **Step 1: `scripts/pull-recordings.sh` and its test first** — `tests/test_pull_recordings.py` runs the script against a fake mount directory (`KWSREC_MOUNT` env override so no real USB is needed):

```python
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pull-recordings.sh"


def _fake_drive(root: Path) -> Path:
    mnt = root / "KWSREC"
    (mnt / "spk03" / "licht").mkdir(parents=True)
    (mnt / "spk03" / "licht" / "001.wav").write_bytes(b"RIFF" + b"\0" * 40)
    (mnt / "spk03" / "session.csv").write_text(
        "prompt,file,ms,peak_dbfs,set,seed,ts\n\"Licht\",spk03/licht/001.wav,900,-6.0,words,7,1234\n"
    )
    (mnt / "recognise.log").write_text("[Log] 1234 Licht 0.91\n")
    return mnt


def _run(mnt: Path, dest: Path):
    env = {**os.environ, "KWSREC_MOUNT": str(mnt), "KWSREC_NO_EJECT": "1"}
    return subprocess.run(["bash", str(SCRIPT), str(dest)], env=env, capture_output=True, text=True)


def test_pull_copies_appends_and_clears(tmp_path):
    mnt = _fake_drive(tmp_path)
    dest = tmp_path / "recordings"
    r = _run(mnt, dest)
    assert r.returncode == 0, r.stderr
    assert (dest / "spk03" / "licht" / "001.wav").exists()
    rows = (dest / "sessions.csv").read_text().splitlines()
    assert rows[0] == "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts"
    assert rows[1].startswith("spk03,") and rows[1].endswith(",1234")
    assert (dest / "logs").glob("recognise-*.log")
    assert not (mnt / "spk03").exists()           # cleared after successful copy


def test_pull_is_idempotent_and_needs_a_drive(tmp_path):
    mnt = _fake_drive(tmp_path)
    dest = tmp_path / "recordings"
    assert _run(mnt, dest).returncode == 0
    assert _run(mnt, dest).returncode == 0        # empty drive → no-op, still success
    assert len((dest / "sessions.csv").read_text().splitlines()) == 2
    r = subprocess.run(["bash", str(SCRIPT), str(dest)], env={**os.environ, "KWSREC_MOUNT": str(tmp_path / "nope"), "KWSREC_NO_EJECT": "1"}, capture_output=True, text=True)
    assert r.returncode == 1 and "KWSREC" in r.stderr
```

Run `uv run pytest tests/test_pull_recordings.py -q` → FAIL (script missing).

- [ ] **Step 2: the script**

```bash
#!/usr/bin/env bash
# Pull recordings off a CoreS3 in USB-drive mode into data/recordings/ and clear the device.
# Usage: scripts/pull-recordings.sh [DEST]   (default data/recordings)
# Env:   KWSREC_MOUNT=/path  override auto-detect;  KWSREC_NO_EJECT=1  skip eject (tests)
set -euo pipefail

dest=${1:-data/recordings}
mnt=${KWSREC_MOUNT:-}
if [[ -z $mnt ]]; then
  for c in /Volumes/KWSREC /media/*/KWSREC /run/media/*/KWSREC; do
    [[ -d $c ]] && mnt=$c && break
  done
fi
if [[ -z $mnt || ! -d $mnt ]]; then
  echo "no KWSREC drive mounted (put the device in USB mode, or set KWSREC_MOUNT)" >&2
  exit 1
fi

mkdir -p "$dest/logs"
pulled=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sessions="$dest/sessions.csv"
[[ -f $sessions ]] || echo "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts" > "$sessions"

shopt -s nullglob
for spk in "$mnt"/spk*/; do
  spk=${spk%/}; name=$(basename "$spk")
  rsync -a --exclude session.csv "$spk/" "$dest/$name/"
  if [[ -f $spk/session.csv ]]; then
    tail -n +2 "$spk/session.csv" | sed "s/^/$name,$pulled,/" >> "$sessions"
  fi
  rm -rf "$spk"
  echo "pulled $name"
done
if [[ -f $mnt/recognise.log ]]; then
  mv "$mnt/recognise.log" "$dest/logs/recognise-${pulled//:/-}.log"
fi

if [[ -z ${KWSREC_NO_EJECT:-} ]]; then
  if command -v diskutil >/dev/null; then diskutil eject "$mnt" >/dev/null
  elif command -v udisksctl >/dev/null; then udisksctl unmount -b "$(findmnt -n -o SOURCE "$mnt")" >/dev/null
  fi
fi
echo "done → $dest"
```

`chmod +x scripts/pull-recordings.sh`. Run the pytest → PASS. Run `shellcheck scripts/pull-recordings.sh` (install via `brew install shellcheck` if missing) → clean. `tail -n +2 | sed` keeps the quoted prompt intact because the device writes one row per line with no embedded newlines.

- [ ] **Step 3: `usb_drive.c`** — esp_tinyusb MSC owns the wear-levelling handle. Change `storage_mount()` to initialise through it so both paths share one handle:

```c
/* storage.c (replace mount/unmount bodies) */
#include "tinyusb_msc_storage.h"      /* esp_tinyusb */

esp_err_t storage_mount(void)
{
    static bool inited;
    if (!inited) {
        ESP_ERROR_CHECK(wl_mount(esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, "storage"), &s_wl));
        tinyusb_msc_spiflash_config_t cfg = {.wl_handle = s_wl};
        ESP_ERROR_CHECK(tinyusb_msc_storage_init_spiflash(&cfg));
        inited = true;
    }
    esp_err_t err = tinyusb_msc_storage_mount("/rec");
    ESP_LOGI(TAG, "mount /rec: %s, free %llu KB", esp_err_to_name(err), storage_free_bytes() / 1024);
    return err;
}

esp_err_t storage_unmount(void) { return tinyusb_msc_storage_unmount(); }
```

```c
/* usb_drive.c */
#include "usb_drive.h"
#include "esp_log.h"
#include "storage.h"
#include "tinyusb.h"
#include "tinyusb_msc_storage.h"

static const char *TAG = "usb";

esp_err_t usb_drive_enter(void)
{
    ESP_RETURN_ON_ERROR(storage_unmount(), TAG, "unmount");
    const tinyusb_config_t cfg = {0};          /* default descriptors; product string set below */
    ESP_RETURN_ON_ERROR(tinyusb_driver_install(&cfg), TAG, "tinyusb install");
    ESP_LOGI(TAG, "exposed /rec as MSC");
    return ESP_OK;
}

esp_err_t usb_drive_exit(void)
{
    ESP_RETURN_ON_ERROR(tinyusb_driver_uninstall(), TAG, "tinyusb uninstall");
    return storage_mount();
}
```

The volume label "KWSREC" is a FAT property, not a USB descriptor: set it once at first mount — after the first `storage_mount()` in `app_main`, if `f_getlabel` returns empty, call `f_setlabel("KWSREC")` (FatFs, `ff.h`; needs `CONFIG_FATFS_USE_LABEL=y` in `sdkconfig.defaults`). The esp_tinyusb API differs between v1.x and v2.x (`tinyusb_msc_storage_init_spiflash` vs `tinyusb_msc_new_storage_spiflash`); use whichever the resolved `managed_components/espressif__esp_tinyusb` ships and say which in the report. Also add `CONFIG_TINYUSB_MSC_ENABLED=y` if it is not already in `sdkconfig.defaults` from Task 1, and add `esp_partition` + `wear_levelling` to `REQUIRES`.

- [ ] **Step 4: `ui/ui_usb.c` + wiring in `main.c`**

```c
/* ui_usb.c */
#include "ui.h"
#include "bsp/esp-bsp.h"
#include "lvgl.h"

static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD); }

void ui_show_usb(void)
{
    bsp_display_lock(0);
    lv_obj_t *scr = lv_obj_create(NULL);
    lv_obj_t *l = lv_label_create(scr);
    lv_label_set_text(l, "USB drive mode\n\nKWSREC is mounted on your computer.\nRun scripts/pull-recordings.sh,\nthen tap Back.");
    lv_obj_set_style_text_align(l, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(l, LV_ALIGN_TOP_MID, 0, 30);
    lv_obj_t *b = lv_button_create(scr);
    lv_obj_set_size(b, 120, 44); lv_obj_align(b, LV_ALIGN_BOTTOM_MID, 0, -16);
    lv_obj_add_event_cb(b, on_back, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bl = lv_label_create(b); lv_label_set_text(bl, "Back"); lv_obj_center(bl);
    lv_screen_load(scr);
    bsp_display_unlock();
}
```

`main.c` `app_set_mode` gains:

```c
    if (s_mode == UI_MODE_USB) ESP_ERROR_CHECK(usb_drive_exit());
    ...
    if (m == UI_MODE_USB) { ui_show_usb(); ESP_ERROR_CHECK(usb_drive_enter()); }
```

Order matters: leaving USB remounts `/rec` *before* the recorder resumes; entering USB pauses the recorder *before* unmounting (the recorder's `REC_CMD_PAUSE` is queued — wait for `record_get_status().phase == REC_IDLE` with a 100 ms poll, max 1 s, before `usb_drive_enter`, so no `fopen` races the unmount).

- [ ] **Step 5: CI shellcheck** — in `.github/workflows/firmware.yml` `host-test` job add a step after the C tests:

```yaml
      - name: shellcheck scripts
        run: shellcheck scripts/*.sh
```

(`ubuntu-latest` ships shellcheck.) Add `scripts/**` to the workflow's `paths:` filter if Task 1 introduced one — per spec §6 the firmware workflow runs unconditionally, so there should be no filter; verify.

- [ ] **Step 6: Build, test, commit**

```bash
uv run pytest tests/test_pull_recordings.py -q && shellcheck scripts/pull-recordings.sh
make -C firmware/test
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG idf.py build
git add firmware scripts/pull-recordings.sh tests/test_pull_recordings.py .github/workflows/firmware.yml
git commit -m "feat(firmware): USB mass-storage mode (KWSREC) + scripts/pull-recordings.sh"
```

---

### Task 7: Recognise mode — TFLM inference + detector + screen + log

**Files:**
- Create: `firmware/main/recognise.cc`, `firmware/main/recognise.h`, `firmware/main/ui/ui_recognise.c`
- Modify: `firmware/main/main.c`, `firmware/main/CMakeLists.txt`

**Interfaces:**
- Consumes: `mfcc_*`, `stream_*` (Task 3), `gen/model_data.h` + `gen/model_config.h` (Task 4), `gen/labels.h`, `audio_read/audio_write_pos` (Task 5), `app_set_mode`.
- Produces:

```c
/* recognise.h (C linkage) */
typedef struct { char word[24]; float conf; uint32_t infer_ms; uint32_t arena_used; uint32_t fired_count; } recognise_status_t;
void recognise_start(void);              /* create task, paused */
void recognise_set_active(bool on);
void recognise_get_status(recognise_status_t *out);

/* ui.h additions */
void ui_recognise_refresh(const recognise_status_t *st);
```

- [ ] **Step 1: `recognise.cc`**

```cpp
#include "recognise.h"
#include <cstring>
#include "audio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "gen/features_config.h"
#include "gen/labels.h"
#include "gen/model_config.h"
#include "gen/model_data.h"
#include "mfcc.h"
#include "stream.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ui/ui.h"

static const char *TAG = "recognise";
static constexpr int kStepSamples = KWS_SAMPLE_RATE / 10;      /* inference every 100 ms */

static SemaphoreHandle_t s_lock;
static recognise_status_t s_st;
static volatile bool s_active;
static uint8_t *s_arena;
static tflite::MicroInterpreter *s_interp;
static FILE *s_log;

static void log_fire(const char *word, float conf)
{
    if (!s_log) s_log = fopen("/rec/recognise.log", "a");
    if (!s_log) return;
    fprintf(s_log, "[Log] %lld %s %.2f\n", esp_timer_get_time() / 1000, word, conf);
    fflush(s_log);
}

static void recognise_task(void *)
{
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddFullyConnected();
    resolver.AddMean(); resolver.AddSoftmax(); resolver.AddReshape(); resolver.AddAdd();
    const tflite::Model *model = tflite::GetModel(g_model);
    assert(model->version() == TFLITE_SCHEMA_VERSION);
    static tflite::MicroInterpreter interp(model, resolver, s_arena, KWS_MODEL_ARENA_BYTES);
    s_interp = &interp;
    assert(interp.AllocateTensors() == kTfLiteOk);
    ESP_LOGI(TAG, "arena used %u / %u", (unsigned)interp.arena_used_bytes(), (unsigned)KWS_MODEL_ARENA_BYTES);
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);

    static stream_t stream;
    static int16_t pcm[KWS_SAMPLE_RATE];
    static float feats[KWS_N_FRAMES][KWS_N_MFCC];
    static float probs[KWS_NUM_LABELS];
    uint32_t cursor = audio_write_pos();

    for (;;) {
        if (!s_active) { vTaskDelay(pdMS_TO_TICKS(50)); cursor = audio_write_pos(); stream_reset(&stream); continue; }
        while (audio_write_pos() < cursor + kStepSamples) vTaskDelay(pdMS_TO_TICKS(5));
        cursor += kStepSamples;
        int64_t t0 = esp_timer_get_time();
        audio_read(cursor, pcm, KWS_SAMPLE_RATE);          /* trailing 1 s window */
        mfcc_compute(pcm, feats);                          /* one-shot; ponytail: streaming ring is a later optimisation */
        mfcc_quantize(feats, in->data.int8, KWS_MODEL_INPUT_SCALE, KWS_MODEL_INPUT_ZERO_POINT);
        assert(interp.Invoke() == kTfLiteOk);
        int best = 0;
        for (int i = 0; i < KWS_NUM_LABELS; i++) {
            probs[i] = (out->data.int8[i] - KWS_MODEL_OUTPUT_ZERO_POINT) * KWS_MODEL_OUTPUT_SCALE;
            if (probs[i] > probs[best]) best = i;
        }
        int fired = stream_push(&stream, probs);
        uint32_t ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);

        xSemaphoreTake(s_lock, portMAX_DELAY);
        s_st.infer_ms = ms; s_st.arena_used = interp.arena_used_bytes();
        if (fired >= 0) { strlcpy(s_st.word, KWS_LABELS[fired], sizeof s_st.word); s_st.conf = probs[fired]; s_st.fired_count++; }
        else if (best != KWS_SILENCE_INDEX) { /* keep last fired word on screen; show live top-1 only in conf */ s_st.conf = probs[best]; }
        recognise_status_t copy = s_st;
        xSemaphoreGive(s_lock);
        if (fired >= 0) { ESP_LOGI(TAG, "fired %s %.2f (%lu ms)", copy.word, copy.conf, (unsigned long)ms); log_fire(copy.word, copy.conf); }
        ui_recognise_refresh(&copy);
    }
}

extern "C" void recognise_start(void)
{
    s_lock = xSemaphoreCreateMutex();
    s_arena = (uint8_t *)heap_caps_malloc(KWS_MODEL_ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    assert(s_arena);
    xTaskCreatePinnedToCore(recognise_task, "recognise", 16384, nullptr, 6, nullptr, 1);
}
extern "C" void recognise_set_active(bool on) { s_active = on; if (!on && s_log) { fclose(s_log); s_log = nullptr; } }
extern "C" void recognise_get_status(recognise_status_t *out) { xSemaphoreTake(s_lock, portMAX_DELAY); *out = s_st; xSemaphoreGive(s_lock); }
```

`recognise.h` wraps its declarations in `#ifdef __cplusplus extern "C" { #endif`. The op list is exactly the 7 ops from Global Constraints; if `AllocateTensors` fails with "Didn't find op for builtin opcode", print the missing op in the report and add it — do not switch to `AllOpsResolver`. Use `mfcc_compute` (one-shot over the whole second) rather than the streaming ring: at 100 ms cadence with a 480-point naive DFT this is ~50 frames × 480² ≈ 11.5 M MACs → roughly 50 ms at 240 MHz, acceptable for phase 1; the streaming path (`mfcc_push_frame`, 5 new frames per step) is the drop-in optimisation if `infer_ms` exceeds 100 ms. Report `infer_ms` from the device log if hardware is available. `recognise_set_active(false)` closes the log so USB mode sees a consistent file.

- [ ] **Step 2: `ui/ui_recognise.c`**

```c
#include "ui.h"
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "lvgl.h"
#include "recognise.h"

static lv_obj_t *l_word, *l_stats;
static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD); }

void ui_show_recognise(void)
{
    bsp_display_lock(0);
    lv_obj_t *scr = lv_obj_create(NULL);
    l_word = lv_label_create(scr);
    lv_obj_set_style_text_font(l_word, &lv_font_montserrat_28, 0);
    lv_label_set_text(l_word, "...");
    lv_obj_align(l_word, LV_ALIGN_CENTER, 0, -30);
    l_stats = lv_label_create(scr);
    lv_obj_align(l_stats, LV_ALIGN_CENTER, 0, 30);
    lv_obj_t *b = lv_button_create(scr);
    lv_obj_set_size(b, 120, 44); lv_obj_align(b, LV_ALIGN_BOTTOM_MID, 0, -16);
    lv_obj_add_event_cb(b, on_back, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bl = lv_label_create(b); lv_label_set_text(bl, "Record"); lv_obj_center(bl);
    lv_screen_load(scr);
    bsp_display_unlock();
}

void ui_recognise_refresh(const recognise_status_t *st)
{
    char buf[80];
    if (!bsp_display_lock(50)) return;
    lv_label_set_text(l_word, st->word[0] ? st->word : "...");
    snprintf(buf, sizeof buf, "conf %.2f   %lu ms   arena %lu B   fired %lu",
             st->conf, (unsigned long)st->infer_ms, (unsigned long)st->arena_used, (unsigned long)st->fired_count);
    lv_label_set_text(l_stats, buf);
    bsp_display_unlock();
}
```

- [ ] **Step 3: wire `main.c`** — `recognise_start()` in `app_main` after `audio_start()`; in `app_set_mode`: leaving RECOGNISE → `recognise_set_active(false)`; entering → `ui_show_recognise(); recognise_set_active(true);`. `CMakeLists.txt` adds `recognise.cc`, `ui/ui_recognise.c`, `usb_drive.c`, `ui/ui_usb.c` to `SRCS`; `esp-tflite-micro` is pulled in by the component manager — if the component is not found by name, add `PRIV_REQUIRES esp-tflite-micro` and match the exact managed-component name.

- [ ] **Step 4: Build + commit**

```bash
make -C firmware/test
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:IDF_TAG idf.py build
git add firmware
git commit -m "feat(firmware): recognise mode — TFLM int8 inference, streaming detector, on-screen stats, /rec/recognise.log"
```

Report the binary size line from the build (`kws-de.bin binary size ...`) and the arena-used figure if run on hardware.

---

### Task 8: `scripts/flash.sh`, README, Pi note

**Files:**
- Create: `scripts/flash.sh`
- Modify: `firmware/README.md`, `README.md` (one "Firmware" paragraph linking to `firmware/README.md`)

- [ ] **Step 1: `scripts/flash.sh`**

```bash
#!/usr/bin/env bash
# Flash a merged CI/CD image onto a CoreS3. Usage: scripts/flash.sh [image.bin] [port]
# Needs esptool (pip install esptool, or the ESP-IDF venv). Hold BOOT/reset if the port is not found.
set -euo pipefail
img=${1:-$(ls -t kws-de-fw-*.bin 2>/dev/null | head -1)}
[[ -n ${img:-} && -f $img ]] || { echo "usage: $0 [kws-de-fw-<sha>.bin] [port]" >&2; exit 1; }
port=${2:-$(ls /dev/cu.usbmodem* /dev/ttyACM* 2>/dev/null | head -1)}
[[ -n ${port:-} ]] || { echo "no serial port found (/dev/cu.usbmodem* or /dev/ttyACM*)" >&2; exit 1; }
esptool.py --chip esp32s3 --port "$port" --baud 921600 write_flash 0x0 "$img"
echo "flashed $img → $port"
```

`chmod +x`, `shellcheck scripts/flash.sh` clean (the CI step from Task 6 covers it).

- [ ] **Step 2: `firmware/README.md`** — replace the Task 1 v1 with sections: What it does (two modes + USB), Build (docker one-liner and native `idf.py`; IDF pin `IDF_TAG` and where it is pinned: `CMakeLists.txt` warning, `firmware.yml`, this README), Flash (CI artifact → `scripts/flash.sh`, or `idf.py flash` in the container with `--device`), Record-mode walkthrough (set selection W/S/N, seed on screen, Redo/Skip/Next, +Spk, what `session.csv` holds, caps 1.5 s/4 s/6 s, clipping → redo), Pull recordings (USB button → `scripts/pull-recordings.sh` → `data/recordings/spkNN/...` + `sessions.csv`), Recognise mode (what the numbers mean; `[Log]` lines in `/rec/recognise.log`), Regenerating headers (`uv run kws-fwgen` after config changes — CI diff-checks; `uv run kws-export --firmware` after retraining — committed as-is), Manual test checklist (spec §7 list, verbatim), Pi note ("a Raspberry Pi with ESP-IDF at `~/esp/esp-idf` can build natively: `git -C ~/esp/esp-idf checkout IDF_TAG && ~/esp/esp-idf/install.sh esp32s3 && . ~/esp/esp-idf/export.sh && cd firmware && idf.py build`" — keep `IDF_TAG` beside any newer IDF already installed there, do not replace it). No speaker names, no local paths, no source-app provenance anywhere.

- [ ] **Step 3: top-level README** — after the existing "Data" or "Evaluation" section, add:

```markdown
## Firmware (M5Stack CoreS3)

`firmware/` is an ESP-IDF app with two modes: a guided recorder that
collects word/sentence/negative takes onto the device's flash (pulled over
USB with `scripts/pull-recordings.sh`), and an on-device recogniser running
the int8 model with the same MFCC front-end and detector as `kws_de.stream`.
Build, flash, and the manual test checklist: [firmware/README.md](firmware/README.md).
```

- [ ] **Step 4: lint + commit**

```bash
npx --yes markdownlint-cli@0.42.0 --config .markdownlint.json '**/*.md'
shellcheck scripts/*.sh
git add scripts/flash.sh firmware/README.md README.md
git commit -m "docs(firmware): flash script, firmware README with manual test checklist, top-level pointer"
```

---

## Self-review notes (controller)

- Spec §1 states/tasks → Tasks 5/6/7 + `app_set_mode`; §2 recorder → Task 5 (VAD ruling recorded there); §3 USB + pull script → Task 6; §4 recognise → Task 7 (`[Log]` format, 100 ms cadence, arena in PSRAM, stats on screen); §5 tree → Tasks 1–8; §6 CI → Task 1 (build), Task 2 (gen-fresh), Task 3 (host-test), Task 6 (shellcheck); §7 manual checklist → Task 8 README; §8 out of scope untouched.
- Type consistency: `prompt_set_t {PROMPT_WORDS, PROMPT_SENTENCES, PROMPT_NEGS}` and `prompt_cap_ms(prompt_set_t)` (Task 3) used by Task 5; `storage_wl_handle()` introduced in Task 5, superseded by the esp_tinyusb-owned handle in Task 6 (Task 6 may delete it); `mfcc_compute(pcm, feats)` + `mfcc_quantize(feats, int8*, scale, zp)` (Task 3) used by Task 7; `KWS_LABELS`, `KWS_SILENCE_INDEX` (Task 2) used by Task 7; `WAV_HEADER_BYTES` = 44 (Task 3) used by Task 5.
- Required checks (`firmware / build`, `host-test`, `gen-fresh`) are a user action on the repo settings after the first firmware PR — not a task.
