# CoreS3 dual-mode firmware: guided recorder + on-device word demo

Date: 2026-09-01
Status: approved design, phase 1
Target: M5Stack CoreS3 (ESP32-S3, 16 MB flash, 8 MB PSRAM, ES7210 dual
mic, 320×240 touch LCD, USB-C OTG). No firmware exists in the repo today;
this is the first.

## 0. Why

Two things the Python side cannot do alone:

1. **Real speech at scale.** `kws_de.recordings` expects
   `data/recordings/<speaker>/<word>/*.wav` but nothing produces them. A
   guided recorder on the device makes a 26-word + ~50-sentence + 20-negative
   session a ten-minute task per speaker, with consistent mic, room and
   format.
2. **Prove the export path.** `kws-export` writes `model_data.h` and
   `metadata.json`; nothing has ever consumed them. A minimal recogniser
   closes that loop and gives hard numbers (inference ms, arena bytes) for
   the paper.

One firmware, two modes, so the device you record with is the device you
test on.

## 1. Architecture

```text
                 ┌────────────── audio task ──────────────┐
  ES7210 mics ──▶│ BSP I2S → 16 kHz mono int16 ring buffer │
                 └────────┬───────────────────┬───────────┘
                          │ (only one drains) │
                 ┌────────▼───────┐  ┌────────▼────────┐
                 │  record task   │  │ recognise task  │
                 │ VAD → WAV+CSV  │  │ MFCC → TFLM →   │
                 │ on /rec        │  │ StreamDetector  │
                 └────────┬───────┘  └────────┬────────┘
                          └──────▶ ui task (LVGL) ◀──────┘
```

- **States:** `RECORD` ⇄ `RECOGNISE` (touch bar, left button toggles), and
  `USB_DRIVE` reachable from `RECORD` only. Exactly one consumer task
  drains the ring buffer; the other is suspended. The audio task never
  stops (pre-roll needs a warm buffer).
- **Storage:** FAT partition `storage`, ~10 MB, wear-levelled
  (`esp_vfs_fat_spiflash_mount_rw_wl`), mounted at `/rec`. The same
  partition is exposed over USB as a mass-storage device in `USB_DRIVE`.
  One owner at a time: entering `USB_DRIVE` unmounts `/rec`; leaving
  remounts it. SD card is unusable: `BSP_SD_SPI_MISO` shares GPIO35 with
  `BSP_LCD_DC`, so SD and LCD cannot coexist on CoreS3.
- **Generated inputs** (committed under `firmware/main/gen/`, produced by
  `kws-export --firmware`): `model_data.h` (int8 TFLite blob),
  `labels.h`, `prompts.h` (word/sentence/negative prompt sets),
  `features_config.h` (MFCC + detector params from `metadata.json`).
  The firmware never parses JSON at runtime.
- **NVS:** speaker counter, last set + index (resume after reboot).
- **Not in phase 1:** Wi-Fi, SD, wake word, grammar, OTA.
- **Build system:** ESP-IDF's `idf.py` (CMake + Ninja). Not negotiable:
  the component manager, Kconfig, partition tooling, esp-bsp, esp-sr and
  esp-tflite-micro are all IDF/CMake components with no Bazel rules. Our
  own CMake is one `project()` and one `idf_component_register()`.

## 2. Record mode

Screen (320×240, LVGL):

```text
┌──────────────────────────────────────────┐
│ words · seed 17            12/26   spk03 │
│                                          │
│              Kühlschrank                 │
│                                          │
│ ▁▂▃▅▆▇▇▆▅▃▂▁  ▏level▕                    │
│                                          │
│ [Redo]        [Skip]          [Next →]   │
│ [⇄ Recognise] [USB] [+ Speaker]          │
└──────────────────────────────────────────┘
```

- **Prompt sets** (from `prompts.h`): `words` = `config.COMMAND_LABELS`
  minus `_unknown_`/`_silence_`; `sentences` = `phrases.py` catalog;
  `negatives` = `config.NEGATIVE_PROMPTS` (new, ~20 fixed German
  sentences that contain no command word, e.g. "wie spät ist es"). Each
  set is shuffled with a per-session seed shown on screen (`seed 17`) so
  the order is reproducible from `session.csv`.
- **Capture:** 300 ms pre-roll from the ring buffer; ESP-SR AFE VAD
  (mode 3) opens the take; 500 ms trailing silence closes it. Caps: 4 s
  words, 6 s sentences/negatives. 8 s with no speech → auto `Redo`.
  Clipping (any sample at ±32767) → level bar red, auto `Redo`. Successful
  take → 700 ms hold → auto-advance.
- **Output:** WAV, 16 kHz, 16-bit mono, standard 44-byte RIFF header.

  ```text
  /rec/spk03/licht/001.wav                     (words: one dir per label)
  /rec/spk03/_phrase_/licht-hinten-an_001.wav  (sentences)
  /rec/spk03/_neg_/wie-spaet-ist-es_001.wav    (negatives)
  /rec/spk03/session.csv                       (prompt,file,ms,peak_dbfs,set,ts)
  ```

  Filenames are ASCII slugs (ä→ae, ß→ss, spaces→-). `ts` is uptime ms
  (no RTC); the pull script adds host time.
- **Speaker ids** are numeric only (`spk03`, NVS counter incremented by
  `[+ Speaker]`). No names anywhere on the device or in the repo.
- **Flash full** (< 200 KB free) → recording disabled, banner says so,
  `[USB]` still works.
- `recordings.py` stays unchanged: it already reads `spk*/<word>/*.wav`.
  `_phrase_` and `_neg_` are skipped by the current loader; a later
  plan wires them into the sentence/grammar evaluation.

## 3. USB drive mode

- `[USB]` → unmount `/rec` → `tinyusb_msc_storage_init_spiflash` on the
  `storage` partition, volume label `KWSREC` → screen shows "USB drive —
  tap to exit". Tap → deinit MSC → remount `/rec` → back to `RECORD`.
- While in `USB_DRIVE`, console log goes to UART on the Grove port (USB-C
  is owned by MSC).
- **`scripts/pull-recordings.sh [mount]`** (host side, Mac and Pi):
  auto-detects `/Volumes/KWSREC` or `/media/*/KWSREC`; `rsync -a`
  `spk*/` → `data/recordings/`; appends each `session.csv` (prefixed with
  speaker id and host date) to `data/recordings/sessions.csv`; deletes
  the device copy only if rsync exited 0; ejects. Idempotent: re-running
  on an empty drive is a no-op.

## 4. Recognise mode (phase 1)

- **Front end:** MFCC in C (`main/mfcc.c`) matching `features.py`:
  16 kHz, 480-sample window, 320 hop, 40 mels, 10 coefficients, 1 s
  window → 49×10, quantised to int8 with the model's input scale/zero
  point. All constants come from `features_config.h`, never hard-coded
  twice.
- **Model:** `tflite::MicroInterpreter` with a `MicroMutableOpResolver`
  registering exactly the gated set from `tests/test_transducer.py`
  (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX, RESHAPE,
  ADD) with ESP-NN kernels. Arena = `metadata.json` arena bytes × 1.2, in
  PSRAM.
- **Post-processing:** C port of `stream.StreamDetector` (`smooth_win`,
  `threshold`, `min_consecutive`, `gap_steps` from `features_config.h`),
  one detector step per inference; inference runs every 100 ms (5 hops)
  over the sliding 1 s window to keep CPU under 50 %. The Python
  detector's `gap_steps`/`min_consecutive` are in inference steps, so
  the values carry over unchanged.
- **Screen:** last detected word, confidence, inference ms, arena bytes
  used. `[Log]` toggles appending `ts,word,conf,ms` to
  `/rec/recognise.log` for offline comparison against the Python
  detector.
- **Phase 2 (separate spec, not built now):** microWakeWord gate,
  `grammar.parse` C port validated against exported Python test vectors,
  intent on screen, n-best lattice decoding per the distill spec §10.

## 5. Repository, build, flash

```text
firmware/
  CMakeLists.txt            project()
  sdkconfig.defaults        PSRAM, TinyUSB MSC, FATFS long names, ESP-NN
  partitions.csv            factory app 3 MB, nvs, storage (FAT, ~10 MB)
  idf_component.yml         esp-bsp m5stack_core_s3, esp-sr, esp-tflite-micro, esp_tinyusb
  main/
    CMakeLists.txt          idf_component_register(...)
    main.c                  state machine, task setup
    audio.c/.h              BSP mic → ring buffer
    record.c/.h             VAD, WAV writer, session.csv
    recognise.c/.h          MFCC → TFLM → detector
    usb_drive.c/.h          MSC mount/unmount
    mfcc.c/.h               pure C, no IDF deps (host-testable)
    stream.c/.h             StreamDetector port, pure C
    wav.c/.h                header writer, pure C
    ui/                     LVGL screens
    gen/                    model_data.h labels.h prompts.h features_config.h
  test/                     host unit tests (linux target)
  README.md                 IDF version pin, flash + pull how-to, manual checklist
scripts/
  flash.sh                  esptool.py --chip esp32s3 write_flash 0x0 <merged.bin>
  pull-recordings.sh
```

- **IDF version:** pinned in `firmware/README.md`, `firmware/CMakeLists.txt`
  (`idf_build_get_property` check) and the CI workflow. Plan Task 1 is a
  spike: try IDF v6.2 (already on buspi); if esp-sr / esp-tflite-micro /
  esp-bsp do not build there, fall back to v5.5 LTS and install it beside
  6.2 on buspi (`~/esp/esp-idf-v5.5`). The Mac gets the same pin via
  `idf-installer` or the VS Code extension.
- **Build hosts:** Mac (primary) and CI. **Flash hosts:** any machine
  with `esptool.py` — Mac, second Mac, buspi — using the merged binary,
  so no IDF install is needed to flash. `scripts/flash.sh` takes the
  bin path and optional port; auto-detects `/dev/cu.usbmodem*` /
  `/dev/ttyACM*`.
- **Python side:** `kws-export --firmware` writes the four `gen/` headers
  from the current model + `config` + `phrases` catalog. Committed
  outputs, so CI builds without training. Adds `config.NEGATIVE_PROMPTS`.

## 6. CI

`.github/workflows/firmware.yml`, triggered on every push and PR (no
path filter — a `config`/`phrases` change regenerates headers, so any
commit can break the firmware build). It lands in plan Task 1 together
with the IDF-version spike, so the firmware is under CI from its first
commit. All three jobs below are to be added to the `main` branch
protection as required checks (user action, one-time):

| Job | What | Gate |
|---|---|---|
| `build` | `espressif/esp-idf-ci-action@v1` at the pinned tag, `idf.py -DIDF_TARGET=esp32s3 build`, then `esptool.py --chip esp32s3 merge_bin -o kws-de-fw-<sha>.bin @flash_args` | build must succeed; artifact uploaded (30-day retention) |
| `host-test` | same action, `idf.py --preview set-target linux && idf.py build && ./build/kws_de_fw_test.elf` in `firmware/test` (Unity) | all tests pass |
| `gen-fresh` | in the Python CI: `uv run kws-export --firmware --out /tmp/gen && diff -r /tmp/gen firmware/main/gen` | committed headers match the exporter; fails when someone changes `config`/`phrases` without regenerating |

Plus existing jobs extended: ruff/pytest already cover `export.py` and
`config.py` changes; markdownlint covers `firmware/README.md`;
`shellcheck` added for the two new scripts.

## 7. Testing

**Host unit tests (`firmware/test`, Unity, run in CI):**

- `mfcc`: 1 s fixture WAV → 49×10 int8; every coefficient within 1
  quantisation step of the Python `features.mfcc` output stored as a
  header by `kws-export --firmware --test-vectors`.
- `stream`: replay the probability sequences from `tests/test_stream.py`
  (exported as a header) and assert identical event lists.
- `wav`: header bytes for 16 kHz/16-bit/mono at 3 sample counts.
- `prompts`: shuffle with seed 17 twice → same order; all three sets
  non-empty; every word prompt is a `COMMAND_LABELS` entry.
- `slug`: `Kühlschrank` → `kuehlschrank`, `Licht hinten an` →
  `licht-hinten-an`.

**Python tests (existing pytest job):**

- `kws-export --firmware` golden: headers are deterministic and contain
  every label / prompt exactly once.
- `NEGATIVE_PROMPTS`: ≥ 15 entries, none contains a `COMMANDS`/`DEVICES`
  word (case-insensitive), all ASCII-sluggable.
- `pull-recordings.sh`: pytest spawns it against a tmp dir laid out like
  the device, asserts files land under `data/recordings/spk03/...`,
  `sessions.csv` is appended, source is emptied.

**On-device manual checklist (`firmware/README.md`):** record 3 words +
1 sentence + 1 negative → USB → pull → `kws-recordings --list` shows them;
toggle to Recognise → say "Licht" → word appears, inference < 30 ms;
`recognise.log` replays through `stream.StreamDetector` with the same
events.

## 8. Out of scope / follow-ups

- Wake-word gate, grammar port, lattice decoding (phase 2 spec).
- Sentence and negative recordings feeding training (loader change,
  separate plan).
- Wi-Fi upload instead of USB (not needed while USB works).
- Any speaker metadata beyond the numeric id.
