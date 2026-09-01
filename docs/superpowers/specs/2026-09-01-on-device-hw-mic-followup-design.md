# Design: on-device performance model + mic/acoustic domain adaptation (follow-up)

Date: 2026-09-01
Status: draft (follow-on spec) — pending review
Builds on: v1 KWS spec (`2026-08-31-kws-de-design.md`), v2 wake+slot spec
(`2026-09-01-voice-control-v2-design.md`). Consumes the trained command + wake models.

## 1. Goal & boundary

Two things the model-factory work cannot answer from a laptop, both requiring the physical
device (M5Stack CoreS3 / ESP32-S3):

1. **A measured performance model** — real inference latency, tensor-arena RAM, CPU load,
   heap/PSRAM, and power/battery on the CoreS3, replacing the current *estimates* (MACs →
   cycles) and *proxies* (budget gates).
2. **Microphone + acoustic domain adaptation** — all training/eval audio so far is MSWC
   (clean read speech) + TTS + ESC-50 noise; **none has passed through the CoreS3 mic or the
   van**. Close that sim-to-real gap: characterize the mic + AFE, adapt the training data to
   what the device actually hears, and validate on real on-device recordings.

Out of scope: a production firmware app, cloud anything, the camper BLE control write (that is
the separate firmware/integration plan). The measurement harness here may be minimal/throwaway.

## 2. Hardware facts (CoreS3)

- **SoC:** ESP32-S3, dual Xtensa LX7 @ 240 MHz with SIMD (ESP-NN INT8 kernels), 512 KB SRAM,
  **8 MB PSRAM**, 16 MB flash.
- **Audio:** **ES7210 codec with dual analog MEMS microphones** → **2-mic** input, so ESP-SR
  AFE can run **beamforming** in addition to AEC + noise-suppression + VAD. This is a lever the
  single-mic assumption in the earlier specs did not account for.
- **Power:** AXP2101 PMU (supports current read-back for on-board power estimate), 500 mAh LiPo.
- **Display:** 2" ILI9342 LCD + LVGL (for the recognised-word demo).

## 3. Part A — measured performance model

An ESP-IDF **measurement harness** (built/flashed on the Pi that holds the toolchain):

- Loads the command + wake INT8 TFLite-Micro models (C-array headers from `kws_de.export`).
- Runs inference over (i) canned fixed input vectors and (ii) live mic audio.
- Measures, per model:
  - **latency** per inference (`esp_timer`, many-run distribution: p50/p95),
  - **tensor arena** actually used (`interpreter.arena_used_bytes()`),
  - **peak heap + PSRAM** occupancy,
  - **CPU load** at the real streaming hop (inference time × hop rate),
  - **power**: average current via AXP2101 (and/or an external USB power meter) in
    always-on-wake idle vs active-command inference,
  - **estimated battery life** on the 500 mAh cell for the always-on wake duty cycle.
- Output: a **measured** performance table, and an **estimate-vs-measured** comparison against
  §Performance-model estimates in the v1/v2 specs (the estimate was ~2–9 ms / ~2–9 % CPU for the
  command model — confirm or correct).

## 4. Part B — microphone + acoustic domain adaptation

The high-value accuracy lever. Three steps:

### B1 — Characterize the mic + AFE
Play known stimuli (sweeps, the command words, noise) near the CoreS3 and capture:
- the dual-mic + ES7210 path **frequency response** and **noise floor**,
- the effect of the **ESP-SR AFE** (AEC/NS/VAD/beamforming) on the captured signal,
- a measured **mic impulse response** (or a matched EQ) for augmentation.

### B2 — Device-matched training data
- **Augment** existing training clips with the measured mic IR / EQ + a model of the AFE
  output, so the model trains on what the device actually feeds it (not clean corpus audio).
- **Record real utterances through the CoreS3** — "Hey Bus" + the command catalog, several
  speakers, in the van and other rooms → the **honest real-device validation set** currently
  missing (today every number is synthetic or clean-corpus).

### B3 — Retrain + on-device eval
- Retrain the command + wake models on device-matched data.
- Evaluate on the **real on-device recordings**: full-intent command accuracy, and the wake
  **false-accepts/hour measured over real ambient audio** (the real always-on metric, vs the
  synthetic FA/hour from microWakeWord's test set).
- On-device **threshold tuning** for the wake cutoff (the frr/faph operating point) against real
  ambient — the calibration knob a laptop cannot set.

### AFE integration
Use ESP-SR's **2-mic AFE** (beamforming + AEC + NS + VAD) as the on-device front-end, and make
the training-data augmentation match its output — so the golden-vector MFCC contract extends
from "host == device MFCC" to "training audio == AFE-processed device audio".

## 5. Metrics

- **Performance:** latency p50/p95 (ms/window), arena (KB), peak heap/PSRAM (KB), CPU (%), avg
  power (mW) idle vs active, estimated battery life (h).
- **Accuracy (real device):** command full-intent accuracy on real-mic recordings; wake
  detection rate + **FA/hour over real ambient**; before/after domain adaptation.

## 6. Method notes / "hardware is never ideal on paper"

The physical world adds knobs a minimal model cannot see and a laptop cannot set: the mic's
frequency response, PDM/ADC artifacts, AFE tuning, the wake cutoff operating point, and the van
acoustics. This spec's whole point is to expose and calibrate them, and to replace estimates
with measurements. Expect the real-device accuracy to sit **below** the synthetic-eval numbers —
quantifying that sim-to-real gap is itself a result.

## 7. Paper hook (E6)

Feeds a new experiment for the research log: **sim-to-real gap** — estimated vs measured
performance, and clean-corpus vs real-mic accuracy. A concrete, honest "what the datasheet and
the eval report don't tell you" section.

## 8. Milestones

1. Measurement harness (ESP-IDF): flash models, measure latency/arena/heap/CPU/power → table.
2. Mic + AFE characterization (frequency response, noise floor, IR/EQ, AFE effect).
3. Device-matched augmentation of the training data (mic IR/EQ + AFE model).
4. Real on-device recordings ("Hey Bus" + catalog, multiple speakers, in-van) → validation set.
5. Retrain command + wake on device-matched data; on-device eval + wake threshold tuning.
6. E6 write-up: estimate-vs-measured perf + clean-vs-real accuracy in `docs/paper-notes.md`.
