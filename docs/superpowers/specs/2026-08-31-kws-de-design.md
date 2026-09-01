# Design: German KWS model for ESP32-S3 (`kws-de`)

Date: 2026-08-31
Status: approved (brainstorming), pending spec review

## 1. Goal & boundary

Produce a **deployable INT8 keyword-spotting artifact** (`model.tflite` + C-array header)
for a fixed set of German command words, **proven usable on an ESP32-S3** (M5Stack CoreS3)
through Mac tests, CI resource-budget gates, and a minimal on-device demo that shows the
recognised word on the display. The consumer is the buspi camper voice-control satellite
(see buspi-config #154). This repo is a **reproducible model factory**, not a product.

Success criteria:

1. `model.tflite` is full-INT8, ≤ 500 KB (target ≤ 150 KB), all ops TFLM-supported.
2. Held-out command accuracy ≥ threshold (set empirically; report per-command + `_unknown_`
   false-accept rate) and an accuracy-vs-SNR curve down into van-noise levels.
3. On-device budgets (size, tensor arena, MACs/inference, estimated latency) pass in CI.
4. Model runs on a real CoreS3 and prints/draws the recognised word (manual HW smoke).
5. Everything reproducible from versioned scripts; no training bytes in git.

## 2. Architecture / data flow

```text
MSWC-de word clips ──┐
German TTS (piper) ──┼─► feature extract (MFCC) ─► DS-CNN (Keras) ─► INT8 quantize (TFLite)
van + ESC-50 noise ──┘        (librosa, matched          │                    │
   (augmentation)              to on-device MFCC)   train / eval         model.tflite
                                                                              │
                                    ┌─────────────────────────────────────────┤
                              Mac + CI: interp + accuracy              model_data.h (xxd)
                              + budget gates + golden vectors                  │
                                                                       firmware/ (ESP-IDF)
                                                                   AFE → MFCC → TFLM → LVGL label
```

Units, each independently testable:

- `kws_de.data` — fetch MSWC-de keyword subset + noise, build augmented dataset, cache
  features.
- `kws_de.features` — MFCC front-end; **must match the on-device implementation**.
- `kws_de.model` — DS-CNN definition.
- `kws_de.train` — training loop (local/manual; not run in CI).
- `kws_de.export` — TFLite INT8 conversion + `model_data.h` + `metadata.json`.
- `kws_de.eval` — accuracy, confusion, SNR sweep, latency estimate → report.
- `kws_de.budgets` — pure assertions on model size / arena / MACs / ops.
- `firmware/` — ESP-IDF demo (AFE → MFCC → TFLM → LVGL).

## 3. Command set

Single-word German commands, each acting as a **toggle** for one camper function. Words are
chosen to be short (2–3 syllables), phonetically distinct, and common enough to appear in
MSWC German (so positives come free; no TTS):

| Command (spoken) | Camper function | Data source |
|---|---|---|
| **Licht** | lights | MSWC-native |
| **Kühlschrank** | cooler / fridge | MSWC-native |
| **Camping** | camping-mode master switch | MSWC-native |
| **Heizung** | heater | MSWC-native |
| **Wasser** | water system | MSWC-native |

plus two auxiliary classes:

- `_unknown_` — random other MSWC German words (open-world rejection).
- `_silence_` — noise-only / background.

Total: 5 command classes + 2 auxiliary = 7. Optional additions (not in the core set):
`Batterie` (battery status), `Dach` (roof — control unverified), and explicit on/off
`Einschalten`/`Ausschalten` if toggle semantics prove insufficient.

Two-word phrases (e.g. "Licht einschalten") are out of scope: they force TTS-only positives
and hurt robustness. Add later only if needed.

## 4. Data (recipe versioned, bytes never committed)

- **Positives + hard negatives:** MSWC German (CC BY 4.0), pulled **per-keyword via
  HuggingFace `datasets` streaming** — only the ~8 command words + a sample of random words
  for `_unknown_`. Downloading the full 1083 h to use 8 words is waste; the subset is a few
  hundred MB and CI-reproducible. A full-corpus run is possible as an opt-in.
- **Voice-diversity fill:** optional German TTS (piper, offline) to boost positive count and
  speaker variety for words with few MSWC clips.
- **Noise:** ESC-50 (~600 MB) + on-site van recordings, for SNR augmentation (§12-D).
- **Not committed:** `data/`, `models/`, `*.npy`, `*.tflite` are gitignored. `kws_de.data`
  (the downloader/builder) **is** committed — reproduce from code. A small deterministic
  **mini-dataset** derived from the same subset logic is used by CI.

## 5. Model

DS-CNN, small (MLPerf-Tiny reference class): ~25 k params, ~2–3 M MACs, INT8 ≈ 40–120 KB.
Depthwise-separable convolutions → TFLM/ESP-NN friendly. Feature front-end: MFCC, 30 ms
window / 20 ms hop, matched host↔device. Rationale and citations in §12.

## 6. Export

`model.tflite` (full-INT8) → `firmware/main/model_data.h` via `xxd -i`. Also emit
`metadata.json` (labels, MFCC params, budgets). Firmware embeds the C array (no SD — the
CoreS3 BSP cannot drive SD and LCD simultaneously).

## 7. Testing / CI + on-device budget gates

- **Mac (pytest):** feature-pipeline correctness, tiny-smoke training converges, TFLite
  interpreter runs, held-out accuracy ≥ threshold.
- **Golden-vector test:** x86 tflite-micro interpreter bit-exact vs reference on fixed
  inputs → proves device kernels agree with the trained model.
- **Budget gates** (the "usable on ESP32" proof, no hardware): `model ≤ 500 KB`,
  `tensor-arena ≤ budget`, `MACs/inf ≤ real-time budget`, **all ops INT8 & TFLM-supported**,
  estimated latency < 30 ms.
- **CI** runs the above on a committed mini-dataset + committed checkpoint (full MSWC
  training is too heavy for CI). Full training is local; the exported artifact is committed.
- Firmware build in CI is optional (espressif idf docker); budget gates already prove fit.

## 8. Firmware demo (minimal, with display)

ESP-IDF app, built/flashed on buspi (toolchain already installed there):

```text
CoreS3 BSP mic → ESP-SR AFE (AEC + NS + VAD) → MFCC (esp-dsp)
  → INT8 KWS (esp-tflite-micro + ESP-NN)
  → on detection: LVGL label shows the German word (large font) + confidence
  → serial log
```

The M5Stack CoreS3 has an official ESP-IDF BSP (`espressif/m5stack_core_s3`) providing LCD
(ILI9342) + LVGL + mic + touch — no manual driver work. ESP-Skainet's ESP-BOX examples
already demonstrate showing a recognised command on the LCD; this reuses that pattern.
Model embedded as a C array (avoids the BSP's SD-vs-LCD limitation). This is the manual
hardware smoke test.

## 9. Performance evaluation

`kws_de.eval` → committed `docs/eval-report.md`:

- per-command accuracy, confusion matrix;
- `_unknown_` false-accept rate;
- **accuracy-vs-SNR curve** (clean → van-noise) — the headline metric; paper accuracy is
  meaningless without it;
- resource table: model size / tensor arena / MACs / estimated latency.

## 10. Tooling

`uv` · tensorflow (+ optional tensorflow-metal for Apple GPU) · librosa / soundfile / numpy ·
HuggingFace `datasets` (MSWC pull) · piper (offline German TTS) · ESC-50 + van recordings ·
esp-idf (buspi) · optional esp-idf docker for CI firmware build.

Hardware: a modern laptop trains this tiny DS-CNN in minutes to low hours — compute is not
the constraint. The keyword-subset keeps the data footprint small (a few hundred MB), so no
special storage is required. Optional GPU acceleration on Apple Silicon via tensorflow-metal.

## 11. Non-goals (YAGNI)

No natural-language / free speech. No cloud recognition. No hardware-in-CI. No two-word
phrases (unless explicitly added). No custom MultiNet acoustic model. No checked-in training
data or model binaries.

## 12. Algorithmic aspects (documented, cited)

The design choices trace the data flow; each stage has a canonical reference.

**A. Task formulation.** KWS as a closed-set classifier over *N* commands plus `_unknown_`
and `_silence_`; 1 s windows; on the streaming path, posterior smoothing + threshold. The
`_unknown_` class (fed random words) is what prevents the van radio triggering a command.
→ Warden, *Speech Commands* (arXiv:1804.03209).

**B. Feature front-end (MFCC).** Pre-emphasis → 30 ms frames @ 20 ms hop → Hann window →
|FFT|² → mel filterbank → log → DCT → keep low-order cepstra. **Host (librosa) and device
(esp-dsp) must produce identical coefficients**; the golden-vector test enforces bit
agreement, because a host/device front-end mismatch is the classic silent failure mode.
→ Davis & Mermelstein, *Comparison of parametric representations…* (IEEE TASSP, 1980).

**C. Model (DS-CNN).** Depthwise-separable convolution = depthwise (one spatial filter per
channel) + pointwise 1×1 channel mix, ~8–9× fewer MACs than dense convolution at comparable
accuracy — the reason it fits a microcontroller. → Zhang et al., *Hello Edge: Keyword
Spotting on Microcontrollers* (arXiv:1711.07128) introduced DS-CNN for KWS; Howard et al.,
*MobileNets* (arXiv:1704.04861) is the separable-conv primitive; Banbury et al., *MLPerf
Tiny Benchmark* (arXiv:2106.07597) is the reference DS-CNN we size against.

**D. Training / noise robustness.** Multi-condition training: mix positives with additive
noise (van recordings + ESC-50) at sampled SNRs, convolve with room impulse responses for
reverberation, and apply SpecAugment time/frequency masking; `_unknown_` sampled from random
MSWC words, `_silence_` from noise-only. This stage is why the model works in a moving
vehicle rather than only on clean data — hence it is the most heavily grounded. → Snyder et
al., *MUSAN* (arXiv:1510.08484); Ko et al., *A Study on Data Augmentation of Reverberant
Speech for Robust Speech Recognition* (ICASSP 2017, DOI:10.1109/ICASSP.2017.7953152); Park
et al., *SpecAugment* (arXiv:1904.08779).

**E. Quantization (full-INT8).** Integer-arithmetic-only inference: per-axis weight
quantization, per-tensor activations, scales/zero-points calibrated on a representative
dataset. Required so the ESP-NN SIMD kernels apply — without full-INT8 the model falls back
to slow reference kernels. → Jacob et al., *Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference* (arXiv:1712.05877).

**F. On-device runtime.** TFLite-Micro interpreter with ESP-NN optimized kernels; streaming
inference = ring buffer of overlapping windows; a non-streaming Keras model is auto-converted
to streaming (Google `kws_streaming` library). → David et al., *TensorFlow Lite Micro*
(arXiv:2010.08678); Rybakov et al., *Streaming Keyword Spotting on Mobile Devices*
(arXiv:2005.06720, Interspeech 2020).

**G. Data corpus.** MSWC: CC BY 4.0, 50 languages, word-aligned 1 s clips; German is a
high-resource split (1083 h). Word-level alignment means the clips are already the tensor
shape KWS wants. → Mazumder et al., *Multilingual Spoken Words Corpus* (NeurIPS Datasets &
Benchmarks 2021, OpenReview c20jiJ5K2H).

**H. Rejected alternative (documented, not built).** Few-shot embedding enrollment — train an
embedding on MSWC, enroll commands from a handful of examples with no retraining. More
flexible, but less CI-deterministic and harder to fit real-time, so we chose the trained
DS-CNN. → Mazumder et al., *Few-Shot Keyword Spotting in Any Language* (arXiv:2104.01454,
Interspeech 2021); Menon et al., *Plug-and-Play Multilingual Few-shot Spoken Words*
(arXiv:2305.03058).

### Reference list

1. Warden, *Speech Commands* — arXiv:1804.03209
2. Mazumder et al., *Multilingual Spoken Words Corpus*, NeurIPS D&B 2021 — OpenReview c20jiJ5K2H
3. Zhang et al., *Hello Edge: KWS on Microcontrollers* — arXiv:1711.07128
4. Howard et al., *MobileNets* — arXiv:1704.04861
5. Banbury et al., *MLPerf Tiny Benchmark* — arXiv:2106.07597
6. Jacob et al., *Integer-Arithmetic-Only Inference* — arXiv:1712.05877
7. David et al., *TensorFlow Lite Micro* — arXiv:2010.08678
8. Rybakov et al., *Streaming KWS on Mobile Devices*, Interspeech 2020 — arXiv:2005.06720
9. Snyder et al., *MUSAN* — arXiv:1510.08484
10. Park et al., *SpecAugment* — arXiv:1904.08779
11. Ko et al., *Data Augmentation of Reverberant Speech*, ICASSP 2017 — DOI:10.1109/ICASSP.2017.7953152
12. Mazumder et al., *Few-Shot KWS in Any Language*, Interspeech 2021 — arXiv:2104.01454
13. Menon et al., *Plug-and-Play Multilingual Few-shot Spoken Words* — arXiv:2305.03058

## 13. Documentation & diagrams

Docs use a Sphinx site (furo theme, myst-parser for Markdown, sphinxcontrib-mermaid,
sphinx-needs for requirement traceability) with architecture diagrams via **`sphinx-likec4`**
(an in-house extension). Toolchain pinned in `docs/requirements.txt`:

```text
sphinx>=7,<8
sphinx-needs>=5,<6
sphinxcontrib-mermaid>=1,<2
myst-parser>=2,<5
furo>=2024.1
sphinx-likec4 @ git+https://github.com/ckeller42/sphinx-likec4
```

`docs/conf.py` sets `extensions = ["sphinx_likec4", ...]`, `likec4_source_dir = "likec4"`,
`html_theme = "furo"`. LikeC4 model lives in **`docs/likec4/*.c4`** (`model.c4`,
`sequences.c4`), embedded in reStructuredText via `.. likec4-view:: <view-id>`. Node ≥ 20 is
required at doc-build time (the extension runs the pinned LikeC4 CLI). Directive reference:
the agent skill in `~/src/sphinx-likec4/skills/` (or its `llms.txt`). Build via
`docs/build_site.sh`; publish via `.github/workflows/docs.yml` (GitHub Pages).

Views to model in `docs/likec4/`:

- **Pipeline view** — data flow (MSWC/TTS/noise → MFCC → DS-CNN → INT8 → artifact).
- **Deployment view** — training host, CI (budget gates), the Pi that builds/flashes the
  firmware, CoreS3 device (AFE → TFLM → LVGL display).
- **Dynamic view (sequence)** — on-device runtime (mic → AFE → MFCC → TFLM → LVGL label),
  a `sequences.c4` dynamic view.

Prose docs (Markdown via myst) describe the algorithmic aspects with citations (§12). The
Sphinx doc build is a separate workflow (node dependency), not part of the main test CI.

## 14. Milestones (detailed plan follows via writing-plans)

1. Scaffold + data pipeline (`kws_de.data`, MSWC subset, noise, feature cache).
2. MFCC front-end + golden vectors (host, later matched on device).
3. DS-CNN + training + smoke test.
4. INT8 export + `model_data.h` + budget gates.
5. Eval + report (accuracy, SNR sweep).
6. Sphinx + sphinx-likec4 docs: LikeC4 model + pipeline/deployment/dynamic views, algorithms
   prose.
7. ESP-IDF firmware demo (AFE → TFLM → LVGL display), HW smoke on CoreS3.
