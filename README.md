# kws-de

German **keyword-spotting (KWS)** model for the **ESP32-S3** (M5Stack CoreS3), for
offline voice control of a camper ([buspi](https://github.com/ckeller42/buspi-config)).
Trains a tiny INT8 DS-CNN on public German speech data, exports a
`model.tflite` + C-array header that runs on-device via TensorFlow Lite Micro + ESP-NN,
and ships a minimal ESP-IDF demo that shows the recognised word on the CoreS3 display.

Design & rationale (with algorithm citations): `docs/superpowers/specs/2026-08-31-kws-de-design.md`.

## Scope

- **In:** data pipeline, DS-CNN training, INT8 export, Mac + CI tests, on-device
  resource-budget gates, performance evaluation, minimal ESP-IDF display demo.
- **Out:** natural-language / free speech, cloud recognition, hardware-in-CI, a custom
  MultiNet acoustic model. See spec §11 (non-goals).

## Data

Training corpora and model binaries are **never committed** (`.gitignore`): `data/`,
`models/`, `*.npy`, `*.tflite`. The **scripts that fetch and build the data are
versioned** — reproduce the dataset from code, not from checked-in bytes.

## Quick start

```bash
uv sync                       # install deps (add --extra metal for Apple-GPU)
uv run kws-data   --fetch     # download MSWC-de keyword subset + noise, cache features
uv run kws-train              # train the DS-CNN
uv run kws-export             # -> models/model.tflite + firmware/main/model_data.h
uv run kws-eval               # -> docs/eval-report.md (accuracy, SNR sweep, budgets)
```

## ESP32 firmware demo

`firmware/` is an ESP-IDF app (built/flashed on buspi, which has the toolchain):
CoreS3 BSP mic -> ESP-SR AFE -> MFCC -> INT8 KWS -> recognised word on the LCD (LVGL).
Model is embedded as a C array (no SD card — the CoreS3 BSP cannot use SD and LCD at once).

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

CI (`.github/workflows/ci.yml`): ruff + pytest/coverage, markdownlint, gitleaks. Work on a
branch, open a PR, wait for CI + CodeRabbit, then merge — never commit straight to `main`.

## References

See spec §12 for the algorithmic background and the full, cited reference list
(Speech Commands, MSWC, Hello Edge / DS-CNN, MobileNets, MLPerf Tiny, integer-only
quantization, TFLite Micro, streaming KWS, MUSAN, SpecAugment, reverberant augmentation).
