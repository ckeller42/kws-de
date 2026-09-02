# kws-de

German **keyword-spotting (KWS)** model for the **ESP32-S3** (M5Stack CoreS3), for
offline voice control of a camper ([buspi](https://github.com/ckeller42/buspi-config)).
Trains a tiny INT8 DS-CNN on public German speech data, exports a
`model.tflite` + C-array header that runs on-device via TensorFlow Lite Micro + ESP-NN,
and ships a minimal ESP-IDF demo that shows the recognised word on the CoreS3 display.

Design & rationale (with algorithm citations): `docs/superpowers/specs/`.
**Docs site** (architecture + LikeC4 diagrams + eval reports): <https://ckeller42.github.io/kws-de/>
— build locally with `pip install -r docs/requirements.txt && python -m sphinx -b html docs docs/_build/html` (node ≥ 20 required).

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

## Firmware (M5Stack CoreS3)

`firmware/` is an ESP-IDF app with two modes: a guided recorder that
collects word/sentence/negative takes onto the device's flash (pulled over
USB with `scripts/pull-recordings.sh`), and an on-device recogniser running
the int8 model with the same MFCC front-end and detector as `kws_de.stream`.
Build, flash, and the manual test checklist: [firmware/README.md](firmware/README.md).

## Development

```bash
./scripts/setup-hooks.sh   # once per clone: activate the git hooks (.githooks/)
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

`setup-hooks.sh` points `core.hooksPath` at `.githooks/`: **pre-commit** runs ruff +
markdownlint (+ gitleaks if installed), **pre-push** runs the tests — the same gates as CI,
so a break is caught locally. Bypass a hook with `--no-verify`. Claude Code sessions also lint
each edited file via a `.claude/` PostToolUse hook.

CI (`.github/workflows/ci.yml`): ruff + pytest/coverage, markdownlint, gitleaks. Work on a
branch, open a PR, wait for CI + CodeRabbit, then merge — never commit straight to `main`.

## References

See spec §12 for the algorithmic background and the full, cited reference list
(Speech Commands, MSWC, Hello Edge / DS-CNN, MobileNets, MLPerf Tiny, integer-only
quantization, TFLite Micro, streaming KWS, MUSAN, SpecAugment, reverberant augmentation).
