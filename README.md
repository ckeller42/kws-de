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

**v3 (real speech):** download the MSWC German audio + splits tarballs
(<https://mlcommons.org/datasets/multilingual-spoken-words/>, CC-BY 4.0,
~18 GB) and extract to `data/mswc/de/` so it contains `clips/` and
`de_splits.csv`. Words no public corpus has (e.g. `Aufstelldach`, a rare
camper-hardware compound) are self-recorded: one word per file,
`data/recordings/<word>/<speaker>_<n>.wav` (phone voice memo is fine, any
sample rate; 5–10 speakers × ~10 takes, quiet and in-vehicle). Then:

```bash
uv run kws-data --fetch --v3 --mswc-root data/mswc/de
uv run kws-dataset build --seed 0 --cache raw_clips_v3.pkl --prefix features_v3
uv run kws-benchmark --features features_v3
uv run kws-distill --features features_v3
```

**TTS backstop voices.** Words still short of 300 real clips are topped up
with TTS across every engine that imports (`say` on macOS, Piper when
`uv sync --extra tts`). Piper voices are whatever is cached under
`data/piper-voices/<name>/<quality>/` (the rhasspy/piper-voices layout);
multi-speaker voices such as `de_DE-mls-medium` (236 speakers) count once
per speaker. To add a voice:

```bash
v=mls; q=medium; id=de_DE-$v-$q; d=data/piper-voices/$v/$q; mkdir -p $d
for ext in onnx onnx.json; do
  curl -fL -o $d/$id.$ext \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/$v/$q/$id.$ext
done
```

Every TTS clip also gets one pitch/tempo-perturbed copy at build time
(±2 semitones, 0.85–1.15× tempo); real clips do not.

## Quick start

```bash
uv sync                       # install deps (--extra metal loads Apple-GPU; slower for these models, see paper-notes)
uv run kws-data   --fetch     # download MSWC-de keyword subset + noise, cache features
uv run kws-train              # train the DS-CNN
uv run kws-export             # -> models/model.tflite + firmware/main/model_data.h
uv run kws-eval               # -> docs/eval-report.md (accuracy, SNR sweep, budgets)
```

`data/` and `models/` are gitignored and can live anywhere: set `KWS_DATA_ROOT`
to a directory containing `data/` and `models/` (e.g. an external SSD) and every
checkout and worktree shares it — no per-worktree symlinks. Unset, they are
repo-relative. Keep frozen dataset/model versions in `<root>/archive/<version>/`
for reproducibility; scripts only ever write to `<root>/data` and `<root>/models`.

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
