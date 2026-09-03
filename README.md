# kws-de

German **keyword-spotting (KWS)** model for the **ESP32-S3** (M5Stack CoreS3), for
offline voice control of a camper ([buspi](https://github.com/ckeller42/buspi-config)).
Trains a tiny INT8 DS-CNN on public German speech data, exports a
`model.tflite` + C-array header that runs on-device via TensorFlow Lite Micro + ESP-NN,
and ships a minimal ESP-IDF demo that shows the recognised word on the CoreS3 display.

Design & rationale (with algorithm citations): `docs/superpowers/specs/`.

## Documentation

Built and deployed on every push to `main`:

- Architecture and paper docs: **<https://ckeller42.github.io/kws-de/>**
- Firmware, models, recording pipeline, requirements traceability and the C API
  (Sphinx + sphinx-needs + Doxygen): **<https://ckeller42.github.io/kws-de/sphinx/>**

## Status (2026-09-03)

What the command model measures on the two device speakers' own real
recordings as each stage lands, and the wake model's real-take result.
Full numbers and provenance: [`docs/sphinx/models.rst`](docs/sphinx/models.rst).

| | spk01 word acc. | spk02 word acc. | False accepts |
|---|---|---|---|
| Command, v2 stock (no device recordings) | 0.19 (held-out) | 0.27 (held-out) | 0/6 |
| Command, v3 PTQ (device recordings in train) | 0.538 (user-customised) | 0.553 (user-customised) | 0/10 |
| Command, v3 QAT | **0.615** (user-customised) | **0.737** (user-customised) | 0/10 |

- **Wake, round 5** (real "Hey Bus" takes added, weighted): 10 of 10 real
  takes fire at the device gate (round 4: 4 of 10); see
  [`docs/sphinx/models.rst`](docs/sphinx/models.rst) for the generic-voice
  trade-off that comes with it.
- **Recogniser step** (CoreS3, per streaming step): 164–181 ms → **82–85 ms**
  after the exact 480-point FFT front end; see
  [`docs/sphinx/firmware.rst`](docs/sphinx/firmware.rst) for the full
  on-device measurement table.
- **On the device today:** the 23-class INT8 QAT command model (17,880 B) and
  the round-5 wake model, both flashed and running; boot menu = Recognition,
  Hey Bus, Record, Hey Bus aufnehmen, USB.

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

`firmware/` is an ESP-IDF app with a 5-button boot menu — Recognition, Hey
Bus, Record, Hey Bus aufnehmen, USB — covering the guided recorder (word/
sentence/negative and wake-word takes onto the device's flash, pulled over
USB with `scripts/pull-recordings.sh`), the on-device command recogniser
(same MFCC front-end and detector as `kws_de.stream`), and an isolated wake
word test mode. Build, flash, and the manual test checklist:
[firmware/README.md](firmware/README.md); the device as it behaves, the
serial console protocol, and on-device measurements:
[`docs/sphinx/firmware.rst`](docs/sphinx/firmware.rst).

## Recording data loop

Beyond MSWC/TTS, the CoreS3's own guided recorder feeds a repeatable loop:
pull a speaker's session, quality-control it with Whisper large-v3
(`qc` extra: `uv sync --extra qc`), fold the approved audio into the v3
dataset, retrain, export, and report both the standard held-out accuracy
and a **user-customised** figure on that speaker's own recordings — kept
strictly separate, never mixed into one number. One command runs the whole
thing end to end:

```bash
export KWS_DATA_ROOT=/path/to/data-root
scripts/data-loop.sh -H <device-host>   # or: export KWSREC_HOST=<device-host>
```

`-H`/`KWSREC_HOST` is the SSH name of the machine the CoreS3 is plugged
into — never hard-coded in the repo; `KWSREC_HOST_PYTHON` names a python on
that host with `pyserial` installed if it isn't `python3` on `PATH`. Details,
the data layout, and the two eval figures:
[docs/sphinx/pipeline.rst](docs/sphinx/pipeline.rst). Every stage prints an
ETA before it starts and records its actual duration after, improving
future predictions (`kws_de.eta`, `kws-eta predict/record/run/watch`, "How
long will it take" in that same doc).

`kws-train --qat` / `kws-export --qat` add a quantisation-aware fine-tune on
top of the plain INT8 export — see
[`docs/sphinx/models.rst`](docs/sphinx/models.rst) for the accuracy numbers.
`--width N` (both CLIs, default 32) resizes every conv/depthwise-separable
channel count in the command model; the same page has the width-sweep
results and why 32 stays the recommendation.

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

`main` is protected: every change goes through a PR, and the branch must be
up to date with `main` before merging. Required checks are the CI jobs —
`test`, `markdownlint`, `gitleaks` (`ci.yml`); `build`, `gen-fresh`,
`host-test` (`firmware.yml`); `build`, `firmware-docs` (`docs.yml`) — the
same gates the local hooks above mirror. Work on a branch, open a PR, wait
for CI + CodeRabbit, then merge — never commit straight to `main`.

## References

See spec §12 for the algorithmic background and the full, cited reference list
(Speech Commands, MSWC, Hello Edge / DS-CNN, MobileNets, MLPerf Tiny, integer-only
quantization, TFLite Micro, streaming KWS, MUSAN, SpecAugment, reverberant augmentation).
