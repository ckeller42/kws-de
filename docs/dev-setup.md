# Development setup (macOS + Linux)

How to get a working kws-de dev environment on either **macOS (Apple Silicon)** or
**Linux (CUDA)**. The training + data tooling is cross-platform; firmware build and
device flashing stay on the Mac (see [firmware/README.md](../firmware/README.md)).

Run `uv run kws-doctor` at any point — it prints exactly what resolved (paths, GPU,
Whisper backend) and is the first thing to check when something looks wrong.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — the package/venv manager this repo uses.
- **Python 3.11+** (uv installs one if needed).
- **macOS:** Apple Silicon for GPU; CPU-only training works everywhere.
- **Linux:** an NVIDIA GPU + driver for GPU training; CPU-only works without one.

## Install

`uv sync` is exact by default: it removes anything not in the selected set, so a
later `uv sync --extra X` drops the extras an earlier one added. Select every
extra you need in **one** command:

```bash
# Full development setup; run once. Each extra is platform-marked, so the same
# command resolves the right packages on macOS and Linux — no per-OS variant.
uv sync --extra gpu --extra qat --extra qc --extra dev
```

Or add `--all-extras`. Bare `uv sync` is core deps only — CPU training works, but
without `dev` the test suite, without `qc` the recording QC, etc. The individual
extras (`gpu`, `qat`, `qc`, `dev`) are choices to compose in that one command,
not steps to run in sequence.

| Extra | macOS installs | Linux installs |
|---|---|---|
| `gpu` | `tensorflow-metal` | `tensorflow[and-cuda]` (bundled CUDA userspace) |
| `qc` | `mlx-whisper` | `faster-whisper` (CTranslate2, CUDA) |
| `metal` | `tensorflow-metal` | — (back-compat alias for `gpu` on macOS) |

TensorFlow is pinned `>=2.16,<2.19` on both OSes (the tensorflow-metal plugin ABI,
and cross-machine determinism). `gpu` is the portable extra; `metal` stays as an
alias for older instructions.

## Storage configuration

`data/` and `models/` are gitignored; their physical home is a per-machine detail.
Each location resolves in this order — **explicit env var → config file → per-OS
default**:

| Location | Env var | `config.toml` key | Default |
|---|---|---|---|
| data root | `KWS_DATA_ROOT` | `data_root` | repo root (macOS) / `~/kws-data` (Linux) |
| noise dir | `KWS_NOISE_DIR` | `noise_dir` | unset → van augmentation off |
| RIR dir | `KWS_RIR_DIR` | `rir_dir` | unset → van augmentation off |

`DATA_DIR` = `<data_root>/data`, `MODELS_DIR` = `<data_root>/models`. A default is
always computed — never a committed absolute path — so a fresh clone works with no
configuration. Setting `KWS_DATA_ROOT` reproduces the classic "one shared root for
every worktree, e.g. an external SSD" behaviour.

**Env var (one shell / CI):**

```bash
export KWS_DATA_ROOT=/path/to/kws-data
```

**Per-machine config file** at `${XDG_CONFIG_HOME:-~/.config}/kws-de/config.toml`
(never committed — a machine setting, not a repo setting):

```toml
# ~/.config/kws-de/config.toml
data_root = "/mnt/data/kws-data"          # Linux example
# data_root = "/Volumes/YourSSD/kws-data" # macOS external-SSD example
noise_dir = "/mnt/data/mww-train/data/fma_16k"   # van-aug noise (16 kHz wavs), optional
rir_dir   = "/mnt/data/mww-train/data/mit_rirs"  # van-aug room IRs (16 kHz wavs), optional
```

Set both `noise_dir` and `rir_dir` to enable van-cabin augmentation for real clips;
leave them unset to keep it off. Verify what resolved with `uv run kws-doctor`.

## GPU setup notes

- **macOS (Metal):** `uv sync --extra gpu` installs `tensorflow-metal`. `kws-doctor`
  reports the GPU as `(Metal)`. Note these models are tiny; Metal is often *slower*
  than CPU here (per-step overhead) — see `docs/paper-notes.md`. CPU is fine.
- **Linux (CUDA):** `uv sync --extra gpu` installs `tensorflow[and-cuda]`, which
  bundles a CUDA userspace. It must match your installed NVIDIA **driver**; a
  mismatch shows up as no GPU in `kws-doctor`. If so, pin to a TF build whose bundled
  CUDA matches the driver. `kws-doctor` reports the GPU as `(CUDA)`.

## Running a train

```bash
uv run kws-data   --fetch     # download MSWC-de subset + cache features
uv run kws-train              # train the DS-CNN (add --qat for quant-aware fine-tune)
uv run kws-export             # -> models/model.tflite + firmware header
uv run kws-eval               # -> accuracy / SNR sweep / budget report
```

See the top-level [README](../README.md) and `docs/sphinx/models.rst` for the recipe
grid and accuracy numbers.

## Recording quality control

Recording QC transcribes with Whisper large-v3 through an injected backend, so the
`qc` extra pulls the right one per OS (mlx-whisper on macOS, faster-whisper on Linux).
`kws_de.qc.default_transcriber()` selects it automatically. See the recording loop in
the [README](../README.md) and `docs/sphinx/pipeline.rst`.

> **Note:** the faster-whisper-vs-mlx label-agreement check on a real clip set is a
> coordinator follow-up (design §2), run on the Linux box; until it passes, macOS
> mlx-whisper is the reference backend. See `docs/paper-notes.md` (E59).

## Self-check

```bash
uv run kws-doctor
```

Read-only. Prints the resolved `data_root` / `data_dir` / `models_dir` / `noise_dir`
/ `rir_dir` and whether each exists and is readable, the platform, Python + TF
versions, GPU availability (Metal or CUDA), and the selected Whisper backend. Run it
first whenever a path or a device looks wrong.

## Tests and lint

```bash
./scripts/setup-hooks.sh                        # once per clone: activate git hooks
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

Same gates as CI. The `dev` extra provides pytest and the pinned ruff.

## Data sync + verify

The `kws-data` tree (recordings, models, ~8 GB) is not in git. `scripts/sync-data.sh`
moves it between machines and cross-checks it by sha256 (design §3). The remote is an
ssh target you pass; the local root is `KWS_DATA_ROOT` / `--data-root` / the per-OS
default `kws_de.config` resolves — nothing is baked in. It is dry-run by default.

```bash
# See what a push would transfer (no changes):
scripts/sync-data.sh --to thinky

# Actually push, then confirm byte-for-byte:
scripts/sync-data.sh --to thinky --run
scripts/sync-data.sh --to thinky --verify   # sha256-manifests both ends, diffs them

# Pull the other way (e.g. refresh the Mac backup from the primary):
scripts/sync-data.sh --from thinky --run
```

`kws-verify` is the manifest/diff underneath, usable on its own — `kws-verify manifest
[ROOT]` writes a sha256 manifest of a tree, `kws-verify diff A B` reports drift and exits
non-zero on any. `sync-data.sh --verify` runs it on both ends (the remote needs `kws_de`
importable). A clean diff is the gate to trust a copy before making the destination the
primary and the source a backup.
