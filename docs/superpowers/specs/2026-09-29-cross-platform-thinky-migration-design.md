# Cross-platform kws-de + thinky migration — design

**Status:** approved (brainstorm 2026-09-29). **Scope:** migration-adjacent only.

## Goal

Make the kws-de data + training tooling run cleanly on both macOS (Apple Silicon)
and Linux (CUDA), with storage locations configurable rather than hardcoded, and
use the move to `thinky` (RTX 3090 Ti, Threadripper 3975WX, Linux) as a
data-consistency cross-check. `thinky` becomes the primary home for the
`kws-data` tree; the current external SSD on the Mac becomes a verified backup.
Firmware build + device flashing stay on the Mac (Docker ESP-IDF + the CoreS3 on
`bar`) — out of scope for the move.

## Non-goals

Standing audit backlog (batched tflite eval, ProcessPool MFCC, ETA-ledger fix,
`_tflite_predict`/`score` dedupe). Not touched here — this is migration + the
cross-platform/config work only.

## Current state

- Storage: `kws_de/config.py` resolves one root from `os.environ["KWS_DATA_ROOT"]`
  (default = repo root), then `DATA_DIR = root/data`, `MODELS_DIR = root/models`.
  Van augmentation reads `KWS_NOISE_DIR` / `KWS_RIR_DIR`. No config file, no
  per-OS default, no resolved-path introspection.
- `pyproject.toml`: `metal` extra = `tensorflow-metal` (Apple GPU only); `qc`
  extra = `mlx-whisper` (Apple only). TF pinned `>=2.16,<2.19` for the
  tensorflow-metal ABI. `qc.py` already accepts an injected `Transcriber`, so the
  Whisper backend is swappable without touching QC logic.
- Determinism: `kws-codegen --check` is byte-exact integer; `kws-fwgen --check`
  compares float MFCC tables within a tolerance (BLAS/CPU-dependent). Both run in
  CI (`gen-fresh`). See `docs/paper-notes.md` codegen-parity notes.

## Design

### 1. Storage resolver

Keep `KWS_DATA_ROOT` as the mechanism; add precedence and introspection.

- Resolution order for the data root and each aux dir (`KWS_NOISE_DIR`,
  `KWS_RIR_DIR`): explicit env var → config file → per-OS default.
- Config file: `~/.config/kws-de/config.toml` (per-machine, never committed;
  `XDG_CONFIG_HOME` honored). Keys: `data_root`, `noise_dir`, `rir_dir`.
- Per-OS default for `data_root` when nothing is set: macOS keeps the existing
  SSD path *only if it already exists*, else `~/kws-data`; Linux `~/kws-data`.
  A default is a fallback, never a committed machine path.
- New `kws-doctor` CLI (`kws_de.doctor:main`): prints resolved data_root /
  data_dir / models_dir / noise_dir / rir_dir, whether each exists and is
  readable, the platform, Python/TF versions, GPU availability (Metal or CUDA),
  and the selected Whisper backend. Read-only. This is the first thing to run
  when a path is wrong (would have made the SSD-EPERM week trivial).

### 2. Cross-platform Python env

- GPU extras, platform-marked so `uv sync --extra gpu` resolves per OS:
  - `tensorflow-metal>=1.1 ; sys_platform == "darwin"`
  - `tensorflow[and-cuda] ; sys_platform == "linux"` (same TF version range)
  - Keep `metal` as an alias for back-compat; document `gpu` as the portable one.
- Whisper backend for QC:
  - `mlx-whisper ; sys_platform == "darwin"`
  - `faster-whisper ; sys_platform == "linux"` (CTranslate2, CUDA-accelerated)
  - `kws_de.qc` gains `faster_whisper_transcriber()` beside the mlx one, and a
    `default_transcriber()` that picks by platform/availability. QC logic
    unchanged (it already takes an injected transcriber).
  - **Agreement check:** the two backends must produce the same QC labels on a
    fixed clip set (same language ID + transcript decision), or the divergence
    is documented and the stricter one used. A test pins this on a small sample.
- TF version range held on Linux too, for cross-machine determinism (§4).

### 3. Data migration + consistency cross-check

- `scripts/sync-data.sh`: rsync `kws-data` over Tailscale, direction-flagged
  (`--to thinky` / `--to backup`), `--data-root` and remote host configurable,
  `--dry-run` default-safe. No hardcoded host or path.
- `kws-verify` (`kws_de.verify:main` or `scripts/verify-data.py`): build a
  per-file sha256 manifest of a tree, or diff two manifests, and report
  missing/extra/changed files. Run on both sides after sync; a clean diff is the
  gate to flip `thinky` to primary.
- Flip: once verify passes, `thinky` is primary; the SSD becomes a backup target
  (reverse sync). Documented, not automated destructively.

### 4. Cross-machine reproducibility check

On `thinky`, after the env is up, run and compare against the Mac/committed:

- `kws-codegen --check firmware/main/gen` — byte-exact integer; must match.
- `kws-fwgen --check firmware/main/gen` — float MFCC tables; the real risk
  (BLAS/numpy differ across machines). Expected within the existing tolerance;
  any drift is documented as a cross-machine caveat, not silenced.
- Parity tests (`test_codegen_parity`, `test_intent`, etc.).
- Fixed-seed `kws-train` run twice on thinky → identical, confirming
  same-machine determinism (the reproducibility the recipe grid relies on).

### 5. Measured GPU benchmark

Install the env on thinky, run one real `kws-train --v2 --width 48 --qat
--qat-epochs 20 --real-weight 3 --seed 0` on the GPU, record wall-clock and
per-epoch against the M4 CPU baseline (~26 min), confirm the export runs and its
`kws-codegen` output matches §4. Replaces the estimate with a number.

### 6. Docs

`docs/dev-setup.md`: both OSes — prerequisites, `uv sync` with the right extras,
storage config (env var / config file / defaults), GPU setup, running a train,
the sync + verify workflow, and `kws-doctor` as the self-check. README links it.

## Sequencing

- Now (SSD not required): §1 resolver + doctor, §2 extras + faster-whisper, §5
  thinky env + benchmark on a small generated dataset, §6 docs, §4 codegen/parity
  checks on thinky.
- Deferred until the Mac SSD is readable again (tmux/TCC restart): §3 actual
  `kws-data` copy + full cross-check, then the primary/backup flip.

## Risks

- `kws-fwgen` float-table drift across machines could fail `gen-fresh` if a
  Linux-generated header is committed. Mitigation: keep header generation on one
  reference machine, or widen/verify the tolerance; §4 measures it before any
  Linux-generated header is committed.
- faster-whisper vs mlx-whisper label disagreement would change which clips pass
  QC. Mitigation: the §2 agreement check gates the swap.
- `tensorflow[and-cuda]` userspace vs the installed driver (595.84) mismatch.
  Mitigation: pin to a TF build whose bundled CUDA matches; verified in §5.
