"""`kws-doctor` — read-only environment self-check (design §1).

Prints where storage resolved (data_root / data_dir / models_dir / noise_dir /
rir_dir) and whether each exists and is readable, the platform, Python/TF
versions, GPU availability (Metal on macOS, CUDA on Linux), and which Whisper
QC backend `default_transcriber` would select. Nothing is loaded that would be
slow or fail on a bare machine: TensorFlow and the Whisper libs are imported
lazily and their absence is reported, never fatal. This is the first thing to
run when a path looks wrong.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from kws_de import config


def _path_status(p: Path | None) -> str:
    if p is None:
        return "unset (feature off)"
    exists = p.exists()
    readable = exists and os.access(p, os.R_OK)
    flags = "exists" if exists else "MISSING"
    if exists:
        flags += ", readable" if readable else ", NOT READABLE"
    return f"{p}  [{flags}]"


def _tf_report() -> tuple[str, str]:
    """(version, gpu) lines for TensorFlow, importing it lazily. Absence is a
    normal answer, not a crash — the resolver + doctor must work with no TF."""
    try:
        import tensorflow as tf  # noqa: PLC0415 - lazy on purpose
    except Exception as exc:  # ImportError, or a plugin ABI load failure
        return f"not importable ({type(exc).__name__}: {exc})", "n/a (no TensorFlow)"
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        return tf.__version__, f"query failed ({type(exc).__name__}: {exc})"
    if not gpus:
        return tf.__version__, "no GPU visible (CPU only)"
    if sys.platform == "darwin":
        kind = " (Metal)"
    elif sys.platform.startswith("linux"):
        kind = " (CUDA)"
    else:
        kind = ""
    plural = "s" if len(gpus) != 1 else ""
    names = ", ".join(g.name for g in gpus)
    return tf.__version__, f"{len(gpus)} GPU{plural} visible{kind}: {names}"


def _whisper_backend() -> str:
    """Which QC transcriber backend is selected for this platform, and whether
    its library is importable — without constructing a model."""
    from kws_de import qc  # noqa: PLC0415 - keep the import graph lazy

    name, lib = qc.default_backend()
    try:
        __import__(lib)
        avail = "installed"
    except Exception:
        avail = f"NOT installed (pip/uv extra missing: {lib})"
    return f"{name} [{lib}: {avail}]"


def report() -> str:
    # Report the exact values the rest of the app uses (module constants),
    # plus the dynamically-resolved aux dirs.
    data_root = config._DATA_ROOT
    lines = [
        "kws-doctor — environment self-check",
        "",
        "Storage (resolved: env var > config.toml > per-OS default)",
        f"  config file : {config._config_file_path()}"
        f"  [{'present' if config._config_file_path().exists() else 'absent'}]",
        f"  data_root   : {_path_status(data_root)}",
        f"  data_dir    : {_path_status(config.DATA_DIR)}",
        f"  models_dir  : {_path_status(config.MODELS_DIR)}",
        f"  noise_dir   : {_path_status(config.noise_dir())}",
        f"  rir_dir     : {_path_status(config.rir_dir())}",
        "",
        "Platform",
        f"  platform    : {platform.platform()}",
        f"  python      : {platform.python_version()} ({sys.executable})",
    ]
    tf_version, gpu = _tf_report()
    lines += [
        f"  tensorflow  : {tf_version}",
        f"  gpu         : {gpu}",
        "",
        "Quality-control (Whisper)",
        f"  qc backend  : {_whisper_backend()}",
    ]
    return "\n".join(lines)


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
