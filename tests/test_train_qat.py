# kws-train --qat / kws-export --qat run as subprocesses, never in-process: both
# CLIs re-exec themselves under TF_USE_LEGACY_KERAS=1 (tfmot only wraps Keras-2
# models) before importing TensorFlow, and `os.execv` replaces the calling
# process image -- calling `main()` directly from pytest would blow away the
# test runner itself.
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from kws_de import config

# The QAT path needs the optional `qat` extra (tfmot + tf_keras); without it the
# re-exec'd trainer cannot import legacy Keras, so skip rather than fail.
pytest.importorskip("tensorflow_model_optimization")
pytest.importorskip("tf_keras")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_toy_npz(data_root: Path, prefix: str) -> None:
    """Tiny, easily-separable 2-class synthetic dataset, same shape convention
    as the real MFCC features (config.N_FRAMES x config.N_MFCC)."""
    rng = np.random.default_rng(0)
    n = 24
    X = rng.standard_normal((n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    X += y[:, None, None] * 3  # strong class-dependent shift -> quick convergence
    data_dir = data_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / f"{prefix}_train.npz", X=X, y=y)
    np.savez(data_dir / f"{prefix}_test.npz", X=X, y=y)


def _run(args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_qat_flag_trains_exports_and_beats_random(tmp_path):
    prefix = "features_toy"
    _write_toy_npz(tmp_path, prefix)
    env = {**os.environ, "KWS_DATA_ROOT": str(tmp_path)}

    # High epoch counts: 24 samples is one batch/epoch, so BatchNormalization's
    # moving mean/var (momentum 0.99) needs many steps before the inference-mode
    # normalisation it freezes into the TFLite export matches the training-mode
    # (batch-statistics) accuracy .fit() reports -- too few epochs here converges
    # the *training* metric to 1.0 while inference-mode predict()/export stays at
    # chance, which would silently pass a broken int8 pipeline.
    train_res = _run(
        [
            "kws_de.train",
            "--v2",
            "--prefix",
            prefix,
            "--out",
            "toy.keras",
            "--epochs",
            "300",
            "--qat",
            "--qat-epochs",
            "150",
        ],
        env,
    )
    assert train_res.returncode == 0, train_res.stdout + train_res.stderr
    assert "QAT final train accuracy" in train_res.stdout

    models_dir = tmp_path / "models"
    float_path = models_dir / "toy.keras"
    qat_dir = models_dir / "toy_qat"
    assert float_path.exists()  # PTQ path's own float model, artefact naming intact
    assert (qat_dir / "saved_model.pb").exists()  # QAT model, SavedModel dir, _qat suffix

    export_res = _run(
        ["kws_de.export", "--v2", "--prefix", prefix, "--model", "toy.keras", "--qat"],
        env,
    )
    assert export_res.returncode == 0, export_res.stdout + export_res.stderr

    tflite_path = models_dir / "command_toy_qat.tflite"
    assert tflite_path.exists()
    blob = tflite_path.read_bytes()

    itp = tf.lite.Interpreter(model_content=blob)
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8  # int8 I/O export, as PTQ

    data = np.load(tmp_path / "data" / f"{prefix}_test.npz")
    scale, zp = inp["quantization"]
    correct = 0
    for x, label in zip(data["X"], data["y"], strict=True):
        q = np.round(x[None, ..., None] / scale + zp).astype(np.int8)
        itp.set_tensor(inp["index"], q)
        itp.invoke()
        pred = int(np.argmax(itp.get_tensor(out["index"])[0]))
        correct += pred == label
    # 2-class toy, well-separated, fine-tuned on this exact data: near-perfect, not
    # just above chance -- proves the QAT model's fake-quant ranges actually made
    # it through the int8 conversion usably, not merely that convert() didn't crash.
    assert correct / len(data["y"]) >= 0.9
