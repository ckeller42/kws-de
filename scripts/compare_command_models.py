"""Compare two command .tflite models on the real-voice approved/qc set (same pattern
docs/paper-notes.md E16 used for the width sweep, first written as a throwaway driver
for E27's TTS-regen comparison — promoted here since E28 needed it again): `kws-eval`
derives its model filename from --prefix/--qat alone and has no --model, so it cannot
score a specific .tflite file directly. This calls the same code path
(`eval_recordings` + `make_command_predict_fn`) with explicit bytes for two models: the
deployed model (parsed out of firmware/main/gen/model_data.h, the bytes the device
actually runs — kws_de.codegen.model_bytes) and a candidate .tflite export.

Usage:
  uv run --no-sync python scripts/compare_command_models.py \
      --candidate models/command_v3_w48_qat.tflite \
      --candidate-test-npz data/features_v3_test.npz \
      --deployed-test-npz data/features_v3_test.npz.pre-regen
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from kws_de import config
from kws_de.codegen import model_bytes
from kws_de.eval import _tflite_predict, eval_recordings, make_command_predict_fn


def int8_test_acc(tflite_bytes: bytes, test_npz: Path) -> tuple[float, int]:
    d = np.load(test_npz)
    X, y = d["X"], d["y"]
    preds = _tflite_predict(tflite_bytes, X)
    return float((preds == y).mean()), len(y)


def score(
    label: str, tflite_bytes: bytes, test_npz: Path, approved: Path, manifest: Path, qc: Path
):
    acc, n = int8_test_acc(tflite_bytes, test_npz)
    predict_fn = make_command_predict_fn(tflite_bytes)
    res = eval_recordings(
        approved, predict_fn, manifest_path=manifest, qc_root=qc if qc.is_dir() else None
    )
    print(f"== {label} ==")
    print(f"bytes={len(tflite_bytes)} sha256={hashlib.sha256(tflite_bytes).hexdigest()[:8]}")
    print(f"INT8 test accuracy: {acc:.4f} (n={n})")
    for fig_label, fig in res["figures"].items():
        for spk, row in fig["isolated"].items():
            print(f"  [{fig_label}] {spk} words n={row['n']} acc={row['acc']:.3f}")
        for spk, row in fig["e2e"].items():
            print(f"  [{fig_label}] {spk} phrases n={row['n']} acc={row['acc']:.3f}")
        for spk, row in fig["false_accepts"].items():
            print(f"  [{fig_label}] {spk} false_accepts n={row['n']} rate={row['rate']:.3f}")
    print()
    return {"bytes": len(tflite_bytes), "int8_test_acc": acc, "int8_test_n": n, "recordings": res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deployed-header", default="firmware/main/gen/model_data.h", type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--candidate-test-npz", required=True, type=Path)
    ap.add_argument("--deployed-test-npz", required=True, type=Path)
    ap.add_argument("--approved", default=config.DATA_DIR / "recordings" / "approved", type=Path)
    ap.add_argument("--manifest", default=config.DATA_DIR / "manifest_v3_qat.json", type=Path)
    ap.add_argument("--qc", default=config.DATA_DIR / "recordings" / "qc", type=Path)
    args = ap.parse_args()

    deployed_bytes = model_bytes(args.deployed_header)
    candidate_bytes = args.candidate.read_bytes()

    score(
        "deployed",
        deployed_bytes,
        args.deployed_test_npz,
        args.approved,
        args.manifest,
        args.qc,
    )
    score(
        "candidate",
        candidate_bytes,
        args.candidate_test_npz,
        args.approved,
        args.manifest,
        args.qc,
    )


if __name__ == "__main__":
    main()
