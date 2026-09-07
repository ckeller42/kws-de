"""Host-only hyper-recipe grid search for the v3 command model
(docs/paper-notes.md -- width sweep was E16, the deploy rule E27/E28).

Sweeps width x real-clip weight (kws-train --real-weight) x QAT epochs, all on
the SAME `features_v3` build (fixed seed=0, whatever `kws-dataset build`
already wrote to `$KWS_DATA_ROOT/data/features_v3_{train,val,test}.npz`) so
the only variables are the three recipe knobs. Each candidate trains +
exports into its own `models/grid/<run>/` directory -- it never writes to the
shared `command_v3_w48_qat.*` deploy path or `firmware/main/gen/`, both of
which every other script in this repo treats as the checked-in/currently-
deployed artefact. Scoring reuses the same real-voice approved set and
`eval_recordings`/`make_command_predict_fn` code path as
`scripts/compare_command_models.py`.

Resumable: a run whose id already has a row in the output CSV is skipped, so
a killed/interrupted grid can be re-launched and picks up where it left off.

Usage:
  uv run --no-sync python scripts/recipe-grid.py
  uv run --no-sync python scripts/recipe-grid.py --csv scripts/recipe-grid.csv --only w48_rw3_qe20
"""

import argparse
import csv
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np

from kws_de import config
from kws_de.codegen import model_bytes
from kws_de.eval import eval_recordings, make_command_predict_fn

REPO_ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 40  # float epochs -- fixed at the recipe's established value (E16, E28)
PREFIX = "features_v3"
APPROVED = config.DATA_DIR / "recordings" / "approved"
MANIFEST = config.DATA_DIR / "manifest_v3_qat.json"
QC = config.DATA_DIR / "recordings" / "qc"
DEPLOYED_HEADER = REPO_ROOT / "firmware" / "main" / "gen" / "model_data.h"
TEST_NPZ = config.DATA_DIR / f"{PREFIX}_test.npz"
TRAIN_NPZ = config.DATA_DIR / f"{PREFIX}_train.npz"
SPEAKERS = ("spk01", "spk02", "spk10", "spk18")  # the standing real-voice scoreboard

# The 8-run grid: width x real-weight x QAT-epochs, float epochs fixed.
GRID = [
    {"width": w, "real_weight": rw, "qat_epochs": qe}
    for w in (32, 48)
    for rw in (1, 3)
    for qe in (10, 20)
]

FIELDS = [
    "run",
    "width",
    "real_weight",
    "qat_epochs",
    "epochs",
    "int8_test_acc",
    "spk01_words",
    "spk02_words",
    "spk10_words",
    "spk18_words",
    "aggregate_words",
    "false_accepts",
    "false_accepts_n",
    "macs",
    "bytes",
    "sha256",
    "train_seconds_predicted",
    "train_seconds_actual",
    "passes",
    "git_sha",
]
GIT_SHA = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
).stdout.strip()


def passes(row: dict) -> bool:
    """The standing deploy rule (E27/E28): aggregate real-voice words >= 0.785, no false
    accepts, and the least-represented speaker (spk18) above the deployed 1/3 floor."""
    return (
        float(row["aggregate_words"]) >= 0.785
        and int(row["false_accepts"]) == 0
        and float(row["spk18_words"]) > 0.333
    )


def run_id(width: int, real_weight: int, qat_epochs: int) -> str:
    return f"w{width}_rw{real_weight}_qe{qat_epochs}"


def existing_runs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as fh:
        return {row["run"] for row in csv.DictReader(fh)}


def append_row(csv_path: Path, row: dict) -> None:
    row = {**row, "passes": "PASS" if passes(row) else "FAIL", "git_sha": GIT_SHA}
    print(f"{row['run']}: {row['passes']}")
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def score(tflite_bytes: bytes, test_npz: Path) -> dict:
    """Same code path as scripts/compare_command_models.py: INT8 held-out test
    accuracy plus per-speaker isolated-word / false-accept figures over the
    real-voice approved set."""
    from kws_de.eval import _tflite_predict

    d = np.load(test_npz)
    X, y = d["X"], d["y"]
    preds = _tflite_predict(tflite_bytes, X)
    int8_acc = float((preds == y).mean())

    predict_fn = make_command_predict_fn(tflite_bytes)
    res = eval_recordings(APPROVED, predict_fn, manifest_path=MANIFEST, qc_root=None)

    words: dict[str, float] = {}
    fa_fired = fa_n = 0
    agg_n = agg_ok = 0
    for fig in res["figures"].values():
        for spk, row in fig["isolated"].items():
            if spk in SPEAKERS:
                words[spk] = row["acc"]
                agg_n += row["n"]
                agg_ok += round(row["acc"] * row["n"])
        for spk, row in fig["false_accepts"].items():
            if spk in SPEAKERS:
                fa_n += row["n"]
                fa_fired += round(row["rate"] * row["n"])

    aggregate_words = agg_ok / agg_n if agg_n else float("nan")
    return {
        "int8_test_acc": int8_acc,
        "words": words,
        "aggregate_words": aggregate_words,
        "aggregate_n": agg_n,
        "false_accepts": fa_fired,
        "false_accepts_n": fa_n,
    }


def macs_of(width: int) -> int:
    """MACs are purely architecture-derived (kws_de.budgets.estimate_macs
    only looks at layer shapes) so a fresh, untrained build_dscnn(width=...)
    gives the exact same number as the actually-trained model -- and, unlike
    tf.keras.models.load_model, needs no TF_USE_LEGACY_KERAS re-exec dance to
    read a QAT run's saved-under-tf_keras .keras file back."""
    from kws_de.budgets import estimate_macs
    from kws_de.model import build_dscnn

    model = build_dscnn(num_classes=len(config.COMMAND_LABELS), width=width)
    return estimate_macs(model)


def sha8(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:8]


def reference_row(label: str, tflite_bytes: bytes, width: int = 48) -> dict:
    s = score(tflite_bytes, TEST_NPZ)
    row = {
        "run": label,
        "width": width,
        "real_weight": "n/a",
        "qat_epochs": "n/a",
        "epochs": "n/a",
        "int8_test_acc": f"{s['int8_test_acc']:.4f}",
        "aggregate_words": f"{s['aggregate_words']:.4f}",
        "false_accepts": s["false_accepts"],
        "false_accepts_n": s["false_accepts_n"],
        "bytes": len(tflite_bytes),
        "sha256": sha8(tflite_bytes),
        "macs": macs_of(width),
        "train_seconds_predicted": "n/a",
        "train_seconds_actual": "n/a",
    }
    for spk in SPEAKERS:
        row[f"{spk}_words"] = f"{s['words'].get(spk, float('nan')):.4f}"
    return row


def run_one(spec: dict, env: dict) -> dict:
    width, real_weight, qat_epochs = spec["width"], spec["real_weight"], spec["qat_epochs"]
    run = run_id(width, real_weight, qat_epochs)
    print(f"\n=== run {run}: width={width} real_weight={real_weight} qat_epochs={qat_epochs} ===")

    out_name = f"command_grid_{run}.keras"
    export_dir = config.MODELS_DIR / "grid" / run

    d = np.load(TRAIN_NPZ)
    real_rows = int((~d["is_tts"]).sum())
    total_rows = d["X"].shape[0] + (real_weight - 1) * real_rows
    predicted_size = EPOCHS * total_rows

    predict_cmd = ["uv", "run", "--no-sync", "kws-eta", "predict", "train", str(predicted_size)]
    pred_out = subprocess.run(
        predict_cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True
    ).stdout.strip()
    print(pred_out)
    predicted_seconds = None
    if "~" in pred_out and "min" in pred_out:
        try:
            predicted_seconds = float(pred_out.split("~")[1].split("min")[0]) * 60
        except ValueError:
            predicted_seconds = None

    train_cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "kws_de.train",
        "--v2",
        "--prefix",
        PREFIX,
        "--epochs",
        str(EPOCHS),
        "--width",
        str(width),
        "--qat",
        "--qat-epochs",
        str(qat_epochs),
        "--real-weight",
        str(real_weight),
        "--out",
        out_name,
    ]
    t0 = time.monotonic()
    subprocess.run(train_cmd, cwd=REPO_ROOT, env=env, check=True)
    train_seconds_actual = time.monotonic() - t0

    export_cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "kws_de.export",
        "--v2",
        "--prefix",
        PREFIX,
        "--width",
        str(width),
        "--qat",
        "--model",
        out_name,
        "--out",
        str(export_dir),
    ]
    subprocess.run(export_cmd, cwd=REPO_ROOT, env=env, check=True)

    suffix = f"_w{width}" if width != 32 else ""
    tflite_path = export_dir / f"command_v3{suffix}_qat.tflite"
    tflite_bytes = tflite_path.read_bytes()
    s = score(tflite_bytes, TEST_NPZ)

    row = {
        "run": run,
        "width": width,
        "real_weight": real_weight,
        "qat_epochs": qat_epochs,
        "epochs": EPOCHS,
        "int8_test_acc": f"{s['int8_test_acc']:.4f}",
        "aggregate_words": f"{s['aggregate_words']:.4f}",
        "false_accepts": s["false_accepts"],
        "false_accepts_n": s["false_accepts_n"],
        "bytes": len(tflite_bytes),
        "sha256": sha8(tflite_bytes),
        "macs": macs_of(width),
        "train_seconds_predicted": (
            f"{predicted_seconds:.1f}" if predicted_seconds is not None else "n/a"
        ),
        "train_seconds_actual": f"{train_seconds_actual:.1f}",
    }
    for spk in SPEAKERS:
        row[f"{spk}_words"] = f"{s['words'].get(spk, float('nan')):.4f}"
    return row


def main() -> None:
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "scripts" / "recipe-grid.csv")
    ap.add_argument(
        "--only", default=None, help="run only this run id (e.g. w48_rw3_qe20), for retries"
    )
    ap.add_argument(
        "--skip-references",
        action="store_true",
        help="skip the deployed-w48 / run-2 reference rows (already recorded)",
    )
    args = ap.parse_args()

    env = dict(os.environ)
    if "KWS_DATA_ROOT" not in env:
        raise SystemExit("KWS_DATA_ROOT must be set (see docs/sphinx or paper-notes.md)")

    done = existing_runs(args.csv)

    if not args.skip_references:
        if "deployed_w48" not in done:
            deployed_bytes = model_bytes(DEPLOYED_HEADER)
            append_row(args.csv, reference_row("deployed_w48", deployed_bytes, width=48))
            print("recorded reference row: deployed_w48")
        if "run2_w48" not in done:
            run2_tflite = config.MODELS_DIR / "command_v3_w48_qat.tflite"
            if run2_tflite.exists():
                append_row(
                    args.csv,
                    reference_row("run2_w48", run2_tflite.read_bytes(), width=48),
                )
                print("recorded reference row: run2_w48")
            else:
                print(f"WARNING: {run2_tflite} not found -- skipping run2_w48 reference row")

    done = existing_runs(args.csv)
    for spec in GRID:
        run = run_id(spec["width"], spec["real_weight"], spec["qat_epochs"])
        if args.only and run != args.only:
            continue
        if run in done:
            print(f"skip {run} (already in {args.csv})")
            continue
        row = run_one(spec, env)
        append_row(args.csv, row)
        print(f"recorded: {row}")


if __name__ == "__main__":
    main()
