"""Phase-1 architecture benchmark: train each device-runnable architecture on
the frozen v2 dataset, INT8-export, and report isolated-word + full-intent
catalog accuracy alongside on-device cost. See
docs/superpowers/plans/2026-09-01-kws-benchmark-phase1.md (Tasks 5-6).
"""

import argparse
import json
import tempfile
from pathlib import Path

from kws_de import config
from kws_de.architectures import ARCHITECTURES
from kws_de.budgets import check_budgets, estimate_macs
from kws_de.data import _TTS_VOICES
from kws_de.dataset import load_split
from kws_de.eval import _keras_predict, _tflite_predict, make_command_predict_fn, run_catalog_eval
from kws_de.export import balanced_calibration, to_int8_tflite
from kws_de.train import train

EPOCHS = 30
SEED = 0

# Architectures that INT8-export within the TFLM op set and actually run on the
# ESP32-S3 -- kwt is excluded (reference-only: MultiHeadAttention/LayerNormalization
# lower to ops outside the TFLM set even though it INT8-exports); see
# kws_de/architectures/kwt.py and tests/test_architectures.py.
DEVICE_RUNNABLE_ARCHITECTURES = ("ds_cnn", "bc_resnet", "matchboxnet")

# The full command catalog (~49 entries) x the full `_TTS_VOICES` (9) x 3
# architectures would shell out to macOS `say` and run the streaming pipeline
# thousands of times. 3 voices keeps this tractable while still exercising
# voice diversity; docs/benchmark.md notes this vs. the 4-voice eval used in
# docs/eval-report-v2.md.
CATALOG_VOICES = _TTS_VOICES[:3]


def render_table(rows: list[dict]) -> str:
    """Pure Markdown comparison table. Float = Keras float32 test accuracy (- if the
    row predates it), Isolated = INT8 test accuracy, so the PTQ gap is a column."""
    header = "| Architecture | Float | Isolated | Catalog | Params | MACs | INT8 | Budget |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        budget = "yes" if r["budget_ok"] else "no"
        flt = f"{r['float_acc']:.3f}" if "float_acc" in r else "-"
        lines.append(
            f"| {r['name']} | {flt} | {r['isolated_acc']:.3f} | {r['catalog_acc']:.3f} | "
            f"{r['params']:,} | {r['macs']:,} | {r['int8_bytes']:,} | {budget} |"
        )
    return "\n".join(lines) + "\n"


def _evaluate_int8(
    name: str, model, X_test, y_test, *, seed: int, calib
) -> dict:  # pragma: no cover - tflite + catalog TTS
    """Shared tail of every device-model evaluation: float test accuracy, INT8
    export on `calib`, INT8 isolated + catalog accuracy, on-device cost."""
    float_acc = float((_keras_predict(model, X_test) == y_test).mean())
    tflite_bytes = to_int8_tflite(model, calib)
    isolated_acc = float((_tflite_predict(tflite_bytes, X_test) == y_test).mean())
    catalog = run_catalog_eval(make_command_predict_fn(tflite_bytes), CATALOG_VOICES, seed=seed)
    try:
        check_budgets(tflite_bytes, model)
        budget_ok = True
    except AssertionError:
        budget_ok = False
    return {
        "name": name,
        "float_acc": float_acc,
        "isolated_acc": isolated_acc,
        "catalog_acc": catalog["overall_accuracy"],
        "catalog_trials": catalog["total_trials"],
        "params": int(model.count_params()),
        "macs": estimate_macs(model),
        "int8_bytes": len(tflite_bytes),
        "budget_ok": budget_ok,
    }


def evaluate_architecture(
    name: str, epochs: int = EPOCHS, seed: int = SEED, features: str = "features"
) -> dict:
    # pragma: no cover - heavy I/O (training + TTS + tflite)
    """Build `name`, train on `load_split("train", features)` (val-selected best
    epoch), then `_evaluate_int8` with a class-balanced calibration set."""
    import tensorflow as tf

    n_classes = len(config.COMMAND_LABELS)
    input_shape = (config.N_FRAMES, config.N_MFCC, 1)

    X_train, y_train, _ = load_split("train", features)
    X_val, y_val, _ = load_split("val", features)
    X_test, y_test, _ = load_split("test", features)

    tf.keras.utils.set_random_seed(seed)
    model = ARCHITECTURES[name](input_shape, n_classes=n_classes)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = str(Path(td) / "best.weights.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
        )
        model, _history = train(
            X_train,
            y_train,
            epochs=epochs,
            seed=seed,
            num_classes=n_classes,
            model=model,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint],
        )
        model.load_weights(ckpt_path)  # best val_accuracy epoch, not necessarily the last

    calib = balanced_calibration(X_train, y_train, seed=seed)
    return _evaluate_int8(name, model, X_test, y_test, seed=seed, calib=calib)


def main() -> None:  # pragma: no cover - I/O wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features")
    args = parser.parse_args()

    repo_root = config.DATA_DIR.parent
    rows = [
        evaluate_architecture(name, epochs=EPOCHS, seed=SEED, features=args.features)
        for name in DEVICE_RUNNABLE_ARCHITECTURES
    ]

    intro = (
        "# KWS architecture benchmark\n\n"
        "Phase-1 comparison of the device-runnable architecture zoo on the frozen "
        f"v2 dataset (`kws_de.dataset.load_split`, prefix `{args.features}`, seed=0). "
        "**Float** = Keras float32 test-set word accuracy, **Isolated** = INT8 "
        "test-set word accuracy (the gap is the PTQ cost). **Catalog** = full-intent "
        "accuracy over the enumerated command catalog (`kws_de.eval.run_catalog_eval`: "
        "TTS-synthesized, clean/no noise, streaming detector + grammar parse), "
        f"{len(CATALOG_VOICES)} voices ({', '.join(CATALOG_VOICES)}) -- reduced from "
        "the full 9-voice `_TTS_VOICES` set (and the 4-voice eval used in "
        "`docs/eval-report-v2.md`) to keep 3 architectures x ~49 catalog entries "
        "tractable; treat Catalog as indicative, not the precision-tuned number. "
        "**Params/MACs/INT8** are on-device cost (`kws_de.budgets`); **Budget** = "
        f"fits the Phase-1 budgets (model <= {config.MAX_MODEL_BYTES:,} bytes, "
        f"MACs <= {config.MAX_MACS:,}, full INT8 I/O).\n\n"
        "KWT (Keyword Transformer) is **reference-only**: it INT8-exports but "
        "`MultiHeadAttention`/`LayerNormalization` lower to BATCH_MATMUL/TRANSPOSE/"
        "GATHER/CONCATENATION/TILE plus float DEQUANTIZE/QUANTIZE bridges, outside "
        "the TFLM op set the other three architectures stay within -- not "
        "device-runnable, so it is excluded from this benchmark run (see "
        "`kws_de/architectures/kwt.py`, "
        "`tests/test_architectures.py::test_kwt_is_not_tflm_device_runnable`).\n\n"
        f"Config: epochs={EPOCHS}, seed={SEED}, dataset manifest seed=0 "
        "(`data/manifest.json`).\n\n"
    )

    (repo_root / "docs" / "benchmark.md").write_text(intro + render_table(rows))
    (repo_root / "benchmark.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print("wrote docs/benchmark.md, benchmark.json")


if __name__ == "__main__":  # pragma: no cover
    main()
