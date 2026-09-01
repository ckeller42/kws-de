"""Knowledge distillation (Hinton et al. 2015): a KWT teacher (accurate, not
device-runnable) -> the unchanged DS-CNN student (deployable).

Both models end in softmax, so "logits" are recovered as log(p): log-softmax
equals the logits up to a per-row constant, and softmax(log p / T) is exact
temperature scaling. The teacher is frozen, so its probabilities are computed
once and carried in the target tensor `[one_hot(y) | teacher_probs]`; the
student then trains with plain `model.fit`. No tfmot, no custom train step.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from kws_de import config
from kws_de.model import build_dscnn

_EPS = 1e-9


def soften(p: np.ndarray, T: float) -> np.ndarray:
    """Temperature-scale a probability row-batch: softmax(log(p) / T)."""
    z = np.log(np.asarray(p, np.float64) + _EPS) / T
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def distill_targets(y: np.ndarray, teacher_probs: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, 2C): hard one-hot label followed by the teacher's (untempered) probs."""
    one_hot = np.eye(n_classes, dtype=np.float32)[np.asarray(y)]
    return np.concatenate([one_hot, np.asarray(teacher_probs, np.float32)], axis=1)


def _tf_soften(p, T):
    return tf.nn.softmax(tf.math.log(p + _EPS) / T, axis=-1)


def make_distill_loss(n_classes: int, T: float, alpha: float):
    """alpha * CE(hard, student) + (1 - alpha) * T^2 * KL(teacher_T || student_T)."""

    def loss(y_true, y_pred):
        hard = y_true[:, :n_classes]
        teacher = y_true[:, n_classes:]
        ce = tf.keras.losses.categorical_crossentropy(hard, y_pred)
        t_soft = _tf_soften(teacher, T)
        s_soft = _tf_soften(y_pred, T)
        kl = tf.reduce_sum(t_soft * (tf.math.log(t_soft + _EPS) - tf.math.log(s_soft + _EPS)), -1)
        return alpha * ce + (1.0 - alpha) * (T**2) * kl

    return loss


def hard_accuracy(n_classes: int):
    """Keras metric on the one-hot half, named `accuracy` so
    `ModelCheckpoint(monitor="val_accuracy")` keeps working."""

    def accuracy(y_true, y_pred):
        return tf.keras.metrics.categorical_accuracy(y_true[:, :n_classes], y_pred)

    return accuracy


def distill(
    X,
    y,
    teacher,
    *,
    epochs: int,
    seed: int,
    num_classes: int | None = None,
    T: float = 4.0,
    alpha: float = 0.5,
    validation_data=None,
    callbacks=None,
):
    """Train a fresh `build_dscnn` student on (X, y) against `teacher`'s frozen
    probabilities. Mirrors `kws_de.train.train` (adam, batch 32, inverse-frequency
    balancing) but via `sample_weight`, since a 2-D target rules out `class_weight`."""
    tf.keras.utils.set_random_seed(seed)
    num_classes = num_classes if num_classes is not None else len(config.COMMAND_LABELS)
    Xc = np.asarray(X, np.float32)[..., None]
    y = np.asarray(y)
    targets = distill_targets(y, teacher.predict(Xc, verbose=0), num_classes)
    counts = np.bincount(y, minlength=num_classes)
    w_class = np.where(counts > 0, len(y) / (num_classes * np.maximum(counts, 1)), 0.0)
    sample_weight = w_class[y].astype(np.float32)

    student = build_dscnn(num_classes=num_classes)
    student.compile(
        optimizer="adam",
        loss=make_distill_loss(num_classes, T, alpha),
        metrics=[hard_accuracy(num_classes)],
    )
    if validation_data is not None:
        Xv, yv = validation_data
        Xv = np.asarray(Xv, np.float32)[..., None]
        validation_data = (Xv, distill_targets(yv, teacher.predict(Xv, verbose=0), num_classes))
    h = student.fit(
        Xc,
        targets,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    return student, h.history


def _best_val_checkpoint(fit):  # pragma: no cover - training I/O
    """Run `fit(callbacks)` with a best-val_accuracy ModelCheckpoint, reload it."""
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "best.weights.h5")
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            path, save_best_only=True, save_weights_only=True, monitor="val_accuracy", mode="max"
        )
        model, history = fit([ckpt])
        model.load_weights(path)
    return model, history


def main() -> None:  # pragma: no cover - training + TTS + tflite
    """`kws-distill`: on one split/seed, train KWT teacher, undistilled DS-CNN
    baseline, distilled DS-CNN; INT8-evaluate the two device models with a
    class-balanced calibration set, plus the baseline with the legacy
    `X_train[:200]` calibration so the PTQ recovery is one table row."""
    from kws_de.architectures import ARCHITECTURES
    from kws_de.benchmark import CATALOG_VOICES, _evaluate_int8, render_table
    from kws_de.dataset import load_split
    from kws_de.eval import _keras_predict
    from kws_de.export import balanced_calibration
    from kws_de.train import train

    ap = argparse.ArgumentParser(prog="kws-distill")
    ap.add_argument("--features", default="features", help="npz prefix (features | features_v3)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    n_classes = len(config.COMMAND_LABELS)
    X_train, y_train, _ = load_split("train", args.features)
    X_val, y_val, _ = load_split("val", args.features)
    X_test, y_test, _ = load_split("test", args.features)
    shape = (config.N_FRAMES, config.N_MFCC, 1)

    tf.keras.utils.set_random_seed(args.seed)
    teacher, _ = _best_val_checkpoint(
        lambda cb: train(
            X_train,
            y_train,
            epochs=args.epochs,
            seed=args.seed,
            num_classes=n_classes,
            model=ARCHITECTURES["kwt"](shape, n_classes=n_classes),
            validation_data=(X_val, y_val),
            callbacks=cb,
        )
    )
    teacher_acc = float((_keras_predict(teacher, X_test) == y_test).mean())

    baseline, _ = _best_val_checkpoint(
        lambda cb: train(
            X_train,
            y_train,
            epochs=args.epochs,
            seed=args.seed,
            num_classes=n_classes,
            validation_data=(X_val, y_val),
            callbacks=cb,
        )
    )
    student, _ = _best_val_checkpoint(
        lambda cb: distill(
            X_train,
            y_train,
            teacher,
            epochs=args.epochs,
            seed=args.seed,
            num_classes=n_classes,
            T=args.T,
            alpha=args.alpha,
            validation_data=(X_val, y_val),
            callbacks=cb,
        )
    )

    calib = balanced_calibration(X_train, y_train, seed=args.seed)
    common = dict(seed=args.seed)
    rows = [
        _evaluate_int8(
            "ds_cnn (first-200 calib)", baseline, X_test, y_test, calib=X_train[:200], **common
        ),
        _evaluate_int8("ds_cnn (balanced calib)", baseline, X_test, y_test, calib=calib, **common),
        _evaluate_int8(
            "ds_cnn distilled (balanced calib)", student, X_test, y_test, calib=calib, **common
        ),
    ]

    repo_root = config.DATA_DIR.parent
    intro = (
        "# Distillation + INT8 calibration report\n\n"
        f"Dataset prefix `{args.features}`, epochs={args.epochs}, seed={args.seed}, "
        f"T={args.T}, alpha={args.alpha}. Teacher = KWT (reference-only, float): "
        f"test accuracy **{teacher_acc:.3f}**. Student = DS-CNN (unchanged "
        "`build_dscnn`). **Float** = Keras float32 test accuracy, **Isolated** = "
        "INT8 test accuracy, **Catalog** = full-intent catalog accuracy "
        f"({len(CATALOG_VOICES)} voices). Rows 1-2 differ only in the PTQ "
        "calibration set (`X_train[:200]` vs `kws_de.export.balanced_calibration`).\n\n"
    )
    (repo_root / "docs" / "distill-report.md").write_text(intro + render_table(rows))
    (repo_root / "docs" / "distill-benchmark.json").write_text(
        json.dumps({"teacher_acc": teacher_acc, "rows": rows}, indent=2, ensure_ascii=False) + "\n"
    )
    print("wrote docs/distill-report.md, docs/distill-benchmark.json")


if __name__ == "__main__":  # pragma: no cover
    main()
