"""Knowledge distillation (Hinton et al. 2015): a KWT teacher (accurate, not
device-runnable) -> the unchanged DS-CNN student (deployable).

Both models end in softmax, so "logits" are recovered as log(p): log-softmax
equals the logits up to a per-row constant, and softmax(log p / T) is exact
temperature scaling. The teacher is frozen, so its probabilities are computed
once and carried in the target tensor `[one_hot(y) | teacher_probs]`; the
student then trains with plain `model.fit`. No tfmot, no custom train step.
"""

import numpy as np
import tensorflow as tf

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

    accuracy.__name__ = "accuracy"
    return accuracy
