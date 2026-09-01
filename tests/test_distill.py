import numpy as np
import tensorflow as tf

from kws_de import config
from kws_de.distill import distill, distill_targets, hard_accuracy, make_distill_loss, soften


def test_soften_identity_at_T1_and_flatter_at_higher_T():
    p = np.array([[0.7, 0.2, 0.1]], np.float32)
    assert np.allclose(soften(p, 1.0), p, atol=1e-5)
    hot = soften(p, 4.0)
    assert np.isclose(hot.sum(), 1.0) and hot.max() < p.max()
    assert np.argmax(hot) == 0  # ordering preserved


def test_distill_targets_concatenates_one_hot_and_teacher():
    y = np.array([2, 0])
    t = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], np.float32)
    out = distill_targets(y, t, 3)
    assert out.shape == (2, 6)
    assert out[0, :3].tolist() == [0, 0, 1] and np.allclose(out[:, 3:], t)


def test_loss_alpha1_is_plain_cross_entropy():
    y = np.array([1, 0])
    t = np.array([[0.5, 0.5], [0.5, 0.5]], np.float32)
    pred = tf.constant([[0.2, 0.8], [0.9, 0.1]], tf.float32)
    y_true = tf.constant(distill_targets(y, t, 2))
    loss = make_distill_loss(2, T=4.0, alpha=1.0)(y_true, pred)
    ce = tf.keras.losses.sparse_categorical_crossentropy(tf.constant(y), pred)
    assert np.allclose(loss.numpy(), ce.numpy(), atol=1e-5)


def test_loss_kl_term_is_zero_when_student_matches_teacher():
    y = np.array([0])
    t = np.array([[0.3, 0.7]], np.float32)
    y_true = tf.constant(distill_targets(y, t, 2))
    pred = tf.constant(t)
    loss = make_distill_loss(2, T=2.0, alpha=0.0)(y_true, pred)
    assert abs(float(loss.numpy()[0])) < 1e-5


def test_hard_accuracy_reads_one_hot_half():
    y_true = tf.constant(distill_targets(np.array([1, 0]), np.zeros((2, 2), np.float32), 2))
    pred = tf.constant([[0.1, 0.9], [0.1, 0.9]], tf.float32)
    acc = hard_accuracy(2)(y_true, pred)
    assert np.isclose(float(tf.reduce_mean(acc)), 0.5)


def _toy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.1, size=(n, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    X[y == 1] += 2.0  # trivially separable
    return X, y


def test_distill_student_learns_separable_toy():
    X, y = _toy()

    # A "teacher" that is just a fixed lookup: returns confident-but-soft probs.
    class Teacher:
        def predict(self, Xc, verbose=0):
            hot = (Xc.reshape(len(Xc), -1).mean(1) > 1.0).astype(np.float32)
            return np.stack([1 - hot, hot], 1) * 0.8 + 0.1

    # build_dscnn's BatchNorm moving stats (momentum 0.99) need well over 8
    # epochs x 2 batches to converge for this low-variance toy input; 8 epochs
    # (as the brief's draft used) leaves `student.predict` (inference-mode BN)
    # collapsed to the majority class even though train-mode "accuracy" already
    # reads 1.0 -- verified against the pre-existing kws_de.train.train too, so
    # it's a BN-momentum/epoch-budget property of build_dscnn, not a distill()
    # bug. 300 epochs is comfortably past the ~200-epoch point where it clears.
    student, history = distill(X, y, Teacher(), epochs=300, seed=0, num_classes=2)
    assert history["accuracy"][-1] > 0.9
    preds = np.argmax(student.predict(X[..., None], verbose=0), 1)
    assert (preds == y).mean() > 0.9
