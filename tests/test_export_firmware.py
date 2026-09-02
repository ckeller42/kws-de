import numpy as np
import pytest

from kws_de import config
from kws_de.export import assert_model_healthy, to_int8_tflite, write_model_config
from kws_de.model import build_dscnn


def test_write_model_config_reports_quant_and_arena(tmp_path):
    rng = np.random.default_rng(0)
    model = build_dscnn(num_classes=len(config.COMMAND_LABELS))
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(model, rep)
    p = tmp_path / "model_config.h"
    info = write_model_config(blob, p)
    txt = p.read_text()
    assert "#define KWS_MODEL_INPUT_SCALE" in txt and "#define KWS_MODEL_INPUT_ZERO_POINT" in txt
    assert f"#define KWS_MODEL_NUM_CLASSES {len(config.COMMAND_LABELS)}" in txt
    assert info["arena_bytes"] % 4096 == 0 and info["arena_bytes"] > 0
    assert f"#define KWS_MODEL_ARENA_BYTES {info['arena_bytes']}" in txt


def test_model_health_gate_catches_broken_models():
    """Regression guard for the broken-model bug: a mode-collapsed or
    random-accuracy model must be rejected before it reaches the firmware."""
    rng = np.random.default_rng(0)
    n = len(config.COMMAND_LABELS)
    y = rng.integers(0, n, size=800)

    # A healthy model: mostly correct, uses the whole label set -> passes.
    good = y.copy()
    good[:80] = (good[:80] + 1) % n  # ~90% accuracy
    h = assert_model_healthy(y, good)
    assert h["accuracy"] > 0.8 and h["predicted_classes"] >= n - 2

    # Mode collapse: predictions cover too few classes. Use a case that is
    # otherwise accurate so the class-count guard is what trips (not the floor).
    few = np.zeros(800, dtype=int)
    few[400:] = 1
    with pytest.raises(ValueError, match="mode collapse"):
        assert_model_healthy(few, few.copy())  # 100% accuracy but only 2 classes

    # ~Random accuracy across all classes: fails the accuracy floor.
    with pytest.raises(ValueError, match="below"):
        assert_model_healthy(y, rng.integers(0, n, size=800))
