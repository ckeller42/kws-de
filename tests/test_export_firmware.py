import re

import numpy as np
import pytest

from kws_de import config
from kws_de.export import (
    assert_model_healthy,
    to_int8_tflite,
    write_model_config,
    write_wake_headers,
)
from kws_de.model import build_dscnn


def test_write_model_config_reports_quant_and_arena(tmp_path):
    rng = np.random.default_rng(0)
    model = build_dscnn(num_classes=len(config.COMMAND_LABELS))
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(model, rep)
    src = tmp_path / "command.tflite"
    src.write_bytes(blob)
    p = tmp_path / "model_config.h"
    info = write_model_config(blob, p, src)
    txt = p.read_text()
    assert "#define KWS_MODEL_INPUT_SCALE" in txt and "#define KWS_MODEL_INPUT_ZERO_POINT" in txt
    assert f"#define KWS_MODEL_NUM_CLASSES {len(config.COMMAND_LABELS)}" in txt
    assert info["arena_bytes"] % 4096 == 0 and info["arena_bytes"] > 0
    assert f"#define KWS_MODEL_ARENA_BYTES {info['arena_bytes']}" in txt
    # Stamp: <name>@<8 hex> <YYYY-MM-DD>, so the device can report its model.
    assert re.search(r'#define KWS_MODEL_ID "command\.tflite@[0-9a-f]{8} \d{4}-\d\d-\d\d"', txt)
    assert f"#define KWS_MODEL_BYTES {len(blob)}" in txt


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


def test_write_wake_headers_emits_model_contract(tmp_path):
    """The wake headers must carry the exact quantisation contract the firmware
    compiles against. Uses the real microWakeWord export (it is not retrained
    here, only read); skipped when the artifact is absent."""
    model = config.MODELS_DIR / "hey_bus.tflite"
    if not model.exists():
        pytest.skip("models/hey_bus.tflite absent")
    info = write_wake_headers(model, tmp_path)
    assert info is not None

    data = (tmp_path / "wake_model_data.h").read_text()
    assert "g_wake_model[]" in data
    assert f"g_wake_model_len = {model.stat().st_size}" in data

    cfg = (tmp_path / "wake_model_config.h").read_text()
    # [1, 3, 40] int8 spectrogram rows in, one uint8 probability out.
    assert "#define KWS_WAKE_FRAMES 3" in cfg
    assert "#define KWS_WAKE_FEATURES 40" in cfg
    assert "#define KWS_WAKE_INPUT_ZERO_POINT -128" in cfg
    # uint8 output quantised as q/256, so prob = q * 1/256 with zero_point 0.
    assert "#define KWS_WAKE_OUTPUT_SCALE 3.90625000e-03f" in cfg
    assert "#define KWS_WAKE_OUTPUT_ZERO_POINT 0" in cfg
    assert info["arena_bytes"] % 4096 == 0
    assert f"#define KWS_WAKE_ARENA_BYTES {info['arena_bytes']}" in cfg
    assert re.search(
        r'#define KWS_WAKE_MODEL_ID "hey_bus\.tflite@[0-9a-f]{8} \d{4}-\d\d-\d\d"', cfg
    )
    assert f"#define KWS_WAKE_MODEL_BYTES {model.stat().st_size}" in cfg


def test_write_wake_headers_skips_a_missing_model(tmp_path):
    assert write_wake_headers(tmp_path / "nope.tflite", tmp_path) is None
    assert not list(tmp_path.iterdir())
