import re
import sys

import numpy as np
import pytest

from kws_de import config, export
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


def test_bootstrap_classes_get_a_separate_lower_floor():
    """E54: classes with zero real training clips (TTS-only bootstrap, e.g. a
    just-promoted word) must not be held to the blanket 50% floor, but a
    realistic bootstrap accuracy (24-29% for run8_s1's 3 new classes) still
    passes its own much lower sanity floor."""
    n = len(config.COMMAND_LABELS)  # 26
    bootstrap = {n - 3, n - 2, n - 1}
    rows_per_class = 40
    y = np.repeat(np.arange(n), rows_per_class)
    y_pred = y.copy()
    # Bootstrap classes score 25% (run8_s1's actual 24-29%) -> not 100%, but
    # excluded from the mature floor and clears its own 15% floor.
    for c in bootstrap:
        rows = np.flatnonzero(y == c)
        y_pred[rows[rows_per_class // 4 :]] = (c + 1) % n

    h = assert_model_healthy(
        y, y_pred, bootstrap_classes=bootstrap, class_names=config.COMMAND_LABELS
    )
    assert h["accuracy"] == 1.0  # mature-only accuracy: untouched by low bootstrap scores
    assert h["bootstrap_accuracy"] == {config.COMMAND_LABELS[c]: 0.25 for c in bootstrap}


def test_bootstrap_floor_still_rejects_a_near_chance_class():
    """A bootstrap class must not be a free pass: near-chance accuracy (export
    corruption, label-index scrambling) still fails, just against its own
    floor instead of the blanket one."""
    n = len(config.COMMAND_LABELS)
    bootstrap = {n - 3, n - 2, n - 1}
    rows_per_class = 40
    y = np.repeat(np.arange(n), rows_per_class)
    y_pred = y.copy()
    broken = n - 1
    rows = np.flatnonzero(y == broken)
    y_pred[rows] = (broken + 1) % n  # 0% accuracy on the broken bootstrap class

    with pytest.raises(ValueError, match="bootstrap class"):
        assert_model_healthy(
            y, y_pred, bootstrap_classes=bootstrap, class_names=config.COMMAND_LABELS
        )


def test_firmware_export_leaves_canonical_files_untouched_on_health_failure(tmp_path, monkeypatch):
    """E54: a failed health check must not leave canonical
    command_*.tflite/_data.h/_metadata.json overwritten with the rejected
    model's bytes while firmware/main/gen still names the old model -- the
    write order must be validate-then-write, not write-then-validate-and-hope
    someone catches a failure by hand."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    config.MODELS_DIR.mkdir()
    config.DATA_DIR.mkdir()
    monkeypatch.chdir(tmp_path)

    n = len(config.COMMAND_LABELS)
    build_dscnn(num_classes=n).save(config.MODELS_DIR / "command.keras")

    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(n), 4)
    X = rng.standard_normal((len(y), config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    is_tts = np.zeros(len(y), dtype=bool)
    np.savez(config.DATA_DIR / "features_test_train.npz", X=X, y=y, is_tts=is_tts)
    np.savez(config.DATA_DIR / "features_test_test.npz", X=X, y=y, is_tts=is_tts)

    canonical = config.MODELS_DIR / "command_test.tflite"
    canonical.write_bytes(b"SENTINEL-TFLITE")
    header = config.MODELS_DIR / "command_test_data.h"
    header.write_text("SENTINEL-HEADER")
    meta = config.MODELS_DIR / "command_test_metadata.json"
    meta.write_text("SENTINEL-META")
    gen_model = tmp_path / "firmware" / "main" / "gen" / "model_data.h"
    gen_model.parent.mkdir(parents=True)
    gen_model.write_text("SENTINEL-GEN")

    monkeypatch.setattr(sys, "argv", ["kws-export", "--firmware", "--prefix", "features_test"])
    with pytest.raises(ValueError):  # untrained model: nowhere near the health floor
        export.main()

    assert canonical.read_bytes() == b"SENTINEL-TFLITE"
    assert header.read_text() == "SENTINEL-HEADER"
    assert meta.read_text() == "SENTINEL-META"
    assert gen_model.read_text() == "SENTINEL-GEN"


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
