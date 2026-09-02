import numpy as np

from kws_de import config
from kws_de.export import to_int8_tflite, write_model_config
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
