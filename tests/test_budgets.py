import numpy as np
from kws_de import config
from kws_de.budgets import check_budgets, estimate_macs, is_full_int8
from kws_de.export import to_int8_tflite
from kws_de.model import build_dscnn


def test_macs_and_budgets_pass_for_small_model():
    m = build_dscnn()
    assert 0 < estimate_macs(m) <= config.MAX_MACS
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)
    assert is_full_int8(blob)
    report = check_budgets(blob, m)
    assert report["model_bytes"] <= config.MAX_MODEL_BYTES
    assert report["int8"] is True
