import numpy as np

from kws_de import config
from kws_de.budgets import check_wake_budgets
from kws_de.export import to_int8_tflite  # from microWakeWord (spec §12)
from kws_de.model import build_dscnn  # stand-in small model; real wake tflite comes


def test_wake_budget_checks_tflite_bytes_only():
    m = build_dscnn()
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(m, rep)  # a small INT8 tflite ~ wake-model size class
    r = check_wake_budgets(blob)
    assert r["model_bytes"] <= 150_000 and r["int8"] is True
