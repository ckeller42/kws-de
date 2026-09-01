import numpy as np
import tensorflow as tf

from kws_de import config
from kws_de.export import balanced_calibration, to_int8_tflite, write_c_array
from kws_de.model import build_dscnn


def test_export_is_full_int8_and_runs(tmp_path):
    rng = np.random.default_rng(0)
    model = build_dscnn()
    rep = rng.standard_normal((8, config.N_FRAMES, config.N_MFCC)).astype(np.float32)
    blob = to_int8_tflite(model, rep)
    itp = tf.lite.Interpreter(model_content=blob)
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8
    x = rng.integers(-128, 127, size=inp["shape"], dtype=np.int8)
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    assert itp.get_tensor(out["index"]).shape[-1] == config.NUM_CLASSES


def test_c_array_header(tmp_path):
    p = tmp_path / "model_data.h"
    write_c_array(b"\x01\x02\x03", p)
    txt = p.read_text()
    assert "g_model[]" in txt and "g_model_len = 3" in txt


def test_balanced_calibration_caps_per_class_and_keeps_every_class():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 3 + [2] * 20)
    X = rng.standard_normal((len(y), 4, 3)).astype(np.float32)
    X[:, 0, 0] = y  # tag each row with its class
    rep = balanced_calibration(X, y, per_class=5, seed=0)
    got = np.bincount(rep[:, 0, 0].astype(int), minlength=3)
    assert got.tolist() == [5, 3, 5]


def test_balanced_calibration_is_deterministic_in_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=200)
    X = rng.standard_normal((200, 2, 2)).astype(np.float32)
    a = balanced_calibration(X, y, per_class=10, seed=7)
    b = balanced_calibration(X, y, per_class=10, seed=7)
    c = balanced_calibration(X, y, per_class=10, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
