import numpy as np
import tensorflow as tf
from kws_de import config
from kws_de.export import to_int8_tflite, write_c_array
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
