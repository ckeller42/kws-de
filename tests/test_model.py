from kws_de import config
from kws_de.model import build_dscnn


def test_output_shape_and_params():
    m = build_dscnn()
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 40_000


def test_width_shrinks_param_count():
    wide = build_dscnn(width=32)
    narrow = build_dscnn(width=16)
    assert wide.output_shape == narrow.output_shape
    assert narrow.count_params() < wide.count_params()
