from kws_de import config
from kws_de.architectures import ARCHITECTURES, get


def test_registry_has_expected_names():
    assert {"ds_cnn", "bc_resnet", "matchboxnet", "kwt"} <= set(ARCHITECTURES)


def test_ds_cnn_builder_shape():
    m = get("ds_cnn")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)


def test_bc_resnet_builds_and_is_small():
    m = get("bc_resnet")((config.N_FRAMES, config.N_MFCC, 1), n_classes=config.NUM_CLASSES)
    assert m.input_shape == (None, config.N_FRAMES, config.N_MFCC, 1)
    assert m.output_shape == (None, config.NUM_CLASSES)
    assert m.count_params() < 60_000
