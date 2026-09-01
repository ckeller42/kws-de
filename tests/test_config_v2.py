from kws_de import config


def test_v2_vocab():
    assert config.WAKE_WORD == "Hey Bus"
    assert config.WAKE_LABELS == ["wake", "_not_"]
    assert config.DEVICES == ["Licht", "Heizung", "Kühlschrank", "Wasser"]
    assert config.ZONES == ["Küche", "Bad", "Decke", "Außen"]
    assert config.ACTIONS == ["an", "aus"]
    assert config.COMMAND_LABELS == (
        config.DEVICES + config.ZONES + config.ACTIONS + ["_unknown_", "_silence_"]
    )


def test_command_index_roundtrip():
    for i, lbl in enumerate(config.COMMAND_LABELS):
        assert config.command_index(lbl) == i


def test_v1_constants_unchanged():
    assert config.COMMANDS == ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
    assert config.NUM_CLASSES == 7
