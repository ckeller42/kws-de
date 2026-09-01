from kws_de import config


def test_v2_vocab():
    assert config.WAKE_WORD == "Hey Bus"
    assert config.WAKE_LABELS == ["wake", "_not_"]
    assert config.DEVICES == [
        "Licht",
        "Kühlschrank",
        "Heizung",
        "Aufstelldach",
        "Campingmodus",
        "USB",
        "Wasser",
        "Energie",
    ]
    assert config.ZONES == ["Küche", "Dach", "Außen", "Lesen"]
    assert "auf" in config.ACTIONS and "heller" in config.ACTIONS and "Eco" in config.ACTIONS
    assert config.ZONED_DEVICES == ["Licht"]
    assert config.COMMAND_LABELS == (
        config.DEVICES + config.ZONES + config.ACTIONS + ["_unknown_", "_silence_"]
    )


def test_device_actions_cover_all_devices_and_use_valid_actions():
    assert set(config.DEVICE_ACTIONS) == set(config.DEVICES)
    for _dev, acts in config.DEVICE_ACTIONS.items():
        assert acts and all(a in config.ACTIONS for a in acts)


def test_command_index_roundtrip():
    for i, lbl in enumerate(config.COMMAND_LABELS):
        assert config.command_index(lbl) == i


def test_v1_constants_unchanged():
    assert config.COMMANDS == ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
    assert config.NUM_CLASSES == 7
