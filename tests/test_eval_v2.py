from kws_de import config
from kws_de.eval import build_catalog, intent_accuracy
from kws_de.grammar import Intent, parse


def test_intent_accuracy_all_slots_must_match():
    t = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "aus")]
    p = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "an")]  # 2nd action wrong
    assert intent_accuracy(t, p) == 0.5


def test_catalog_covers_every_device_action_and_all_are_valid():
    catalog = build_catalog()
    # bare "device action" present for every (device, action) pair
    expected_bare = {(dev, act) for dev in config.DEVICES for act in config.DEVICE_ACTIONS[dev]}
    bare_in_catalog = {(i.device, i.action) for i in catalog if i.zone is None}
    assert bare_in_catalog == expected_bare
    # every catalog entry parses back to itself via the real grammar (no bogus entries)
    for intent in catalog:
        toks = [intent.device, *([intent.zone] if intent.zone else []), intent.action]
        assert parse(toks) == intent


def test_catalog_zones_only_on_zoned_devices():
    catalog = build_catalog()
    zoned = {i.device for i in catalog if i.zone is not None}
    assert zoned == set(config.ZONED_DEVICES)
