from kws_de.grammar import Intent, Rejection, parse


def test_device_zone_action():
    assert parse(["Licht", "Küche", "an"]) == Intent("Licht", "Küche", "an")


def test_device_action_no_zone():
    assert parse(["Licht", "an"]) == Intent("Licht", None, "an")


def test_ignores_unknown_and_silence():
    assert parse(["_silence_", "Licht", "_unknown_", "an"]) == Intent("Licht", None, "an")


def test_missing_action_rejected():
    assert isinstance(parse(["Licht", "Küche"]), Rejection)


def test_missing_device_rejected():
    assert isinstance(parse(["Küche", "an"]), Rejection)


def test_out_of_order_rejected():
    assert isinstance(parse(["an", "Licht"]), Rejection)


def test_duplicate_slot_rejected():
    assert isinstance(parse(["Licht", "Heizung", "an"]), Rejection)


def test_action_invalid_for_device_rejected():
    assert isinstance(parse(["Aufstelldach", "an"]), Rejection)  # roof has no an/aus


def test_roof_auf_valid():
    assert parse(["Aufstelldach", "auf"]) == Intent("Aufstelldach", None, "auf")


def test_zone_on_non_zoned_device_rejected():
    assert isinstance(parse(["Heizung", "Küche", "an"]), Rejection)


def test_light_zone_brightness_valid():
    assert parse(["Licht", "Küche", "heller"]) == Intent("Licht", "Küche", "heller")


def test_energy_mode_valid():
    assert parse(["Energie", "Eco"]) == Intent("Energie", None, "Eco")
