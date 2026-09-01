from kws_de.tts import _piper_voice_path, voice_combos


def test_voice_combos_round_robin_spans_engines():
    combos = voice_combos(6, ["say", "piper"])
    engines_used = [e for e, _, _ in combos]
    assert set(engines_used[:2]) == {"say", "piper"}  # alternates engines first
    assert len(combos) == 6


def test_voice_combos_caps_at_n():
    assert len(voice_combos(3, ["say"])) == 3


def test_voice_combos_single_engine_distinct():
    combos = voice_combos(4, ["piper"])
    assert all(e == "piper" for e, _, _ in combos)
    assert len(set(combos)) == 4  # distinct (engine,voice,rate) while the pool allows


def test_piper_voice_path_resolves_name_and_quality():
    path = _piper_voice_path("de_DE-thorsten-medium")
    assert path.parts[-3:] == ("thorsten", "medium", "de_DE-thorsten-medium.onnx")


def test_piper_voice_path_handles_underscored_name():
    # "eva_k" itself contains an underscore but no hyphen, so the (name, quality) split
    # on the LAST hyphen must still land on "eva_k" / "x_low", not split mid-name.
    path = _piper_voice_path("de_DE-eva_k-x_low")
    assert path.parts[-3:] == ("eva_k", "x_low", "de_DE-eva_k-x_low.onnx")
