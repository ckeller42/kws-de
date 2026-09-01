from kws_de.tts import voice_combos


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
