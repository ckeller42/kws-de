import json

from kws_de import config
from kws_de.tts import ENGINE_VOICES, _piper_voice_path, engine_voices, piper_voices, voice_combos


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


def _fake_voice(root, name, quality, num_speakers=1):
    d = root / name / quality
    d.mkdir(parents=True)
    vid = f"de_DE-{name}-{quality}"
    (d / f"{vid}.onnx").write_bytes(b"")
    (d / f"{vid}.onnx.json").write_text(json.dumps({"num_speakers": num_speakers}))
    return vid


def test_piper_voices_scans_cache_and_expands_multi_speaker(tmp_path):
    _fake_voice(tmp_path, "thorsten", "medium")
    _fake_voice(tmp_path, "mls", "medium", num_speakers=3)
    assert piper_voices(tmp_path) == [
        "de_DE-mls-medium#0",
        "de_DE-mls-medium#1",
        "de_DE-mls-medium#2",
        "de_DE-thorsten-medium",
    ]
    assert piper_voices(tmp_path / "missing") == []


def test_engine_voices_uses_cache_then_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert engine_voices("piper") == ENGINE_VOICES["piper"]  # empty cache -> static list
    _fake_voice(tmp_path / "piper-voices", "kerstin", "low")
    assert engine_voices("piper") == ["de_DE-kerstin-low"]
    assert engine_voices("say") == ENGINE_VOICES["say"]


def test_piper_voice_path_strips_speaker_suffix():
    p = _piper_voice_path("de_DE-mls-medium#17")
    assert p.name == "de_DE-mls-medium.onnx" and p.parent.name == "medium"


def test_voice_combos_spread_over_voices_before_rates():
    combos = voice_combos(len(ENGINE_VOICES["say"]), ["say"])
    assert len({v for _, v, _ in combos}) == len(ENGINE_VOICES["say"])  # every voice once
    assert len({r for _, _, r in combos}) == 1  # ...at a single rate
