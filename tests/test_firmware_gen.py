import pathlib
import re

import numpy as np

from kws_de import config, features, firmware_gen


def test_negative_prompts_contain_no_command_words():
    vocab = {w.lower() for w in config.DEVICES + config.ZONES + config.ACTIONS}
    assert len(config.NEGATIVE_PROMPTS) >= 15
    for p in config.NEGATIVE_PROMPTS:
        assert not (set(p.lower().split()) & vocab), p


def test_slug_is_ascii_and_stable():
    assert firmware_gen.slug("Kühlschrank") == "kuehlschrank"
    assert firmware_gen.slug("Licht Außen an") == "licht-aussen-an"
    assert firmware_gen.slug("wie spät ist es") == "wie-spaet-ist-es"
    assert re.fullmatch(r"[a-z0-9-]+", firmware_gen.slug("Straße  weiß"))


def test_prompt_sets_cover_labels_and_catalog():
    words, sentences, negs, wake = firmware_gen.prompt_sets()
    assert [w for w, _ in words] == [
        label for label in config.COMMAND_LABELS if not label.startswith("_")
    ]
    assert len(sentences) == len(firmware_gen.build_catalog())
    assert len(negs) == len(config.NEGATIVE_PROMPTS)
    assert len({s for _, s in words + sentences + negs}) == len(words + sentences + negs)
    wake_prompt = (config.WAKE_WORD, firmware_gen.slug(config.WAKE_WORD))
    assert wake == [wake_prompt] * config.WAKE_PROMPT_REPEATS


def test_sentence_prompts_say_prozent_for_light_levels():
    """Sentence prompts are spoken commands: light levels get the natural
    'Prozent' (a filler the grammar ignores), nothing else does; every valid
    intent in the catalog is present exactly once."""
    from kws_de.eval import build_catalog, intent_text

    sents = [intent_text(it) for it in build_catalog()]
    assert len(sents) == len(set(sents)) == 49
    levels = [s for s in sents if any(lv in s.split() for lv in config.LIGHT_LEVELS)]
    assert levels and all(s.endswith(" Prozent") for s in levels)
    assert not any("Prozent" in s for s in sents if s not in levels)
    assert "Licht Küche fünfzig Prozent" in sents
    assert "Heizung wärmer" in sents and "Aufstelldach auf" in sents
    # zones only ever attach to Licht (camper business logic)
    zoned_non_light = [
        s for s in sents if not s.startswith("Licht") and set(s.split()) & set(config.ZONES)
    ]
    assert not zoned_non_light


def test_c_tables_reproduce_librosa_mfcc():
    win, mel, dct = firmware_gen.mfcc_tables()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32) * 0.1
    ref = features.mfcc(x)
    got = firmware_gen.mfcc_reference(x, win, mel, dct)
    assert np.allclose(got, ref, atol=1e-3)


def test_mel_bands_reproduce_the_dense_filterbank_exactly():
    """The banded form drops only exact zeros, so the mel energies it produces
    must be bit-identical to the dense matmul, not merely close."""
    _, mel, _ = firmware_gen.mfcc_tables()
    start, length, weights = firmware_gen.mel_bands(mel)
    assert weights.size == int(length.sum())
    rebuilt = np.zeros_like(mel)
    off = 0
    for m in range(mel.shape[0]):
        rebuilt[m, start[m] : start[m] + length[m]] = weights[off : off + length[m]]
        off += length[m]
    assert np.array_equal(rebuilt, mel)


def test_generate_is_deterministic_and_complete(tmp_path):
    firmware_gen.generate(tmp_path)
    firmware_gen.generate(tmp_path / "again")
    for name in ("labels.h", "prompts.h", "features_config.h", "test_vectors.h"):
        a = (tmp_path / name).read_text()
        assert a == (tmp_path / "again" / name).read_text()
    labels = (tmp_path / "labels.h").read_text()
    assert f"#define KWS_NUM_LABELS {len(config.COMMAND_LABELS)}" in labels
    assert f"#define KWS_SILENCE_INDEX {config.COMMAND_LABELS.index('_silence_')}" in labels
    assert f"#define KWS_UNKNOWN_INDEX {config.COMMAND_LABELS.index('_unknown_')}" in labels
    fc = (tmp_path / "features_config.h").read_text()
    assert "#define KWS_N_BINS 241" in fc
    assert "KWS_MEL_START[40]" in fc and "KWS_MEL_LEN[40]" in fc
    assert "#define KWS_MEL_NNZ 459" in fc and "KWS_MEL_W[459]" in fc
    tv = (tmp_path / "test_vectors.h").read_text()
    assert "TV_PCM[16000]" in tv and "TV_MFCC[49][10]" in tv


def test_check_passes_on_fresh_and_catches_changes(tmp_path):
    firmware_gen.generate(tmp_path)
    assert firmware_gen.check(tmp_path) == []

    # a real value change (>> the hardware-noise tolerance) is caught
    fc = tmp_path / "features_config.h"
    text = fc.read_text()
    m = next(re.finditer(r"-?\d+\.\d+e[+-]\d+f", text))
    fc.write_text(text[: m.start()] + f"{float(m.group()[:-1]) + 1.0:.5e}f" + text[m.end() :])
    assert "features_config.h" in firmware_gen.check(tmp_path)

    # a structural change (renamed macro) is caught even with identical floats
    firmware_gen.generate(tmp_path)
    labels = tmp_path / "labels.h"
    labels.write_text(labels.read_text().replace("KWS_NUM_LABELS", "KWS_LABEL_COUNT"))
    assert "labels.h" in firmware_gen.check(tmp_path)


def test_wake_int8_matches_microwakeword_requantisation():
    """int8 = (v*256 + 333)/666 - 128, clamped — the firmware's wakefront.c
    must agree exactly, so pin the formula's edges and a mid value here."""
    assert firmware_gen.wake_int8(0) == -128  # silence floors at INT8_MIN
    assert firmware_gen.wake_int8(333) == 0  # half the range lands mid-scale
    assert firmware_gen.wake_int8(666) == 127  # 26.0 in float terms saturates
    assert firmware_gen.wake_int8(65535) == 127  # and stays clamped well beyond it
    assert all(firmware_gen.wake_int8(v) <= firmware_gen.wake_int8(v + 1) for v in range(700))


def test_check_catches_an_embedded_model_that_does_not_match_its_stamp(tmp_path):
    """The two model headers are written together by kws-export --firmware but
    are two files; a half-regeneration leaves the device running a model the
    KWS_MODEL_ID beside it does not describe, and everything downstream --
    including the generated inference, built from model_data.h -- is then
    self-consistently wrong."""
    gen = pathlib.Path(__file__).resolve().parents[1] / "firmware" / "main" / "gen"
    firmware_gen.generate(tmp_path)
    for f in ("model_config.h", "model_data.h"):
        (tmp_path / f).write_text((gen / f).read_text())
    assert firmware_gen.check(tmp_path) == []
    # Flip one byte of the embedded model: same length, different hash.
    data = (tmp_path / "model_data.h").read_text()
    head, sep, rest = data.partition("{")
    first, comma, tail = rest.partition(",")
    (tmp_path / "model_data.h").write_text(f"{head}{sep}{int(first) ^ 1}{comma}{tail}")
    assert any("model_data.h" in name for name in firmware_gen.check(tmp_path))
