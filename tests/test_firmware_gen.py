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
    words, sentences, negs = firmware_gen.prompt_sets()
    assert [w for w, _ in words] == [
        label for label in config.COMMAND_LABELS if not label.startswith("_")
    ]
    assert len(sentences) == len(firmware_gen.build_catalog())
    assert len(negs) == len(config.NEGATIVE_PROMPTS)
    assert len({s for _, s in words + sentences + negs}) == len(words + sentences + negs)


def test_c_tables_reproduce_librosa_mfcc():
    win, mel, dct = firmware_gen.mfcc_tables()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32) * 0.1
    ref = features.mfcc(x)
    got = firmware_gen.mfcc_reference(x, win, mel, dct)
    assert np.allclose(got, ref, atol=1e-3)


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
    assert "#define KWS_N_BINS 241" in fc and "KWS_MEL[40][241]" in fc
    tv = (tmp_path / "test_vectors.h").read_text()
    assert "TV_PCM[16000]" in tv and "TV_MFCC[49][10]" in tv


def test_check_passes_on_fresh_and_catches_changes(tmp_path):
    firmware_gen.generate(tmp_path)
    assert firmware_gen.check(tmp_path) == []

    # a real value change (>> the hardware-noise tolerance) is caught
    fc = tmp_path / "features_config.h"
    text = fc.read_text()
    m = next(
        x for x in re.finditer(r"-?\d+\.\d+e[+-]\d+f", text) if abs(float(x.group()[:-1])) > 0.1
    )
    fc.write_text(text[: m.start()] + f"{float(m.group()[:-1]) + 0.01:.5e}f" + text[m.end() :])
    assert "features_config.h" in firmware_gen.check(tmp_path)

    # a structural change (renamed macro) is caught even with identical floats
    firmware_gen.generate(tmp_path)
    labels = tmp_path / "labels.h"
    labels.write_text(labels.read_text().replace("KWS_NUM_LABELS", "KWS_LABEL_COUNT"))
    assert "labels.h" in firmware_gen.check(tmp_path)
