import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kws_de import config, qc


def _wav(path: Path, sig: np.ndarray, sr: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sig.astype(np.float32), sr, subtype="PCM_16")
    return path


def _tone(ms=800, amp=0.3, sr=16000):
    t = np.arange(int(sr * ms / 1000)) / sr
    return amp * np.sin(2 * np.pi * 440 * t)


def test_normalise_umlauts_sharp_s_prozent_and_punctuation():
    assert qc.normalise("Licht Küche fünfzig Prozent.") == ["licht", "küche", "fünfzig"]
    assert qc.normalise("Außen") == qc.normalise("Aussen") == ["aussen"]


def test_required_tokens_per_set():
    assert qc.required_tokens("Kühlschrank", "words") == ["kühlschrank"]
    assert qc.required_tokens("Licht Küche fünfzig Prozent", "sentences") == [
        "licht",
        "küche",
        "fünfzig",
    ]
    assert qc.required_tokens("wie spät ist es", "negatives") == []


def test_content_gate_rules():
    assert qc.content_gate("words", "Licht", "licht") == (1.0, None)
    # edit distance 1, >5 letters
    assert qc.content_gate("words", "Kühlschrank", "kühlschrenk")[1] is None
    # short word (<=5 letters): exact only
    assert qc.content_gate("words", "Licht", "nicht")[1].startswith("wrong_word")
    assert qc.content_gate("sentences", "Licht Küche an", "licht küche an")[0] == 1.0
    score, reason = qc.content_gate("sentences", "Licht Küche an", "küche licht an")
    assert reason == "missing:licht küche an (order)" or reason.startswith("missing")
    # filler ok
    assert qc.content_gate("sentences", "Licht Küche an", "licht bitte küche an") == (1.0, None)
    assert qc.content_gate("negatives", "wie spät ist es", "wie spät ist es") == (1.0, None)
    reason = qc.content_gate("negatives", "wie spät ist es", "mach das licht an")[1]
    assert reason == "contains_command:licht"


def test_content_gate_numerals_heard_as_digits():
    # Whisper large-v3 writes the light levels as numerals, not the German number words.
    assert qc.content_gate("words", "fünfzig", "50") == (1.0, None)
    assert qc.content_gate("sentences", "Licht Küche fünfzig Prozent", "Licht Küche 50%") == (
        1.0,
        None,
    )


def test_content_gate_glued_keywords_still_order_sensitive_and_short_word_exact():
    # sentence: keywords glued into one Whisper token still match, in order
    assert qc.content_gate("sentences", "Licht Dach heller", "Lichtdach heller") == (1.0, None)
    reason = qc.content_gate("sentences", "Licht Dach heller", "Licht heller Dach")[1]
    assert reason.startswith("missing")
    # words scope has only ONE required token ("licht") - "an" isn't required here, so
    # gluing it on must NOT match (that would be the boundary-check regression: a short
    # keyword false-positiving inside an unrelated glued word).
    assert qc.content_gate("words", "Licht", "lichtan")[1].startswith("wrong_word")
    # but a short (<=5 letter) keyword never fuzzy-matches a different word
    assert qc.content_gate("words", "Licht", "nicht")[1].startswith("wrong_word")


def test_content_gate_short_keyword_boundary_check():
    # a short (<=5 letter) required keyword must never match merely because it occurs as
    # a substring inside an unrelated, longer heard word
    assert qc.content_gate("words", "an", "dank")[1].startswith("wrong_word")
    # ...but gluing IS allowed when the neighbour is itself another required keyword of
    # the same prompt (sentences scope: "Licht an" requires both "licht" and "an")
    assert qc.content_gate("sentences", "Licht an", "Lichtan") == (1.0, None)
    # a short keyword still doesn't match when the glued word isn't a run of required
    # tokens ("nichtdach" isn't "licht"+"dach" - it doesn't even start with "licht")
    reason = qc.content_gate("sentences", "Licht Dach", "nichtdach")[1]
    assert reason.startswith("missing")


def test_content_gate_negatives_two_letter_keyword_needs_whole_token_or_repeat():
    # a lone 2-letter keyword hallucination doesn't reject...
    assert qc.content_gate("negatives", "wann fahren wir los", "An den fahren wir los") == (
        1.0,
        None,
    )
    # ...but a real (>=3 letter) command keyword still does...
    assert qc.content_gate("negatives", "x", "Licht an bitte")[1] == "contains_command:licht"
    # ...and a 2-letter keyword appearing twice still does
    assert qc.content_gate("negatives", "x", "an und an")[1] == "contains_command:an"


def test_required_tokens_and_content_gate_wake():
    assert qc.required_tokens("Hey Bus", "wake") == ["hey", "bus"]
    assert qc.content_gate("wake", "Hey Bus", "Hey Bus") == (1.0, None)
    assert qc.content_gate("wake", "Hey Bus", "Hej Boss") == (1.0, None)
    assert qc.content_gate("wake", "Hey Bus", "Hallo")[1].startswith("wrong_word")


def test_audio_gate_ok_clipped_quiet_short(tmp_path):
    ok = _wav(tmp_path / "ok.wav", _tone())
    m, reason = qc.audio_gate(ok, "words")
    assert reason is None
    assert m["sr"] == 16000 and 700 <= m["dur_ms"] <= 900 and m["peak_dbfs"] < -0.5
    clip = _wav(tmp_path / "clip.wav", np.clip(_tone(amp=3.0), -1, 1))
    assert qc.audio_gate(clip, "words")[1] == "clipped"
    assert qc.audio_gate(_wav(tmp_path / "quiet.wav", _tone(amp=0.001)), "words")[1] == "too_quiet"
    assert qc.audio_gate(_wav(tmp_path / "short.wav", _tone(ms=100)), "words")[1] == "too_short"
    assert qc.audio_gate(_wav(tmp_path / "long.wav", _tone(ms=4500)), "words")[1] == "too_long"
    assert qc.audio_gate(_wav(tmp_path / "long6.wav", _tone(ms=4500)), "sentences")[1] is None


def test_audio_gate_click_is_not_clipping(tmp_path):
    # A 2-sample full-scale click (a seam glitch, a pop) must not throw away a
    # 3.5 s take; real clipping parks hundreds of samples at the rail.
    sig = _tone(ms=3500)
    sig[16489:16491] = 1.0
    assert qc.audio_gate(_wav(tmp_path / "click.wav", sig), "field")[1] is None
    sig[16000:16400] = 1.0  # 400 rail samples = 1.1 % -> clipped
    assert qc.audio_gate(_wav(tmp_path / "rail.wav", sig), "field")[1] == "clipped"


def test_audio_gate_unreadable_file_is_rejection_not_crash(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav\x00")  # 10 bytes garbage, not a real RIFF/WAV
    m, reason = qc.audio_gate(bad, "words")
    assert m == {}
    assert reason.startswith("unreadable")


def test_judge_and_sessions_roundtrip(tmp_path):
    inc = tmp_path / "incoming" / "2026-09-02-1500"
    _wav(inc / "spk02" / "licht" / "001.wav", _tone())
    _wav(inc / "spk02" / "_neg_" / "wie-spaet-ist-es_001.wav", _tone())
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk02,2026-09-02T15:00:00,Licht,spk02/licht/001.wav,800,-10.0,words,1,123\n"
        'spk02,2026-09-02T15:00:00,"wie spät ist es",'
        "spk02/_neg_/wie-spaet-ist-es_001.wav,800,-10.0,negatives,1,124\n"
    )
    takes = qc.read_sessions(inc)
    assert [(t.set, t.speaker, t.prompt) for t in takes] == [
        ("words", "spk02", "Licht"),
        ("negatives", "spk02", "wie spät ist es"),
    ]
    heard = {"001.wav": "Licht", "wie-spaet-ist-es_001.wav": "mach das licht an"}

    def transcriber(p: Path):
        return {"text": heard[p.name], "words": []}

    rows = [qc.judge(t, transcriber)[0] for t in takes]
    assert rows[0].verdict == "approve" and rows[0].match_score == 1.0
    assert rows[1].verdict == "reject" and rows[1].reason == "contains_command:licht"
    out = tmp_path / "qc.csv"
    qc.write_qc_csv(rows, out)
    got = list(csv.DictReader(out.open()))
    assert [r["verdict"] for r in got] == ["approve", "reject"]
    assert got[0]["file"].endswith("licht/001.wav")


def test_label_for_token_maps_normalised_token_back_to_config_label():
    # Task 6 (prompt_intent) needs this to recover the original config label
    # (case, umlauts) from a normalised transcript token.
    assert qc.label_for_token("licht") == "Licht"
    assert qc.label_for_token("kuehlschrank") is None  # normalised form only, not raw
    assert qc.label_for_token("küche") == "Küche"
    assert qc.label_for_token("aussen") == "Außen"  # ß -> ss normalisation
    assert qc.label_for_token("an") == "an"
    assert qc.label_for_token("hundert") == "hundert"
    assert qc.label_for_token("nonexistent") is None


def test_segment_word_centres_and_pads():
    sr = 16000
    sig = np.zeros(sr * 2, dtype=np.float32)
    sig[sr : sr + 1600] = 0.5  # 100 ms burst at 1.0 s
    seg = qc.segment_word(sig, sr, 1.0, 1.1)
    assert seg.shape == (config.CLIP_SAMPLES,)
    assert np.argmax(np.abs(seg)) in range(
        config.CLIP_SAMPLES // 2 - 1000, config.CLIP_SAMPLES // 2 + 1000
    )
    edge = qc.segment_word(sig, sr, 0.0, 0.1)  # window would start before 0
    assert edge.shape == (config.CLIP_SAMPLES,) and np.all(edge[:4000] == 0)


def test_label_for_token():
    assert qc.label_for_token("küche") == "Küche" and qc.label_for_token("licht") == "Licht"
    assert qc.label_for_token("fünfzig") == "fünfzig"


def test_field_wake_split_cuts_the_wake_phrase_and_returns_the_command():
    tr = {
        "text": "Hey Bus Licht Küche an",
        "words": [
            {"word": "Hey", "start": 0.10, "end": 0.35},
            {"word": "Bus", "start": 0.36, "end": 0.60},
            {"word": "Licht", "start": 1.40, "end": 1.70},
            {"word": "Küche", "start": 1.75, "end": 2.05},
            {"word": "an", "start": 2.10, "end": 2.30},
        ],
    }
    cut_s, tokens = qc.field_wake_split(tr)
    assert cut_s == pytest.approx(0.75)  # end of "Bus" + WAKE_TAIL_S
    assert tokens == ["licht", "küche", "an"]


def test_field_wake_split_ignores_a_late_or_absent_wake_phrase():
    # the phrase matches but lands after WAKE_MAX_S -> not this take's wake word
    late = {
        "text": "Hey Bus an",
        "words": [
            {"word": "Hey", "start": 2.60, "end": 2.85},
            {"word": "Bus", "start": 2.86, "end": 3.20},
            {"word": "an", "start": 3.30, "end": 3.50},
        ],
    }
    assert qc.field_wake_split(late) == (None, ["hey", "bus", "an"])
    # no wake phrase at all: nothing is cut, everything is command material
    plain = {"text": "Licht an", "words": [{"word": "Licht", "start": 0.1, "end": 0.4}]}
    assert qc.field_wake_split(plain) == (None, ["licht", "an"])


def test_field_wake_split_accepts_a_single_glued_wake_span():
    # Whisper sometimes emits the whole phrase as ONE span; gluing only words[0:2]
    # would then produce "heybuslicht", which never matches -> wake positive lost.
    tr = {
        "text": "HeyBus Licht Küche an",
        "words": [
            {"word": "HeyBus", "start": 0.10, "end": 0.60},
            {"word": "Licht", "start": 1.40, "end": 1.70},
            {"word": "Küche", "start": 1.75, "end": 2.05},
            {"word": "an", "start": 2.10, "end": 2.30},
        ],
    }
    assert qc.field_wake_split(tr) == (pytest.approx(0.75), ["licht", "küche", "an"])


def test_field_intent_splits_a_welded_compound_but_not_ordinary_speech():
    from kws_de.grammar import Intent, Rejection

    # Whisper welds German compounds; "Lichtküche" is a real transcript from this
    # task's field smoke, and losing it files a correct command as _unknown_.
    assert qc.field_intent(["lichtküche", "an"]) == Intent("Licht", "Küche", "an")
    assert qc.field_intent(["lichtan"]) == Intent("Licht", None, "an")
    assert isinstance(qc.field_intent(["ich", "habe", "angst"]), Rejection)
    # only a token that decomposes COMPLETELY into vocabulary words is split, so
    # ordinary German words that merely START with a keyword are left whole
    v = qc.vocab()
    for word in ("dank", "anzug", "angst", "banane", "ankommen"):
        assert qc._split_glued(word, v) == [word]
    # ...but real verbs that DO decompose completely ("auslesen" -> "aus lesen")
    # are split, and the safety then comes from the grammar, not the splitter.
    # Pin that interaction: a wrong split must never reach a training label.
    for word in ("auslesen", "anlesen"):
        assert len(qc._split_glued(word, v)) == 2
        assert isinstance(qc.field_intent([word]), Rejection)


def test_field_intent_uses_the_same_grammar_as_the_device():
    from kws_de.grammar import Intent, Rejection

    assert qc.field_intent(["licht", "küche", "an"]) == Intent("Licht", "Küche", "an")
    # filler is dropped, exactly as the sentence prompts' token filter does
    assert qc.field_intent(["mach", "licht", "bitte", "an"]) == Intent("Licht", None, "an")
    # ordinary speech has no command tokens at all -> a Rejection, i.e. kept as
    # negative / _unknown_ material, never dropped
    assert isinstance(qc.field_intent(["wann", "fahren", "wir", "los"]), Rejection)


def _phrase_transcriber(p: Path):
    if "_phrase_" in str(p):
        return {
            "text": "Licht Küche an",
            "words": [
                {"word": "Licht", "start": 0.2, "end": 0.5},
                {"word": "Küche", "start": 0.6, "end": 0.9},
                {"word": "an", "start": 1.0, "end": 1.2},
            ],
        }
    return {"text": "Licht" if "licht" in str(p) else "hallo welt", "words": []}


def test_run_qc_word_naming_avoids_bare_vs_phrase_collision_and_is_idempotent(tmp_path):
    # bare "Licht" word take AND a phrase containing "Licht" share take number 001 ->
    # both must land on distinct approved paths, not alias to the same file.
    inc = tmp_path / "incoming" / "s1"
    _wav(inc / "spk02" / "licht" / "001.wav", _tone())
    _wav(inc / "spk02" / "_phrase_" / "licht-kueche-an_001.wav", _tone(ms=1500))
    _wav(inc / "spk02" / "_neg_" / "hallo-welt_001.wav", _tone())
    _wav(inc / "spk02" / "_neg_" / "hallo-welt_002.wav", np.clip(_tone(amp=3.0), -1, 1))  # clipped
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk02,t,Licht,spk02/licht/001.wav,800,-10,words,1,1\n"
        "spk02,t,Licht Küche an,spk02/_phrase_/licht-kueche-an_001.wav,1500,-10,sentences,1,2\n"
        "spk02,t,hallo welt,spk02/_neg_/hallo-welt_001.wav,800,-10,negatives,1,3\n"
        "spk02,t,hallo welt,spk02/_neg_/hallo-welt_002.wav,800,-1,negatives,1,4\n"
    )

    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _phrase_transcriber)
    assert counts == {
        "takes": 4,
        "approved": 3,
        "rejected": 1,
        "words_written": 4,
        "words_skipped": 0,
        "wake_written": 0,
        "field_takes": 0,
        "field_approved": 0,
        "field_truncated": 0,
        "field_parsable": 0,
        "field_agree": 0,
    }
    licht_files = sorted((appr / "words" / "Licht").glob("*.wav"))
    assert len(licht_files) == 2  # bare take + phrase-segmented word, distinct files
    assert (appr / "words" / "Küche" / "spk02_001.wav").exists()
    assert (appr / "words" / "an" / "spk02_001.wav").exists()
    assert (appr / "phrases" / "spk02" / "licht-kueche-an_001.wav").exists()
    assert (appr / "negatives" / "spk02" / "hallo-welt_001.wav").exists()
    assert not (appr / "negatives" / "spk02" / "hallo-welt_002.wav").exists()
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an" and idx[0]["speaker"] == "spk02"
    assert (qcd / "report.md").read_text().count("reject") >= 1
    words = list(csv.DictReader((qcd / "words.csv").open()))
    assert {w["word"] for w in words} == {"Licht", "Küche", "an"}
    assert counts["words_written"] == len(list((appr / "words").rglob("*.wav")))

    # re-run the SAME stamp: no duplication, no growth in file count
    counts2 = qc.run_qc(inc, qcd, appr, _phrase_transcriber)
    assert counts2 == counts
    assert len(list((appr / "words" / "Licht").glob("*.wav"))) == 2
    assert len(list((appr / "phrases" / "spk02").glob("*.wav"))) == 1
    assert len(list(csv.DictReader((appr / "phrases" / "index.csv").open()))) == 1


def test_run_qc_two_stamps_same_speaker_dont_alias_or_duplicate(tmp_path):
    # each stamp also records the SAME phrase (same prompt/slug, same take number) ->
    # phrases/negatives must be as collision-proof across stamps as words are.
    def sessions(inc: Path, take_no: str):
        _wav(inc / "spk02" / "licht" / f"{take_no}.wav", _tone())
        _wav(inc / "spk02" / "_phrase_" / f"licht-kueche-an_{take_no}.wav", _tone(ms=1500))
        (inc / "sessions.csv").write_text(
            "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
            f"spk02,t,Licht,spk02/licht/{take_no}.wav,800,-10,words,1,1\n"
            f"spk02,t,Licht Küche an,spk02/_phrase_/licht-kueche-an_{take_no}.wav,"
            "1500,-10,sentences,1,2\n"
        )

    inc1, inc2 = tmp_path / "incoming" / "s1", tmp_path / "incoming" / "s2"
    sessions(inc1, "001")
    sessions(inc2, "001")  # each session numbers its own takes from 001
    qcd1, qcd2, appr = tmp_path / "qc" / "s1", tmp_path / "qc" / "s2", tmp_path / "approved"

    def transcriber(p: Path):
        # empty word spans -> content gate still approves the phrase (text matches),
        # but no word segmentation happens; keeps this test focused on the phrase
        # copy + index row, not on word-clip segmentation (covered elsewhere).
        if "_phrase_" in str(p):
            return {"text": "Licht Küche an", "words": []}
        return {"text": "Licht", "words": []}

    c1 = qc.run_qc(inc1, qcd1, appr, transcriber)
    c2 = qc.run_qc(inc2, qcd2, appr, transcriber)
    assert c1["words_written"] == c2["words_written"] == 1
    licht = appr / "words" / "Licht"
    assert len(list(licht.glob("*.wav"))) == 2  # both stamps' word clips coexist
    phrases = appr / "phrases" / "spk02"
    assert len(list(phrases.glob("*.wav"))) == 2  # both stamps' phrase clips coexist
    idx_path = appr / "phrases" / "index.csv"
    assert len(list(csv.DictReader(idx_path.open()))) == 2

    # re-running stamp s1 alone must not touch s2's approved output
    before_s2 = {f.name: f.read_bytes() for f in licht.glob("spk02_*.wav")}
    before_phrase_s2 = {f.name: f.read_bytes() for f in phrases.glob("*.wav")}
    idx_before = {r["file"] for r in csv.DictReader(idx_path.open())}

    c1_again = qc.run_qc(inc1, qcd1, appr, transcriber)
    assert c1_again["words_written"] == 1
    assert len(list(licht.glob("*.wav"))) == 2  # s1's old clip replaced, not added
    survivors = {f.name: f.read_bytes() for f in licht.glob("spk02_*.wav")}
    # at least one file from before the re-run is untouched (s2's)
    assert set(before_s2.items()) & set(survivors.items())

    assert len(list(phrases.glob("*.wav"))) == 2  # s1's old phrase replaced, not added
    survivors_phrase = {f.name: f.read_bytes() for f in phrases.glob("*.wav")}
    assert set(before_phrase_s2.items()) & set(survivors_phrase.items())  # s2's phrase intact
    idx_after = {r["file"] for r in csv.DictReader(idx_path.open())}
    assert len(idx_after) == 2
    assert idx_before & idx_after  # s2's index row survived unchanged


def test_run_qc_unmapped_word_token_is_skipped_not_mislabelled(tmp_path):
    inc = tmp_path / "incoming" / "s1"
    _wav(inc / "spk02" / "blau" / "001.wav", _tone())
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk02,t,Blau,spk02/blau/001.wav,800,-10,words,1,1\n"
    )

    def transcriber(p: Path):
        return {"text": "Blau", "words": []}

    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["approved"] == 1  # content gate approved it
    assert counts["words_written"] == 0
    assert counts["words_skipped"] == 1
    assert not (appr / "words").exists() or not list((appr / "words").rglob("*.wav"))


def test_run_qc_segmentation_gap_reported_when_word_spans_miss_a_token(tmp_path):
    inc = tmp_path / "incoming" / "s1"
    _wav(inc / "spk02" / "_phrase_" / "licht-kueche-an_001.wav", _tone(ms=1500))
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk02,t,Licht Küche an,spk02/_phrase_/licht-kueche-an_001.wav,1500,-10,sentences,1,1\n"
    )

    def transcriber(p: Path):
        # text-level transcript matches (approves content gate), but the word-level
        # spans Whisper returned are missing the last token -> segmentation gap.
        return {
            "text": "Licht Küche an",
            "words": [
                {"word": "Licht", "start": 0.2, "end": 0.5},
                {"word": "Küche", "start": 0.6, "end": 0.9},
            ],
        }

    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["approved"] == 1
    assert counts["words_written"] == 2
    assert counts["words_skipped"] == 1
    assert (appr / "words" / "Licht" / "spk02_001.wav").exists()
    assert (appr / "words" / "Küche" / "spk02_001.wav").exists()
    assert not (appr / "words" / "an").exists()
    report = (qcd / "report.md").read_text()
    assert "## Segmentation gaps" in report
    assert "licht-kueche-an_001.wav" in report.split("## Segmentation gaps")[1]


def test_run_qc_isolates_a_transcriber_error_to_one_row(tmp_path):
    inc = tmp_path / "incoming" / "s1"
    _wav(inc / "spk01" / "licht" / "001.wav", _tone())
    _wav(inc / "spk01" / "_neg_" / "boom_001.wav", _tone())
    _wav(inc / "spk01" / "kueche" / "001.wav", _tone())
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk01,t,Licht,spk01/licht/001.wav,800,-10,words,1,1\n"
        "spk01,t,boom,spk01/_neg_/boom_001.wav,800,-10,negatives,1,2\n"
        "spk01,t,Küche,spk01/kueche/001.wav,800,-10,words,1,3\n"
    )
    calls = []

    def transcriber(p: Path):
        calls.append(p)
        if len(calls) == 2:  # second take: simulate a transcriber blow-up
            raise RuntimeError("boom")
        return {"text": "Licht" if "licht" in str(p) else "Küche", "words": []}

    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["takes"] == 3  # all three takes judged, none dropped
    rows = list(csv.DictReader((qcd / "qc.csv").open()))
    assert rows[0]["verdict"] == "approve"  # first take: unaffected
    assert rows[1]["verdict"] == "reject" and rows[1]["reason"] == "error: RuntimeError"
    assert rows[2]["verdict"] == "approve"  # third take: batch continued past the error


def test_run_qc_writes_approved_wake_set_and_is_idempotent(tmp_path):
    inc = tmp_path / "incoming" / "s1"
    _wav(inc / "spk07" / "hey-bus" / "001.wav", _tone())
    _wav(inc / "spk07" / "hey-bus" / "002.wav", _tone())
    _wav(inc / "spk07" / "hey-bus" / "003.wav", _tone())
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk07,t,Hey Bus,spk07/hey-bus/001.wav,800,-10,wake,1,1\n"
        "spk07,t,Hey Bus,spk07/hey-bus/002.wav,800,-10,wake,1,2\n"
        "spk07,t,Hey Bus,spk07/hey-bus/003.wav,800,-10,wake,1,3\n"
    )
    heard = {"001.wav": "Hey Bus", "002.wav": "Hey Bus", "003.wav": "Hallo"}

    def transcriber(p: Path):
        return {"text": heard[p.name], "words": []}

    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["wake_written"] == 2
    assert counts["approved"] == 2 and counts["rejected"] == 1
    files = sorted((appr / "wake" / "spk07").glob("*.wav"))
    assert len(files) == 2
    idx = list(csv.DictReader((appr / "wake" / "index.csv").open()))
    assert len(idx) == 2
    assert all(r["prompt"] == "Hey Bus" and r["speaker"] == "spk07" for r in idx)
    assert "wake" in (qcd / "report.md").read_text()

    # re-run the same stamp: no duplication
    counts2 = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts2 == counts
    assert len(list((appr / "wake" / "spk07").glob("*.wav"))) == 2
    assert len(list(csv.DictReader((appr / "wake" / "index.csv").open()))) == 2


def test_audio_gate_field_take_may_be_longer_than_one_window(tmp_path):
    # a window EXTENDS on every fire inside it, so a take is not fixed at 3500 ms;
    # the firmware's own ceiling is FIELD_MAX_TAKE_SAMPLES (9.8 s). Rejecting a
    # 5 s two-fire take as too_long would throw away data spec §1 says is kept.
    assert qc.audio_gate(_wav(tmp_path / "f5.wav", _tone(ms=5000)), "field")[1] is None
    assert qc.audio_gate(_wav(tmp_path / "f10.wav", _tone(ms=9900)), "field")[1] == "too_long"


def _field_session(
    tmp_path,
    device_intent: str,
    device_words: str = "Licht:0.93|an:0.88",
    window_ms: int = 2500,
    ms: int = 4000,
) -> Path:
    # 4000 ms of audio = FIELD_PREROLL_MS (1500) + a 2500 ms window, i.e. the
    # whole window fitted the ring: not truncated. A larger window_ms with the
    # same wav is what a ring-truncated take looks like on the host.
    inc = tmp_path / "incoming" / "f1"
    _wav(inc / "field" / "spk05" / "1-123456.wav", _tone(ms=ms))
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,"
        "fire_ms,wake_prob,device_intent,device_words,window_ms\n"
        f"spk05,t,,field/spk05/1-123456.wav,{ms},-10,field,,123456,123456,0.910,"
        f"{device_intent},{device_words},{window_ms}\n"
    )
    return inc


def _field_transcriber(p: Path):
    return {
        "text": "Hey Bus Licht Küche an",
        "words": [
            {"word": "Hey", "start": 0.10, "end": 0.35},
            {"word": "Bus", "start": 0.36, "end": 0.60},
            {"word": "Licht", "start": 1.40, "end": 1.70},
            {"word": "Küche", "start": 1.75, "end": 2.05},
            {"word": "an", "start": 2.10, "end": 2.30},
        ],
    }


def test_run_qc_field_take_splits_wake_labels_by_grammar_and_scores_agreement(tmp_path):
    inc = _field_session(tmp_path, "Licht Küche an")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)

    assert counts["field_takes"] == 1
    assert counts["field_approved"] == 1
    assert counts["field_truncated"] == 0
    assert counts["field_parsable"] == 1
    assert counts["field_agree"] == 1
    assert counts["wake_written"] == 1

    # the wake phrase became a wake clip, cut at the end of "Bus" + 0.15 s
    wake_files = sorted((appr / "wake" / "spk05").glob("*.wav"))
    assert len(wake_files) == 1
    sig, sr = sf.read(wake_files[0], always_2d=True)
    assert 0.70 <= len(sig) / sr <= 0.80

    # the command became an approved phrase with the grammar-derived prompt,
    # segmented into word clips exactly like a guided sentence take
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an" and idx[0]["speaker"] == "spk05"
    phrase = appr / "phrases" / "spk05" / "1-123456_001.wav"
    assert phrase.exists()
    assert {p.parent.name for p in (appr / "words").rglob("*.wav")} == {"Licht", "Küche", "an"}

    # the phrase clip is the COMMAND, not the whole take: it starts after the
    # wake phrase (0.60 + 0.15 s) and ends 0.3 s past the last word (2.30 s), so
    # ~1.85 s, not the full 3.5 s. eval streams this clip end to end, and the
    # pre-roll plus trailing silence would be streamed with it.
    psig, psr = sf.read(phrase, always_2d=True)
    assert 1.80 <= len(psig) / psr <= 1.90

    # provenance: the device's own intent is recorded and scored, never used
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["set"] == "field" and row["verdict"] == "approve"
    assert row["device_intent"] == "Licht Küche an"
    assert row["agrees"] == "1"
    # the derived label reaches qc.csv, so "parsable" is readable off the row
    # itself (non-empty prompt) rather than inferred from `agrees`
    assert row["prompt"] == "Licht Küche an"
    assert "## Field" in (qcd / "report.md").read_text()

    # the field wake clip takes a different write path (sf.write of a slice) than
    # the guided one (read_bytes), so pin its idempotence too
    counts2 = qc.run_qc(inc, qcd, appr, _field_transcriber)
    assert counts2 == counts
    assert len(list((appr / "wake" / "spk05").glob("*.wav"))) == 1
    assert len(list((appr / "phrases" / "spk05").glob("*.wav"))) == 1
    assert len(list(csv.DictReader((appr / "wake" / "index.csv").open()))) == 1


def test_run_qc_field_agreement_rate_counts_only_takes_the_device_answered(tmp_path):
    # one compared take that agrees + one truncated take with no device answer:
    # the rate is 1.000 over the COMPARED takes, not 0.500 over the parsable ones.
    inc = tmp_path / "incoming" / "f1"
    _wav(inc / "field" / "spk05" / "1-123456.wav", _tone(ms=4000))
    _wav(inc / "field" / "spk05" / "1-123457.wav", _tone(ms=4000))
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,"
        "fire_ms,wake_prob,device_intent,device_words,window_ms\n"
        "spk05,t,,field/spk05/1-123456.wav,4000,-10,field,,123456,123456,0.910,"
        "Licht Küche an,Licht:0.93|an:0.88,2500\n"
        "spk05,t,,field/spk05/1-123457.wav,4000,-10,field,,123457,123457,0.910,,,9000\n"
    )
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)
    assert counts["field_takes"] == 2 and counts["field_parsable"] == 2
    assert counts["field_agree"] == 1
    # the second take held 4000 ms of a 1500 + 9000 ms span: the ring cut it, and
    # THAT is why it carries no device answer. Readable on the host at last.
    assert counts["field_truncated"] == 1
    report = (qcd / "report.md").read_text()
    assert "device-Whisper agreement 1.000" in report
    assert "1 ring-truncated" in report
    rows = {r["file"].rsplit("/", 1)[-1]: r for r in csv.DictReader((qcd / "qc.csv").open())}
    assert rows["1-123456.wav"]["truncated"] == "0"
    assert rows["1-123457.wav"]["truncated"] == "1"


def test_run_qc_field_take_records_disagreement_without_relabelling(tmp_path):
    inc = _field_session(tmp_path, "Licht an")  # device missed the zone
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)
    assert counts["field_parsable"] == 1 and counts["field_agree"] == 0
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["agrees"] == "0"
    # the LABEL still comes from Whisper + grammar, never from the device
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an"


def test_run_qc_truncated_field_take_has_no_device_answer_to_compare(tmp_path):
    # a take the ring cut short carries empty device_intent/device_words: the
    # device gave no answer, so there is nothing to agree or disagree with —
    # but Whisper still labels it and it is still filed.
    inc = _field_session(tmp_path, "", "", window_ms=9000)
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)
    assert counts["field_takes"] == 1 and counts["field_parsable"] == 1
    assert counts["field_agree"] == 0
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["verdict"] == "approve"
    assert row["device_intent"] == "" and row["agrees"] == ""
    # and the reason is now readable on the host: 4000 ms of audio for a
    # 1500 + 9000 ms span means the ring cut it, not that nothing was heard.
    assert row["truncated"] == "1" and counts["field_truncated"] == 1
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an"


def test_run_qc_rounding_short_field_take_is_not_truncated(tmp_path):
    # 4000 ms of audio against a 1500 + 2530 ms span: the 30 ms is tick/sample
    # rounding on the device (seen on every real take), not a ring cut.
    inc = _field_session(tmp_path, "Licht Küche an", window_ms=2530)
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["truncated"] == "0" and counts["field_truncated"] == 0


def test_run_qc_older_session_with_shorter_preroll_is_not_truncated(tmp_path):
    # A session recorded by a 1.0 s pre-roll build: 3500 ms = 1000 + 2500. The
    # pre-roll is inferred from the session, not assumed to be today's.
    inc = _field_session(tmp_path, "Licht Küche an", window_ms=2500, ms=3500)
    takes = qc.read_sessions(inc)
    assert takes[0].preroll_ms == 1000
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)
    assert counts["field_truncated"] == 0


def test_run_qc_field_take_that_does_not_parse_is_kept_as_a_negative(tmp_path):
    inc = _field_session(tmp_path, "")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"

    def transcriber(p: Path):
        return {
            "text": "wann fahren wir los",
            "words": [{"word": "wann", "start": 0.2, "end": 0.5}],
        }

    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["field_takes"] == 1 and counts["field_parsable"] == 0
    assert counts["wake_written"] == 0  # no wake phrase in the transcript
    idx = list(csv.DictReader((appr / "negatives" / "index.csv").open()))
    assert idx[0]["prompt"] == "wann fahren wir los"  # the transcript is the prompt
    assert (appr / "negatives" / "spk05" / "1-123456_001.wav").exists()
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["verdict"] == "approve" and row["agrees"] == ""
    assert row["prompt"] == ""  # nothing parsed -> no derived label


def test_run_qc_unparsable_field_take_with_command_words_is_not_filed_as_a_negative(tmp_path):
    # "an Licht Küche" is a real command spoken out of order: the grammar rejects
    # it, but filing it under negatives/ would teach the model that a genuine
    # command is _unknown_ AND score a correct recognition as a false accept.
    inc = _field_session(tmp_path, "")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"

    def transcriber(p: Path):
        return {
            "text": "an Licht Küche",
            "words": [{"word": "an", "start": 0.2, "end": 0.5}],
        }

    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["field_takes"] == 1 and counts["field_parsable"] == 0
    assert not (appr / "negatives").exists()
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["prompt"] == "" and row["agrees"] == ""
    assert "1 unparsed (vocab present)" in (qcd / "report.md").read_text()


def test_run_qc_field_take_with_an_empty_transcript_is_rejected(tmp_path):
    inc = _field_session(tmp_path, "")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, lambda p: {"text": "", "words": []})
    assert counts["rejected"] == 1 and counts["field_parsable"] == 0
    # "field takes" is every field row, rejected ones included; "approved" is the
    # separate number. Both reports count it this way, so they cannot disagree.
    assert counts["field_takes"] == 1 and counts["field_approved"] == 0
    assert "1 field takes, 0 approved" in (qcd / "report.md").read_text()
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["reason"] == "empty_transcript"


def test_cli_missing_sessions_csv_exits_2(tmp_path, monkeypatch):
    inc = tmp_path / "incoming" / "nope"
    inc.mkdir(parents=True)
    monkeypatch.setattr("sys.argv", ["kws-qc", str(inc)])
    with pytest.raises(SystemExit) as exc:
        qc.main()
    assert exc.value.code == 2


def test_cli_dry_run_lists_takes_without_model(tmp_path, capsys, monkeypatch):
    inc = tmp_path / "incoming" / "s2"
    _wav(inc / "spk03" / "licht" / "001.wav", _tone())
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk03,t,Licht,spk03/licht/001.wav,800,-10,words,1,1\n"
    )
    monkeypatch.setattr("sys.argv", ["kws-qc", str(inc), "--dry-run"])
    qc.main()
    out = capsys.readouterr().out
    assert "1 takes" in out and "licht/001.wav" in out


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None, reason="mlx-whisper not installed"
)
def test_whisper_transcriber_pads_audio_and_shifts_word_offsets_back(tmp_path, monkeypatch):
    import mlx_whisper

    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured["len"] = len(audio)
        captured["initial_prompt"] = kwargs.get("initial_prompt")
        return {"text": "x", "segments": [{"words": [{"word": "x", "start": 0.6, "end": 0.7}]}]}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)
    tr = qc.whisper_transcriber("dummy-model")
    wav = _wav(tmp_path / "t.wav", _tone(ms=800))
    out = tr(wav)
    assert captured["len"] == 800 * 16 + 2 * 8000  # 800ms audio + 500ms pad each side @16kHz
    assert out["words"][0]["start"] == pytest.approx(0.1)
    # narrow prompt: only the words Whisper actually mangles, not the whole vocabulary
    assert "Hey Bus" in captured["initial_prompt"]
    assert "fünfzig" in captured["initial_prompt"]
    assert "Licht" not in captured["initial_prompt"]


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None, reason="mlx-whisper not installed"
)
def test_whisper_transcriber_clamps_word_offset_at_zero(tmp_path, monkeypatch):
    import mlx_whisper

    def fake_transcribe(audio, **kwargs):
        # word starts at 0.3s in the padded audio - before the 0.5s pad boundary
        return {"text": "x", "segments": [{"words": [{"word": "x", "start": 0.3, "end": 0.4}]}]}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)
    tr = qc.whisper_transcriber("dummy-model")
    wav = _wav(tmp_path / "t.wav", _tone(ms=800))
    out = tr(wav)
    assert out["words"][0]["start"] == 0.0
    assert out["words"][0]["end"] == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None, reason="mlx-whisper not installed"
)
def test_whisper_transcriber_smoke(tmp_path):
    tr = qc.whisper_transcriber("mlx-community/whisper-tiny-mlx")  # tiny: quick smoke only
    out = tr(_wav(tmp_path / "t.wav", _tone(ms=1200)))
    assert set(out) >= {"text", "words"}
