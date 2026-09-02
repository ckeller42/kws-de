import csv
from pathlib import Path

import numpy as np
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


def test_run_qc_writes_approved_tree_and_is_idempotent(tmp_path):
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

    def transcriber(p: Path):
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

    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts == {"takes": 4, "approved": 3, "rejected": 1, "words_written": 4}
    assert (appr / "words" / "Licht" / "spk02_001.wav").exists()  # bare word take
    assert (appr / "words" / "Küche" / "spk02_001.wav").exists()  # segmented from the phrase
    assert (appr / "words" / "an" / "spk02_001.wav").exists()
    assert (appr / "phrases" / "spk02" / "licht-kueche-an_001.wav").exists()
    assert (appr / "negatives" / "spk02" / "hallo-welt_001.wav").exists()
    assert not (appr / "negatives" / "spk02" / "hallo-welt_002.wav").exists()
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an" and idx[0]["speaker"] == "spk02"
    assert (qcd / "report.md").read_text().count("reject") >= 1
    words = list(csv.DictReader((qcd / "words.csv").open()))
    assert {w["word"] for w in words} == {"Licht", "Küche", "an"}
    # idempotent: a second run produces the same tree and counts
    assert qc.run_qc(inc, qcd, appr, transcriber) == counts
