import json

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de import eval as ev
from kws_de.grammar import Intent


def test_prompt_intent():
    assert ev.prompt_intent("Licht Küche fünfzig Prozent") == Intent("Licht", "Küche", "fünfzig")


def _build_approved(tmp_path):
    """spk02: one word clip, one phrase clip, one negative clip. spk03: one word
    clip only. A stub predict_fn that always says "Licht" (isolated 100%, e2e
    Rejection(missing action), negatives fire)."""
    root = tmp_path / "approved"
    licht = config.COMMAND_LABELS.index("Licht")
    sil = config.COMMAND_LABELS.index("_silence_")
    for spk in ("spk02", "spk03"):
        d = root / "words" / "Licht"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{spk}_001.wav", np.zeros(16000, np.float32), 16000, subtype="PCM_16")
    (root / "phrases" / "spk02").mkdir(parents=True)
    sf.write(
        root / "phrases" / "spk02" / "licht-an_001.wav",
        np.zeros(32000, np.float32),
        16000,
        subtype="PCM_16",
    )
    # `file` is relative to approved/ — exactly what qc.run_qc writes
    # (`str(dst.relative_to(approved))`, see test_eval_recordings_on_a_run_qc_tree).
    (root / "phrases" / "index.csv").write_text(
        "file,prompt,speaker\nphrases/spk02/licht-an_001.wav,Licht an,spk02\n"
    )
    (root / "negatives" / "spk02").mkdir(parents=True)
    sf.write(
        root / "negatives" / "spk02" / "hallo_001.wav",
        np.zeros(16000, np.float32),
        16000,
        subtype="PCM_16",
    )
    (root / "negatives" / "index.csv").write_text(
        "file,prompt,speaker\nnegatives/spk02/hallo_001.wav,hallo,spk02\n"
    )

    def predict_fn(window):
        p = np.zeros(len(config.COMMAND_LABELS), np.float32)
        p[licht] = 0.9
        p[sil] = 0.1
        return p

    return root, predict_fn


def test_eval_recordings_splits_by_manifest(tmp_path):
    root, predict_fn = _build_approved(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"splits": {"train": {"speakers": ["spk02"]}}}))

    res = ev.eval_recordings(root, predict_fn, manifest_path=manifest_path)
    assert res["manifest_found"] is True
    # manifest_path is outside config.DATA_DIR here (a bare tmp_path, not the data
    # root), so the safe display form falls back to just the basename -- never the
    # absolute tmp_path (see test_eval_recordings_manifest_path_is_data_root_relative
    # for the data-root-relative production case).
    assert res["manifest_path"] == "manifest.json"

    trained = res["figures"]["user-customised, in-training"]
    held_out = res["figures"]["held-out"]

    # spk02's word + negative clips trained the model -> in-training. spk02's
    # phrase clip is never training material (phrases aren't built into the
    # dataset), so it's held-out even though spk02 is a trained speaker.
    assert trained["isolated"]["spk02"]["n"] == 1 and trained["isolated"]["spk02"]["acc"] == 1.0
    assert trained["false_accepts"]["spk02"]["n"] == 1
    assert "spk02" not in trained["e2e"]
    assert set(trained["isolated"]) == {"spk02"}

    # spk03 never trained -> its word clip is held-out, alongside spk02's phrase.
    assert held_out["isolated"]["spk03"]["n"] == 1
    assert held_out["e2e"]["spk02"]["n"] == 1 and held_out["e2e"]["spk02"]["acc"] == 0.0
    assert "spk02" not in held_out["isolated"]
    assert "spk03" not in held_out["e2e"] and "spk03" not in held_out["false_accepts"]

    n_trained, spk_trained = ev._figure_totals(trained)
    n_held, spk_held = ev._figure_totals(held_out)
    assert n_trained == 2 and spk_trained == {"spk02"}
    assert n_held == 2 and spk_held == {"spk02", "spk03"}

    md = ev.render_recordings_section(res)
    assert "Training manifest checked: `manifest.json`" in md
    assert "## user-customised, in-training" in md and "## held-out" in md
    assert "2 clips across 1 speakers" in md  # in-training figure
    assert "2 clips across 2 speakers" in md  # held-out figure
    assert "speaker-level, not per-clip" in md  # honest-match disclosure


def test_eval_recordings_manifest_path_is_data_root_relative(tmp_path, monkeypatch):
    """Blocker fix: a manifest under config.DATA_DIR must render/serialize as
    `data/manifest.json`, never the absolute (machine-local, username-bearing)
    path -- checked in both the rendered section and the JSON sidecar content
    (`json.dumps(res)`, exactly what main() writes to `<out>.recordings.json`)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    root, predict_fn = _build_approved(tmp_path)
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"built_at": "2026-09-02T12:00:00+00:00", "splits": {"train": {"speakers": ["spk02"]}}}
        )
    )

    res = ev.eval_recordings(root, predict_fn, manifest_path=manifest_path)
    assert res["manifest_path"] == "data/manifest.json"
    assert res["manifest_built_at"] == "2026-09-02T12:00:00+00:00"

    md = ev.render_recordings_section(res)
    sidecar = json.dumps(res)
    for text in (md, sidecar):
        assert "/Users" not in text
        assert str(tmp_path) not in text
    assert "data/manifest.json" in md
    assert "2026-09-02T12:00:00+00:00" in md


def test_render_names_the_model_it_measured(tmp_path, monkeypatch):
    """SF-5: the report must name the model file actually measured, data-root-relative."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    root, predict_fn = _build_approved(tmp_path)
    res = ev.eval_recordings(root, predict_fn)
    res["model_path"] = ev._relative_to_data_root(tmp_path / "models" / "command_v3.tflite")
    res["model_sha256"] = "abc123def456789"
    res["evaluated_at"] = "2026-09-02T18:00:00+00:00"

    md = ev.render_recordings_section(res)
    assert "models/command_v3.tflite" in md and "abc123def456" in md
    assert "/Users" not in md and str(tmp_path) not in md


def test_replace_recordings_section_rewrites_in_place():
    """SF-6: a second run replaces the section instead of appending a second copy,
    so the markdown can never disagree with the (always overwritten) JSON sidecar."""
    report = ev.replace_recordings_section("# v3 report\n", "## held-out\nfirst\n")
    assert report.startswith("# v3 report\n")
    again = ev.replace_recordings_section(report, "## held-out\nsecond\n")
    assert again.count("## held-out") == 1
    assert "first" not in again and "second" in again
    assert again.count(ev.RECORDINGS_SECTION_START) == 1


def test_eval_recordings_on_a_run_qc_tree(tmp_path):
    """Producer and consumer in one test: `qc.run_qc` writes the approved tree
    (index `file` column relative to `approved/`), `eval_recordings` reads it.
    Any drift between the two — a changed index convention, a changed directory
    layout — fails here instead of only on a real run."""
    from kws_de import qc

    inc = tmp_path / "incoming" / "s1"
    sr = 16000
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.8)) / sr)).astype(np.float32)
    takes = ("spk02/licht/001.wav", "spk02/_phrase_/licht-an_001.wav", "spk02/_neg_/hallo_001.wav")
    for rel in takes:
        p = inc / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, tone, sr, subtype="PCM_16")
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk02,t,Licht,spk02/licht/001.wav,800,-10,words,1,1\n"
        "spk02,t,Licht an,spk02/_phrase_/licht-an_001.wav,800,-10,sentences,1,2\n"
        "spk02,t,hallo welt,spk02/_neg_/hallo_001.wav,800,-10,negatives,1,3\n"
    )

    def transcriber(p):
        if "_phrase_" in str(p):
            return {"text": "Licht an", "words": []}
        return {"text": "Licht" if "licht" in str(p) else "hallo welt", "words": []}

    approved = tmp_path / "approved"
    qc.run_qc(inc, tmp_path / "qc" / "s1", approved, transcriber)

    licht = config.COMMAND_LABELS.index("Licht")

    def predict_fn(window):
        p = np.zeros(len(config.COMMAND_LABELS), np.float32)
        p[licht] = 0.9
        return p

    res = ev.eval_recordings(approved, predict_fn)
    held = res["figures"][ev.HELD_OUT]
    assert held["isolated"]["spk02"]["n"] == 1
    assert held["e2e"]["spk02"]["n"] == 1  # the phrase clip was actually read
    assert held["false_accepts"]["spk02"]["n"] == 1  # so was the negative


def test_eval_recordings_no_manifest_all_held_out(tmp_path):
    root, predict_fn = _build_approved(tmp_path)

    res = ev.eval_recordings(root, predict_fn)  # no manifest_path passed
    assert res["manifest_found"] is False
    assert res["manifest_path"] is None

    trained = res["figures"]["user-customised, in-training"]
    held_out = res["figures"]["held-out"]
    n_trained, spk_trained = ev._figure_totals(trained)
    n_held, spk_held = ev._figure_totals(held_out)
    assert n_trained == 0 and spk_trained == set()
    assert spk_held == {"spk02", "spk03"}
    assert held_out["isolated"]["spk02"]["n"] == 1 and held_out["isolated"]["spk03"]["n"] == 1
    assert held_out["e2e"]["spk02"]["n"] == 1
    assert held_out["false_accepts"]["spk02"]["n"] == 1

    md = ev.render_recordings_section(res)
    assert "No training manifest given; all clips held-out." in md
    assert "## user-customised, in-training" in md and "## held-out" in md


# --- Field section (Task 5) --------------------------------------------------
#
# Contract shipped by Task 4 (task-4-report.md "Fix round 1", qc.csv semantics
# table; ruling R-9 in progress.md) — NOT the task-5 brief's superseded version:
#   parsable  <=> row["prompt"] non-empty (agrees-based test in the brief was
#                 wrong: a row can be parsable but uncompared, see spk02/c below)
#   compared  <=> row["agrees"] in {"0", "1"}; "" means unparsable OR the device
#                 gave no answer (e.g. a ring-truncated take)
#   agreement <=> agrees=="1" / compared (NOT / parsable)
#   unfiled ("N unparsed (vocab present)") lives only in each stamp's
#   report.md Field line -- qc.csv carries no column for it (an unfiled row and
#   a filed-negative row are identical in qc.csv: verdict=approve, prompt="").
#   takes     <=> EVERY field row, approved or not; approved is its own number,
#                 counted the same way as qc.py's own report.md Field line
#   truncated <=> row["truncated"] == "1": the device's ring cut the take short


def _qc_root(tmp_path):
    """Two stamps' qc.csv + report.md. spk02: 5 field takes, 4 approved -- 3
    parsable (one of which, c.wav, is parsable but NOT compared: the device gave
    no answer, because the ring truncated it), 2 compared, 1 agreeing, plus one
    REJECTED take that still counts as a take -- plus a guided `words` row that
    must be ignored. spk03: 1 field take, parsable and agreeing."""
    root = tmp_path / "qc"
    cols = (
        "file,set,prompt,speaker,verdict,reason,transcript,match_score,"
        "rms_dbfs,peak_dbfs,dur_ms,device_intent,agrees,truncated\n"
    )
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "qc.csv").write_text(
        cols
        + "a.wav,field,Licht an,spk02,approve,,Hey Bus Licht an,1.0,-20,-6,3500,Licht an,1,0\n"
        + "b.wav,field,Licht an,spk02,approve,,Hey Bus Licht aus,1.0,-20,-6,3500,Licht aus,0,0\n"
        + "c.wav,field,Heizung waermer,spk02,approve,,Hey Bus Heizung waermer,1.0,-20,-6,"
        + "3500,,,1\n"
        + "d.wav,field,,spk02,approve,,wann fahren wir los,1.0,-20,-6,3500,,,0\n"
        + "g.wav,field,,spk02,reject,empty_transcript,,0.0,-20,-6,3500,,,0\n"
        + "e.wav,words,Licht,spk02,approve,,Licht,1.0,-20,-6,800,,,\n"
    )
    (root / "s1" / "report.md").write_text(
        "# QC s1\n\n## Field\n\n5 field takes, 4 approved, 3 parsable, 1 wake clips, "
        "1 unparsed (vocab present), 1 ring-truncated, "
        "device-Whisper agreement 0.500 over 2 compared.\n"
    )
    (root / "s2").mkdir(parents=True)
    (root / "s2" / "qc.csv").write_text(
        cols + "f.wav,field,Heizung waermer,spk03,approve,,Hey Bus Heizung waermer,1.0,-20,-6,"
        "3500,Heizung waermer,1,0\n"
    )
    (root / "s2" / "report.md").write_text(
        "# QC s2\n\n## Field\n\n1 field takes, 1 approved, 1 parsable, 1 wake clips, "
        "0 unparsed (vocab present), 0 ring-truncated, "
        "device-Whisper agreement 1.000 over 1 compared.\n"
    )
    return root


def test_field_figures_count_takes_parsable_compared_and_agreement(tmp_path):
    fig = ev.field_figures(_qc_root(tmp_path))
    # takes counts the rejected row too; approved is reported beside it, so the
    # two reports cannot quote different numbers for "field takes" (S5).
    assert fig["takes"] == 6
    assert fig["approved"] == 5
    assert fig["truncated"] == 1
    assert fig["parsable"] == 4
    assert fig["compared"] == 3
    assert fig["agree"] == 2
    assert fig["unfiled"] == 1
    assert fig["per_speaker"]["spk02"] == {
        "takes": 5,
        "approved": 4,
        "truncated": 1,
        "parsable": 3,
        "compared": 2,
        "agree": 1,
    }
    assert fig["per_speaker"]["spk03"] == {
        "takes": 1,
        "approved": 1,
        "truncated": 0,
        "parsable": 1,
        "compared": 1,
        "agree": 1,
    }


def test_recordings_section_carries_the_field_table(tmp_path):
    root, predict_fn = _build_approved(tmp_path)
    res = ev.eval_recordings(root, predict_fn, qc_root=_qc_root(tmp_path))
    md = ev.render_recordings_section(res)
    assert "## Field" in md
    assert "6 field takes, 5 approved, 4 parsable" in md
    assert "1 unparsed (vocab present)" in md
    assert "1 ring-truncated" in md
    # spk02: agree/compared = 1/2 = 0.500 (NOT 1/3 -- parsable is not the
    # denominator); spk03: 1/1 = 1.000.
    assert "| spk02 | 5 | 4 | 3 | 0.500 |" in md
    assert "| spk03 | 1 | 1 | 1 | 1.000 |" in md
    assert "AT CAPTURE TIME" in md  # says whose accuracy this is


def test_field_table_says_n_a_rather_than_nan_when_nothing_was_compared(tmp_path):
    # a speaker whose takes the device never answered has no rate to print; qc.py
    # prints "n/a" for exactly this case, so eval must not print `nan`.
    fig = {
        "per_speaker": {"spk09": dict.fromkeys(ev._FIELD_KEYS, 0) | {"takes": 2, "approved": 2}},
        **dict.fromkeys(ev._FIELD_KEYS, 0),
        "takes": 2,
        "approved": 2,
        "unfiled": 0,
    }
    md = ev.render_field_section(fig)
    assert "| spk09 | 2 | 2 | 0 | n/a |" in md
    assert "nan" not in md


def test_recordings_section_has_no_field_table_without_field_takes(tmp_path):
    root, predict_fn = _build_approved(tmp_path)
    res = ev.eval_recordings(root, predict_fn)
    assert res["field"]["takes"] == 0
    assert "## Field" not in ev.render_recordings_section(res)
