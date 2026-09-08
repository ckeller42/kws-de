import importlib.util
import pathlib

import numpy as np
import soundfile as sf

_SPEC = importlib.util.spec_from_file_location(
    "audit_approved", pathlib.Path(__file__).parent.parent / "scripts" / "audit-approved.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _tone(ms=1000, sr=16000):
    t = np.arange(int(sr * ms / 1000)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _tree(root):
    """A guided words/Licht clip and a context/Licht clip (E48's two buckets)."""
    for tree in ("words", "context"):
        d = root / "approved" / tree / "Licht"
        d.mkdir(parents=True)
        sf.write(d / "spk01_001.wav", _tone(), 16000, subtype="PCM_16")


def test_audit_reports_words_and_context_as_separate_sets(tmp_path, monkeypatch, capsys):
    from kws_de import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    approved = tmp_path / "recordings" / "approved"
    _tree(tmp_path / "recordings")
    monkeypatch.setattr("sys.argv", ["audit-approved.py", "--no-transcribe", str(approved)])

    rc = audit.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "words" in audit.SETS and "context" in audit.SETS
    assert "words         1   Licht 1" in out
    assert "context       1   Licht 1" in out


def test_audit_word_content_check_splits_guided_vs_context(tmp_path, monkeypatch, capsys):
    from kws_de import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    approved = tmp_path / "recordings" / "approved"
    _tree(tmp_path / "recordings")
    monkeypatch.setattr("sys.argv", ["audit-approved.py", str(approved)])

    def fake_transcriber(_enabled):
        return lambda wav: {"text": "Licht", "words": [{"word": "Licht", "start": 0.4, "end": 0.6}]}

    monkeypatch.setattr(audit, "transcriber_or_none", fake_transcriber)
    rc = audit.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "guided" in out and "context" in out
    section = out.split("## Word content check")[1]
    assert "guided" in section and "context" in section
