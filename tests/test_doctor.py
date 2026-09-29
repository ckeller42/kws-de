"""`kws-doctor` is a read-only self-check (design §1): it must exit 0 and print
the resolved storage paths even on a bare machine (no data root on disk, TF not
importable). Reloaded around a tmp KWS_DATA_ROOT and restored afterwards."""

import importlib

import pytest

from kws_de import config, doctor


@pytest.fixture
def under_root(monkeypatch, tmp_path):
    monkeypatch.setenv("KWS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("KWS_NOISE_DIR", raising=False)
    monkeypatch.delenv("KWS_RIR_DIR", raising=False)
    importlib.reload(config)
    importlib.reload(doctor)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)
    importlib.reload(doctor)


def test_main_exits_zero_and_prints_resolved_paths(under_root, capsys):
    rc = doctor.main()
    assert rc == 0
    out = capsys.readouterr().out
    # The tmp data root and its derived data/ + models/ dirs are printed.
    assert str(under_root) in out
    assert str(under_root / "data") in out
    assert str(under_root / "models") in out
    # Section headers and the never-fatal TF/GPU/backend lines are present.
    for token in (
        "data_root",
        "data_dir",
        "models_dir",
        "noise_dir",
        "rir_dir",
        "platform",
        "tensorflow",
        "gpu",
        "qc backend",
    ):
        assert token in out


def test_report_survives_missing_tf(monkeypatch, under_root):
    # Simulate TensorFlow absent: import must be caught, report still produced.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ImportError("simulated: no tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    text = doctor.report()
    assert "not importable" in text
    assert "no TensorFlow" in text
