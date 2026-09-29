"""Whisper backend selection (design §2): mlx-whisper on macOS, faster-whisper on
Linux, chosen by platform with a graceful ImportError fallback. Tested without
loading any model — the real builders are monkeypatched with stubs, so this runs
identically on macOS and Linux CI."""

import pytest

from kws_de import qc


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", ("mlx-whisper", "mlx_whisper")),
        ("linux", ("faster-whisper", "faster_whisper")),
        ("linux2", ("faster-whisper", "faster_whisper")),
    ],
)
def test_default_backend_by_platform(monkeypatch, platform, expected):
    monkeypatch.setattr(qc.sys, "platform", platform)
    assert qc.default_backend() == expected


def _stub(tag):
    def build(language="de"):
        return lambda path: {"text": "", "words": [], "language": tag}

    return build


def test_default_transcriber_picks_mlx_on_macos(monkeypatch):
    monkeypatch.setattr(qc.sys, "platform", "darwin")
    monkeypatch.setattr(qc, "whisper_transcriber", _stub("mlx"))
    monkeypatch.setattr(qc, "faster_whisper_transcriber", _stub("faster"))
    tr = qc.default_transcriber()
    assert tr(None)["language"] == "mlx"


def test_default_transcriber_picks_faster_on_linux(monkeypatch):
    monkeypatch.setattr(qc.sys, "platform", "linux")
    monkeypatch.setattr(qc, "whisper_transcriber", _stub("mlx"))
    monkeypatch.setattr(qc, "faster_whisper_transcriber", _stub("faster"))
    tr = qc.default_transcriber()
    assert tr(None)["language"] == "faster"


def test_default_transcriber_falls_back_on_importerror(monkeypatch):
    # macOS preferring mlx, but mlx not installed → falls back to faster-whisper.
    monkeypatch.setattr(qc.sys, "platform", "darwin")

    def missing(language="de"):
        raise ImportError("no mlx here")

    monkeypatch.setattr(qc, "whisper_transcriber", missing)
    monkeypatch.setattr(qc, "faster_whisper_transcriber", _stub("faster"))
    tr = qc.default_transcriber()
    assert tr(None)["language"] == "faster"


def test_default_transcriber_raises_when_no_backend(monkeypatch):
    monkeypatch.setattr(qc.sys, "platform", "linux")

    def missing(language="de"):
        raise ImportError("nope")

    monkeypatch.setattr(qc, "whisper_transcriber", missing)
    monkeypatch.setattr(qc, "faster_whisper_transcriber", missing)
    with pytest.raises(ImportError, match="no Whisper backend"):
        qc.default_transcriber()
