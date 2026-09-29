import pytest

from kws_de.qc import _load_whisper_model


class _Ctor:
    """Stand-in for faster_whisper.WhisperModel: fails with the given error on the
    first call, records every call, succeeds afterwards."""

    def __init__(self, first_error=None):
        self.first_error = first_error
        self.calls = []

    def __call__(self, model_id, device, compute_type):
        self.calls.append((model_id, device, compute_type))
        if self.first_error is not None and len(self.calls) == 1:
            raise self.first_error
        return ("model", model_id, device, compute_type)


def test_cuda_oom_on_auto_falls_back_to_cpu_int8(capsys):
    ctor = _Ctor(RuntimeError("CUDA failed with error out of memory"))
    got = _load_whisper_model(ctor, "large-v3", "auto", "default")
    assert got == ("model", "large-v3", "cpu", "int8")
    assert ctor.calls == [("large-v3", "auto", "default"), ("large-v3", "cpu", "int8")]
    assert "retrying on CPU" in capsys.readouterr().out


def test_explicit_cpu_request_never_retries():
    ctor = _Ctor(RuntimeError("CUDA failed with error out of memory"))
    with pytest.raises(RuntimeError, match="out of memory"):
        _load_whisper_model(ctor, "large-v3", "cpu", "int8")
    assert len(ctor.calls) == 1


def test_non_oom_error_is_reported_not_masked():
    ctor = _Ctor(RuntimeError("cublas handle creation failed"))
    with pytest.raises(RuntimeError, match="cublas"):
        _load_whisper_model(ctor, "large-v3", "auto", "default")
    assert len(ctor.calls) == 1


def test_success_first_time_makes_one_call():
    ctor = _Ctor()
    assert _load_whisper_model(ctor, "large-v3", "auto", "default")[2] == "auto"
    assert len(ctor.calls) == 1
