import pytest

from kws_de.qc import _load_whisper_model, _with_cpu_fallback


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
    got, on_cpu = _load_whisper_model(ctor, "large-v3", "auto", "default")
    assert got == ("model", "large-v3", "cpu", "int8") and on_cpu
    assert ctor.calls == [("large-v3", "auto", "default"), ("large-v3", "cpu", "int8")]
    assert "retrying on CPU" in capsys.readouterr().out


def test_explicit_cpu_request_never_retries():
    ctor = _Ctor(RuntimeError("CUDA failed with error out of memory"))
    with pytest.raises(RuntimeError, match="out of memory"):
        _load_whisper_model(ctor, "large-v3", "cpu", "int8")
    assert len(ctor.calls) == 1


def test_missing_cuda_library_falls_back_too(capsys):
    ctor = _Ctor(RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"))
    got, on_cpu = _load_whisper_model(ctor, "large-v3", "cuda", "float16")
    assert got[2:] == ("cpu", "int8") and on_cpu
    assert "libcublas" in capsys.readouterr().out


def test_success_first_time_makes_one_call():
    ctor = _Ctor()
    got, on_cpu = _load_whisper_model(ctor, "large-v3", "auto", "default")
    assert got[2] == "auto" and not on_cpu
    assert len(ctor.calls) == 1


def test_on_cpu_reads_the_device_the_model_landed_on():
    class Inner:
        device = "cpu"

    class Model:
        model = Inner()

    _, on_cpu = _load_whisper_model(lambda *a, **k: Model(), "large-v3", "auto", "default")
    assert on_cpu  # "auto" resolved to CPU: a later failure must not "fall back" to CPU again


def test_mid_transcription_gpu_failure_switches_to_cpu_once(capsys):
    state = {"model": "gpu", "cpu": False}
    calls = []

    def run(model):
        calls.append(model)
        if model == "gpu":
            raise RuntimeError("CUDA failed with error out of memory")
        return f"text from {model}"

    assert _with_cpu_fallback(state, run, lambda: "cpu") == "text from cpu"
    assert calls == ["gpu", "cpu"] and state == {"model": "cpu", "cpu": True}
    assert "switching to CPU" in capsys.readouterr().out
    assert _with_cpu_fallback(state, run, lambda: "cpu2") == "text from cpu"  # kept, not rebuilt


def test_cpu_model_errors_propagate():
    state = {"model": "cpu", "cpu": True}

    def run(_model):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _with_cpu_fallback(state, run, lambda: "cpu")
