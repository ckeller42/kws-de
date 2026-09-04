from pathlib import Path


class WakeDetector:
    """Cutoff + debounce wrapper around microWakeWord's per-step wake
    probability (spec §12) — fires once when `prob >= cutoff`, then
    suppresses for `refractory` steps."""

    def __init__(self, cutoff: float = 0.8, refractory: int = 20):
        self.cutoff = cutoff
        self.refractory = refractory
        self._cooldown = 0

    def push(self, prob: float) -> bool:
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if prob >= self.cutoff:
            self._cooldown = self.refractory
            return True
        return False


def gate_fired(probs, cutoff: float = 0.85, consecutive: int = 2) -> bool:
    """The firmware's wake gate: `consecutive` steps at or above `cutoff` in a row.

    Kept separate from `WakeDetector` because the gate is what an offline probe
    asks of a whole clip, while `WakeDetector` is the online one-shot debouncer.
    """
    run = 0
    for p in probs:
        run = run + 1 if p >= cutoff else 0
        if run >= consecutive:
            return True
    return False


# The streaming model's receptive field is ~1.9 s, and it starts from the ring-buffer
# state its CALL_ONCE init subgraph writes. A clip shorter than that must therefore be
# fed with real leading context or the wake word is scored against a half-filled ring:
# tightly-cut field clips (0.2-0.7 s) read as silence without it.
WAKE_CONTEXT_S = 2.0


def stream_wake_probs(model_path, pcm, context_s: float = WAKE_CONTEXT_S):
    # pragma: no cover - needs the trained artifact
    """Per-step wake probabilities for int16 `pcm`, exactly as the firmware sees them.

    Builds a fresh interpreter per call on purpose. `Interpreter.reset_all_variables()`
    zeroes the ring buffers instead of re-running CALL_ONCE, which does not restore the
    trained initial state and leaves scores dependent on which clip was scored before —
    so there is deliberately no way to reuse an interpreter through this function.
    """
    import numpy as np
    import tensorflow as tf

    from kws_de.firmware_gen import wake_features

    pad = np.zeros(int(context_s * 16000), np.int16)
    rows = wake_features(np.concatenate([pad, np.asarray(pcm, np.int16), pad]))

    itp = tf.lite.Interpreter(model_path=str(model_path))
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    probs = []
    for i in range(0, len(rows) - 2, 3):
        itp.set_tensor(inp["index"], rows[i : i + 3].reshape(inp["shape"]).astype(np.int8))
        itp.invoke()
        probs.append(float(itp.get_tensor(out["index"]).ravel()[0]) * out["quantization"][0])
    return np.array(probs)


def load_wake_tflite(path):  # pragma: no cover - needs the trained artifact
    import numpy as np
    import tensorflow as tf

    itp = tf.lite.Interpreter(model_path=str(path))
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]

    def predict_fn(features):
        itp.set_tensor(inp["index"], np.asarray(features, inp["dtype"]).reshape(inp["shape"]))
        itp.invoke()
        return float(itp.get_tensor(out["index"]).ravel()[-1])

    return predict_fn


def train_hey_bus(out_dir) -> Path:  # pragma: no cover - invokes microWakeWord trainer
    # Use microWakeWord's training framework: generate "Hey Bus" positives via Piper TTS,
    # combine with its ambient/negative sets, train the streaming model, export TFLite-Micro.
    # See https://github.com/kahrendt/microWakeWord (+ microwakeword-trainer). Produces
    # hey_bus.tflite in out_dir.
    raise NotImplementedError("wire microWakeWord trainer; see spec §12")
