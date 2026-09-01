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
