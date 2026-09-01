# Local microWakeWord training — "Hey Bus"

Trains the always-on wake word locally (no Colab). microWakeWord is Python-3.10-only, so it
runs in its **own venv**, separate from the 3.11 `kws-de` package. GPU is optional — it trains
on CPU / Apple-Metal (slower, a one-time cost).

Verified: microWakeWord installs and imports cleanly in a local `uv`-managed 3.10 venv on
Apple Silicon (TensorFlow 2.21, pymicro-features, python-stretch, webrtcvad).

## Setup

```bash
./setup.sh          # creates .venv (py3.10) and installs microWakeWord from git
```

## Train

microWakeWord's trainer:

1. **Positives** — generate "Hey Bus" samples with **Piper TTS** (many voices / rates), plus
   confusables. (Piper is pulled in by microWakeWord.)
2. **Negatives / ambient** — microWakeWord's ambient + negative feature sets.
3. **Train** the streaming (MixConv) model on CPU/Metal → export a TFLite-Micro model.
4. Copy the result to `models/hey_bus.tflite` (gitignored) — consumed by
   `kws_de.wake.load_wake_tflite` at runtime and budget-checked by `kws_de.budgets.check_wake_budgets`.

See <https://github.com/kahrendt/microWakeWord> for the current trainer API and config schema.
`kws_de.wake.train_hey_bus` shells out to this venv to run the training.

## Notes

- The `.venv/`, downloaded datasets, and `*.tflite` here are **gitignored** — only this README
  and `setup.sh` are versioned (recipe, not bytes).
- Training time on CPU/Metal is minutes-to-hours for the ~30k Piper samples; reduce the sample
  count for a quick first model and note it in the eval report.
