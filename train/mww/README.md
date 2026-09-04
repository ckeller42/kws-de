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

## Retraining on real recordings (round 6 recipe)

Once the device has produced real "Hey Bus" takes, `scripts/wake-retrain.sh` drives the whole
round from this repo. It never writes inside the repo and never installs a model — it leaves a
candidate at `$KWS_DATA_ROOT/models/hey_bus_<round>.tflite` for a human to compare and promote.

```bash
export KWS_DATA_ROOT=...            # shared data root
export MWW_DIR=...                  # this training directory (its own py3.10 .venv)
export WAKE_ROUND=r6
export WAKE_HOLDOUT=<session-stamp> # session kept out of train AND validation
export WAKE_SIL_DIR=...             # speech-free field takes -> room-noise negatives
scripts/wake-retrain.sh --stage-only   # inspect the split first
scripts/wake-retrain.sh
```

Three rules carry the round, and each exists because an earlier round got it wrong.

### 1. The held-out unit is a recording session, not a clip

Splitting one session at random measures memorisation: every clip in it shares a room, a mic
gain and a firmware build. `wake-retrain.sh` reads each session's `qc/<stamp>/written.txt` — the
list of approved files that session produced — and keeps the whole of `WAKE_HOLDOUT` out of both
train and validation. Report held-out numbers separately from in-training ones; they are not the
same measurement.

### 2. State the real share — but do not cap it

microWakeWord's `sampling_weight` is a global weight in one `random.choices` draw over every
feature set, so a set's share of the positives is `w / sum(positive w)`. Round 5 gave ten real
clips `sampling_weight: 5.0` against the TTS positives' `2.0` — **71 % of every positive batch
drawn from ten unique recordings**. That looks like overfitting, and round 6 tested capping it.

**It is not overfitting, and capping it is harmful.** Recall stays pinned at the ceiling (every
model fires on every real positive, held-out session included, at peak 0.996) while false wakes
rise monotonically as the real share falls — 1–4, 15 and 33 false fires out of 48 TTS non-wake
clips at 71 %, 50 % and 30 %. Nine thousand Piper "hey bus" clips can be satisfied by a loose
"German speech with a stressed front syllable" feature, because the hard negatives opposing them
are Piper too; ten real recordings through the device's own microphone are a far narrower target,
and that narrowness is what rejects "hallo bus" and "der bus kommt gleich". Keep the real share
high. If overfitting to few speakers is the worry, add speakers, not weight.

Give real audio its own feature dir anyway, so the share is a number in the config rather than
an accident of how many clips happen to exist — and state it in the report.

### 3. Hard negatives come from the same takes as the positives

Command phrases cut from the field takes *after* the wake phrase, and speech-free room-noise
takes, go through the same augmentation as the positives (`gen_features_real.py`) into their own
`truth: false` dirs. Same voice, same mic, same room — the negative that generic English ambient
sets cannot supply.

**Check what you are about to call a negative.** The session cutter writes the *whole take*
whenever the wake word is not a leading cut — when the take has no pre-roll, or when the speaker
says "Hey Bus" again at the end. Clips filed under `phrases/` and `negatives/` can therefore
still contain the wake word, and training on those teaches the model *not* to wake. Pass those in
`WAKE_EXCLUDE`. Two cheap checks find them: a clip whose duration equals its source take's
duration was never cut at all, and `scripts/wake_probe.py` firing on something filed as a
negative is either a false accept or a mislabelled clip — look before assuming which.

The same warning applies to the positives. A session recorded without pre-roll starts *after*
the device's own wake detection, so its `wake/` clips hold only the tail fragment of the phrase
— 0.2–0.3 s, already at full level in the first frame, with the "Hey" missing entirely. Plot the
100 ms frame energies of a take before trusting its cut: a good one opens with 200 ms of near
silence. Training on fragments teaches the model to fire on a 0.2 s syllable, which is the
false-accept failure mode that costs the most. Exclude them.

### 4. Trim trailing silence off the real positives, or the wake fires a second late

`gen_features_real.py` builds `Clips` with `remove_silence=False`, and
`truncation_strategy: truncate_start` keeps the last 1,500 ms of each 3.2 s augmentation window.
So a real take that carries a second of trailing silence after the phrase puts the positive
label a second after the phrase *ends*, and the model dutifully learns to answer there — even on
clips that have no trailing silence, because it learned the offset, not the clip.

This is exactly what the guided takes did: 1.78–1.92 s files whose phrase ends at 0.76–1.16 s,
i.e. **~1.04 s of trailing silence**, producing a measured 0.94–1.13 s delay between phrase end
and wake. The model answers with two probability humps and the gate fires on the second one.
Trimming the clips to the phrase (energy endpoint, ~0.25 s lead / 0.20 s tail) — or setting
`remove_silence=True` — removes the whole second and lifts the first hump from 0.32 to 0.996,
with no change to any accept or reject.

Measure it, don't assume it: splice each clip into several seconds of real room tone so the
microfrontend's noise and PCAN estimates are warm, then report fire time minus the *phrase*
endpoint (energy-based), not minus the file end. A wake word that answers a second after you
stop speaking feels broken in a way no accuracy number captures.

### 5. Gate the synthetic clips before they become features

The Piper/`say` positives and negatives are most of the training set, and nobody ever
listens to them. macOS `say` silently substitutes an English voice when the German one is
missing — or when the name is ambiguous, `say -v Eddy` being the *English* Eddy on a
machine that has both — so a whole "German" voice can come out English with no error
anywhere. That reached a device test once (paper notes E23).

`wake-retrain.sh` therefore runs `kws-tts-check` over `WAKE_TTS_DIRS` (default
`generated_samples_v3/{positives,negatives}`) before `gen_features_real.py`, and a failing
clip stops the round:

```bash
uv run --no-sync kws-tts-check <dir> --quarantine
```

It reads the `manifest.csv` that `kws_de.tts.synthesize` writes beside the clips, so a set
generated before that existed has to be regenerated to be checkable at all. The per-voice
summary is the useful part: a voice that is not German shows up as 100 % failed, not as
scattered bad luck.

### Probing

Use `scripts/wake_probe.py` (device gate 0.85 x 2 consecutive steps), not an ad-hoc loop:

- it builds a **fresh interpreter per clip**, because `reset_all_variables()` zeroes the ring
  buffers rather than re-running the `CALL_ONCE` init subgraph, which leaves every score
  dependent on which clip happened to be scored before it;
- it prepends `WAKE_CONTEXT_S` (2 s) of leading context, because the streaming receptive field is
  ~1.9 s and a tightly-cut 0.2–0.7 s field clip otherwise scores against a half-filled ring.

Neither artifact shows on 1.8 s guided takes, which is how both survived five rounds.

## Notes

- The `.venv/`, downloaded datasets, and `*.tflite` here are **gitignored** — only this README
  and `setup.sh` are versioned (recipe, not bytes).
- Training time on CPU/Metal is minutes-to-hours for the ~30k Piper samples; reduce the sample
  count for a quick first model and note it in the eval report.
- Keep the architecture flags fixed across rounds (`--pointwise_filters 64,64,64,64`,
  `--first_conv_filters 32`, `--stride 3`): the width/depth sweep found bigger models fire 2–4x
  more often on German non-wake speech, and widening pushes the arena out of internal SRAM.
