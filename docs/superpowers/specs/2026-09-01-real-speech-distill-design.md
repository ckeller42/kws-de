# Real speech, distillation, INT8 recovery — design

**Date:** 2026-09-01 · **Branch:** `feat/real-speech-distill` · **Baseline:** `main` @ cf785a5

## 1. Problem

The v2 command model (DS-CNN, 23 classes, 20 KB INT8) is trained on a
dataset where **17 of 21 command words have zero real speech** — they are
100 % macOS-`say` TTS (`data/raw_clips_merged.pkl`, verified 2026-09-01):

| real ≥ 300 | partial | **0 real (TTS only)** |
|---|---|---|
| Licht, Kühlschrank, aus, auf | Heizung 120, Außen 158 | Aufstelldach, Küche, Dach, Lesen, an, zu, heller, dunkler, wärmer, kälter, leise, fünfundzwanzig, fünfzig, fünfundsiebzig, hundert |

Consequences measured in E7: isolated-word 0.834 / catalog 0.544 (DS-CNN);
float→INT8 drop 1.63 % (0.8661 → 0.8498, n = 5474).

Root cause of the data gap: `kws_de.data._fetch_mswc` *streams* the
HuggingFace `MLCommons/ml_spoken_words` `de_wav` split with
`safety_cap=300_000` and early-stops per word. MSWC-de has ~48 k keyword
folders (17.95 GB); most of our words were simply never reached.

## 2. Goals / non-goals

Goals, in priority order:

1. **Real speech for the command words** — deterministic, from public
   CC-BY/CC0 sources, rebuilding the frozen speaker-disjoint dataset.
2. **Knowledge distillation** KWT teacher → DS-CNN student, benchmarked
   against the undistilled student on the same data/seed.
3. **INT8 recovery** — close the 1.63 % PTQ gap cheaply (class-balanced
   calibration); hand-rolled QAT only if a gap that matters survives.

Non-goals: new architectures; changing the vocabulary; CV-de forced
alignment (conditional follow-up, §3.3); the wake stage; on-device
firmware.

## 3. Lane 1 — real speech

### 3.1 MSWC-de tarball mining (`kws_de/mswc.py`, new)

Sources (CC-BY 4.0, MLCommons):

- audio `https://mswc.mlcommons-storage.org/audio/de.tar.gz` (17.95 GB)
- splits `https://mswc.mlcommons-storage.org/splits/de.tar.gz`
  (`de_splits.csv`: `SET,LINK,WORD,VALID,SPEAKER,GENDER`)

Extracted once by the user under `data/mswc/de/` (gitignored; local path
is a machine detail and is **not** documented beyond `data/`). Layout:
`clips/<keyword>/<clip>.opus`, 1 s @ 16 kHz mono.

```python
def mine(root: Path, words: list[str], *, n_per_word: int = 300,
         n_unknown: int = 2000, unknown_per_word_cap: int = 5,
         seed: int = 0) -> dict[str, list[tuple[np.ndarray, str]]]
```

- Keyword folder lookup is **case-insensitive on the folder name**
  (`Küche` → `küche`); the returned label is the `config` spelling.
- Only rows with `VALID == "TRUE"` in the splits csv; speaker id taken
  from the csv `SPEAKER` column (real MSWC ids, never `tts:`-prefixed).
- Per word: shuffle the valid clip list with `numpy.random.default_rng(seed)`
  and take the first `n_per_word` (deterministic, speaker-diverse enough
  because MSWC clips are one-per-utterance).
- `_unknown_`: `n_unknown` clips drawn from keyword folders **not** in
  `words`, at most `unknown_per_word_cap` per keyword, keywords visited in
  a seeded shuffle of the folder list (diversity, not alphabetical bias).
- Decode with `soundfile.read`; if libsndfile lacks opus, fall back to
  `ffmpeg -i <f> -f f32le -ac 1 -ar 16000 -` via `subprocess`. Resample only
  if the file is not 16 kHz.
- Returns the **same dict shape** as `_fetch_mswc`, so the rest of the
  pipeline (`_fill_with_tts`, `split_three_way`, `dataset.build`) is
  unchanged.
- Pure helper split out for tests: `_pick(paths: list[Path], n: int, rng)`
  and `_folder_index(root) -> dict[str, Path]` (lower-cased folder → path).

`kws-data --fetch --mswc-root data/mswc/de --v3` mines locally instead of
streaming, writes `data/raw_clips_v3.pkl` with the same
`{"clips": ..., "scanned": "mswc-tarball"}` payload. It does **not** merge
`raw_clips_merged.pkl`: the streamed clips are a subset of the tarball, so
the tarball alone is the real-speech source. `_fill_with_tts` then tops up
only what is still short — TTS becomes the backstop it was meant to be.

### 3.2 Self-recorded clips (`kws_de/recordings.py`, new)

For words absent from MSWC (expected: `Aufstelldach`, a rare hardware
compound; possibly `an`/`zu` if MSWC filtered 2-letter words).

- Drop-in folder `data/recordings/<word>/<speaker>_<n>.wav|m4a`
  (gitignored). Any sample rate; converted with `librosa.load(sr=16000)`.
- `load_recordings(root: Path, words: list[str]) -> dict[str, list[(clip, "rec:<speaker>")]]`:
  trim leading/trailing silence (`librosa.effects.trim(top_db=30)`),
  centre in a `config.CLIP_SAMPLES` window (zero-pad / crop symmetrically).
- Speaker id `rec:<speaker>` keeps the speaker-disjoint split honest
  (`split_by_speaker` already groups on the id; `_origin_flags` counts
  `rec:` as real — only `tts:` is synthetic).
- README section: how to record (phone voice memo, 1 word per file, 5–10
  speakers × ~10 takes, quiet + in-vehicle), naming, where to put files.
  No mention of where the word comes from beyond "camper hardware".

`kws-data --fetch --v3` also merges `load_recordings(config.DATA_DIR / "recordings", words)`.

### 3.3 Conditional follow-up (not built now)

If after 3.1 + 3.2 any word still has < 100 real clips, the next lever is
MSWC `alignments/de.tar.gz` + Common Voice German slicing. Decision is
made from the v3 datasheet, not speculatively.

### 3.4 Dataset v3

`kws-dataset build --seed 0 --cache raw_clips_v3.pkl` (add `--cache` arg to
`dataset.main`) → `data/features_v3_{train,val,test}.npz`,
`data/manifest_v3.json`. `load_split(name, prefix="features")` gains a
`prefix` kwarg; `benchmark`/`distill` take `--features v3`. `DATASHEET.md`
gains a v3 table (real vs TTS per word) generated by the existing
datasheet code path.

### 3.5 TTS breadth + perturbation (added 2026-09-01, after §3.1–3.4 were built)

Audit of `raw_clips_merged.pkl`: the TTS backstop is macOS `say` only —
9 voices × 9 rates, 6 622 clips. Piper was wired in `kws_de/tts.py` but
not installed when v2 was built. And TTS speaker ids are
`tts:{engine}:{voice}:{rate}`, so the same voice at two rates lands in
train *and* test — TTS rows in the speaker-disjoint split are not
disjoint. More clips of the same 9 voices would not help; more *voices*
and more *acoustic variety* would.

**Piper breadth (`kws_de/tts.py`, `kws_de/data.py`):**

- Voices are discovered from the cache (`data/piper-voices/<name>/<quality>/<id>.onnx`),
  not hard-coded; `ENGINE_VOICES["piper"]` stays as the fallback when the
  cache is empty (fresh checkout, CI). Multi-speaker voices (the
  `.onnx.json` `num_speakers > 1`: `de_DE-mls-medium` has 236,
  `de_DE-thorsten_emotional-medium` 8) expand to one voice id per speaker,
  `<id>#<speaker_id>`. With `kerstin`, `mls`, `pavoque`, `thorsten_emotional`
  added to the four cached voices: ~250 Piper voices next to 9 `say`.
- Piper gets tempo natively: `SynthesisConfig(length_scale=160/rate)`
  (Piper's own knob, 160 wpm = 1.0), so the shared `RATES` grid means the
  same thing for both engines. `(engine, voice, rate)` combos are ordered
  rates-outer so a per-word draw spreads over voices first.
- TTS speaker id for the split becomes `tts:{engine}:{voice}` — rate is
  augmentation, not identity. v2 feature files are frozen and untouched;
  v3 is a new manifest anyway.

**Perturbation (`kws_de/augment.py`, `kws_de/data.py`):**

- `perturb(sig, n_steps, rate, sr=16000)` — `librosa.effects.pitch_shift`
  + `time_stretch`; pure, deterministic in its arguments.
- `build_dataset(..., synthetic=None)`: `synthetic` is `{label: [bool]}`
  aligned to `clips`; each flagged clip gets one extra copy with
  `n_steps ~ U(-2, 2)` semitones and `rate ~ U(0.85, 1.15)` drawn from the
  build rng, pushed through the same clean + per-SNR pipeline. Rows per
  TTS clip double; real clips are untouched (they carry their own
  variety, and the real:TTS row ratio should not get worse).
  `_origin_flags(..., perturb_tts=True)` mirrors the doubled rows.
  `dataset.assemble` passes both. v1 callers pass nothing → byte-identical
  output.
- Cost: ~50 ms per TTS clip at build time.

Not done: voice cloning / XTTS (deps, GPU, licence questions);
perturbing real clips (measure v3 first; add if isolated < 0.90).

## 4. Lane 2 — distillation (`kws_de/distill.py`, new)

Teacher: `ARCHITECTURES["kwt"]` (reference-only, accurate). Student:
`build_dscnn(num_classes)` unchanged (deployable, softmax head).

```python
def soften(p: np.ndarray, T: float) -> np.ndarray      # softmax(log(p+1e-9)/T), pure
def distill_targets(y: np.ndarray, teacher_probs: np.ndarray, n_classes: int) -> np.ndarray
    # -> (N, 2*n_classes): [one_hot(y) | teacher_probs]
def make_distill_loss(n_classes: int, T: float, alpha: float)
    # Keras loss(y_true(N,2C), y_pred(N,C)):
    #   alpha * CE(one_hot, y_pred)
    # + (1-alpha) * T**2 * KL(soften(teacher,T) || soften(y_pred,T))
def distill(X, y, teacher, *, epochs, seed, T=4.0, alpha=0.5,
            validation_data=None, callbacks=None) -> (student_model, history)
```

- Teacher probabilities are **precomputed once** (`teacher.predict`) — the
  teacher is frozen, so no online forward pass; the student trains with
  plain `model.fit` on the concatenated target.
- Softmax outputs are re-tempered via `log(p)`: log-softmax equals logits
  up to a per-row constant, so `softmax(log p / T)` is exact temperature
  scaling. No change to `model.py` or `kwt.py`.
- Class balance via `sample_weight` (inverse frequency, same formula as
  `train()`), because `class_weight` needs a 1-D label target.
- `validation_data` is evaluated with a plain accuracy metric on the hard
  label (`argmax(y_true[:, :C])`) so `ModelCheckpoint(monitor="val_accuracy")`
  keeps working.

`kws-distill --features v3 --epochs 40 --seed 0` runs, on the same split
and seed: teacher (KWT) float; student baseline (existing `train`);
student distilled. Each device model goes through the shared
`benchmark._evaluate_int8(model, X_train, y_train, X_test, y_test, seed, calib)`
(extracted from `evaluate_architecture`, which then calls it) and is
reported with `benchmark.render_table` → `docs/distill-report.md` +
`docs/distill-benchmark.json` (untracked outputs, like the transducer
report). Success = distilled INT8 iso and catalog vs baseline, stated
honestly either way.

## 5. Lane 3 — INT8 recovery (`kws_de/export.py`)

```python
def balanced_calibration(X, y, *, per_class: int = 20, seed: int = 0) -> np.ndarray
```

Stratified: up to `per_class` rows per class, seeded shuffle within class,
concatenated. Replaces `X_train[:200]` in `export.main`,
`benchmark.evaluate_architecture` and `distill`. `_evaluate_int8` also
reports `float_acc` next to `isolated_acc` (INT8) so the gap is a column,
not a one-off script. `kws-distill` runs the baseline student with both
calibration schemes (`first-200`, `balanced-460`) so the recovery is one
table row.

Decision gate, recorded in the paper: if the balanced gap is still > 1 %
absolute, a hand-rolled fake-quant QAT (Keras 3, no tfmot) becomes the next
spec. Otherwise QAT is closed as unnecessary.

## 6. Cross-cutting constraints

- No training data or model binaries committed; `data/` and `models/`
  stay gitignored (`data/mswc/`, `data/recordings/` included).
- Public repo never names the product/app the vocabulary was derived
  from, nor machine-local paths (external volumes, symlinks).
- Ruff `==0.16.5`, markdownlint, gitleaks, pytest ≥ 85 % on the CI cov
  modules; pure helpers get unit tests, network/audio I/O is
  `# pragma: no cover`.
- Reproducibility: every random choice takes an explicit `seed`; the v3
  manifest hash is the contract.
- PR → CI green on the live head SHA → user merges.

## 7. Testing

- `tests/test_mswc.py`: `_folder_index` case-insensitive; `_pick`
  deterministic + bounded; `mine` on a tmp fixture tree (2 words, 3 tiny
  wavs written by the test, csv with one `VALID=FALSE` row) returns the
  right counts, speaker ids, excludes invalid, `_unknown_` respects
  per-word cap.
- `tests/test_recordings.py`: centring/padding to `CLIP_SAMPLES`, speaker
  id prefix, unknown word folders ignored.
- `tests/test_distill.py`: `soften(p, 1) == p`; `soften` higher T is
  flatter (lower max); `distill_targets` shape/one-hot; loss at
  `alpha=1` equals CE; distilled 2-class toy student reaches > 0.9 train
  accuracy in a few epochs on separable data.
- `tests/test_export.py`: `balanced_calibration` returns ≤ per_class per
  class, all classes present, deterministic in seed.
- Existing tests unchanged and green.

## 8. Paper

`docs/paper.md`: new dataset v3 section (real-speech coverage table
before/after), E9 distillation, E10 calibration/INT8 gap, updated
conclusion. `docs/paper-notes.md` gets one entry per landed result.
