# Offline German Voice Control on a Microcontroller: Keyword Spotting with a Slot Grammar on the ESP32-S3

**Project:** kws-de — <https://github.com/ckeller42/kws-de>
**Status:** working draft. All numbers are from committed evaluation reports in this repository;
synthetic-data caveats are stated in place. Author/affiliation line to be completed by the author.

---

## Abstract

We present an open, reproducible pipeline for **German**, **intent-level** voice control that runs
**fully offline on a microcontroller** (ESP32-S3, M5Stack CoreS3). The gap it fills is concrete:
the vendor on-device stack (ESP-SR/MultiNet) supports only Chinese and English, and the one
on-device German speech-to-intent engine (Picovoice Rhino) is closed-source. Our system pairs a
tiny always-on wake-word model with a streaming keyword detector whose events a **pure,
device-specific slot grammar** (`device → zone? → action`) composes into a validated intent. A
single-word model reaches **91.1 % real-speech INT8 accuracy** on the words for which real German
data exists — comparable to MultiNet's reported ~85–95 % English — in **19 KB / 2.07 M MACs**,
full-INT8. Every model passes CI **resource-budget gates** that prove device-fit without hardware.
We contribute three methodological points: (i) **honest synthetic-data provenance** — of 24 grounded
command words only 7 occur in the public corpus, so 17 are text-to-speech, and we report per-word
real-vs-TTS counts and compute headline metrics on real speech only; (ii) a **full-command-catalog
end-to-end evaluation** (audio → MFCC → streaming detector → grammar → intent) that caught a
dataset-construction bug a per-clip accuracy metric structurally could not; and (iii) a documented,
**reusable dataset** with speaker-disjoint train/val/test splits and a datasheet. We also show a
custom German wake word ("Hey Bus") trains **locally on a laptop in minutes** (no GPU/Colab),
and we report an instructive **negative result**: the obvious "train on word-boundary windows"
fix *reduces* accuracy unless the added negatives are class-balanced.

## 1. Introduction

Camper vans and similar always-on edge settings need hands-free control (lights, heater, fridge,
pop-up roof) where cloud connectivity is intermittent or absent and a Linux single-board computer
cannot stay awake. The natural device is a cheap microcontroller with a microphone that listens
continuously and wakes heavier logic only on demand.

Doing this in **German** at the **intent** level (not just a wake word, but "Licht Küche an" →
`turn on the kitchen light`) on an **MCU**, with **open** tooling, is unserved:

- Espressif's on-device speech stack, **ESP-SR / MultiNet**, recognises fixed command sets but
  ships models for **Chinese and English only**.
- **Picovoice Rhino** does on-device speech-to-intent with slot-filling and supports German on
  Arm Cortex-M — but it is **closed-source**.
- Open alternatives (Rhasspy, Vosk, DeepSpeech) are speech-to-*text*, Raspberry-Pi-class, not
  MCU slot-filling.

**Contributions.**

1. A reproducible **model factory** for German KWS on the ESP32-S3: public-corpus subset → MFCC →
   depthwise-separable CNN → full-INT8 TFLite-Micro, with CI resource-budget gates.
2. An **honest-provenance** methodology for synthetic data, and a **full-catalog end-to-end**
   evaluation that exposes failures per-clip metrics hide.
3. A two-stage **open** architecture — a reused wake engine gating a streaming detector + a pure
   slot grammar — as the open-source counterpart to closed speech-to-intent.
4. A documented, **reusable dataset** (speaker-disjoint splits + datasheet), and a **negative
   result** on transition-window training that we believe is broadly useful.

## 2. Related work

**Small-footprint KWS.** DS-CNN established depthwise-separable convolutions as the MCU-KWS
workhorse (Zhang et al., 2017); later encoders push accuracy at tiny parameter counts —
BC-ResNet (Kim et al., 2021), MatchboxNet (Majumdar & Ginsburg, 2020), and the Keyword Transformer
(Berg et al., 2021). A recent survey collects the field (2506.11169).

**Streaming and connected KWS.** Recognising *sequences* of words in continuous speech is
addressed by streaming sequence models — sequence-to-sequence (He et al., 2017), CTC and RNN-T
transducers, and recent CTC-transducer hybrids (MFA-KWS, 2025). Streaming conversion of
non-streaming models is treated by Rybakov et al. (2020).

**On-device stacks.** ESP-SR/MultiNet (Espressif) and microWakeWord (the ESPHome/Home-Assistant
wake engine) target the ESP32-S3; Picovoice Rhino is the closed reference for on-device German
speech-to-intent.

**Data.** The Multilingual Spoken Words Corpus (MSWC; Mazumder et al., 2021) supplies word-aligned
clips in 50 languages (German is a high-resource split). We document the resulting dataset following
Datasheets for Datasets (Gebru et al., 2018).

## 3. System: vocabulary and grammar

The command vocabulary is grounded in the camper's actually-controllable functions. The final set is
**4 devices**, one of which (lighting) carries **zones** and **brightness levels**:

| Device | Actions | Zones |
|---|---|---|
| Licht (lights) | an, aus, heller, dunkler, 25/50/75/100 % | Küche, Dach, Außen, Lesen |
| Kühlschrank (fridge) | an, aus, leise | — |
| Heizung (heater) | an, aus, wärmer, kälter | — |
| Aufstelldach (pop-up roof) | auf, zu | — |

A **pure, device-specific slot grammar** parses an ordered keyword-event sequence into an intent
`(device, zone?, action)`. Validity is *learned by no model*: the grammar rejects out-of-order
sequences, missing slots, an action not allowed for its device ("Aufstelldach an"), and a zone on a
non-zoned device ("Heizung Küche"). Keeping validity in a small pure function makes it exhaustively
unit-testable and portable to firmware unchanged. Words that denote no controllable action (the water
tank is a level *readout*, not a control) are excluded — grounded in the control logic, not guessed.

## 4. Dataset

### 4.1 Coverage study (why TTS is structural)

We streamed ~2.5 M MSWC-German examples. Of the 24 grounded command words, **only 7 have real
clips** (Licht, Kühlschrank, Heizung, aus, auf, Außen[158], and Wasser[300, found on a deeper scan]);
**17 — all zone words, "an", and every level/mode word — have zero**. Synthetic fill is therefore
*structural*, not a shortcut: a slot vocabulary drawn from control semantics simply is not a subset
of a read-speech corpus's frequent-word list.

### 4.2 Construction and provenance

Real clips come from MSWC (CC-BY-4.0); missing words are filled with offline German TTS
(macOS `say`; Piper neural voices). Noise augmentation uses ESC-50. **Every experiment reports
per-word real-vs-TTS counts**, and **headline accuracy is computed on real speech only** — TTS-only
words are flagged and excluded from the real-speech number, because a synthetic voice is not a
speaker. This is the paper's central honesty discipline.

### 4.3 Splits and reproducibility

Splits are **speaker-disjoint**: real words split by `speaker_id`; TTS words by a synthetic speaker
id (`engine:voice:rate`) so a voice never straddles train and test. The reusable dataset (in
progress) adds a **validation split** (so model selection never touches test), a **manifest**
(per-word counts, config, content hashes — verifiable without shipping audio), a **deterministic
rebuild** from one seed, and a **datasheet** (Gebru et al.) stating that 15 of the 21 command words
are synthetic-only and that a real-microphone benchmark remains future work.

### 4.4 Statistics and examples

The frozen v2 feature set (seed 0) holds **28 259 one-second examples** over 23 classes: 21 command
words plus `_unknown_` (other MSWC-German words, at most a few clips per word so no single
distractor dominates) and `_silence_` (noise-only). Each source clip yields four rows — a randomly
time-shifted clean copy and ESC-50 noise mixes at 20, 10 and 0 dB SNR — so 1 200 rows ≈ 300 source
clips per word. Splits are speaker-disjoint (§4.3):

| Split | Rows | Real | TTS |
|---|---|---|---|
| train | 20 116 | 6 544 | 13 572 |
| val | 4 101 | 1 241 | 2 860 |
| test | 4 042 | 1 186 | 2 856 |

Provenance per word (rows, all splits):

| Provenance | Words | Rows per word |
|---|---|---|
| real only (MSWC) | Licht, Kühlschrank, aus, auf | 1 200 real |
| mixed | Heizung (480 real + 720 TTS), Außen (632 + 568) | 1 200 |
| TTS only | Aufstelldach, Küche, Dach, Lesen, an, zu, heller, dunkler, wärmer, kälter, leise, fünfundzwanzig, fünfzig, fünfundsiebzig, hundert | 1 200 TTS |
| non-command | `_unknown_` 2 400 real, `_silence_` 659 real | – |

Real-speech numbers exclude every TTS row; in this set that leaves six words and the two
non-command classes, while the fifteen TTS-only words shape the confusion structure but never the
headline number. Synthetic speakers carry ids such as `tts:piper:de_DE-thorsten-medium`
(v2 ids also carried the speaking rate) so the split holds out whole voices.

Concise examples of what the system sees and decides:

- **Word clip → class.** A 1 s clip of "Kühlschrank" → MFCC 49×10 → class `Kühlschrank`; an MSWC
  clip of "Fenster" → `_unknown_`; 1 s of ESC-50 rain → `_silence_`.
- **Event sequence → intent (grammar, §3).** `[Licht, Küche, an]` →
  `Intent(device=Licht, zone=Küche, action=an)`; `[Heizung, wärmer]` →
  `Intent(Heizung, –, wärmer)`; `[Licht, fünfzig]` → `Intent(Licht, –, fünfzig)` (50 %).
- **Rejections (no model involved).** `[Kühlschrank, heller]` → `Rejection("heller invalid for
  Kühlschrank")`; `[Küche, an]` → `Rejection("zone out of order")`.
- **Catalog (E3).** The grammar's 49 valid intents — every (device, zone?, action) the van can
  execute — synthesised as spoken phrases per voice, e.g. "Licht Außen aus", "Aufstelldach zu".

## 5. Method

**Front-end.** 16 kHz, 1 s clips → MFCC (30 ms periodic-Hann window / 20 ms hop, 480-point FFT
(241 bins), 40 Slaney mel bands, log with an 80 dB floor, DCT-II, 10 cepstra → a 49×10 feature map).
The host front-end is librosa; the device front-end is a table-driven C port (window, mel and DCT
matrices generated from the same Python configuration) pinned to the host by a fixed-input
**golden-vector test** — the classic silent train/deploy mismatch becomes a unit test.

**Model architecture.** The deployed command recogniser is a depthwise-separable CNN in the
"Hello Edge" family (Zhang et al., 2017), kept deliberately plain so every layer lowers to a
TFLite-Micro builtin:

| Stage | Layer | Output | Params |
|---|---|---|---|
| input | MFCC map (frames × cepstra × 1) | 49×10×1 | – |
| stem | Conv2D 3×3, 32 filters, no bias → BatchNorm → ReLU | 49×10×32 | 288 + 128 |
| block ×3 | DepthwiseConv2D 3×3 → BN → ReLU → Conv2D 1×1, 32 filters → BN → ReLU | 49×10×32 | 3 × (288 + 128 + 1 024 + 128) |
| pool | global average over time and cepstra (`MEAN`) | 32 | – |
| head | Dense 32 → 23, softmax | 23 | 759 |

5 879 parameters, 2.07 M MACs per 1 s window (the 1×1 pointwise convolutions dominate:
49·10·32·32 ≈ 0.5 M each). There is no striding, pooling or dropout inside the stack — the feature
map stays at full 49×10 resolution and the receptive field grows only through the four 3×3 stages
(9 frames ≈ 190 ms of context per output cell, with global pooling supplying the rest). BatchNorm is
folded into the convolutions at export; the INT8 graph uses exactly `CONV_2D`, `DEPTHWISE_CONV_2D`,
`MEAN`, `FULLY_CONNECTED` and `SOFTMAX` and is 20.2 KB. Output classes are the 21 command words plus
`_unknown_` and `_silence_` (§3). Three alternative encoders (MatchboxNet, BC-ResNet, the Keyword
Transformer) are benchmarked against it in §6.7.

**Teacher and distillation (E9).** The teacher is a Keyword Transformer (Berg et al., 2021): each of
the 49 MFCC frames is linearly projected to d = 64, a learned class token is prepended and learned
positional embeddings added, then 3 pre-LayerNorm encoder blocks (4-head self-attention with key
dimension 16, MLP 64 → 128 → 64 with ReLU, residuals) and a final LayerNorm; the class-token output
feeds a Dense softmax (106 k parameters, float only — its attention/LayerNorm ops do not lower to
INT8 TFLite-Micro, §6.7). The student is the unchanged DS-CNN above, trained on
α · CE(y, p_s) + (1 − α) · T² · KL(p_t^T ‖ p_s^T) with T = 4, α = 0.5, where p^T denotes the
softmax at temperature T (Hinton et al., 2015).

**Quantization and budget gates.** Full-INT8 post-training quantization with a class-balanced
representative set (E10), exported to TFLite-Micro (ESP-NN kernels). **CI budget gates** assert
model ≤ 500 KB, MACs ≤ 3 M, INT8-only I/O, and that every op is device-runnable — "fits the MCU" as
a test, no hardware required. Batch size 32 through E8, 128 from E9 on (throughput; §notes).

**Two-stage runtime.** An always-on **wake detector** ("Hey Bus", microWakeWord) gates the heavier
**command recogniser**, which runs streaming inside the post-wake window. The command model's
per-window posteriors are decoded by **edge-triggered run-based decoding**: a run of consecutive
steps sharing the same qualifying top-1 label fires that label *once* when the run reaches
`min_consecutive` steps, with **no global cooldown** — a different label may fire immediately, and
the same label refires only after ≥ `gap_steps` non-matching steps. This replaced a level-triggered
threshold + global-refractory scheme that conflated *same-word debounce* and *next-word gating* into
one knob and could satisfy neither (§6.2). Emitted events feed the pure grammar (§3).

**Wake word, trained locally.** microWakeWord is documented as Python-3.10 + Colab/GPU; we trained a
custom German "Hey Bus" model **end-to-end on an M4 laptop** (CPU/Metal) in ~6m41s, positives from
Piper TTS.

## 6. Experiments and results

All numbers are from committed evaluation reports. "Catalog full-intent accuracy" scores every valid
command end-to-end (synthesised audio → MFCC → streaming detector → grammar → intent); one wrong or
missed word anywhere fails the whole entry — a strictly harder metric than per-word accuracy.

### 6.1 E1 — single-word recognition (real speech)

| Metric | Value |
|---|---|
| **Headline real-speech INT8 accuracy** (TTS excluded) | **91.1 %** (n = 775, mixed 20/10/0 dB) |
| Float, same subset | 91.7 % |
| Full-model INT8 (incl. TTS-augmented classes) | 93.2 % |
| `_unknown_` false-accept | 0.0 % |
| SNR sweep (commands, INT8) | 93.0 % clean → 84.6 % @ 0 dB |
| Model | 19,256 B · 2,069,984 MACs · full-INT8 · 5,351 params |
| Reference | ESP-SR MultiNet ~85–95 % English, clean |

The real-speech INT8 number sits in MultiNet's English range while being measured under noise (a
harder condition) and in German. **Data mattered more than modelling:** finding real "Wasser" clips
on a deeper corpus scan moved the headline 87.9 % → 91.1 % with no model change.

### 6.2 E3 — end-to-end command catalog, and an ablation arc

The end-to-end catalog on the streaming-plus-grammar system produced the paper's most instructive
result — a five-point arc, each step eliminating one named cause:

| Stage | Catalog full-intent | Cause eliminated |
|---|---|---|
| Initial | **0.000** | dataset asymmetry — model learned the noise floor, not words (§6.4) |
| + symmetric domains, ±200 ms shift | **0.066** | data fixed; isolated words now 0.99+ |
| + edge-triggered decoder | **0.362** | same-word re-fire, swallowed next word, 1-step ghosts |
| + naive transition-window negatives | **0.197** *(regression)* | over-corrected → recall loss (§6.3) |
| + **balanced** transition negatives + class-weights | **0.689** | recovers recall; zone slot 0.156 → 0.789 |

The final model (23 classes, 20 KB INT8) is strong on the zoned/levelled **Licht** commands (per-entry
0.75–1.0, including the new brightness words) but weak on some non-Licht devices (**Kühlschrank 0.00**
despite 300 real clips — consistent with the streaming detector segmenting the long compound word).
We report this openly: the catalog number is lighting-dominated, and per-device robustness is the
main open problem.

### 6.3 A useful negative result

Adding word-**boundary** windows to training (labelled `_unknown_`) to suppress the "ghost word at a
boundary" failure *reduced* catalog accuracy 0.362 → 0.197. A probe showed why: the transition
`_unknown_` negatives out-weighted each word class ~5×, the model predicted `_unknown_` 17.4 % vs
7.7 % true, and clip-level accuracy fell 88.5 % → 78.1 %. It over-corrected toward "say nothing",
killing recall — worst on multi-word phrases. **Balancing** the negatives (cut to ~⅓ the volume) plus
inverse-frequency class weighting reversed it decisively (→ 0.689). Lesson: *the obvious data-side fix
for boundary ghosts hurts unless the added negatives are class-balanced.*

### 6.4 The bug the end-to-end metric caught

Initially, `build_dataset` added command clips **only noise-mixed** but `_unknown_` clips **only
clean**. The model learned a trivial shortcut — "clean ⇒ `_unknown_`, noisy ⇒ some command" —
orthogonal to recognising words. **Per-clip held-out accuracy was 88.5 %**, because the held-out set
shared the same asymmetry: a perfectly consistent, perfectly wrong signal. The **end-to-end catalog
eval scored 0.000** and exposed it; the giveaway was the SNR sweep *improving* as noise worsened.
This is the case for composition-level evaluation: a per-clip metric that inherits the training
data's flawed assumption cannot see the flaw.

### 6.5 E4 — wake word, local training

| Metric | Value |
|---|---|
| Model | 62,304 B INT8 streaming TFLite (passes ≤ 150 KB wake budget) |
| Training | full 10,000-step config, ~6m41s on M4 CPU/Metal |
| Positives | 2,000 synthetic (Piper `de_DE-mls-medium`) |
| Best checkpoint | recall 71.6 %, precision 100 %, ~2.9 FA/hour |
| @ cutoff 0.99 | false-reject 0.39, **2.0 false-accepts/hour** |

First-pass and untuned (39 % miss at the low-FA operating point; 100 % synthetic positives), but it
establishes feasibility with numbers: a **custom German wake word trains on a laptop in minutes**.

### 6.6 E5 — voice-diversity ablation (null in-domain; generalization untested)

Hypothesis: for synthetic KWS data, **voice diversity** dominates per-voice fidelity. We added
Piper neural voices to the `say`-only training data (the 17 TTS-only words), all else fixed.
Result: catalog full-intent accuracy was **unchanged at 0.689**, and the weak non-Licht devices
were unmoved (Kühlschrank 0.00, Heizung ≈ 0.25). **Methodological catch:** the catalog test
synthesises phrases with the same `say` voices present in *both* trainings, so this measures
**in-domain** accuracy — not the **cross-voice generalization** the hypothesis targets. A fair test
requires **voice-disjoint** train/test (whole voices held out) — precisely what the reusable
dataset's speaker-disjoint splits (§4.3, Phase 0) are built to provide. Honest reading: *no
in-domain gain from diversity here; the generalization claim is untested until a voice-held-out
protocol runs.* The multi-engine TTS infrastructure is now in place for that test. This null result
is itself a lesson — a diversity ablation with an in-domain test set cannot detect the effect it
is designed to measure.

### 6.7 E7 — architecture benchmark

Four small-footprint encoders on the frozen dataset (§4.3), trained identically (30 epochs,
seed 0, class-weighted, selected on val, reported on test). Catalog numbers use a 3-voice subset
(147 trials/arch) on the clean dataset (no transition augmentation), so they are lower than the
tuned frame-classifier of §6.2 — the value here is the apples-to-apples ranking, not the absolute
catalog number.

| Architecture | Isolated | Catalog | Params | MACs | INT8 | Device-runnable |
|---|---|---|---|---|---|---|
| DS-CNN | 0.834 | **0.544** | 5,879 | 2.07 M | 20 KB | ✓ |
| BC-ResNet | 0.773 | 0.102 | 4,919 | 1.39 M | 31 KB | ✓ |
| MatchboxNet | **0.903** | 0.245 | 12,957 | **0.47 M** | 43 KB | ✓ |
| Keyword-Transformer | — | — | 106 k | — | 173 KB | **✗ (non-TFLM ops)** |

Findings: (i) the **ranking depends on the metric** — MatchboxNet wins isolated-word accuracy and
is the most MAC-efficient (0.47 M), yet DS-CNN wins the end-to-end catalog; again, isolated accuracy
is not the task. (ii) BC-ResNet, strong on Google-Speech-Commands in the literature, underperforms
at this tiny scale / 30-epoch budget — a caution against importing leaderboard rankings unchanged.
(iii) the **Keyword-Transformer INT8-exports but is not device-runnable**: its attention ops
(BATCH_MATMUL, GATHER, TRANSPOSE, …) fall outside the TFLM/ESP-NN kernel set — a concrete reminder
that "small and accurate" is necessary but not sufficient for an MCU; the **op-set is the gate**,
which is exactly why our budget test asserts it. All CNN encoders fit the budget. We select
**MatchboxNet** as the streaming-CTC encoder for §6.8 (best isolated accuracy, lowest MACs, and a
time-native 1-D structure).

### 6.8 E8 — streaming CTC transducer (negative/preliminary result)

The literature's fix for connected commands is a streaming sequence model that transcribes the
keyword sequence and learns alignment natively, rather than a frame classifier + a hand-rolled
decoder. We built a **streaming CTC** recogniser: the MatchboxNet encoder (stride-1 in time, no
global pool) + a per-frame CTC head over `blank + the 21 keyword tokens`, trained with
`tf.nn.ctc_loss` on **392 synthesised `device [zone] action` phrases** (8× the catalog, from
train-split clips only), 60 epochs, decoded greedily into the same pure grammar.

The result is a negative one, and instructive. Training loss fell (370 → 28), yet greedy decoding
**collapsed to empty sequences** — the model emits near-all-blank — so catalog full-intent was
**0.000** vs the frame-classifier's 0.689. We separate the two causes, because they resolve very
differently:

1. **CTC is data-hungry (open).** 392 phrases is far too few to learn alignment for 21 tokens;
   all-blank collapse is the classic small-data CTC failure. Generating orders-of-magnitude more
   phrase data — a strength of the synthetic pipeline — is the first fix, and the reason the
   accuracy number above is preliminary rather than a verdict on the architecture.
2. **The op-set/export blocker (resolved).** As first built, the streaming head was a
   `TimeDistributed(Dense)` that unrolled into a `tf.while` loop and emitted a `TensorListReserve`
   op the INT8-builtins-only converter (no `SELECT_TF_OPS`) could not legalize — the same op-set
   gate that ruled out the Keyword Transformer (§6.7). Replacing it with a **1×1 Conv2D per-frame
   head** (identical per-frame projection, one static `CONV_2D`) and exporting a **fixed-T,
   batch-1 clone** — the honest on-device shape, one chunk at a time behind a ring buffer, with
   weights transferred from the variable-length trained model — makes the encoder export at
   **42.9 KB, full INT8, with every op inside the TFLM builtin set** (`CONV_2D`,
   `DEPTHWISE_CONV_2D`, `ADD`, `RESHAPE`). The streaming transducer is therefore now
   **device-runnable**; only the data-scale gap (cause 1) stands between it and a real comparison.

The **frame-classifier + grammar (0.689, 20 KB INT8) remains the working, deployable system today**;
the streaming transducer is now a device-runnable but under-trained direction whose one remaining
blocker this experiment names precisely — phrase-data scale.

### E9 — Knowledge distillation (KWT → DS-CNN)

On the frozen v2 split (§4.3), we distilled the KWT teacher (§6.7; INT8-exports but is not
device-runnable) into the unchanged DS-CNN student (`kws_de.distill.distill`, `T=4.0`, `alpha=0.5`,
40 epochs, seed 0, batch 128 — see
`docs/superpowers/specs/2026-09-01-real-speech-distill-design.md` §4). Teacher float test accuracy:
**0.894**.

| Architecture | Float | Isolated | Catalog | Params | MACs | INT8 | Budget |
|---|---|---|---|---|---|---|---|
| ds_cnn (first-200 calib) | 0.862 | 0.842 | 0.218 | 5,879 | 2,070,496 | 20,224 | yes |
| ds_cnn (balanced calib) | 0.862 | 0.853 | 0.259 | 5,879 | 2,070,496 | 20,224 | yes |
| ds_cnn distilled (balanced calib) | 0.842 | 0.833 | 0.667 | 5,879 | 2,070,496 | 20,272 | yes |

Distillation did **not** beat the baseline on the isolated per-clip metric — float accuracy fell
0.862 → 0.842 and INT8-isolated fell 0.853 → 0.833 (both −2.0 points, balanced calibration on both
rows). But it produced the paper's largest single-change win on the metric that reflects the
deployed system: full-intent **catalog accuracy rose 0.259 → 0.667**, +40.8 points, 2.6× the
undistilled baseline. This echoes §6.2/§6.7's lesson that isolated accuracy is not the task — the
KWT teacher's softened targets appear to move the student's decision boundaries in a way that
materially helps stream+grammar composition (fewer boundary-transition ghosts, §6.2/§6.3) even
while very slightly hurting raw per-clip accuracy. We report this honestly as a system-level win,
not an unqualified win on every metric.

### E10 — INT8 calibration

Rows 1–2 above differ only in the PTQ calibration set (`X_train[:200]` vs class-balanced
`kws_de.export.balanced_calibration`, spec §5). Float→INT8 gap: first-200 calibration 0.862 → 0.842
(2.0 points); balanced calibration 0.862 → 0.853 (0.9 points) — balanced calibration recovers 1.1 of
the 2.0-point gap (55 %) on this run. (E7's originally reported 1.63-point gap, 0.8661 → 0.8498, was
measured on an earlier ds_cnn run at 30 epochs/batch 32; this run's own first-200 row, 40
epochs/batch 128, is the baseline the recovery above is measured against.)

Per the spec's decision gate (§5): a hand-rolled fake-quant QAT becomes the next spec only if the
balanced-calibration gap exceeds 1 % absolute. The measured balanced gap is **0.9 %**, under the
threshold, so **QAT is closed as unnecessary** — balanced calibration alone is sufficient.

## 7. Discussion

The two-stage split — a cheap always-on wake gate before a heavier command model — is what makes
the always-on power budget plausible, and keeping intent validity in a pure grammar rather than the
model made the system testable and portable. The results argue two methodological points beyond the
specific numbers: **honest provenance** (real-speech-only headlines with per-word TTS labelling)
prevents synthetic-data self-deception, and **end-to-end evaluation** catches construction bugs that
per-clip metrics structurally cannot. The ablation arc (§6.2) is itself a contribution: a reproducible
sequence in which each fix isolates one failure mode, including a negative result. The
architecture benchmark (§6.7) sharpens the deployability point — the Keyword Transformer is small
and accurate yet not device-runnable, so the **op-set, not the parameter count, is the true MCU
gate** — and the streaming transducer (§6.8), the literature's connected-command fix, once its head
was reformulated to clear that same op-set gate, exports device-runnable but does not yet beat the
frame-classifier at our data scale, a reminder that a principled architecture still needs enough
data to win *on-device*.

## 8. Limitations and future work

- **Synthetic data.** 15 of 21 command words are TTS-only; those per-word numbers are not real-speech
  performance. A recorded real-speaker (and real-microphone) test set is planned.
- **Per-device robustness.** The catalog result is lighting-dominated; long compound words
  (Kühlschrank) fail under the current frame-classification + hand-rolled decoder.
- **The streaming transducer (E8): export solved, data open.** The export blocker is closed — a
  1×1-Conv2D per-frame head plus a fixed-T, batch-1 export makes the encoder device-runnable
  (42.9 KB, full INT8, TFLM builtins only). The remaining blocker is data: the naive CTC model
  collapsed to all-blank at 392 phrases. The follow-up is concrete — generate orders-of-magnitude
  more phrase data (the synthetic pipeline scales), possibly with a label-prior or entropy
  regulariser to avoid blank collapse, then re-run the on-device comparison. RNN-T (implicit LM) is
  the further step.
- **Sim-to-real (E6) is the biggest open item.** Measured on-device latency/RAM/power and
  real-microphone accuracy on the CoreS3 (dual-MEMS + ES7210 + ESP-SR 2-mic AFE) are specified as a
  follow-up; we expect real accuracy below the synthetic eval and see quantifying that gap as a result.

## 9. Conclusion

A German, intent-level voice assistant can run fully offline on a commodity ESP32-S3 with open
tooling: a 19–20 KB INT8 model, a tiny locally-trained wake word, and a pure slot grammar, reaching
91.1 % real-speech single-word accuracy and a lighting-dominated 0.689 end-to-end command-catalog
accuracy. The contributions we most want to travel are methodological — honest synthetic-data
provenance, end-to-end evaluation that catches what per-clip metrics miss, budget gates that make
"fits the MCU" a test, and reproducible ablations (with negative results that name their own next
steps) — plus a documented, reusable German MCU-KWS dataset. We also benchmarked four encoders on
that dataset (MatchboxNet the most MAC-efficient; the Keyword Transformer accurate but not
device-runnable) and built a streaming CTC transducer which — once its head was reformulated to a
1×1-Conv2D and exported at fixed length — runs device-side at 42.9 KB full INT8, leaving phrase-data
scale as the one remaining thing a connected-command model must fix here. Distilling the KWT teacher
into the deployed DS-CNN (E9) traded a small isolated-accuracy loss for a large catalog-accuracy
gain (0.259 → 0.667), and class-balanced PTQ calibration (E10) recovered the INT8 gap to 0.9
points — under the spec's 1 %-gate, closing QAT as unnecessary for now.
Everything is open-source at the repository above.

## References

1. Zhang et al. *Hello Edge: Keyword Spotting on Microcontrollers.* arXiv:1711.07128.
2. Kim et al. *Broadcasted Residual Learning for Efficient Keyword Spotting.* arXiv:2106.04140.
3. Majumdar & Ginsburg. *MatchboxNet.* arXiv:2004.08531.
4. Berg et al. *Keyword Transformer.* arXiv:2104.00769.
5. He et al. *Streaming Small-Footprint KWS with Sequence-to-Sequence Models.* arXiv:1710.09617.
6. *MFA-KWS* (CTC-Transducer). arXiv:2505.19577.
7. Rybakov et al. *Streaming Keyword Spotting on Mobile Devices.* arXiv:2005.06720.
8. *Advances in Small-Footprint KWS: A Comprehensive Review.* arXiv:2506.11169.
9. Mazumder et al. *Multilingual Spoken Words Corpus.* NeurIPS Datasets & Benchmarks, 2021.
10. Warden. *Speech Commands.* arXiv:1804.03209.
11. Jacob et al. *Quantization and Training for Integer-Arithmetic-Only Inference.* arXiv:1712.05877.
12. David et al. *TensorFlow Lite Micro.* arXiv:2010.08678.
13. Gebru et al. *Datasheets for Datasets.* arXiv:1803.09010.
