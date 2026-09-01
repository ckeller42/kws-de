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
rebuild** from one seed, and a **datasheet** (Gebru et al.) stating that 17/23 words are synthetic
and that a real-microphone benchmark remains future work.

## 5. Method

**Front-end.** 16 kHz, 1 s clips → MFCC (30 ms window / 20 ms hop, 40 mel, 10 cepstra → 49×10). The
host (librosa) and device (esp-dsp) front-ends are pinned bit-for-bit by a fixed-input **golden-vector
test** — the classic silent train/deploy mismatch becomes a unit test.

**Model and quantization.** A small DS-CNN (~5 k params), full-INT8 via representative-dataset
calibration, exported to TFLite-Micro (ESP-NN kernels). **CI budget gates** assert model ≤ 500 KB,
MACs ≤ 3 M, INT8-only I/O, and that every op is device-runnable — "fits the MCU" as a test, no
hardware required.

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

### 6.8 E8 — streaming CTC transducer (the new model) *(in progress)*

The literature's fix for connected commands is a streaming sequence model that transcribes the
keyword sequence and learns alignment natively, rather than a frame classifier + a hand-rolled
decoder. We build a small **streaming CTC** recogniser (MatchboxNet encoder + a per-frame CTC head
over `blank + the keyword tokens`), decode greedily into the same pure grammar, and evaluate on the
same catalog against the frame-classifier baseline. *(Result pending the training run; reported here
when it lands — the target is the connected-command failures, especially the long-word and
boundary-ghost cases §6.2 left open.)*

## 7. Discussion

The two-stage split — a cheap always-on wake gate before a heavier command model — is what makes
the always-on power budget plausible, and keeping intent validity in a pure grammar rather than the
model made the system testable and portable. The results argue two methodological points beyond the
specific numbers: **honest provenance** (real-speech-only headlines with per-word TTS labelling)
prevents synthetic-data self-deception, and **end-to-end evaluation** catches construction bugs that
per-clip metrics structurally cannot. The ablation arc (§6.2) is itself a contribution: a reproducible
sequence in which each fix isolates one failure mode, including a negative result.

## 8. Limitations and future work

- **Synthetic data.** 17 of 23 words are TTS-only; those per-word numbers are not real-speech
  performance. A recorded real-speaker (and real-microphone) test set is planned.
- **Sim-to-real (E6).** All performance figures are *estimated* (MAC→cycle) or *proxied* (budget
  gates). Measured on-device latency/RAM/power and clean-corpus-vs-real-mic accuracy on the CoreS3
  (dual-MEMS + ES7210 + ESP-SR 2-mic AFE) are specified as a follow-up; we expect real accuracy below
  the synthetic eval and see quantifying that gap as a result.
- **Per-device robustness.** The catalog result is lighting-dominated; long compound words
  (Kühlschrank) fail under the current frame-classification + hand-rolled decoder.
- **Architecture (planned).** A benchmark (DS-CNN / BC-ResNet / MatchboxNet / Keyword-Transformer)
  on the reusable dataset, and a **streaming CTC/RNN-T transducer** — the literature-backed fix for
  connected commands — are specified as the next phases; the transducer would replace the hand-rolled
  decoder with a learned aligner that handles boundaries and long words natively.

## 9. Conclusion

A German, intent-level voice assistant can run fully offline on a commodity ESP32-S3 with open
tooling: a 19–20 KB INT8 model, a tiny locally-trained wake word, and a pure slot grammar, reaching
91.1 % real-speech single-word accuracy and a lighting-dominated 0.689 end-to-end command-catalog
accuracy. The contributions we most want to travel are methodological — honest synthetic-data
provenance, end-to-end evaluation that catches what per-clip metrics miss, budget gates that make
"fits the MCU" a test, and a reproducible ablation (with a negative result) — plus a documented,
reusable German MCU-KWS dataset. Everything is open-source at the repository above.

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
