# Offline German Voice Control on a Microcontroller: Keyword Spotting with a Slot Grammar on the ESP32-S3

**Project:** kws-de — <https://github.com/ckeller42/kws-de>

Every number below comes from a committed evaluation report or an on-device measurement in this
repository, and names the model, the audio and the split policy behind it.

---

## Abstract

We present an open, reproducible pipeline for **German**, **intent-level** voice control that runs
**fully offline on a microcontroller** (ESP32-S3, M5Stack CoreS3), and report what it does on the
target hardware. The gap is concrete: the vendor stack (ESP-SR/MultiNet) ships Chinese and English
only, and the one on-device German speech-to-intent engine (Picovoice Rhino) is closed-source. A
tiny always-on wake model gates a streaming keyword detector whose events a **pure, device-specific
slot grammar** (`device → zone? → action`) composes into a validated intent; the deployed 23-class
command model is **17,880 B** full-INT8 at 2.07 M MACs, one streaming step measures **45 ms** on the
device, and wake-gating takes always-on inference cost from 315 to **97 ms per wall second** at an
assumed one interaction per 10 s. Our headline result is a gap, not a peak: the stock model scores
≈0.9 on its own synthetic held-out split but recognises **0.19 and 0.27** of two speakers' real
microphone speech; folding those speakers' own recordings into training raises them to **0.615 and
0.737** as *user-customised, in-training* figures, never quoted as generalisation. Four negative
results carry
the method: a wake model that learned *TTS-vs-real* rather than the phrase, invisible to its own
held-out recall; a transition-window regression; a voice-diversity ablation that fell 0.689 → 0.245
while gaining +0.56 on one device; and a width sweep that buys parameters with real-voice accuracy.

## 1. Introduction

Camper vans and similar always-on edge settings need hands-free control (lights, heater, fridge,
pop-up roof) where cloud connectivity is intermittent and a Linux single-board computer cannot stay
awake. Doing this in **German** at the **intent** level (not just a wake word, but "Licht Küche an" →
`turn on the kitchen light`) on an **MCU** with **open** tooling is unserved: ESP-SR/MultiNet [10]
recognises fixed command sets but ships **Chinese and English only**; Picovoice Rhino [11] does
on-device German speech-to-intent but is **closed-source**; the open alternatives (Rhasspy [12],
Vosk, DeepSpeech [13]) are speech-to-*text*, Pi-class, not MCU slot filling.

**Contributions.** (1) A reproducible **model factory** for German KWS on the ESP32-S3: public
corpus → MFCC → depthwise-separable CNN → full-INT8 TFLite-Micro with CI budget gates, **running on
the target hardware**. (2) **Honest provenance** for synthetic data, extended to the test axis: every
number names its audio and split policy, and real-speech figures carry two never-mixed labels,
*held-out* and *user-customised, in-training*. (3) A two-stage **open** architecture: a reused wake
engine gating a streaming detector plus a pure slot grammar. (4) A device-to-dataset **recording and
QC loop**, and four negative results that each name their own cause.

## 2. Related work

**Small-footprint KWS.** DS-CNN established depthwise-separable convolutions as the MCU-KWS
workhorse [1]; later encoders push accuracy at tiny parameter counts — BC-ResNet [2],
MatchboxNet [3], the Keyword Transformer [4]. Speech Commands [14] is the field's benchmark,
MLPerf Tiny [15] its resource-constrained harness, and a survey collects the area [8].

**Streaming and connected KWS.** Recognising *sequences* of words in continuous speech is addressed
by streaming sequence models — sequence-to-sequence [5], CTC and RNN-T transducers, CTC-transducer
hybrids [6] — and streaming conversion of non-streaming models by Rybakov et al. [7].

**Synthetic, low-resource and personalised KWS.** Topping up limited real speech with synthesised
speech is established practice [16], and few-shot multilingual KWS is the standard answer when a
target word has no corpus coverage [17]. Ours is the extreme case — 15 of 21 command words have *no*
real clips — so what we measure is the synthetic-to-real gap itself, and we close it for the device's
main users rather than for an arbitrary speaker, labelling the numbers accordingly.

**On-device stacks, SLU and data.** ESP-SR/MultiNet [10] and microWakeWord [9] target the ESP32-S3;
Rhino [11] is the closed reference for on-device German speech-to-intent, and edge SLU with slot
filling has been shown at Pi class [18]. Integer-only inference and quantization-aware training
follow Jacob et al. [19]; the deployable op set is TFLite-Micro's [20]. MSWC [21] supplies real
clips, ESC-50 [22] noise, macOS `say` and Piper [23] synthetic fill; the dataset follows Datasheets
for Datasets [24].

## 3. System: vocabulary and grammar

The vocabulary is grounded in the camper's actually-controllable functions: **4 devices**, one of
which (lighting) carries **zones** and **brightness levels**.

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
unit-testable and portable to firmware unchanged. Words denoting no controllable action (the water
tank is a level *readout*) are excluded.

## 4. Dataset

### 4.1 Coverage study (why TTS is structural)

We streamed ~2.5 M MSWC-German examples. Of the 24 grounded command words, **only 7 have real clips**
(Licht, Kühlschrank, Heizung, Wasser, aus, auf, Außen); **17 — all zone words, "an", every level and
mode word — have zero**. Synthetic fill is *structural*: a slot vocabulary drawn from control
semantics is not a subset of a read-speech corpus's frequent-word list.

### 4.2 Construction and provenance

Real clips come from MSWC (CC-BY-4.0) and, from v3 on, the device's own recorder (§4.5); missing
words are filled with offline German TTS, noise augmentation with ESC-50. **Every experiment reports
per-word real-vs-TTS counts** and **headline single-word accuracy is computed on real speech only** —
a synthetic voice is not a speaker. §6 extends the discipline to the test axis.

### 4.3 Splits and reproducibility

Splits are **speaker-disjoint**: real words by `speaker_id`, TTS words by a synthetic speaker id. In
the frozen v2 features behind every §6.1–§6.9 number that id was `tts:{engine}:{voice}:{rate}`, so
*the same voice at two speaking rates could land in both train and test*. This is a real leak,
flagged wherever a v2 number is quoted; v3 closes it by dropping rate from the id. The dataset also
carries a **validation split**, a **manifest** (per-word counts, config, content hashes — verifiable
without shipping audio), a deterministic rebuild from the pinned clip cache rather than from
re-synthesis, and a **datasheet** [24].

### 4.4 Statistics

The frozen v2 feature set (seed 0) holds **28,259 one-second examples** over 23 classes: 21 command
words plus `_unknown_` (other MSWC-German words) and `_silence_` (noise-only).

| Split | Rows | Real | TTS |
|---|---|---|---|
| train | 20,116 | 6,544 | 13,572 |
| val | 4,101 | 1,241 | 2,860 |
| test | 4,042 | 1,186 | 2,856 |

| Provenance | Words | Rows per word |
|---|---|---|
| real only (MSWC) | Licht, Kühlschrank, aus, auf | 1,200 real |
| mixed | Heizung (480 real + 720 TTS), Außen (632 + 568) | 1,200 |
| TTS only | Aufstelldach, Küche, Dach, Lesen, an, zu, heller, dunkler, wärmer, kälter, leise, fünfundzwanzig, fünfzig, fünfundsiebzig, hundert | 1,200 TTS |
| non-command | `_unknown_` 2,400 real, `_silence_` 659 real | – |

Real-speech numbers exclude every TTS row. The **catalog** (§6.2) is the grammar's 49 valid intents —
every (device, zone?, action) the van can execute — synthesised as spoken phrases per voice.

### 4.5 The recording and QC loop (method)

Real microphone data is collected by the device and folded back into training by one pipeline: a
**guided recorder** on the CoreS3 (speaker ids `spkNN`, two reads per prompt, energy-VAD
end-pointing) → `scripts/ingest.sh` into a stamped, never-deleted staging tree → `kws-qc`, an audio
gate (format, duration, level, clipping) then a Whisper large-v3 **content gate** that also segments
approved sentence takes into 1 s word clips on Whisper's word spans → `kws-dataset build` → train →
export → `kws-eval --recordings` (§6.11), chained by `scripts/data-loop.sh`. On the first session
(208 takes, two speakers) it approved **65/208 (31 %)**, up from 48/208 — three QC fixes, not a looser
gate: Whisper writes light levels as numerals ("50" for "fünfzig"), glues keywords into one token
("Lichtdach"), and one hallucinated two-letter keyword was rejecting clean negatives; a fourth fix
requiring whole-token matches moved the count *back down* 69 → 65 by removing false approvals. The
recorder produced one finding of its own: sentence takes were rejected 75/102 for missing words
because the VAD's *fixed* 500 ms trailing hangover is shorter than a natural reading pause (median
take 840 ms against 1,020 ms for words), fixed with a per-prompt-set hangover and a 200 ms
minimum-speech filter.

## 5. Method

**Front-end.** 16 kHz, 1 s clips → MFCC (30 ms periodic-Hann window / 20 ms hop, 480-point FFT,
241 bins, 40 Slaney mel bands, log with an 80 dB floor, DCT-II, 10 cepstra → a 49×10 map). The host
front-end is librosa [25]; the device front-end is a table-driven C port generated from the same
Python configuration and pinned to the host by a fixed-input **golden-vector test**, measured
on-device in §6.10.

**Model architecture.** The command recogniser is a depthwise-separable CNN in the "Hello Edge"
family [1], kept plain so every layer lowers to a TFLite-Micro builtin:

| Stage | Layer | Output | Params |
|---|---|---|---|
| input | MFCC map (frames × cepstra × 1) | 49×10×1 | – |
| stem | Conv2D 3×3, 32 filters, no bias → BatchNorm → ReLU | 49×10×32 | 288 + 128 |
| block ×3 | DepthwiseConv2D 3×3 → BN → ReLU → Conv2D 1×1, 32 filters → BN → ReLU | 49×10×32 | 3 × (288 + 128 + 1,024 + 128) |
| pool | global average over time and cepstra (`MEAN`) | 32 | – |
| head | Dense 32 → 23, softmax | 23 | 759 |

5,879 parameters, 2,070,496 MACs per 1 s window; nothing strides or pools inside the stack, so the
receptive field grows only through the four 3×3 stages (≈190 ms of context per output cell). The
INT8 graph uses exactly `CONV_2D`, `DEPTHWISE_CONV_2D`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`; the v2
export is 20,216 B, the model flashed today the v3 quantization-aware export at **17,880 B** (§6.8).
The distillation teacher (§6.7) is a Keyword Transformer [4] (d = 64, 3 pre-LayerNorm blocks, 4
heads, 106 k parameters, float only); the student is this DS-CNN unchanged, trained on
α · CE(y, p_s) + (1 − α) · T² · KL(p_t^T ‖ p_s^T), T = 4, α = 0.5, 40 epochs, seed 0 [26].

**Quantization and budget gates.** Full-INT8 post-training quantization with a class-balanced
representative set, exported to TFLite-Micro [20] (ESP-NN kernels); quantization-aware training,
where used, wraps the architecture with `tensorflow-model-optimization` fake-quant and fine-tunes 10
epochs at Adam(1e-5) from the float weights [19]. **CI budget gates** assert model ≤ 500 KB, MACs
≤ 3 M, INT8-only I/O and a device-runnable op set: they prove **loadability** without hardware and
say nothing about real-time fit, which only the device measures (§6.10). A **model-health gate**
(§6.2) blocks a broken export from reaching the firmware header. Batch size is 32 for §6.5, 128
elsewhere.

**Two-stage runtime.** An always-on **wake detector** ("Hey Bus", microWakeWord [9], trained
end-to-end on an M4 laptop in ~6m41s with no Colab or GPU) is designed to gate the heavier command
recogniser inside the post-wake window; only the small model runs continuously, which is what makes
the always-on power budget plausible — an argument now measured rather than assumed (§6.10). The
wake-gated hand-off runs on the device, but **only its cost has been measured**: every accuracy figure
in §6 comes from the two stages run in isolation, in the firmware's wake-only and recogniser-only
modes. Posteriors are decoded by **edge-triggered run-based decoding**: a run of consecutive steps sharing
the same qualifying top-1 label fires *once* at `min_consecutive` steps, with **no global cooldown**
— a different label may fire immediately, the same label only after ≥ `gap_steps` non-matching steps.
This replaced a level-triggered threshold plus global refractory that conflated *same-word debounce*
with *next-word gating* (§6.2).

## 6. Experiments and results

Three different things get called "accuracy" here and they are not comparable, so each is labelled at
every use:

| Label | Audio | Split policy | Reported in |
|---|---|---|---|
| synthetic held-out, isolated | MSWC + TTS clips, noise-mixed | voice/speaker-disjoint test split (v2 rows carry the §4.3 leak) | §6.1, §6.5–§6.9 |
| synthetic catalog, end-to-end | TTS-synthesised full phrases | test voices also in training — **in-domain** | §6.2, §6.4–§6.7 |
| real speech, held-out | device microphone | speaker has no clip in train | §6.3, §6.11 |
| real speech, user-customised | device microphone | speaker's own clips are in train | §6.3, §6.8–§6.9, §6.11 |

"Catalog full-intent accuracy" scores every valid command end-to-end (audio → MFCC → streaming
detector → grammar → intent); one wrong or missed word fails the entry.

### 6.1 E1 — single-word recognition, v1 vocabulary

The project's first model is **not** the deployed one: it is a **7-class, 5-toggle-word v1 model**
(5,351 params, 19,256 B, 2,069,984 MACs) over a vocabulary including *Camping* and *Wasser*, both
dropped when E3 grounded the vocabulary in the van's controllable functions. On the real-speech
subset of its own held-out split, TTS excluded:

| Metric (v1, 7 classes) | Value |
|---|---|
| Real-speech INT8 accuracy (TTS excluded) | **91.1 %** (n = 775, mixed 20/10/0 dB) |
| Float, same subset | 91.7 % |
| Full-model INT8 (incl. TTS-augmented classes) | 93.2 % |
| `_unknown_` false-accept | 0.0 % |
| SNR sweep (commands, INT8) | 93.0 % clean → 84.6 % @ 0 dB |

That is real German speech under noise on a five-word toggle vocabulary — not the 23-class task, not
the deployed 17,880 B model, and not comparable to MultiNet's vendor-reported ~85–95 % on English
connected commands, which differs in language, task and corpus at once. We quote it for one thing:
**data mattered more than modelling** — finding real "Wasser" clips on a deeper corpus scan moved it
87.9 % → 91.1 % with no model change.

### 6.2 E3 — end-to-end command catalog, and an ablation arc

The catalog produced the paper's most instructive arc, each step eliminating one named cause
(synthetic catalog, in-domain; v2 features, §4.3 leak):

| Stage | Catalog full-intent | Cause eliminated |
|---|---|---|
| Initial | **0.000** | dataset asymmetry — model learned the noise floor, not words |
| + symmetric domains, ±200 ms shift | **0.066** | data fixed; isolated words now 0.99+ |
| + edge-triggered decoder | **0.362** | same-word re-fire, swallowed next word, 1-step ghosts |
| + naive transition-window negatives | **0.197** *(regression)* | over-corrected → recall loss |
| + **balanced** transition negatives + class-weights | **0.689** | recovers recall; zone slot 0.156 → 0.789 |

**Row 1 — the bug the end-to-end metric caught.** `build_dataset` initially added command clips
**only noise-mixed** but `_unknown_` clips **only clean**, so the model learned "clean ⇒ `_unknown_`,
noisy ⇒ some command". **Per-clip held-out accuracy was 88.5 %** because the held-out set shared the
asymmetry, while the **catalog scored 0.000**; the giveaway was the SNR sweep *improving* as noise
worsened. A per-clip metric that inherits the training data's flawed assumption cannot see the flaw —
met again one layer up in §6.3, and now a shipped guard: after a mode-collapsed model (0.3 % accuracy
on its own training set, ~3 classes ever predicted) "succeeded" by exit code and was flashed,
`kws-export` runs `assert_model_healthy` (≥ 50 % held-out accuracy, ≥ 10 predicted classes) before
writing the firmware header.

**Row 4 — a useful negative result.** Adding word-**boundary** windows labelled `_unknown_` *reduced*
catalog accuracy 0.362 → 0.197: the transition negatives out-weighted each word class ~5×, the model
predicted `_unknown_` 17.4 % vs 7.7 % true, and clip-level accuracy fell 88.5 % → 78.1 %
(coincidentally row 1's figure, on a different model). It over-corrected toward "say nothing";
**balancing** the negatives (~⅓ the volume) plus inverse-frequency class weighting reversed it
decisively. Lesson: *the obvious data-side fix for boundary ghosts hurts unless the added negatives
are class-balanced.* The resulting v2 model (20,216 B INT8) is strong on the zoned and levelled
**Licht** commands (per-entry 0.75–1.0) but weak elsewhere (**Kühlschrank 0.00** despite 300 real
clips, plausibly but unprovenly the detector segmenting a long compound); with Licht 40 of 49
entries, 0.689 is lighting-dominated.

### 6.3 E4 — the wake word: a synthetic-vs-real shortcut, and how it closed

**Round 1 (feasibility).** microWakeWord on 2,000 synthetic positives (Piper `de_DE-mls-medium`),
upstream 10,000-step config: 62,304 B INT8 streaming TFLite, val recall **71.65 %**, precision 100 %;
at cutoff 0.99, false-reject 0.39 and 2.0 false-accepts/hour.

**Round 2 — the held-out metric was uninformative.** On the device that model *never fired*: per-2 s
peak probability 0.00–0.13 while a person said "Hey Bus", although the front end is bit-exact against
the trainer's own extractor (§6.10). A host probe through the identical int8 path explained it: the
model output ≥ 0.99 for *any* Piper sentence in its training voice ("licht küche an": 73 steps
≥ 0.99) and ≈ 0.004 for "hey bus" in unseen Piper voices. With all positives synthetic and all
negatives real recordings, the cheapest separating feature was **TTS-vs-real, not the phrase**. The
71.65 % was not a weak number but a meaningless one: a held-out metric drawn from the same synthetic
distribution cannot see a synthetic-vs-real shortcut. This is §6.2's lesson one layer up, and the
reason for the two-label policy of §6.11.

**Round 4** retrained with TTS hard negatives (near-misses, the command vocabulary, everyday
sentences), reverb augmentation and multi-voice positives (9,000 + 9,000 clips, 20k steps, 58,080 B):
3 of 4 unseen probe voices fire (round 1: 1 of 4), device 2 s peak **0.83–0.99** on "Hey Bus" against
≤ 0.44 on silence and room noise, so the gate moved from 0.99 to **0.85 × 2 consecutive steps**.
**Round 5** added ten real "Hey Bus" takes from two sessions of a main user, QC-approved through §4.5,
as their own feature set at sampling weight 5:

| Probe (gate 0.85 × 2 steps) | Round 4 | Round 5 |
|---|---|---|
| Real "Hey Bus" takes fired | 4 of 10 | **10 of 10** (peak 0.996 every clip) |
| One-session variant, on the *other*, unseen session | — | 5 of 5 |
| TTS non-wake worst peak | 0.988 | 0.758 |
| Generic synthetic "hey bus" via laptop speaker | 3 of 3 (0.96–0.99) | 0 of 3 (0.59–0.64) |

The last row is the stated price, **generic-voice margin**: the wake model is customised to the
device's main users, the same policy the command model follows (§6.11), and each further main user's
five takes go through the same loop. The false-accept rate on real conversational speech is
**unmeasured**.

### 6.4 E5 — voice diversity: a net regression containing a large per-device gain

Hypothesis: for synthetic KWS data, **voice diversity** dominates per-voice fidelity. We added four
Piper neural voices to the `say`-only training data, cycling the balanced engine pool back up to the
baseline's clip target so volume is matched (29,442 vs 29,985 train rows). Result (synthetic catalog,
in-domain): **full-intent accuracy fell 0.689 → 0.245** (zone slot 0.789 → 0.219; clip-level held-out
0.866 → 0.779). The effect is far from uniform, and that is the finding:

| Device (share of the 49-entry catalog) | say-only | say + Piper | Δ |
|---|---|---|---|
| Licht (40/49, 82 %) | 0.794 | 0.200 | **−0.594** |
| Kühlschrank (3/49) | 0.000 | 0.000 | 0.000 |
| Heizung (4/49) | 0.188 | **0.750** | **+0.563** |
| Aufstelldach (2/49) | 0.625 | 0.500 | −0.125 |

The hypothesis is confirmed exactly where it was expected to matter — **Heizung**, one of the two weak
non-Licht devices, improved sharply — while the net regression is entirely a Licht story: Licht
carries 82 % of catalog trials and collapsed, swamping that gain. A plausible, unverified mechanism:
one shared 23-class classifier rather than a head per device, so Piper's more varied renderings of the
*other* classes shifted the shared boundary against Licht, whose clips are 100 % real MSWC.
Kühlschrank was flat in both configurations, so that failure was never a diversity problem. Two
caveats: the catalog test uses `say` voices present in *both* trainings, so this measures **in-domain**
accuracy, not the cross-voice generalization the hypothesis targets; and an earlier pass that left the
balanced pool at its natural size collapsed to 0.066 — a **data-volume** confound (17,862 vs 29,442
train rows), not the number reported here.

### 6.5 E7 — architecture benchmark

Four encoders on the frozen v2 dataset (leak included), trained identically (30 epochs, seed 0,
class-weighted, val-selected, test-reported). Catalog numbers use a 3-voice subset (147 trials/arch)
on the clean dataset, so they sit below §6.2's tuned system; the value is the ranking.

| Architecture | Isolated | Catalog | Params | MACs | INT8 | Device-runnable |
|---|---|---|---|---|---|---|
| DS-CNN | 0.834 | **0.544** | 5,879 | 2.07 M | 20,216 B | ✓ |
| BC-ResNet | 0.773 | 0.102 | 4,919 | 1.39 M | 31 KB | ✓ |
| MatchboxNet | **0.903** | 0.245 | 12,957 | **0.47 M** | 43 KB | ✓ |
| Keyword-Transformer | — | — | 106 k | — | 173 KB | **✗ (non-TFLM ops)** |

The **ranking depends on the metric**: MatchboxNet wins isolated accuracy and is the most
MAC-efficient, yet DS-CNN wins the end-to-end catalog. BC-ResNet, strong on Speech Commands in the
literature [2], underperforms at this scale — a caution against importing leaderboard rankings
unchanged. And the Keyword Transformer INT8-exports but is **not device-runnable**: its attention ops
fall outside the TFLM/ESP-NN kernel set [20], so the **op set is the gate**, not the parameter count.

### 6.6 E8 — streaming CTC transducer (negative/preliminary)

The literature's fix for connected commands is a streaming sequence model that learns alignment
natively [5, 6]. We built one: the MatchboxNet encoder plus a per-frame CTC head over
`blank + the 21 keyword tokens`, trained on **392 synthesised `device [zone] action` phrases**,
decoded greedily into the same grammar. Training loss fell (370 → 28), yet greedy decoding
**collapsed to empty sequences** — catalog full-intent **0.000** against the frame classifier's 0.689.
Two causes resolve differently. **CTC is data-hungry (open):** 392 phrases is far too few for 21
tokens, and all-blank collapse is the classic small-data CTC failure, so this is preliminary rather
than a verdict on the architecture. **The export blocker (resolved):** a `TimeDistributed(Dense)`
head unrolled into a `tf.while` loop the INT8-builtins-only converter could not legalize — the gate
that also ruled out the Keyword Transformer; a **1×1 Conv2D per-frame head** plus a **fixed-T,
batch-1 export clone** exports at **42.9 KB** full INT8, every op a TFLM builtin.

### 6.7 E9 — knowledge distillation (KWT → DS-CNN)

On the frozen v2 split (leak included), the KWT teacher (float test accuracy **0.894**) was distilled
into the unchanged DS-CNN student:

| Model | Float | Isolated | Catalog | INT8 size |
|---|---|---|---|---|
| ds_cnn (first-200 calibration) | 0.862 | 0.842 | 0.218 | 20,224 B |
| ds_cnn (balanced calibration) | 0.862 | 0.853 | 0.259 | 20,224 B |
| ds_cnn distilled (balanced calibration) | 0.842 | 0.833 | **0.667** | 20,272 B |

Distillation did **not** beat the baseline per-clip — float 0.862 → 0.842 and INT8-isolated
0.853 → 0.833, both −2.0 points — yet produced the largest single-change win on the metric that
reflects the deployed system: catalog full-intent **0.259 → 0.667**, +40.8 points. Isolated accuracy
is not the task. Protocol note: this catalog figure is §6.5's 147-trial protocol, *not* the 196-trial
transition-augmented protocol behind §6.2's 0.689; the two must not be compared.

### 6.8 E10 — INT8 calibration, then quantization-aware training

On the v2 rows above the float→INT8 gap is 2.0 points with a first-200 representative set and
**0.9 points with a class-balanced** one. On that evidence we initially closed quantization-aware
training as unnecessary. **That decision was wrong**, and the v3 data says so directly (same
architecture, same held-out split, 2026-09-03):

| Model | Synthetic held-out test accuracy | Size |
|---|---|---|
| Float (`command_v3`) | 89.4 % | 178,142 B |
| INT8 PTQ, balanced calibration | 88.0 % | 18,296 B |
| INT8 QAT, 10 fine-tune epochs | **91.2 %** | **17,880 B** |

QAT recovers all of PTQ's 1.4-point loss **and adds 1.8 points over the float model**: the fake-quant
fine-tune found a better minimum for the quantised graph, not merely a less lossy one. It moves real
speech the same way rather than trading it — on the same two device speakers (*user-customised,
in-training*) isolated-word accuracy goes 0.538 → **0.615** (spk01, n = 13) and 0.553 → **0.737**
(spk02, n = 38), false accepts flat at 0/10 for both. The 17,880 B QAT export is the model flashed
today. A gap threshold measured on one dataset was not a safe basis for closing a technique.

### 6.9 DS-CNN width sweep (negative result)

Widths 24 and 16 were trained on `features_v3` with the width-32 QAT recipe above; width 12 was
skipped once 16 missed by a wide margin. Real-speech columns are *user-customised, in-training*.

| Width | INT8 test acc | Params | MACs | Size | spk01 (n = 13) | spk02 (n = 38) | False accepts (n = 10) |
|---|---|---|---|---|---|---|---|
| 32 (deployed) | **91.2 %** | 5,879 | 2,070,496 | 17,880 B | **0.615** | **0.737** | 0 |
| 24 | 88.7 % | 3,839 | 1,270,632 | 14,528 B | 0.462 | 0.605 | 0 |
| 16 | 84.7 % | 2,183 | 658,928 | 11,528 B | 0.385 | 0.500 | 0 |
| 12 | skipped | 1,499 | 423,636 | — | — | — | — |

**Keep width 32.** Width 24 misses a ≤ 1.0-point synthetic-accuracy bar by 2.5 points, and — the
reason this is worth reporting — *both* narrower widths lose real-speaker accuracy on *both* speakers.
Narrowing trades real-voice recognition, not a fraction of a synthetic point.

### 6.10 E6 — on the device

The system runs on a real M5Stack CoreS3 (ESP-IDF 5.5.5, built reproducibly in Docker, ~950 KB
image). Measured on hardware, 2026-09-03:

| Measurement | Value |
|---|---|
| Command recogniser step (9–10 frames) | **45–46 ms** (164–181 ms naive DFT → 82–85 ms exact FFT → 45 ms) |
| Command model `Invoke` (TFLM) | **41.4 ms** (from 52.9 ms), of which **13.5 ms** is the reference-C `MEAN` |
| Command front end, per new frame | **0.46 ms** (8.5 → 3.09 → 2.01 → 0.46 across two waves, 18×) |
| Wake step | **1.9 ms** (3.5 ms before this wave; 4.9 ms before the 64 KB data cache) |
| Command arena, generated vs. measured need | 139,264 B → **55,024 B** (TFLM reuses buffers) |
| Free internal RAM when the models start | 148,895 B |
| Inference cost, always-on vs. wake-gated | 315 ms/s → **97 ms/s** at an assumed one interaction per 10 s |
| MFCC deviation, C vs. Python reference | 5.4e-4 max abs (1.3e-6 of peak) |
| Quantised int8 tensor fed to the command model | **0 LSB (identical)** |
| Wake front end vs. `pymicro-features` golden vector | **0 LSB exact** (98 × 40) |

The front end, not the model, dominated first: a streaming log-mel ring pushing only new frames took a
step from 1,001 ms to 173 ms, leaving a naive 480-point DFT. 480 is not a power of two, which is why
the DFT was there — but 480 = 2⁵·3·5 is an *exact* kissfft mixed radix, so the transform vendored for
the wake front end now serves both; zero-padding to 512 was rejected because it moves bin spacing away
from the trained mel filters. A second wave took the step 85 → 45 ms, and every step of it was a
memory or bandwidth fact rather than a model change: the generated arena was the desktop planner's
sum of all tensors × 1.2 (139,264 B) against the 55,024 B TFLM actually uses, and right-sizing it into
internal SRAM alone bought 85 → 66 ms; the vendor BSP's own defaults (quad-I/O flash, 1 kHz tick)
66 → 57 ms; and a **banded mel filterbank** — the dense 40 × 241 table is 95 % exact zeros, 38.5 KB of
flash read in full for every frame — 57 → 44 ms, cutting the front end 2.01 → 0.46 ms per frame while
staying bit-identical (only exact zeros dropped, parity unchanged). Only one arena fits internal SRAM,
so it goes to the always-on wake model; doubling the data cache to 64 KB is what makes that
affordable, cutting the wake step 4.9 → 1.9 ms with 60 % less spread and the cost of evicting the
command arena to PSRAM from 12.4 to 2.3 ms, for 43 → 46 ms on the recogniser. Pinning both inference
tasks off the UI core bought variance, not mean (wake spread −17 %).

Wake-gating is the deployment shape and is now measured: always-on recognition costs **315 ms of
inference per wall second**, while running only the wake model and opening a 2.5 s recogniser window
per fire costs **97 ms/s** at an assumed rate of one interaction per 10 s, with the ratio scaling with
that rate. Four candidate optimisations were rejected by their own measurement, which is this wave's
lesson — three of the four were only refutable because the instrumentation was built first. Replacing
the reference-C `MEAN` with an int8 average pool is the biggest prize on the table (a third of
`Invoke`) and fails on arithmetic: TFLite's int8 average pool has no output rescale, so the pooled
embedding inherits the *input's* scale, moving the output **90 LSB** and accuracy
**74.75 % → 74.17 %**. esp-nn's skip-nudge bought 0.7 ms for a ±1 LSB no host test can audit; a 32 KB
instruction cache bought 0.08 ms for 16 KB of SRAM; and copying the wake weights (58,080 B) into
internal RAM does not fit. The device now prints a model stamp at boot
(`command_v3_qat.tflite@fc36da9f`, `hey_bus.tflite@dd9db24f`) plus a fingerprint of the golden vector
through the real interpreter, because device-side arithmetic changes are otherwise invisible to a host
test. The timings above were measured on the v2 PTQ model (74.75 % test accuracy) and are unchanged on
the shipped QAT v3 model (§6.8) — same architecture and width. Two caveats: the CI gates proved
**loadability**, not latency; and the real bring-up cost was board-specific memory configuration
invisible to a host build (quad, not octal, SPI PSRAM; a USB mass-storage buffer at least the
4,096-byte wear-levelling sector).

### 6.11 E6 — real microphone speech, and the reporting policy

`kws-eval --recordings` reports real-recordings accuracy under exactly two labels, matched at speaker
level against the training manifest and **never mixed**: *held-out* (the speaker has no clip in train)
and *user-customised, in-training* (the speaker's own clips are in train). Phrase clips are always
held out.

| Model | spk01 word acc. | spk02 word acc. | False accepts | Label |
|---|---|---|---|---|
| v2 stock (no device recordings; 2026-09-02) | **0.19** (n = 16) | **0.27** (n = 45) | 0/6 | *held-out* |
| v3 PTQ (device recordings in train; 2026-09-03) | 0.538 (n = 13) | 0.553 (n = 38) | 0/10 | *user-customised* |
| v3 QAT (2026-09-03) | **0.615** (n = 13) | **0.737** (n = 38) | 0/10 | *user-customised* |

The first row is the paper's central number and it is a gap: **a model reporting ≈0.9 on its own
held-out MSWC/TTS split recognises roughly a quarter of what the real microphone hears.** The failure
mode is legible in the trace — with a demonstrably healthy model, real microphone speech of a command
word is still classified `_unknown_` at 0.7–0.8 — and none of it is visible from the synthetic split.
Rows 2 and 3 answer a *narrower* question, how well the model knows the people who trained it: this
is a **deliberately user-customised** assistant, on the command side and, with an explicit price, on
the wake side (§6.3). The limits: n is small, phrase accuracy end-to-end on real speech is **0 of 4**,
and 0 false accepts on 10 recorded negatives says nothing about conversational speech.

## 7. Limitations and future work

- **Synthetic data.** 15 of 21 command words are TTS-only; those per-word numbers are not real-speech
  performance. The frozen v2 features additionally carry the split leak of §4.3; v3 closes it.
- **Real-speech evaluation is thin.** Two speakers, 13 and 38 word clips, phrases 0 of 4 end to end,
  and the false-accept rate on real *conversational* speech — command and wake model alike — is
  unmeasured. Grouped speaker k-fold is the planned protocol once ≥ 5 speaker groups cover every word.
- **The combined mode is measured only for cost.** The wake-gated hand-off runs and its duty cycle is
  measured (315 → 97 ms/s, at an assumed interaction rate), but every accuracy figure in §6 comes from
  the two stages in isolation; combined-mode recognition accuracy is unmeasured.
- **Per-device robustness.** The catalog result is lighting-dominated; long compound words fail
  end-to-end, and the segmentation explanation remains an unprobed conjecture.
- **The streaming transducer (E8).** Export is solved, phrase-data scale is not; the follow-up is
  orders-of-magnitude more phrase data, possibly with an entropy regulariser against blank collapse.
- **Decoding.** Detector thresholds commit before the grammar can weigh in; an n-best lattice parse
  over the existing posteriors, gated on false-accept rate, is specified but unrun.
- **The largest remaining on-device cost is a generic kernel.** 13.5 ms of a 41.4 ms `Invoke` is the
  reference-C `MEAN`; an optimised mean kernel would claim it without touching the model, and the
  rejected average-pool substitute (§6.10) shows why the model must not be changed to get it.

## 8. Conclusion

A German, intent-level voice assistant runs fully offline on a commodity ESP32-S3: a 17,880 B INT8
model at 45 ms per streaming step, a locally-trained wake word, and a pure slot grammar keeping intent
validity out of the model. What we most want to travel is methodological: provenance stated on the
test axis as well as the training axis, and a device-to-dataset loop that turned the project's
biggest assumption into a measurement. That measurement is the finding — on real speech the stock
model recognised a quarter of what it claimed on its own split, and closing that gap for named users
is a design choice with a stated price.

## References

1. Zhang et al. *Hello Edge: Keyword Spotting on Microcontrollers.* arXiv:1711.07128.
2. Kim et al. *Broadcasted Residual Learning for Efficient Keyword Spotting.* arXiv:2106.04140.
3. Majumdar & Ginsburg. *MatchboxNet.* arXiv:2004.08531.
4. Berg et al. *Keyword Transformer.* arXiv:2104.00769.
5. He et al. *Streaming Small-Footprint KWS with Sequence-to-Sequence Models.* arXiv:1710.09617.
6. *MFA-KWS* (CTC-Transducer). arXiv:2505.19577.
7. Rybakov et al. *Streaming Keyword Spotting on Mobile Devices.* arXiv:2005.06720.
8. *Advances in Small-Footprint KWS: A Comprehensive Review.* arXiv:2506.11169.
9. Ahrendt. *microWakeWord.* <https://github.com/kahrendt/microWakeWord>.
10. Espressif. *ESP-SR (WakeNet / MultiNet).* <https://github.com/espressif/esp-sr>.
11. Picovoice. *Rhino Speech-to-Intent.* <https://github.com/Picovoice/rhino>.
12. *Rhasspy — offline voice assistant toolkit.* <https://github.com/rhasspy>.
13. Hannun et al. *Deep Speech: Scaling up End-to-End Speech Recognition.* arXiv:1412.5567.
14. Warden. *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209.
15. Banbury et al. *MLPerf Tiny Benchmark.* arXiv:2106.07597.
16. Lin et al. *Training Keyword Spotters with Limited and Synthesized Speech Data.* arXiv:2002.01322.
17. Mazumder et al. *Few-Shot Keyword Spotting in Any Language.* arXiv:2104.01454.
18. Saade et al. *Spoken Language Understanding on the Edge.* arXiv:1810.12735.
19. Jacob et al. *Quantization and Training of Neural Networks for Integer-Arithmetic-Only
    Inference.* arXiv:1712.05877.
20. David et al. *TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems.*
    arXiv:2010.08678.
21. Mazumder et al. *Multilingual Spoken Words Corpus.* NeurIPS Datasets & Benchmarks, 2021.
22. Piczak. *ESC-50: Dataset for Environmental Sound Classification.* ACM Multimedia, 2015.
23. *Piper — a fast, local neural text-to-speech system.* <https://github.com/rhasspy/piper>.
24. Gebru et al. *Datasheets for Datasets.* arXiv:1803.09010.
25. McFee et al. *librosa: Audio and Music Signal Analysis in Python.* SciPy, 2015.
26. Hinton, Vinyals & Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
