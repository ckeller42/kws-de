# Offline German Voice Control on a Microcontroller: Keyword Spotting with a Slot Grammar on the ESP32-S3

**Project:** kws-de — <https://github.com/ckeller42/kws-de>

Every number below comes from a committed evaluation report or an on-device measurement, and names its
model, audio and split policy.

---

## Abstract

We present an open, reproducible pipeline for **German**, **intent-level** voice control that runs
**fully offline on a microcontroller** (ESP32-S3, M5Stack CoreS3), and report what it does on the
target hardware. The gap is concrete: the vendor stack (ESP-SR/MultiNet) ships Chinese and English
only, and the one on-device German speech-to-intent engine (Picovoice Rhino) is closed-source. A tiny
always-on wake model gates a streaming keyword detector whose events a **pure, device-specific slot
grammar** (`device → zone? → action`) composes into a validated intent. The deployed 23-class command
model is **17,880 B** full-INT8, one streaming step measures **45 ms** on the device, and wake-gating
takes always-on inference cost from 315 to **97 ms per wall second** at an assumed one interaction per
10 s. Our headline result is a gap, not a peak: the stock model scores ≈0.9 on its own synthetic split
but recognises **0.19 and 0.27** of two speakers' real microphone speech, and folding those speakers'
recordings into training raises them to **0.615 and 0.737** as *user-customised, in-training* figures,
never quoted as generalisation. Four negative results carry the method: a wake
model that learned *TTS-vs-real* rather than the phrase, invisible to its own held-out recall; a
transition-window regression; a voice-diversity ablation that fell 0.689 → 0.245 while gaining +0.56
on one device; and a width sweep that buys parameters with real-voice accuracy.

## 1. Introduction

Camper vans and similar always-on edge settings need hands-free control where cloud connectivity is
intermittent and a Linux single-board computer cannot stay awake. Doing this in **German** at the **intent** level ("Licht Küche an" → `turn on the kitchen
light`) on an **MCU** with **open** tooling is unserved: ESP-SR/MultiNet [10] ships **Chinese and
English only**; Picovoice Rhino [11] does on-device German speech-to-intent but is **closed-source**;
the open alternatives (Rhasspy [12], Vosk, DeepSpeech [13]) are speech-to-*text*, Pi-class, not MCU
slot filling.

**Contributions.** (1) A reproducible **model factory** for German KWS on the ESP32-S3: public
corpus → MFCC → depthwise-separable CNN → full-INT8 TFLite-Micro with CI budget gates, **running on
the target hardware**. (2) **Honest provenance** for synthetic data, extended to the test axis, with
real-speech figures under two never-mixed labels. (3) A two-stage **open** architecture: a reused wake
engine gating a streaming detector plus a pure slot grammar. (4) A device-to-dataset **recording and
QC loop**, and four negative results that name their own cause.

## 2. Related work

**Small-footprint KWS.** DS-CNN made depthwise-separable convolutions the MCU-KWS workhorse [1];
BC-ResNet [2], MatchboxNet [3] and the Keyword Transformer [4] push accuracy at tiny parameter counts.
Speech Commands [14] is the benchmark, MLPerf Tiny [15] the resource-constrained harness, and a survey
collects the area [8].

**Streaming and connected KWS.** Word *sequences* in continuous speech are handled by streaming
sequence models — sequence-to-sequence [5], CTC-transducer hybrids [6] — and by streaming conversion
of non-streaming models [7].

**Synthetic, low-resource and personalised KWS.** Synthesised speech is established practice for
under-covered keywords [16], and few-shot multilingual KWS the standard answer when a word has no
corpus coverage [17]. Ours is the extreme case, so what we measure is the synthetic-to-real gap
itself — closed for the device's main users, with the numbers labelled accordingly.

**On-device stacks, SLU and data.** ESP-SR/MultiNet [10] and microWakeWord [9] target the ESP32-S3;
Rhino [11] is the closed reference for on-device German speech-to-intent; edge SLU with slot filling
is shown at Pi class [18]. Integer-only inference and quantization-aware training follow Jacob
et al. [19] and the deployable op set is TFLite-Micro's [20]. MSWC [21] supplies real clips, ESC-50
[22] noise, Piper and macOS `say` [23] synthetic fill; the dataset follows Datasheets for
Datasets [24].

## 3. System: vocabulary and grammar

The vocabulary is grounded in the camper's actually-controllable functions, **4 devices** of which one
carries zones and brightness levels. **Licht** takes *an, aus, heller, dunkler* and the levels *25/50/75/100 %* in four zones (*Küche,
Dach, Außen, Lesen*); **Kühlschrank** takes *an, aus, leise*, **Heizung** *an, aus, wärmer, kälter*
and **Aufstelldach** *auf, zu*.

A **pure, device-specific slot grammar** parses an ordered keyword-event sequence into an intent
`(device, zone?, action)`. Validity is *learned by no model*: the grammar rejects out-of-order
sequences, missing slots, an action not allowed for its device ("Aufstelldach an"), and a zone on a
non-zoned device ("Heizung Küche"). Keeping validity in a small pure function makes it exhaustively
testable and portable to firmware unchanged.

## 4. Dataset

### 4.1 Coverage study (why TTS is structural)

We streamed ~2.5 M MSWC-German examples. Of the 24 grounded command words, **only 7 have real clips**
(Licht, Kühlschrank, Heizung, Wasser, aus, auf, Außen); **17 — all zone words, "an", every level and
mode word — have zero**. Synthetic fill is *structural*: a vocabulary drawn from control semantics is
not a subset of a corpus's frequent words.

### 4.2 Construction and provenance

Real clips come from MSWC (CC-BY-4.0) and, from v3 on, the device's recorder (§4.5); missing words are
filled with offline German TTS, noise augmentation with ESC-50. **Every experiment reports
per-word real-vs-TTS counts** and **headline single-word accuracy is computed on real speech only**:
a synthetic voice is not a speaker.

### 4.3 Splits and reproducibility

Splits are **speaker-disjoint**: real words by `speaker_id`, TTS words by a synthetic speaker id. In
the frozen v2 features behind every §6.1–§6.9 number that id was `tts:{engine}:{voice}:{rate}`, so
*one voice at two speaking rates could land in both train and test*. This is a real leak, flagged
wherever a v2 number is quoted; v3 closes it by dropping rate from the id. The dataset also carries a
**validation split**, a **manifest** (counts, config, content hashes — verifiable without shipping
audio), a deterministic rebuild from the pinned clip cache, and a **datasheet** [24].

### 4.4 Statistics

The frozen v2 feature set (seed 0) holds **28,259 one-second examples** over 23 classes — 21 command
words plus `_unknown_` and `_silence_` — split 20,116 / 4,101 / 4,042 rows into train / val / test, of
which 6,544 / 1,241 / 1,186 are real speech.

Every command word carries 1,200 rows. Four are 100 % real MSWC (Licht, Kühlschrank, aus, auf), two
are mixed (Heizung 480 real + 720 TTS, Außen 632 + 568) and the other fifteen — every zone, level and
mode word, plus `an` and Aufstelldach — are 100 % TTS; `_unknown_` holds 2,400 real rows and
`_silence_` 659. Real-speech numbers exclude every TTS row. The **catalog** (§6.2) is the grammar's
49 valid intents, synthesised per voice.

### 4.5 The recording and QC loop (method)

Real microphone data is collected by the device and folded back into training by one pipeline: a
**guided recorder** on the CoreS3 (`spkNN` ids, two reads per prompt, energy-VAD end-pointing) → `scripts/ingest.sh` → `kws-qc`, an audio gate then a Whisper large-v3 **content gate**
that also segments approved sentence takes into 1 s word clips → `kws-dataset build` → train →
export → `kws-eval --recordings` (§6.11), chained by `scripts/data-loop.sh`. On the first session (208 takes, two speakers) it approved **65/208
(31 %)**, up from 48/208 — three QC fixes, not a looser gate: Whisper writes light levels as numerals,
glues keywords into one token, and one hallucinated two-letter keyword was rejecting clean negatives.
A fourth fix requiring whole-token matches moved the count *back down* 69 → 65 by removing false
approvals. The recorder produced one finding of its own:
sentence takes were rejected 75/102 for missing words because the VAD's *fixed* 500 ms trailing
hangover is shorter than a natural reading pause, fixed with a per-prompt-set hangover and a 200 ms
minimum-speech filter.

## 5. Method

**Front-end.** 16 kHz, 1 s clips → MFCC (30 ms periodic-Hann window / 20 ms hop, 480-point FFT,
40 Slaney mel bands, log with an 80 dB floor, DCT-II, 10 cepstra → a 49×10 map). The host front-end is
librosa [25]; the device front-end is a table-driven C port from the same configuration, pinned to the
host by a fixed-input **golden-vector test** and measured on-device in §6.10.

**Model architecture.** The command recogniser is a depthwise-separable CNN in the "Hello Edge"
family [1], kept plain so every layer lowers to a TFLite-Micro builtin: a 3×3 / 32-filter stem, three
depthwise-separable blocks at the same width, a global `MEAN` over time and cepstra, and a Dense
32 → 23 softmax — **5,879 parameters, 2,070,496 MACs** per 1 s window (full layer table: models page
of the project docs). Nothing strides or pools, so the receptive field grows only through the four
3×3 stages (≈190 ms per output cell). The INT8 graph uses exactly `CONV_2D`, `DEPTHWISE_CONV_2D`,
`MEAN`, `FULLY_CONNECTED`, `SOFTMAX`. The v2 export is 20,216 B; the model flashed today is the v3
quantization-aware export (§6.8). The distillation teacher (§6.7) is a
Keyword Transformer [4] — d = 64, 3 pre-LayerNorm blocks, 4 heads, 106 k parameters, float only — and
the student is this DS-CNN unchanged, trained on α · CE(y, p_s) + (1 − α) · T² · KL(p_t^T ‖ p_s^T),
T = 4, α = 0.5, 40 epochs, seed 0 [26].

**Quantization and budget gates.** Full-INT8 post-training quantization with a class-balanced
representative set, exported to TFLite-Micro [20]. Quantization-aware training, where used, wraps the
architecture with `tensorflow-model-optimization` fake-quant and fine-tunes 10 epochs at Adam(1e-5)
from the float weights [19]. **CI budget gates** assert model ≤ 500 KB, MACs ≤ 3 M, INT8-only I/O and
a device-runnable op set: they prove **loadability** without hardware, not real-time fit, which only
the device measures (§6.10). A **model-health gate** (§6.2) blocks a broken export from reaching the
firmware header. Batch size is 32 for §6.5, 128 elsewhere.

**Two-stage runtime.** An always-on **wake detector** ("Hey Bus", microWakeWord [9], trained
end-to-end on an M4 laptop in ~6m41s with no Colab or GPU) gates the heavier command recogniser inside
the post-wake window. Only the small model runs continuously, which is what makes the always-on power
budget plausible — now measured rather than assumed (§6.10). The hand-off runs on the device, but
**only its cost has been measured**: every accuracy figure in §6 comes from the two stages in
isolation. Posteriors are decoded by **edge-triggered run-based decoding**: a run of
consecutive steps sharing the same qualifying top-1 label fires *once* at `min_consecutive` steps,
with **no global cooldown**, so a different label may fire immediately and the same label only after
≥ `gap_steps` non-matching steps. This replaced a level-triggered threshold plus global refractory
that conflated *same-word debounce* with *next-word gating* (§6.2).

## 6. Experiments and results

Three different things get called "accuracy" here and are not comparable, so each is labelled at every
use:

| Label | Audio | Split policy |
|---|---|---|
| synthetic held-out | MSWC + TTS clips, noise-mixed | voice/speaker-disjoint split (v2 rows carry the §4.3 leak) |
| synthetic catalog | TTS-synthesised phrases | test voices also in training — **in-domain** |
| real speech, held-out | device microphone | speaker not in train |
| real speech, user-customised | device microphone | speaker's own clips in train |

"Catalog full-intent accuracy" scores every valid command end-to-end (audio → MFCC → detector →
grammar → intent); one wrong or missed word fails the entry.

### 6.1 E1 — single-word recognition, v1 vocabulary

The project's first model is **not** the deployed one: it is a **7-class, 5-toggle-word v1 model**
(5,351 params, 19,256 B) over a vocabulary including *Camping* and *Wasser*, both dropped when E3
grounded the vocabulary in the van's controllable functions. On the real-speech subset of its own
held-out split:

| Metric (v1, 7 classes) | Value |
|---|---|
| Real-speech accuracy: float / INT8 / INT8 with TTS classes | 91.7 / **91.1** / 93.2 % (n = 775, 20/10/0 dB) |
| `_unknown_` false-accept | 0.0 % |
| SNR sweep, INT8 | 93.0 % clean → 84.6 % @ 0 dB |

That is a five-word toggle vocabulary, not the 23-class task, not the deployed model, and not
comparable to MultiNet's vendor-reported ~85–95 % on English connected commands, which differs in
language, task and corpus at once. We quote it for one thing: **data mattered more than modelling** —
finding real "Wasser" clips on a deeper scan moved it 87.9 % → 91.1 % with no model change.

### 6.2 E3 — end-to-end command catalog, and an ablation arc

The catalog produced the paper's most instructive arc, each step eliminating one cause (synthetic
catalog, in-domain; v2 features, §4.3 leak):

| Stage | Catalog full-intent | Cause eliminated |
|---|---|---|
| Initial | **0.000** | dataset asymmetry — learned the noise floor, not words |
| + symmetric domains, ±200 ms shift | **0.066** | data fixed; isolated words 0.99+ |
| + edge-triggered decoder | **0.362** | re-fire, swallowed next word, 1-step ghosts |
| + naive transition negatives | **0.197** *(regression)* | over-corrected → recall loss |
| + **balanced** negatives + class-weights | **0.689** | recall recovered; zone slot 0.156 → 0.789 |

**Row 1 — the bug the end-to-end metric caught.** `build_dataset` initially added command clips
**only noise-mixed** but `_unknown_` clips **only clean**, so the model learned "clean ⇒ `_unknown_`,
noisy ⇒ some command". Per-clip held-out accuracy stayed at 88.5 % because the held-out set shared the
asymmetry; the giveaway was the SNR sweep *improving* as noise worsened. A per-clip metric inheriting
the training data's assumption cannot see its flaw — met again one layer up in §6.3.
It is now a shipped guard: after a mode-collapsed model (0.3 % accuracy on its own training set) was
flashed having "succeeded" by exit code, `kws-export` runs `assert_model_healthy` first.

**Row 4 — a useful negative result.** Adding word-**boundary** windows labelled `_unknown_` *reduced*
catalog accuracy: the transition negatives out-weighted each word class ~5×, the model predicted
`_unknown_` 17.4 % against 7.7 % true, and clip-level accuracy fell 88.5 % → 78.1 %. It over-corrected
toward "say nothing"; **balancing** them plus inverse-frequency class weighting reversed it. Lesson:
*the obvious data-side fix for boundary ghosts hurts unless the added negatives are class-balanced.*
The resulting v2 model is strong on the zoned and levelled **Licht** commands but weak elsewhere —
**Kühlschrank 0.00** despite 300 real clips, plausibly but unprovenly the detector segmenting a long
compound. Licht is 40 of 49 entries, so the number is lighting-dominated.

### 6.3 E4 — the wake word: a synthetic-vs-real shortcut, and how it closed

**Round 1 (feasibility).** microWakeWord on 2,000 synthetic positives, upstream 10,000-step config:
62,304 B INT8 streaming TFLite, val recall **71.65 %**, precision 100 %, and at cutoff 0.99 2.0
false-accepts/hour.

**Round 2 — the held-out metric was uninformative.** On the device that model *never fired*: per-2 s
peak probability 0.00–0.13 while a person said "Hey Bus", although the front end is bit-exact against
the trainer's own extractor (§6.10). A host probe through the identical int8 path explained it: the
model output ≥ 0.99 for *any* Piper sentence in its training voice and ≈ 0.004 for "hey bus" in unseen
voices. With all positives synthetic and all negatives real recordings, the cheapest separating
feature was **TTS-vs-real, not the phrase**. The 71.65 % was not weak but meaningless: a held-out
metric drawn from the same synthetic distribution cannot see a synthetic-vs-real shortcut. This is
§6.2's lesson one layer up, and why §6.11's two-label policy exists.

**Round 4** retrained with TTS hard negatives, reverb augmentation and multi-voice positives
(9,000 + 9,000 clips, 58,080 B). It fires in 3 of 4 unseen probe voices against round 1's 1 of 4, and
peaks 0.83–0.99 on the device against ≤ 0.44 on silence and room noise, which moved the gate from 0.99
to **0.85 × 2 consecutive steps**. **Round 5** added ten real "Hey Bus" takes from a main user,
QC-approved through §4.5, as their own feature set at sampling weight 5:

| Probe (gate 0.85 × 2) | Round 4 | Round 5 |
|---|---|---|
| Real "Hey Bus" takes fired | 4 of 10 | **10 of 10** (peak 0.996) |
| One-session variant, on the unseen session | — | 5 of 5 |
| TTS non-wake worst peak | 0.988 | 0.758 |
| Generic synthetic "hey bus", laptop speaker | 3 of 3 (0.96–0.99) | 0 of 3 (0.59–0.64) |

The last row is the stated price, **generic-voice margin**: the wake model is customised to the
device's main users, the policy the command model follows (§6.11), and each further user's five takes
go through the same loop. The false-accept rate on conversational speech is **unmeasured**.

### 6.4 E5 — voice diversity: a net regression containing a large per-device gain

Hypothesis: for synthetic KWS data, **voice diversity** dominates per-voice fidelity. We added four
Piper neural voices to the `say`-only training data, cycling the balanced pool back up to the
baseline's clip target so volume is matched (29,442 vs 29,985 train rows). Result (synthetic catalog,
in-domain): **full-intent accuracy fell 0.689 → 0.245** (zone slot 0.789 → 0.219; clip-level
0.866 → 0.779). The effect is not uniform, and that is the finding. **Heizung**, one of the two weak
non-Licht devices, is where the hypothesis was expected to matter and improved sharply,
0.188 → **0.750**; **Licht** collapsed 0.794 → 0.200, Aufstelldach slipped 0.625 → 0.500 and
Kühlschrank stayed flat at 0.000 (per-entry breakdown: models page of the project docs). The net regression is
therefore a Licht story: Licht carries 82 % of catalog trials, so its collapse swamps Heizung's gain. A plausible, unverified mechanism: one shared 23-class classifier, so Piper's
varied renderings of the *other* classes shifted the shared boundary against Licht, whose clips are
100 % real MSWC. Kühlschrank's flatness shows that failure was never a diversity problem. Two caveats: the catalog test uses `say` voices present in *both*
trainings, so this is **in-domain** accuracy rather than the cross-voice generalization the hypothesis
targets; and an earlier pass that left the balanced pool at its natural size collapsed to 0.066, a
**data-volume** confound rather than the number reported here.

### 6.5 E7 — architecture benchmark

Four encoders on the frozen v2 dataset (leak included), trained identically (30 epochs, seed 0,
class-weighted, val-selected, test-reported). Catalog numbers use a 3-voice subset (147 trials/arch)
on the clean dataset, below §6.2's tuned system; the ranking is the point.

| Architecture | Isolated | Catalog | MACs | INT8 | Device-runnable |
|---|---|---|---|---|---|
| DS-CNN | 0.834 | **0.544** | 2.07 M | 20,216 B | ✓ |
| BC-ResNet | 0.773 | 0.102 | 1.39 M | 31 KB | ✓ |
| MatchboxNet | **0.903** | 0.245 | **0.47 M** | 43 KB | ✓ |
| Keyword-Transformer | — | — | — | 173 KB | **✗ (non-TFLM ops)** |

The **ranking depends on the metric**: MatchboxNet wins isolated accuracy and is the most
MAC-efficient, yet DS-CNN wins the end-to-end catalog. BC-ResNet, strong on Speech Commands in the
literature [2], underperforms at this scale — a caution against importing leaderboard rankings. The
Keyword Transformer INT8-exports but is **not device-runnable**, its attention ops falling outside the
TFLM kernel set [20]: the **op set is the gate**, not the parameter count.

### 6.6 E8 — streaming CTC transducer (negative/preliminary)

The literature's fix for connected commands is a streaming sequence model that learns alignment
natively [5, 6]. We built one: the MatchboxNet encoder plus a per-frame CTC head over
`blank + the 21 keyword tokens`, trained on **392 synthesised phrases** and decoded greedily into the
same grammar. Training loss fell (370 → 28), yet greedy decoding
**collapsed to empty sequences** — catalog full-intent **0.000** against the frame classifier's 0.689.
Two causes, resolving differently. **CTC is data-hungry (open):** 392 phrases is far too few for 21
tokens, and all-blank collapse is the classic small-data failure, so this is preliminary, not a
verdict on the architecture. **The export blocker (resolved):** a `TimeDistributed(Dense)` head
unrolled into a `tf.while` loop the INT8-builtins-only converter could not legalize, the same gate
that ruled out the Keyword Transformer; a **1×1 Conv2D per-frame head** and a **fixed-T, batch-1
export clone** export at **42.9 KB** full INT8, every op a TFLM builtin.

### 6.7 E9 — knowledge distillation (KWT → DS-CNN)

On the frozen v2 split, the KWT teacher (float test accuracy **0.894**) was distilled into the
unchanged DS-CNN student:

| Model | Float | Isolated | Catalog | INT8 |
|---|---|---|---|---|
| ds_cnn, first-200 calibration | 0.862 | 0.842 | 0.218 | 20,224 B |
| ds_cnn, balanced calibration | 0.862 | 0.853 | 0.259 | 20,224 B |
| ds_cnn distilled, balanced | 0.842 | 0.833 | **0.667** | 20,272 B |

Distillation did **not** beat the baseline per-clip, losing 2.0 points on both float and
INT8-isolated, yet produced the largest single-change win on the metric that reflects the deployed
system: catalog full-intent **0.259 → 0.667**. Isolated accuracy is not the task. Protocol note: this
figure is §6.5's 147-trial protocol, *not* the 196-trial one behind §6.2's 0.689; do not compare
them.

### 6.8 E10 — INT8 calibration, then quantization-aware training

On the v2 rows above the float→INT8 gap is 2.0 points with a first-200 representative set and
**0.9 points with a class-balanced** one, on which evidence we initially closed quantization-aware
training as unnecessary. **That decision was wrong**, and the v3 data says so directly (same
architecture, same held-out split, 2026-09-03):

| v3: float → PTQ → QAT | Synthetic held-out accuracy | Size |
|---|---|---|
| `command_v3` → balanced PTQ → 10 QAT epochs | 89.4 → 88.0 → **91.2 %** | 178,142 → 18,296 → **17,880 B** |

QAT recovers all of PTQ's loss **and adds 1.8 points over the float model**: the fake-quant fine-tune
found a better minimum for the quantised graph, not just a less lossy one. It moves real speech the
same way rather than trading it — on the same two device speakers (*user-customised, in-training*)
isolated-word accuracy goes 0.538 → **0.615** and 0.553 → **0.737**, false accepts flat at 0/10. This
export is the model flashed today. A gap threshold measured on one dataset was not a safe basis for
closing a technique. Full table: models page of the project docs.

### 6.9 DS-CNN width sweep (negative result)

Widths 24 and 16 were trained with the width-32 QAT recipe above; 12 was skipped once 16 missed by a
wide margin. Real-speech figures are *user-customised, in-training*.

| Width 32 (deployed) → 24 → 16 | INT8 test | spk01 | spk02 | FA |
|---|---|---|---|---|
| 5,879 → 3,839 → 2,183 params | **91.2** → 88.7 → 84.7 % | **0.615** → 0.462 → 0.385 | **0.737** → 0.605 → 0.500 | 0/10 |

**Keep width 32.** Width 24 misses a ≤ 1.0-point synthetic-accuracy bar by 2.5 points, and — the
reason this is worth reporting — *both* narrower widths lose real-speaker accuracy on *both* speakers:
narrowing trades real-voice recognition, not a synthetic fraction. Full table: models page
of the project docs.

### 6.10 E6 — on the device

The system runs on a real M5Stack CoreS3 (ESP-IDF 5.5.5, built reproducibly in Docker). Wake-gating is
the deployment shape and is now measured rather than assumed: only the wake model runs continuously,
each fire opening a 2.5 s recogniser window. The real bring-up cost was board-specific memory config
invisible to a host build — quad, not octal, SPI PSRAM. Measured on hardware, 2026-09-03:

| Measurement | Value |
|---|---|
| Recogniser step | **45–46 ms** (164–181 DFT → 82–85 exact FFT → 45) |
| `Invoke` (TFLM) | **41.4 ms** (from 52.9), **13.5** of it reference-C `MEAN` |
| Front end, per frame | **0.46 ms** (8.5 → 3.09 → 2.01 → 0.46, 18× over two waves) |
| Wake step | **1.9 ms** (3.5 before this wave, 4.9 before the 64 KB cache) |
| Command arena, generated vs. needed | 139,264 → **55,024 B** |
| Inference cost, always-on vs. wake-gated | 315 → **97 ms/s**, one interaction per 10 s |
| MFCC deviation, C vs. Python | 5.4e-4 max abs |
| Quantised int8 tensor into the model | **0 LSB** |
| Wake front end vs. golden vector | **0 LSB** (98 × 40) |

The front end, not the model, dominated first. A streaming log-mel ring pushing only new frames took
the step from 1,001 ms to 173 ms, leaving a naive 480-point DFT — there because 480 is not a power of
two, but 480 = 2⁵·3·5 is an *exact* kissfft mixed radix, so the transform vendored for the wake front
end now serves both. Zero-padding to 512 was rejected: it moves bin spacing off the trained mel
filters. A second wave then halved the step again through memory and bandwidth alone. Right-sizing the arena, emitted by the generator as the desktop planner's sum of
all tensors × 1.2, moved it into internal SRAM for 85 → 66 ms. The vendor BSP's defaults, quad-I/O
flash and a 1 kHz tick, gave 66 → 57 ms. A **banded mel filterbank** gave 57 → 44 ms: the dense
40 × 241 table is 95 % exact zeros read in full every frame, and dropping only exact zeros leaves the
features bit-identical.

Only one arena fits internal SRAM, and it goes to the always-on wake model — which doubling the data
cache to 64 KB makes affordable, at 43 → 46 ms on the recogniser. Pinning the inference tasks off the
UI core bought variance, not mean. Four optimisations were then rejected by their own measurement,
three only because the instrumentation was built first. An int8 average pool replacing the
reference-C `MEAN` was the biggest prize and fails on arithmetic: with no output rescale the pooled
embedding inherits the input's scale, moving the output **90 LSB** and accuracy
**74.75 % → 74.17 %**. Skip-nudge bought 0.7 ms for a ±1 LSB no host test can audit; a 32 KB
instruction cache bought 0.08 ms for 16 KB of SRAM; the wake weights do not fit internal RAM. The
device now stamps its models and a golden-vector fingerprint at boot, since device arithmetic is
invisible to a host test. These timings come from the v2 PTQ model and are unchanged on the shipped
QAT v3 model (§6.8), and the CI gates proved **loadability**, not latency.

### 6.11 E6 — real microphone speech, and the reporting policy

`kws-eval --recordings` reports real-recordings accuracy under exactly two labels, matched at speaker
level against the training manifest and **never mixed**: *held-out* and *user-customised*.

| Model | spk01 | spk02 | False accepts |
|---|---|---|---|
| v2 stock, no device recordings — *held-out* | **0.19** (n = 16) | **0.27** (n = 45) | 0/6 |
| v3 PTQ, recordings in train — *user-customised* | 0.538 (n = 13) | 0.553 (n = 38) | 0/10 |
| v3 QAT — *user-customised* | **0.615** (n = 13) | **0.737** (n = 38) | 0/10 |

The first row is the paper's central number and it is a gap: **a model reporting ≈0.9 on its own
held-out MSWC/TTS split recognises roughly a quarter of what the real microphone hears.** The failure
mode is legible — a healthy model still classifies real command speech as `_unknown_` at 0.7–0.8 —
and invisible from the synthetic split. Rows 2 and 3 answer a *narrower* question, how
well the model knows the people who trained it: this is a **deliberately user-customised** assistant,
on the command side and, with an explicit price, on the wake side (§6.3).
The limits: n is small, phrase accuracy on real speech is **0 of 4**, and zero false accepts on ten
negatives says nothing about conversation.

## 7. Limitations and future work

- **Synthetic data.** 15 of 21 command words are TTS-only, so those per-word numbers are not
  real-speech performance; the frozen v2 features also carry the split leak of §4.3, which v3 closes.
- **Real-speech evaluation is thin.** Two speakers, phrases 0 of 4 end to end, and the false-accept
  rate on real *conversational* speech unmeasured for command and wake model alike; grouped speaker
  k-fold is planned once enough speaker groups cover every word.
- **The combined mode is measured only for cost.** The wake-gated hand-off runs and its duty cycle is
  measured, at an assumed interaction rate; its recognition accuracy is not.
- **Per-device robustness.** The catalog result is lighting-dominated and long compound words fail
  end-to-end, on an unprobed segmentation conjecture.
- **The streaming transducer (E8).** Export is solved, phrase-data scale is not; the follow-up is much
  more phrase data, possibly with an entropy regulariser against blank collapse.
- **Decoding.** Detector thresholds commit before the grammar weighs in; an n-best lattice parse over
  the existing posteriors is specified but unrun.
- **The largest on-device cost left is a generic kernel.** A third of `Invoke` is the reference-C
  `MEAN`; an optimised mean kernel claims it without touching the model, which §6.10's rejected
  substitute shows must not change.

## 8. Conclusion

A German, intent-level voice assistant runs fully offline on a commodity ESP32-S3: a 17,880 B INT8
model at 45 ms per streaming step, a locally-trained wake word, and a pure slot grammar keeping intent
validity out of the model. What we most want to travel is methodological: provenance stated on the
test axis as well as the training axis, and a device-to-dataset loop that turned the project's biggest
assumption into a measurement. That measurement is the finding — on real speech the stock model
recognised a quarter of what it claimed on its own split, and closing that gap for named users is a
design choice with a stated price.

## References

1. Zhang et al. *Hello Edge.* arXiv:1711.07128.
2. Kim et al. *Broadcasted Residual Learning.* arXiv:2106.04140.
3. Majumdar & Ginsburg. *MatchboxNet.* arXiv:2004.08531.
4. Berg et al. *Keyword Transformer.* arXiv:2104.00769.
5. He et al. *Streaming Small-Footprint KWS with seq2seq.* arXiv:1710.09617.
6. *MFA-KWS* (CTC-Transducer). arXiv:2505.19577.
7. Rybakov et al. *Streaming Keyword Spotting on Mobile Devices.* arXiv:2005.06720.
8. *Advances in Small-Footprint KWS.* arXiv:2506.11169.
9. Ahrendt. *microWakeWord.* <https://github.com/kahrendt/microWakeWord>.
10. Espressif. *ESP-SR.* <https://github.com/espressif/esp-sr>.
11. Picovoice. *Rhino.* <https://github.com/Picovoice/rhino>.
12. *Rhasspy.* <https://github.com/rhasspy>.
13. Hannun et al. *Deep Speech.* arXiv:1412.5567.
14. Warden. *Speech Commands.* arXiv:1804.03209.
15. Banbury et al. *MLPerf Tiny Benchmark.* arXiv:2106.07597.
16. Lin et al. *Keyword Spotters from Limited and Synthesized Speech.* arXiv:2002.01322.
17. Mazumder et al. *Few-Shot Keyword Spotting in Any Language.* arXiv:2104.01454.
18. Saade et al. *Spoken Language Understanding on the Edge.* arXiv:1810.12735.
19. Jacob et al. *Integer-Arithmetic-Only Inference.* arXiv:1712.05877.
20. David et al. *TensorFlow Lite Micro.* arXiv:2010.08678.
21. Mazumder et al. *Multilingual Spoken Words Corpus.* NeurIPS D&B, 2021.
22. Piczak. *ESC-50.* ACM Multimedia, 2015.
23. *Piper.* <https://github.com/rhasspy/piper>.
24. Gebru et al. *Datasheets for Datasets.* arXiv:1803.09010.
25. McFee et al. *librosa.* SciPy, 2015.
26. Hinton, Vinyals & Dean. *Distilling the Knowledge.* arXiv:1503.02531.
