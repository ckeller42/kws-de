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
grammar** (`device → zone? → action`) composes into a validated intent. Both models run as
**generated, bit-exact C over esp-nn** with no interpreter on the device: the deployed 23-class
command model is a width-48 DS-CNN, **25,832 B** full-INT8 quantization-aware, one streaming step
measures **46–47 ms** and the wake step **1.25 ms**. Our headline results are two gaps, not a peak.
The stock model scores ≈0.9 on its own synthetic split but recognises **0.19 and 0.27** of two
speakers' real microphone words; folding six users' recordings into training and **up-weighting real
clips 3×** raises isolated-word accuracy to **0.936** on the four-speaker deploy scoreboard (n = 233,
0/32 false accepts) as a *user-customised, in-training* figure, never quoted as generalisation. Yet
the same model turns only **≈0.10** of read command sentences into the exact intent (14/139): the
word→sentence path — segmentation, decoder, grammar — is the bottleneck, not the classifier. Negative
results carry the method: a wake model that learned *TTS-vs-real* rather than the phrase, invisible
to its own held-out recall; a transition-window regression; a voice-diversity ablation that fell
0.689 → 0.245; a width-40 model that is smaller in MACs yet asks for twice the scratch memory of
width 48; and a per-clip augmentation change that failed the deploy rule.

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
real-speech figures under two never-mixed labels, and a machine-checked quality gate on the synthetic
half. (3) A two-stage **open** architecture: a reused wake engine gating a streaming detector plus a
pure slot grammar, both models compiled to straight-line esp-nn C that is bit-exact against the
interpreter it replaced. (4) A device-to-dataset **recording, field-capture and QC loop** with a
fixed deploy rule, and the negative results it produced, each naming its own cause.

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

Real clips come from MSWC (CC-BY-4.0) and, from v3 on, the device's recorder and field capture
(§4.5–4.6); missing words are filled with offline German TTS (Piper and macOS `say`), noise
augmentation with ESC-50 and, from the 2026-09-06 build, opt-in van-cabin augmentation (three SNRs
of cabin noise through a room impulse response per real clip). **Every experiment reports per-word
real-vs-TTS counts** and **headline single-word accuracy is computed on real speech only**: a
synthetic voice is not a speaker. Since §4.6 every synthetic voice must also pass a language and
content gate before its clips enter the build.

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

The **v3 build behind the deployed model** (2026-09-06, seed 0) is larger and differently sourced:
**38,646 / 6,417 / 11,291** train / val / test rows, of which 13,774 / 1,993 / 2,163 are real
speech. Its real material is 2,378 MSWC clips plus **285 device clips** from six speakers merged at
build time (Licht 76, an 26, Außen 22, aus 19, Kühlschrank 17, Heizung 12, Lesen 12, `_unknown_`
104, the rest single digits), each real clip expanded by the noise and van augmentations; every
TTS clip was regenerated through the §4.6 gate from a pool of 252 admitted voices. Because van
augmentation multiplies real clips and the TTS top-up fills every word to 300 clips, a row count
is no longer a clip count; the manifest carries both.

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

### 4.6 Field capture, elicitation, and the synthetic-data gate

Guided takes are read speech. Three later additions widen what the loop can see.

**Field capture.** With an opt-in switch, every wake fire in assistant mode keeps one *field take*:
the pre-roll before the fire plus the command window it opened, copied out of the always-on audio
ring and written only *after* the window closes, because a FAT write costs 100–300 ms — more than a
recognise step — and would have changed the behaviour being captured (measured: the step inside a
capturing window stays at 38–42 ms; writes land ~1 s after the window). The label never comes from
the device: Whisper transcribes the take, the wake phrase is cut off as a wake positive, and the
rest goes through the *same* `grammar.parse` the firmware uses; a valid intent becomes a phrase
label, anything else is kept as `_unknown_` material. What the device *did* answer is stored beside
that label, giving a **field agreement** figure no synthetic test can. Three things had to be found
by measurement: the takes were all "clipped" because the device recorded its own 1 kHz confirmation
tone through a microphone centimetres from the speaker (muted while capture is armed); the wake
phrase was cut at the take's first sample because the model's detection latency was 1.06 s median,
not the 0.2–0.3 s three files had budgeted for (pre-roll 1.0 → 2.5 s, written as its three terms);
and because a take exists only where the wake fired, capture arms the wake gate at 0.60 instead of
the shipped 0.85 and reconstructs the shipped decision on the host, so near-misses and false alarms
become countable — a lower bound on misses, never the miss rate.

**Elicitation.** A third recorder mode shows a *situation* ("Es ist dunkel in der Küche.") and
never the words, so the answer is natural command speech, wake phrase included; each prompt carries
an expected intent that QC scores against but never relabels to.

**The synthetic-data gate.** A device test driven by "German" `say` clips turned out to be English
throughout: `say` silently substitutes a voice for a missing one, and eight of nine voice names in
the pool exist in both languages. A human ear was the only check in the pipeline. The fix is at two
levels. Per clip, a Whisper pass whose *detected* language must be `de` (forcing German answers "de"
for English audio) plus the content rule — which on a 0.4 s single word rejects mostly good German:
`heller`, `zu`, `kälter` fail on language misdetection and `an` on a duration floor, so the first
regenerated cache dropped **67 %** of attempts and four words lost every clip. Per voice, one fixed
calibration sentence must transcribe back at ≥ 0.85 token match, after which a passing voice's
clips need only a cheap duration gate; that took the drop to **6 %** and admitted **252 of 259**
candidate voices (51 with the first calibration sentence, whose word "Küche" Whisper mishears as
"Kirche"; the sentence was rewritten without it). The general lesson: every silent fallback in a
data pipeline is a labelling bug waiting to happen.

## 5. Method

**Front-end.** 16 kHz, 1 s clips → MFCC (30 ms periodic-Hann window / 20 ms hop, 480-point FFT,
40 Slaney mel bands, log with an 80 dB floor, DCT-II, 10 cepstra → a 49×10 map). The host front-end is
librosa [25]; the device front-end is a table-driven C port from the same configuration, pinned to the
host by a fixed-input **golden-vector test** and measured on-device in §6.10.

**Model architecture.** The command recogniser is a depthwise-separable CNN in the "Hello Edge"
family [1], kept plain so every layer lowers to a TFLite-Micro builtin: a 3×3 stem, three
depthwise-separable blocks at the same width *w*, a global `MEAN` over time and cepstra, and a Dense
*w* → 23 softmax. Nothing strides or pools, so the receptive field grows only through the four 3×3
stages (≈190 ms per output cell). The INT8 graph uses exactly `CONV_2D`, `DEPTHWISE_CONV_2D`,
`MEAN`, `FULLY_CONNECTED`, `SOFTMAX` — ten ops. Every experiment through §6.11 uses **w = 32:
5,879 parameters, 2,070,496 MACs** per 1 s window (v2 export 20,216 B, v3 QAT 17,880 B). The model
deployed today is **w = 48: 11,111 parameters, 4,234,704 MACs, 25,832 B** (§6.13; full layer
table: models page of the project docs). The distillation teacher (§6.7) is a
Keyword Transformer [4] — d = 64, 3 pre-LayerNorm blocks, 4 heads, 106 k parameters, float only — and
the student is this DS-CNN unchanged, trained on α · CE(y, p_s) + (1 − α) · T² · KL(p_t^T ‖ p_s^T),
T = 4, α = 0.5, 40 epochs, seed 0 [26].

**Quantization and budget gates.** Full-INT8 post-training quantization with a class-balanced
representative set, exported to TFLite-Micro [20]. Quantization-aware training, where used, wraps the
architecture with `tensorflow-model-optimization` fake-quant and fine-tunes 10 epochs (20 in the
deployed recipe) at Adam(1e-5) from the float weights [19]. The deployed recipe adds one data-side
knob, **`--real-weight 3`**: every real-speech row (device and MSWC) is repeated three times before
class weights are computed, so the real-vs-synthetic mix seen in training is controlled separately
from label balance (§6.13). **CI budget gates** assert model ≤ 500 KB, MACs ≤ 5 M (raised from 3 M
for width 48; the binding constraint is the 100 ms step, of which MACs are a proxy), INT8-only I/O
and a device-runnable op set: they prove **loadability** without hardware, not real-time fit, which
only the device measures (§6.10). A **model-health gate** (§6.2) blocks a broken export from reaching
the firmware header. Batch size is 32 for §6.5, 128 elsewhere. Training is deterministic at a fixed
seed: the same recipe on the same cache reproduces the export byte for byte.

**Generated inference.** Neither model runs through the TFLite-Micro interpreter on the device. A
generator reads the flatbuffer *as embedded in the firmware header* — not the `.tflite` on disk,
which had already drifted from it once — and emits each graph as a flat C function of direct
esp-nn calls: ten straight-line kernels and one static arena for the command model, the streaming
ring buffers as plain static arrays for the wake model, a 256-entry LUT for `LOGISTIC`, and one
shared esp-nn scratch region sized to the widest op of either model. The scratch size is a Python
port of esp-nn's own sizing function; at boot the firmware asks the real function on the real chip
and refuses the generated path if the answer is ever larger, because that failure is a silent
overrun rather than a crash. Bit-exactness against the interpreter is checked at three levels:
synthetic smoke vectors in CI, every real held-out window on the host (`0/1564 bytes differ`), and a
parity build on live microphone features on the device. §6.12 measures what it buys.

**Two-stage runtime.** An always-on **wake detector** ("Hey Bus", microWakeWord [9], trained
end-to-end on an M4 laptop in ~6m41s with no Colab or GPU) gates the heavier command recogniser inside
the post-wake window. Only the small model runs continuously, which is what makes the always-on power
budget plausible — now measured rather than assumed (§6.10). The wake gate is 0.85 on two
consecutive steps, held closed for ~1 s after every front-end reset while the noise and PCAN
estimates settle (a fire at 0.965 on −58 dBFS silence 40 s after boot forced that), and a command
fire inside the first 450 ms of a window is dropped, because a wake model that answers 0.14 s after
"Bus" leaves the recogniser's first retrospective slice scoring the tail of the wake phrase itself
("aus" was the first device word in 12 of 17 real takes before that gate). At the window's close
the device parses the fired words with a C port of the grammar, verified against the Python parse
on a committed case table, and — when the plain parse fails — retries once with the decoder's
runner-up word at each `_unknown_` slot, accepting only if exactly one substitution makes it valid.
Every accuracy figure through §6.14 comes from the two stages in isolation; §6.15 measures the
composed path. Posteriors are decoded by **edge-triggered run-based decoding**: a run of
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
export was the model flashed on 2026-09-03; §6.13 follows it through four more deploys. A gap
threshold measured on one dataset was not a safe basis for closing a technique. Full table: models
page of the project docs.

### 6.9 DS-CNN width sweep, downward (negative result)

Widths 24 and 16 were trained with the width-32 QAT recipe above; 12 was skipped once 16 missed by a
wide margin. Real-speech figures are *user-customised, in-training*.

| Width 32 (then deployed) → 24 → 16 | INT8 test | spk01 | spk02 | FA |
|---|---|---|---|---|
| 5,879 → 3,839 → 2,183 params | **91.2** → 88.7 → 84.7 % | **0.615** → 0.462 → 0.385 | **0.737** → 0.605 → 0.500 | 0/10 |

**Keep width 32** was the verdict, because width 24 misses a ≤ 1.0-point synthetic-accuracy bar by
2.5 points, and — the reason this is worth reporting — *both* narrower widths lose real-speaker
accuracy on *both* speakers: narrowing trades real-voice recognition, not a synthetic fraction. The
sweep had only gone in one direction; §6.13 goes the other way. Full table: models page of the
project docs.

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
invisible to a host test. These timings come from the v2 PTQ model through the TFLM interpreter and
were unchanged on the QAT v3 model (§6.8); §6.12 removes the interpreter, and the CI gates proved
**loadability**, not latency.

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
The limits at this point: n is small, phrase accuracy on real speech is **0 of 4**, and zero false
accepts on ten negatives says nothing about conversation. §6.13 and §6.15 revisit all three.

### 6.12 Generated inference: removing the interpreter (measured on the CoreS3, 2026-09-04/06)

Both models were moved from `tflite::MicroInterpreter` to the generated esp-nn C of §5, each build
measured in one session against the interpreter build it replaced (medians over trace windows):

| | Interpreter | Generated | Δ |
|---|---|---|---|
| Wake step | 1,891 µs | **1,250 µs** | −34 % |
| Wake within-window spread | ±502 µs | ±97 µs | |
| Command step (w32) | 46.0 ms | **31 ms** | −33 % |
| Command `Invoke` (w32) | 41.7 ms | 27.3 ms | −35 % |
| Wake model memory | 40,960 B heap arena + 1 KB variables | 128 B arena + 4,200 B ring state | |
| Command arena (w32) | 65,536 B PSRAM (54,824 used) | 31,360 B PSRAM | |
| Shared esp-nn scratch, internal | — | 19,888 B (29,824 B at w48) | |
| App image | 1,165,872 B | 1,002,560 B | −163 KB |
| Output vs. interpreter, live audio | | 0 bytes differ | |

**Why it is faster is not better kernels — they are the same esp-nn kernels.** The interpreter's own
timers put ~1,090 µs of the wake `Invoke` in kernels and ~640 µs in per-op dispatch, resource-variable
bookkeeping and the reference-C glue ops (`CONCATENATION`, `STRIDED_SLICE`, `QUANTIZE`, `LOGISTIC`);
the generated function keeps the kernels and replaces the glue with `memmove` on the rings. The
command model tells the same story at ten times the scale — 28.4 ms of kernels, 13 ms of dispatch
and reference-C `MEAN` — which is also why the spec's "2× faster" target is *not* met: 1.45× is what
removing all interpreter overhead is worth on a genuinely arithmetic-bound graph. The lever left is
the model, not the runtime. Two lessons from the deploy shape. Moving a model from an interpreter to
generated code converts a heap allocation into a linker placement, and a linker placement has no
failure path: an internal-SRAM arena that looked free left 8,431 B at recogniser start and the record
task silently never spawned, which is how every task-creation return value in the firmware came to be
checked. And a review found each model carving its esp-nn scratch out of its own arena while esp-nn
reaches scratch through file-static globals shared by both — a 4,336 B write past the end of the wake
arena whenever the wake task preempted the recogniser; hence the single shared region and a mutex.

**Per-layer profile.** With every generated kernel call wrapped in a cycle counter, the w48 command
model's 42.3 ms `Invoke` is accounted for to within 38 µs, and the table names the speed lever: the
**first conv, on a single input channel, takes 15.1 ms — 36 % of the invoke — for < 5 % of the MACs**
(14 MAC/µs against 254–262 for the three pointwise convs that follow). One input channel cannot fill
esp-nn's channel-packed vector path. On the wake model the residual outside the kernels is 17 % of a
1.35 ms step, because a few-hundred-MAC op is dominated by its call setup. Neither is optimised yet.

### 6.13 Command model: capacity, data recipe, and a deploy rule (2026-09-04 → 09-07)

After §6.11, five command-model deploys and five rejected candidates followed, all scored the same
way: `eval_recordings` over the full QC-approved set, every speaker labelled against the training
manifest, plus a **deploy rule** fixed before the second candidate was trained — *aggregate
real-voice words ≥ the deployed model's, false accepts no worse, and the weakest speaker
improved* — applied as two hard filters and a ranking objective. Isolated-word accuracy per speaker
(n in the header; all *user-customised, in-training* unless marked held-out):

| Model (w = width) | spk01 (13) | spk02 (38) | spk10 (146) | spk18 (36) | all | FA |
|---|---|---|---|---|---|---|
| v3 QAT w32, 2 speakers (§6.11) | 0.615 | 0.737 | 0.479 ho | — | 0.538 | 3/29 |
| + spk10 in training, w32 | 0.538 | 0.605 | 0.678 | — | 0.655 | 1/29 |
| **w48**, same data | 0.923 | 0.895 | 0.856 | 0.333 ho | 0.785 | **0/32** |
| w48, TTS regenerated, per-clip gate | 0.846 | 0.895 | 0.815 | 0.667 ho | 0.807 | 1/32 |
| w48, voice gate + van augmentation | 0.769 | 0.684 | 0.829 | 0.667 ho | 0.777 | 0/32 |
| w48, `--real-weight 3`, QAT 20 (grid) | 1.000 | 0.974 | 0.932 | 0.778 ho | 0.919 | 0/32 |
| **w48, same recipe, + field takes** (deployed) | **1.000** | **1.000** | **0.952** | **0.778** | **0.936** | **0/32** |
| + per-clip van draws (run 4) | 0.923 | 0.974 | 0.938 | 0.750 | 0.914 | 2/32 |

Four findings. **Capacity, not crowding.** Adding a third speaker at width 32 cost both existing
speakers 8 and 13 points; width 48 recovered both past their two-speaker numbers *and* lifted the
third, which is what "the model ran out of parameters" predicts and "the third speaker crowded the
others out" does not. Width 40 was trained too and must not be deployed at any accuracy: 40 channels
miss esp-nn's 16-alignment fast path, so a model 28 % smaller in MACs asks for **59,632 B** of
internal scratch against width 48's 29,824 B. On the device the wider model's `Invoke` rose 1.55× on
2.05× the MACs — MAC-scaling predicted 59 ms, the chip measured **46–47 ms**; a fine way to decide
whether to measure, a bad number for a table. **The synthetic split and the microphone disagree.**
The w32 retrain moved held-out test accuracy 91.2 → 90.7 % while real-voice words went
0.538 → 0.655; the regenerated-TTS candidates score 62–72 % on their own-era test sets while
beating the deployed model on real speech. The TTS-dominated split is not a proxy for a microphone,
though it is not anti-correlated either — it saw the width change. **Real-clip weighting is the
dominant lever, not width.** An 8-run grid (width {32, 48} × real-weight {1, 3} × QAT epochs
{10, 20}) on one fixed dataset: every `real-weight 3` row beats its counterpart by 0.15–0.32
aggregate points, width by ~0.1–0.15 at matched recipe, and QAT length reads as a small
false-accept/aggregate trade on top. Three rounds of *data* work — regenerating the TTS cache, the
voice gate, van augmentation — each cleared the false-accept and weakest-speaker bars and missed the
aggregate by a few points; one *recipe* change cleared all three by the widest margin yet, at
identical MACs and bytes, and survived a from-scratch rebuild with 285 new field clips folded in
(0.919 → **0.936**, the deployed `86b7105e`). The plain-recipe retrain on the same rebuild is the
control: 0.807 with a new false accept, so more real data alone did not do it. **A negative result,
reported as one.** Two hygiene fixes — a fresh (noise, RIR) pair per real clip instead of one per
build, and best-validation-epoch selection — were re-run with the deployed recipe: the checkpoint
was a no-op (best epoch 40/40), and the per-clip draws scored below the deployed model on all three
rule figures (0.914, 2/32 false accepts). The same hygiene pass took the dataset build from ~25 min
to **97 s** by replacing direct convolution with `fftconvolve`, which is what makes the obvious next
step — the same recipe at several seeds — cheap.

The rule's weakness is stated with it. Its cells are small — spk18 is 36 clips, spk01 13, one clip
per word — and a single false accept on spk10's 19 negatives appeared and vanished between otherwise
identical runs twice, so the rule can neither distinguish a 0.02 regression from seed noise nor
reject a candidate on one negative with confidence. Seed variance is being measured; until it is,
the rule is a ratchet, not a test.

### 6.14 Wake word, rounds 5–7: real clips as a safety knob, an alignment bug, and real false fires

Round 5 (§6.3) gave ten real "Hey Bus" takes 71 % of every positive batch, which reads like textbook
overfitting; round 6 was run to correct it and found the opposite. Cutting the real share to 50 %
and 30 % left recall pinned at the ceiling (every real positive, held-out session included, at
0.996) and moved only the safety side: TTS near-miss false fires 1–5 → 15 → 33 of 48, and three
ordinary commands in the held-out session fired the wake word. Ten real recordings are a far
narrower target than nine thousand Piper clips, and that narrowness is what rejects the near-miss
family; the fix for two-speaker overfitting is more real speakers, not less real weight.

The same rounds found why the wake word felt slow. Spliced into real room tone and measured from the
phrase *end*, the deployed model fired **1.06 s** late (median; 1.20 s worst), on a second
probability hump — the first, at the phrase end, peaked below the gate. The guided takes carry
~1.04 s of trailing silence, the trainer keeps the *last* 1.5 s of each window, so the positive label
sits a second after the phrase and the model learned the offset. Trimming the real clips before
feature generation took latency to **≤ 0.04 s** at identical size, and its one regression — TTS
near-misses 9/48 — was closed by synthesising that family ("hallo bus", "der bus kommt gleich") as
hard negatives in four voices, which improved the *unseen* voices too (2/36, against 11/36). Two
measurement bugs were fatal only once clips got short: the probe reused one interpreter whose
`CALL_ONCE` init subgraph runs once per lifetime, so a clip scored 0.031, 0.316 or 0.996 depending on
what ran before it, and it padded 0.5 s of silence against a 1.9 s receptive field. Every
round-1–5 conclusion survived the corrected probe.

Round 7 is the first trained on the deployed model's **own real false fires**: one conversation
session with nobody addressing the device (spk19, spk20) produced 16 fires, 10 on ordinary speech.
Split 50/50 by clip, the held-out halves became the acceptance rows:

| Gate | Round 6d (deployed) | Round 7 |
|---|---|---|
| Real false-fire speech, held out (5) | 5/5 | **1/5** |
| Near-silence fires, held out (3) | 3/3 | 1/3 (the residual is a clipped clip) |
| New real positives, held out (13) | 12/13 | **13/13** |
| Held-out real non-wake (9), worst peak | 0/9, 0.402 | 0/9, 0.285 |
| TTS non-wake, seen (46) / unseen (36) | 4 / 2 | 5 / 4 |
| Fire latency, held out (median) | 0.14 s | **0.02 s** |

The cost lands where the round expected, on the synthetic gate, and it is the same trade round 5's
user-customisation decision already made once. The wake model is, deliberately, tuned to the people
and the room it lives with; the false-accept rate on *generic* German conversation remains
unmeasured beyond 2.9 minutes of room tone and one session.

### 6.15 Sentence-level performance: the honest headline gap (2026-09-07)

Everything above scores isolated words or the wake phrase. The deployed model was re-measured today
on the full QC-approved set with `kws-eval --recordings --width 48 --qat`, same manifest as §6.13;
phrases are read command sentences (guided and field-derived) decoded through the streaming detector
and the grammar exactly as on the device, and a phrase counts only if the *exact* intent comes out:

| Speaker | Isolated words | E2E phrase → exact intent | Negatives, false accepts |
|---|---|---|---|
| spk01 | 1.000 (13) | — | — |
| spk02 | 1.000 (38) | 0.000 (4) | 0/10 |
| spk10 | 0.952 (146) | 0.082 (97) | 0/19 |
| spk18 | 0.778 (36) | 0.176 (17) | 0/3 |
| spk19 | 0.688 (16) | 0.222 (9) | 0/10 |
| spk20 | 0.800 (20) | 0.083 (12) | 0/1 |
| **All** | **0.911 (269)** | **≈ 0.10 (14/139)** | **0/43** |

The classifier is at 0.94 on words on the deploy scoreboard, 0.91 over all six speakers, and
0 false accepts in 43; the streaming decoder over the same speakers' read sentences yields ≈ 0.10
exact intents. Every command-model change in §6.13 moved phrase intent by at most a few points
(spk10: 0.062 → 0.082 → 0.113 → 0.082 across four models), and a read-only sweep of the decoder's
threshold and hangover over 88 field takes moved device–transcript agreement from 3/18 to 4/18 with
0/70 false fires. **The word→sentence path — segmentation, run-based decoding, the grammar's
all-or-nothing parse — is the bottleneck, not the classifier.** The field figures say the same thing
from the device side: of 125 field takes, 68 passed QC and 36 carried a parsable intent (0.288),
and the device's own answer agreed with the transcript on 0.125 (spk18), 0.500 (spk19) and 0.500
(spk20) of the parsable takes — measured against the model deployed at capture time (`8fa81d08`),
not today's — with 18 false alarms at either wake gate. §6.2's catalog result (0.689 on synthetic
phrases, in-domain) was never a real-speech number, and this is the real-speech number.

## 7. Limitations and future work

- **The sentence gap is the finding and the open problem.** Isolated words at 0.91–0.94, exact
  intents from read sentences at ≈ 0.10 (§6.15). Nothing in the classifier's ablation arc moved it;
  the word→sentence path has had one read-only decoder sweep. The candidates, in order: an n-best
  lattice parse over the existing posteriors so the grammar weighs in before the decoder commits
  (specified, unrun), per-word segmentation diagnostics on the 125 failing phrases, and the
  streaming transducer of §6.6 once phrase data is no longer 392 synthetic sentences.
- **Real-speech evaluation is thin, and the deploy rule inherits that.** Six speakers, most cells
  13–38 clips, one clip per word for two of them; a single false accept on 19 negatives flips the
  rule and has done so between identical runs. Seed variance is being measured; grouped speaker
  k-fold is planned once enough speaker groups cover every word. Every real-voice number is
  *user-customised*: the held-out speakers of one round are in training the next, by design.
- **Synthetic data.** 15 of 21 command words have no real clips beyond the device's users, so those
  per-word numbers are not real-speech performance; the voice gate proves a voice is German and
  intelligible, not that its rendering of a 0.3 s word matches how people say it.
- **The combined mode is measured for cost and field agreement, not recall.** A field take exists
  only where the wake fired, so missed wakes remain unobservable; the false-accept rate on generic
  German conversation is one session and 2.9 minutes of room tone.
- **Per-device robustness.** The catalog result is lighting-dominated and long compound words fail
  end-to-end, on an unprobed segmentation conjecture.
- **The largest on-device cost left is one layer.** The single-channel first conv is 36 % of the
  command `Invoke` for < 5 % of its MACs (§6.12); a wider first stride or folding it into the
  following depthwise is the identified lever, unrun. The spec's sub-1 ms wake step is not met at
  1.25 ms and the runtime cannot close it — 1.1 ms of it is kernel time.

## 8. Conclusion

A German, intent-level voice assistant runs fully offline on a commodity ESP32-S3: a 25,832 B INT8
model at 46–47 ms per streaming step and a 1.25 ms wake step, both as generated C that is bit-exact
against the interpreter it replaced, a locally-trained wake word, and a pure slot grammar keeping
intent validity out of the model. What we most want to travel is methodological: provenance stated
on the test axis as well as the training axis, a machine check for the one property of synthetic
data nobody verifies by ear, and a device-to-dataset loop with a fixed deploy rule that turned the
project's biggest assumptions into measurements. Those measurements are the findings. On real speech
the stock model recognised a quarter of what it claimed on its own split; closing that gap for named
users took capacity and a 3× weight on real clips more than it took data, and is a design choice with
a stated price. And a word classifier at 0.94 composes into an intent recogniser at 0.10: the next
gap is not in the model.

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
