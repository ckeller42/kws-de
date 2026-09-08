# Paper / talk notes — running research log

Raw material for a concise conference-style paper + slide deck. Every substantive result,
design decision, and lesson lands here **as it happens** — with real numbers only.
Working title: *"Offline German voice control on a microcontroller: keyword spotting with a
slot grammar on the ESP32-S3."*

## 1. Problem & motivation

- Goal: hands-free German control of camper functions (lights, heater, fridge, roof, …),
  **fully offline** on an always-on, low-power device (ESP32-S3, M5Stack CoreS3).
- Constraint driving everything: no cloud, no Pi awake 24/7 — the voice satellite must be
  the cheap always-on part.
- Gap: no open-source on-device speech-to-intent for MCUs exists. Vendor stacks:
  ESP-SR/MultiNet = Chinese/English only (no German); Picovoice Rhino does German
  slot-filling but is closed-source. → Our niche: **open, German, MCU-class, intent-level**.

## 2. Contributions (draft list)

1. A reproducible **model factory** for German KWS on ESP32-S3: MSWC subset fetch → MFCC →
   DS-CNN → full-INT8 TFLite-Micro, with CI **resource-budget gates** (size/MACs/INT8/ops)
   that prove device-fit without hardware.
2. An honest treatment of **synthetic-data provenance**: per-word real/TTS counts, headline
   metrics computed on real speech only, TTS-heavy classes explicitly flagged.
3. A **two-stage architecture**: reused wake-word engine (microWakeWord, shown to train
   locally on Apple Silicon — no Colab/GPU) gating a streaming keyword detector + a **pure,
   device-specific grammar** (`device → zone? → action`, per-device action validity) — the
   open-source alternative to closed speech-to-intent.
4. A **full-command-catalog end-to-end evaluation**: every valid intent synthesized and run
   through audio → MFCC → streaming detector → grammar → intent.

## 3. Experiments & results (real, measured)

### E1 — v1 single-word KWS (5 toggle words, 7 classes)

| Metric | Value |
|---|---|
| Headline real-speech INT8 accuracy (TTS excluded) | **91.1 %** (n=775, mixed 20/10/0 dB SNR) |
| Float accuracy (same subset) | 91.7 % |
| Full-model INT8 (incl. TTS-augmented classes) | 93.2 % |
| `_unknown_` false-accept | 0.0 % |
| SNR sweep (commands, INT8) | 93.0 % clean → 84.6 % @ 0 dB |
| Model | 19,256 B, 2,069,984 MACs, full-INT8, 5,351 params |
| Reference point | ESP-SR MultiNet ~85–95 % English, clean speech |

Per-word INT8: Kühlschrank 96.3, Camping 98.4 (TTS-heavy ⚠), Wasser 93.2, Heizung 90.9,
Licht 85.7 (short word ↔ silence confusions — see confusion matrix in eval-report).

Data provenance (the honesty story): MSWC-de real clips found: Licht 300, Kühlschrank 300,
Heizung 120, Camping 22, Wasser 0→300 (a deeper scan found real clips mid-run; headline
moved 87.9 % → 91.1 % — *worth a slide: data > model tweaks*).

### E2 — MSWC-de coverage study (why TTS is unavoidable)

Scanned ~2.5 M MSWC-de examples: of the 24 grounded v2 command words, only **7 have real
clips** (Licht, Kühlschrank, Heizung, Wasser, aus, auf, Außen[158]); 17 incl. all zone words,
`an`, and every level/mode word have **zero**. → Synthetic fill is structural, not a hack.

### E3 — v2 grounded catalog model (23 classes) + end-to-end catalog eval

- Vocabulary grounded in the camper's real controllable functions (8 devices, 4 light
  zones, 12 actions, device-specific validity map) — invalid combinations are rejected by
  the grammar, not learned by the model.
- **Before fix** (asymmetric train-set domains — see §5 war story): clip-level held-out
  accuracy 88.5%, but full command-catalog end-to-end accuracy **0.000** (152 trials =
  38 catalog entries x 4 voices). Model output `_unknown_` at ~1.00 on every streaming
  window, including word-centered ones.
- **After fix** (symmetric clean+noise domains for every class + +/-200ms random
  time-shift augmentation, retrained 40 epochs, train accuracy 92.6%): full-intent
  catalog accuracy **0.066** (device 0.066, action 0.066, zone 0.031). Real, but still
  far from usable.
- Root-caused the residual gap with a single-phrase probe (`Licht Küche an`, per-step
  top-2 posteriors dumped through the actual `KeywordStream`): individual words ARE
  recognized correctly and confidently in isolation (Licht 0.996, Küche 0.99, an
  0.996) — this is NOT a data/training problem anymore. It's a stream-composition
  problem: (1) posterior smoothing (`smooth_win=3`) makes a word's detection "linger"
  into the next word's window, causing the SAME word to re-fire a step or two after the
  audio has already moved on; (2) the resulting refractory cooldown then blocks the
  actual next word's confidence peak from ever being sampled; (3) windows straddling a
  word boundary (tail of one word + silence gap + head of the next) produce confident
  but wrong predictions for words that aren't even in the audio (e.g. a ghost
  "Heizung" between "Küche" and "an") — the model was never trained on inter-word
  transition windows, only isolated single-word clips. Net effect: `grammar.parse`
  sees duplicate-device / duplicate-action / missing-slot patterns and rejects.
  Tried one extra round of stream-parameter tuning (`smooth_win`/`threshold`/
  `refractory` combos) on the same phrase — none resolved both failure modes at once
  (loosening refractory to stop under-firing just causes more duplicate over-firing).
  Diagnosis: the level-triggered threshold + global refractory conflates two
  different jobs (debounce a lingering same-word detection vs. gate the next
  word) into one cooldown knob, so no single value can satisfy both.
- **Decoder fix** (edge-triggered run-based decoding, `kws_de/stream.py::KeywordStream`
  rewritten — see §4): full-intent catalog accuracy **0.066 -> 0.362** (152 trials,
  same 38 catalog entries x 4 voices, default params `step_ms=100, smooth_win=3,
  threshold=0.5, min_consecutive=2, gap_steps=2`; device 0.362, action 0.362, zone
  0.156). One tuning pass over `(step_ms, smooth_win, threshold, min_consecutive,
  gap_steps)` (6 combos, posteriors cached across combos to reuse the same TTS audio)
  found no combo beating the defaults:

  | step_ms | smooth_win | threshold | min_consecutive | gap_steps | accuracy |
  |---|---|---|---|---|---|
  | 100 | 3 | 0.5 | 2 | 2 | **0.362** (default) |
  | 100 | 2 | 0.5 | 2 | 2 | 0.362 |
  | 100 | 3 | 0.6 | 2 | 2 | 0.296 |
  | 50 | 3 | 0.5 | 3 | 3 | 0.342 |
  | 50 | 3 | 0.5 | 2 | 2 | 0.243 |
  | 100 | 3 | 0.5 | 3 | 2 | 0.309 |

  Per-entry pattern: bare `device action` (2-token) entries mostly score 0.25-1.0
  (several perfect: Heizung, USB, Energie-Normal), while zoned `Licht` (3-token)
  entries mostly score 0.0-0.25 (zone-slot accuracy only 0.156) — each additional
  word adds another boundary-transition window where a 1-step ghost or a
  same-word-linger can still slip through even with `min_consecutive=2`. Residual
  gap is consistent with the run-based decoder having fixed the two *decoding*
  failure modes (same-word re-fire, next-word swallowing) while the *upstream*
  cause of boundary-transition ghosts — the model never trained on inter-word
  transition audio — remains, and compounds with phrase length. Flagged as a
  follow-up (multi-word transition augmentation), not further stream-param tuning.

- **Transition-aware training (NEGATIVE RESULT)** — added inter-word boundary windows
  labeled `_unknown_` + in-context positives, retrained: full-intent catalog accuracy
  **0.362 → 0.197** (REGRESSION), zone slot → 0.000. Root cause (probe): the `_unknown_`
  transition negatives out-weighted the word positives (model predicted `_unknown_` 17.4%
  vs 7.7% true; clip-level acc 88.5% → 78.1%). The model over-corrected toward "say nothing",
  killing recall — worst on multi-word phrases where any missed word fails the whole intent.
  **Lesson: naive transition-negative labeling trades boundary-precision for recall and loses
  net.** Correct fix (future work): class-balance the transition negatives (cap at ~half the
  per-word positive volume) and/or class-weight; or move boundary handling entirely into the
  decoder. Best committed model remains the decoder-fix (0.362). — a genuinely publishable
  ablation: the obvious data fix makes it worse.

- **Balanced transition fix + final command set (BEST so far)** — cut transition `_unknown_`
  negatives (n_pairs 2000->600) + inverse-frequency `class_weight` in training, on the trimmed
  command set (4 devices; dropped readout-only Wasser + Campingmodus/USB/Energie; added Licht
  brightness levels 25/50/75/100 %). Full-intent catalog accuracy **0.362 -> 0.689** (196 trials,
  49 entries x 4 voices; zone slot **0.156 -> 0.789**). This reverses the E3 negative result:
  balanced correctly, transition-aware training + class-weighting is a large net win.
  BUT the 0.689 is **Licht-dominated** (40 of 49 catalog entries are Licht, mostly 0.75-1.0
  incl. the new brightness words); the non-Licht devices are weak — **Kühlschrank 0.00 (all 3),
  Heizung ~0.25, Aufstelldach 0.6** — a per-word/streaming issue worth its own probe (Kühlschrank
  is a long word with 300 REAL clips, yet 0/4 end-to-end -> likely stream-segmentation of the
  long compound, not data). Honest headline: strong on lights, per-device work remains.

### E4 — wake word, local training (DONE, real numbers)

microWakeWord (the ESPHome/HA wake engine) is documented as Python-3.10 + Colab/GPU. We
trained a custom German **"Hey Bus"** wake model **end-to-end locally on an M4 laptop** — no
Colab, no discrete GPU:

| Metric | Value |
|---|---|
| Model | 62,304 B INT8 streaming TFLite (input [1,3,40] int8, output [1,1]); passes the <=150 KB wake budget |
| Training | full 10,000-step upstream config, ~6m41s on CPU/Metal |
| Positives | 2,000 synthetic (Piper `de_DE-mls-medium`, "hey bus"/"hej bus"); negatives = mWW's ~5.9 GB ambient/no-speech/speech sets |
| Best checkpoint | val recall 71.65 %, precision 100 %, avg-viable-recall 0.649, ~2.9 false-accepts/hour |
| @ cutoff 0.99 | false-reject 0.39, **2.0 false-accepts/hour** |

First-pass / untuned: 39 % miss at the only low-FA cutoff, 100 % synthetic positives (no real
human "Hey Bus"), no RIR reverb aug. Needs on-device threshold tuning + real recordings. But
the **feasibility claim is now proven with numbers**: a custom German wake word trains on a
laptop in minutes. (5 packaging/dependency blockers were fixed inside the training venv only —
mWW's pip package ships without its `layers/`+`audio/` subpackages, `datasets>=4` drops script
loading, PyTorch 2.6 `weights_only` breaks Piper checkpoint loading, etc. — worth a footnote.)

### E5 — TTS voice diversity (real ablation, DONE)

Hypothesis: synthetic-data quality for KWS is dominated by **voice diversity**, not
per-voice fidelity. Setup: macOS `say`-only baseline (9 voices) vs multi-engine
(say + Piper, 4 German neural voices: thorsten-medium, eva_k-x_low, ramona-low,
karlsson-low), balanced round-robin via `kws_de.tts.voice_combos`, cycled back
through the balanced pool to match the baseline's 300-clips/word target so
training-set volume is (nearly) identical (29442/6017 rows vs baseline's
29985/5474) — isolating diversity, not data quantity, as the variable.

**Result: diversity did NOT help overall — full-intent catalog accuracy fell hard,
0.689 -> 0.245** (device/action slot 0.714 -> 0.245, zone slot 0.789 -> 0.219;
clip-level held-out accuracy 0.866 -> 0.779). But the effect is not uniform, and
where it isn't uniform is informative:

| Device (weight in 49-entry catalog) | say-only | say+piper | Delta |
|---|---|---|---|
| Licht (40/49, 82%) | 0.794 | 0.200 | **-0.594** |
| Kühlschrank (3/49) | 0.000 | 0.000 | 0.000 |
| Heizung (4/49) | 0.188 | **0.750** | **+0.563** |
| Aufstelldach (2/49) | 0.625 | 0.500 | -0.125 |

The hypothesis is confirmed exactly where it was expected to matter most: Heizung
— one of the two weak non-Licht devices flagged in E3 — improved sharply (+0.56)
with multi-engine training data. It does nothing for the other weak device:
Kühlschrank stayed at a flat 0.000 in both configurations, so that failure was
never a diversity problem (something else — grammar/decoder or Kühlschrank's own
100%-real-clip vocabulary — is the actual blocker there, unexamined here). The
net regression is entirely a Licht story: Licht carries 82% of catalog trials and
its accuracy collapsed 0.794 -> 0.200, swamping Heizung's gain in the overall
average. Likely mechanism (not verified further, out of scope for this ablation):
the command model is a single shared 23-class classifier, not one head per
device — Piper's more heavily-accented/varied renderings on the *other* classes
plausibly shifted the shared decision boundary against Licht's real-speech
distribution (Licht is 100% real MSWC clips, unaffected by the TTS-fill change
itself). A per-device or curriculum-weighted engine mix (Piper only where it
demonstrably helps, e.g. Heizung) is the natural follow-up; not attempted here.

Honest caveat on ablation cleanliness: an earlier pass of this same experiment,
run with the balanced-pool cap left at its natural size (32-72 clips/word instead
of cycling back up to 300), collapsed to 0.066 — almost entirely a **data-volume**
confound (train rows 17862 vs the matched-volume run's 29442), not a diversity
effect; that run is not the number reported above. The cycling fix
(`_tts_combo_plan` repeats the balanced engine-pool combos, relying on Piper's
per-call stochastic `noise_scale` to keep repeats non-identical) closed that
volume gap before the real 0.689 -> 0.245 comparison was drawn.

### E6 — sim-to-real gap (planned, needs the physical CoreS3)

Estimated vs MEASURED on-device performance (latency/arena/CPU/power) and clean-corpus vs
real-mic accuracy. All numbers so far are synthetic/clean-corpus; the CoreS3's dual-MEMS +
ES7210 + ESP-SR 2-mic AFE and the van acoustics are an unmodeled domain. Spec:
`docs/superpowers/specs/2026-09-01-on-device-hw-mic-followup-design.md`. Expected result: real
accuracy below synthetic eval; quantifying that gap is the contribution.

### E7 — runtime optimisation wave 2 (2026-09-03, measured on the CoreS3)

Everything below is measured on the device, before and after, in one session. Wave 1 had left
the recogniser step at 82–85 ms with `Invoke` at 52–53 ms and no visibility into what those
53 ms were made of.

**Instrumentation first.** esp-tflite-micro's esp-nn kernel wrappers already accumulate
microseconds per replaced op; zeroing them around `Invoke` and logging the residual
(`Invoke` minus the kernels) turned the whole exercise from guesswork into arithmetic.
Baseline: `Invoke` 52.9 ms = conv 21.9 + depthwise 17.5 + FC 0.17 + softmax 0.22 +
**residual 13.2 ms**. The audit had estimated the residual (the reference-C `MEAN`) at ~3 ms;
it is a quarter of the whole inference. Wake: step 3477 ± 185 µs.

**Arenas right-sized (2026-09-03).** The generated arena size was the desktop interpreter's
sum of all int8 tensors × 1.2 — 139,264 B against the 55,024 B TFLM actually uses, because
the planner reuses buffers. Pinning the size to the measured need let the command arena into
internal SRAM: **step 85 → 66 ms, `Invoke` 52.9 → 40.5 ms (−23 %)**, depthwise alone
17.5 → 8.5 ms. The residual did not move (13.2 → 13.5 ms), which is the clean confirmation
that `MEAN` is compute-bound in reference C rather than starved of bandwidth.

**Quad-I/O flash + 1 kHz tick (2026-09-03).** Both are defaults Espressif's own CoreS3 BSP
ships. **Step 66 → 57 ms, front end 3.09 → 2.01 ms per frame**, `Invoke` unchanged — the
front-end gain is flash bandwidth on the mel table, and `Invoke` not moving confirms the arena
is no longer the bottleneck. The 1 kHz tick turns the wake catch-up loop's `vTaskDelay(1)`
from a 10 ms sleep into a 1 ms yield: wake step 5.53 → 4.81 ms.

**Banded mel filterbank (2026-09-03).** A triangular mel filter is non-zero over 4–33 of the
241 FFT bins, so the dense `KWS_MEL[40][241]` was 95 % exact zeros — 38.5 KB of flash rodata
read *in full for every frame*. Emitting per-band `(start, len, weights)` (459 floats) cut the
**front end 2.01 → 0.46 ms per frame** and the **step 57 → 44 ms**. Bit-identical, not
approximate: only exact zeros are dropped and the surviving terms accumulate in the same
order, so host MFCC parity is unchanged to the digit (5.4e-4, 0 LSB). Across both waves the
front end went **8.5 → 3.09 → 2.01 → 0.46 ms per frame, an 18x cut**.

**64 KB data cache, and who gets the SRAM (2026-09-03).** Only one arena fits internal SRAM —
the largest contiguous DRAM region is ~76 KB against 64 + 40 KB of arenas — so this is a
choice. Doubling the data cache (32 KB of internal RAM) covers the wake path's real working
set of a 31 KB arena plus a 58 KB weight blob: **wake step 4910 ± 1197 → 1902 ± 481 µs, 2.6x
faster with 60 % less spread**. It also makes PSRAM much cheaper — moving the command arena
out of SRAM costs 12.4 ms at a 32 KB cache and 2.3 ms at 64 KB — which is what makes giving
the SRAM to the always-on model affordable. Recogniser step 43 → 46 ms in exchange.

**Core pinning (2026-09-03).** Both inference tasks moved off LVGL's core 1 (priority 4, 5 ms
tick, preempting a 40 ms `Invoke` repeatedly) to core 0. Kept on the *variance*, which is what
preemption actually costs: wake step 4790 ± 1450 → 4910 ± 1197 µs (mean +2.5 %, spread −17 %),
recogniser `Invoke` 39.8 → 39.1 ms.

**Wake-gated duty cycle — the deployment shape (2026-09-03).** Always-on recognition is a
measurement baseline, not a design: measured, it costs **315 ms of inference per wall
second**. Assist mode runs only the wake model continuously (~1.9 ms per 30 ms of audio) and
opens a 2.5 s window for the recogniser on each wake fire. Measured with one interaction per
10 s: **253/1000 of wall time active, 97 ms of inference per wall second — 3.2x less CPU**,
and the ratio scales with interaction rate (at one interaction a minute it is ~16 ms/s, a 20x
cut). Both modes emit the same `KWS_DUTY` log line so the two are directly comparable.

**Shipped model (2026-09-03).** The final firmware carries the QAT v3 command model
(`kws-export --qat --prefix features_v3`), **INT8 test accuracy 0.9123**, against 0.7475 for
the v2 post-training-quantised model the optimisation work was measured on. Same architecture
and width, so the timings are unchanged (`Invoke` 41.4 ms, step 45–46 ms, front end 0.485 ms
per frame, wake step 1.95 ms) — the accuracy comes from QAT and the v3 dataset, not from
anything in this wave, and the wave's numbers are not affected by the swap.

**Net across wave 2: recogniser step 85 → 46 ms, front end 3.09 → 0.46 ms per frame, wake
step 3.48 → 1.90 ms — and the duty-cycle design takes always-on inference cost from 315 to
97 ms/s at one interaction per 10 s.**

#### Rejected, with the number that rejected it

- **`MEAN` → `AVERAGE_POOL_2D`.** The 13.5 ms residual is a third of `Invoke`, so this was the
  biggest single prize on the table. It fails on arithmetic, not speed: TFLite's int8 average
  pool has no output rescale, so the converter must give its output the *input's* scale
  (0.265) where `MEAN` picks its own (0.021). Quantising the pooled 32-value embedding — the
  vector the classifier reads — 12.6x more coarsely moved **90 LSB maximum output delta,
  4,879 of 5,474 test rows changed, accuracy 74.75 % → 74.17 %**. A depthwise convolution with
  constant 1/490 weights *is* bit-exact (0 LSB on all 5,474 rows, identical accuracy) because
  it requantises like any other conv — but esp-nn's s16 scratch for a 49×10 filter needs a
  **78,464 B arena**, which does not fit internal SRAM at all, forfeiting a larger win. `MEAN`
  stays. The clean follow-up is esp-nn's unused `esp_nn_mean_s8_esp32s3.c`, which would claim
  the 13.5 ms without touching the model.
- **`CONFIG_NN_SKIP_NUDGE`.** Advertised at "~20 %"; measured at **0.7 ms** (`Invoke`
  40.0 → 39.3 ms, step 44 → 43 ms). Not worth a documented ±1 LSB on all ~188k requantisations
  per `Invoke` when no host test can audit it — the host runs neither esp-nn nor TFLM. This is
  why the boot log now prints a numeric fingerprint (the golden MFCC vector through the real
  interpreter, as 23 int8 outputs): device arithmetic changes are otherwise invisible.
- **32 KB instruction cache.** With the data cache already at 64 KB: recogniser `Invoke`
  41.6 → 41.5 ms (inside the noise) and wake step 1902 → 1818 µs. 0.08 ms for 16 KB of SRAM.
- **Wake weights copied to internal RAM.** Needs 58,080 B; 47,539 B internal remains after the
  arenas (largest block 31,744 B), and copying them would leave 17.8 KB — under the 24 KB
  floor. The 64 KB data cache already covers those weights, which is most of why the wake step
  is 1.9 ms.

#### Lesson worth a slide

Three of the four rejections were rejected by a *measurement that did not exist before this
wave*. The kernel timers turned "MEAN is ~3 ms" into "MEAN is 13.5 ms"; the boot fingerprint
turned "skip-nudge is probably fine" into a question the host cannot answer. Optimisation
work on a microcontroller is mostly building the instrument.

## 4. Method details worth a figure

- Two-stage always-on architecture (LikeC4 diagrams in `docs/likec4/` — reuse for slides).
- Golden-vector MFCC contract: host (librosa) and device (esp-dsp) front-ends pinned
  bit-level by a fixed-input fixture test — kills the classic silent train/deploy mismatch.
- Budget gates as CI: "fits the MCU" as a unit test (≤500 KB / ≤3 M MACs / INT8-only ops).
- Speaker-disjoint splits everywhere; TTS clips split by voice+rate combo (a synthetic
  "speaker") so voices never straddle train/test.
- Streaming detector: posterior smoothing (trailing mean) + threshold, decoded with
  **edge-triggered run-based decoding**: a run of consecutive steps sharing the same
  qualifying top-1 label fires its label once, as soon as the run reaches
  `min_consecutive` steps, with no global cooldown — a different label's run may fire
  immediately after its own run qualifies, and the same label may fire again only
  after >= `gap_steps` non-matching steps since its run ended. Replaced an earlier
  level-triggered threshold + global-refractory debounce (found + fixed a real
  off-by-one there first: decrement-then-check let a sustained word re-fire) that
  conflated same-word debounce and next-word gating into one cooldown knob and could
  not satisfy both at once (see §3 E3).

## 5. War stories / lessons (talk material)

- **Data beats scanning:** Wasser had 0 clips after 2.5 M examples scanned; more scanning
  was pure waste — the fix was TTS + (later) a lucky deeper index. Know when coverage has
  converged.
- **Silent failure modes:** `subprocess.run("say", …)` without a timeout deadlocked a
  12-way thread pool → whole data build hung at 0.1 % CPU. Bound every external call.
- **`python -m pkg.mod` ≠ console script** when the module lacks a `__main__` block —
  an entire train→export→eval chain "succeeded" as a no-op. Verify artifacts, not exit codes.
- **License landscape for German TTS** (HF survey): permissive core = Piper + Parler-TTS
  (Apache); XTTS-v2/MMS are non-commercial — matters for a public/commercial artifact.
- **The model learned the noise floor, not the words:** `build_dataset` added command-word
  clips ONLY noise-mixed (one row per SNR, no clean copy) but `_unknown_` clips ONLY
  clean. The model didn't learn to recognize words at all — it learned "clean audio
  implies `_unknown_`, noisy audio implies some command," a trivial shortcut orthogonal
  to the actual task. Clip-level held-out accuracy was **88.5%** on this broken data,
  because the held-out test set shared the same asymmetry — a perfectly consistent,
  perfectly wrong signal. The giveaway was the SNR sweep improving as noise got *worse*
  (0.000 clean, better at high noise) — backwards for a real word-recognition model. The
  end-to-end command-catalog eval (full audio -> stream -> grammar -> intent, scored
  0.000) caught what the clip-level split-eval metric could not, because clip-level eval
  inherited the same broken assumption the training data did. Fix: every class sees the
  same audio domains (one clean + one noise-mixed copy per SNR), plus random time-shift
  augmentation so words are recognizable at any window offset, not just clip-start.

## 6. Slide-deck skeleton (draft)

1. The van + the problem (photo, "no cloud in the mountains")
2. Why not X? (MultiNet no German / Rhino closed / Rhasspy needs a Pi)
3. Architecture (LikeC4 deployment + device views)
4. The data problem: MSWC coverage chart (7 of 24 words) → TTS strategy
5. Honest metrics: real-speech headline vs TTS-inflated numbers (E1 table)
6. E2E catalog eval: "Licht Küche an" through the whole pipe (sequence diagram)
7. Budget gates: fitting 19 KB / 2 M MACs (CI screenshot)
8. Results (E1 + E3 tables, SNR sweep plot)
9. Lessons (Section 5)
10. Open source: repo + what's next (firmware, on-device demo)

---
*Update discipline: append to §3/§5 with every landed run; numbers only from committed
eval reports.*

### E7 — architecture benchmark (DONE)

Frozen dataset, 30ep/seed0, class-weighted, val-selected, test-reported. Catalog = 3-voice subset (147 trials/arch), clean dataset (no transition aug) so lower than the 0.689 tuned model — apples-to-apples ranking:

| arch | isolated | catalog | params | MACs | INT8 | device |
|---|---|---|---|---|---|---|
| ds_cnn | 0.834 | 0.544 | 5879 | 2.07M | 20KB | yes |
| bc_resnet | 0.773 | 0.102 | 4919 | 1.39M | 31KB | yes |
| matchboxnet | 0.903 | 0.245 | 12957 | 0.47M | 43KB | yes |
| kwt | - | - | 106k | - | 173KB | NO (non-TFLM ops) |

Findings: metric-dependent ranking (matchboxnet best isolated+lowest MACs; ds_cnn best catalog); bc_resnet underperforms at tiny scale; KWT INT8-exports but non-device-runnable (op-set is the gate). Chose matchboxnet as CTC encoder for E8.

### E8 — streaming CTC transducer (the new model)

MatchboxNet encoder + per-frame CTC head, 392 phrases/60ep. NEGATIVE (accuracy): loss 370->28 but greedy decode collapses to all-blank -> catalog 0.000 vs 0.689. Originally two blockers: (1) CTC data-hungry (392 phrases too few, all-blank collapse); (2) streaming encoder won't INT8-export (TensorListReserve, non-TFLM — same op-set gate as KWT). Frame-classifier+grammar (0.689, 20KB) stays the deployable system. Lesson: a principled arch still needs data scale + export-friendly form to win on-device.

### E8b — export blocker RESOLVED (fix/ctc-export)

Blocker (2) fixed. Root cause: `TimeDistributed(Dense)` head unrolled to a `tf.while` loop -> `TensorListReserve`, unlegalizable under INT8-builtins-only (no SELECT_TF_OPS). Fix, all in `build_ctc_encoder`: (a) head -> `1x1 Conv2D` (identical per-frame projection, one static CONV_2D, no loop); (b) `t_frames` param — None=variable-T training graph unchanged, concrete=fixed-T **batch-1** export clone (weights T/batch-independent -> `set_weights` transfers trained weights); (c) static reshapes (freq->channels, head squeeze) instead of tf.shape-reshape/Permute. Result: exports 42.9KB full-INT8, op set = {CONV_2D, DEPTHWISE_CONV_2D, ADD, RESHAPE, DELEGATE} — all TFLM builtins. Batch-1 was the last mile: a None batch made TFLite recompute Reshape shapes at runtime via SHAPE/STRIDED_SLICE/PACK; fixing batch folds them out. Fixed-T + batch-1 = the honest on-device shape (one chunk, ring buffer). TDD: `test_ctc_encoder_fixed_t_int8_exports_tflm_clean` asserts zero non-TFLM ops + full-int8 (reproduces E8's ConverterError as the red test). Accuracy still 0.000 (blocker (1), data — deliberately out of scope for this fix). Paper §6.8 now: cause 1 (data) open, cause 2 (export) resolved.

### Training throughput (feat/real-speech-distill)

2026-09-01. Machine: Apple M4, 10 cores, 16 GB. `uv run kws-train --epochs 2` on the frozen v2 features (`data/features_{train,val,test}.npz`), wall-clock total for the 2-epoch run (includes fixed npz-load/model-build/save overhead, not isolated per-step time):

| config | s/epoch |
|---|---|
| CPU, batch 32 | 11.4 (22.8 s / 2 epochs) |
| CPU, batch 128 | 9.8 (19.5 s / 2 epochs) |
| Metal, batch 128 | not run — plugin failed to load (TF 2.21; resolved below) |

Metal decision: **dropped** (`uv sync`, no `--extra metal`). `uv sync --extra dev --extra tts --extra metal` resolved and installed `tensorflow-metal==1.2.0` cleanly, but importing `tensorflow` then raised `NotFoundError` at plugin-load time: the Metal plugin dylib could not resolve TF's internal `_pywrap_tensorflow_internal` symbol library.

`tensorflow-metal` (last published for TF ≤2.16-era ABI) doesn't load against TF 2.21 — an ABI break in the plugin loader, not a config issue on this machine. No GPU device was ever listed, so the decision rule's bar (device present AND ≥1.3x CPU@128) can't even be evaluated; restored with `uv sync --extra dev --extra tts` (no `--extra metal`). `uv.lock` unchanged by any of the sync calls (extras already resolved in the lock).

CPU@128 vs CPU@32: ~1.17x faster — modest, as expected for models this tiny (per-step overhead, not compute, dominates at batch 32).

batch 128 from E9 on; E7 numbers were batch 32 — not re-run.

**Update 2026-09-02 (fix/tf-metal-pin).** Pinned `tensorflow>=2.16,<2.19` (resolves to 2.18.1; Keras stays 3.15) so `tensorflow-metal` 1.2.0 loads — `tests/test_metal.py` asserts a GPU device is listed whenever the metal extra is installed on Apple silicon (red on 2.21, green on 2.18). Two findings once it ran:

1. Metal's CTC kernel returns NaN (`test_ctc_train_smoke_loss_decreases`); `transducer._ctc_loss` now pins `tf.nn.ctc_loss` to CPU on every backend — the op is negligible next to the encoder.
2. It is slower. Per-epoch time on the frozen v2 train split (20,116 rows, batch 128, `train()` timed directly, fixed overhead subtracted via a 2-vs-4-epoch difference):

| model | CPU | Metal | Metal/CPU |
|---|---|---|---|
| DS-CNN (5.9 k params) | 5.1 s | 5.6 s | 0.91× |
| KWT teacher (106 k params) | 6.1 s | 10.6 s | 0.58× |

Decision rule was "keep Metal only if ≥1.3× CPU@128" — fails on both models: kernel-launch overhead dominates at 49×10 inputs and the MFCC inputs are already precomputed, so there is no GPU-shaped work. `uv sync` without `--extra metal` stays the default; the extra now works for anyone who wants it, and the pin is what keeps it working.

### E9/E10 — distillation + balanced calibration (feat/real-speech-distill)

2026-09-01. Frozen v2 features (`data/features_{train,val,test}.npz`, 23 classes). Command:
`uv run kws-distill --features features --epochs 40 --seed 0` (~15 min wall-clock on Apple M4 CPU,
batch 128: KWT teacher + DS-CNN baseline + DS-CNN distilled student, each 40 epochs, plus INT8
export/eval ×3 and a 3-voice catalog TTS pass per row). Report: `docs/distill-report.md` /
`docs/distill-benchmark.json` (untracked, like the transducer report). Teacher (KWT) float test
accuracy: **0.894**.

| Architecture | Float | Isolated | Catalog | Params | MACs | INT8 | Budget |
|---|---|---|---|---|---|---|---|
| ds_cnn (first-200 calib) | 0.862 | 0.842 | 0.218 | 5,879 | 2,070,496 | 20,224 | yes |
| ds_cnn (balanced calib) | 0.862 | 0.853 | 0.259 | 5,879 | 2,070,496 | 20,224 | yes |
| ds_cnn distilled (balanced calib) | 0.842 | 0.833 | 0.667 | 5,879 | 2,070,496 | 20,272 | yes |

E9 (distillation): isolated accuracy fell (float 0.862->0.842, INT8 0.853->0.833, both -2.0 pts) but
catalog jumped 0.259->0.667 (+40.8 pts, 2.6x) — a system-level win despite a slightly worse per-clip
number; consistent with §6.2's "isolated accuracy is not the task."

E10 (calibration): float->INT8 gap 2.0 pts with `X_train[:200]` calib, 0.9 pts with balanced calib —
recovers 1.1 of 2.0 pts (55%) on this run. (E7's originally-quoted 1.63-pt gap was a different run,
30ep/batch32; this run's own first-200 row, 40ep/batch128, is the baseline the recovery is measured
against.)

QAT decision (spec §5 gate: >1% absolute balanced-calib gap -> QAT next spec, else closed): measured
balanced gap **0.9% < 1%** -> **QAT closed as unnecessary**.

### TTS breadth + perturbation (feat/real-speech-distill)

Audit of `raw_clips_merged.pkl` (the v2 build) found the TTS backstop was macOS `say` only: 9 voices
x 9 rates, 6 622 clips — and TTS speaker ids were `tts:{engine}:{voice}:{rate}`, so the same voice at
two rates could land in both train and test (a rate-in-speaker-id split leak, TTS rows in the
speaker-disjoint split were not actually disjoint). Fix (Tasks 10-11): Piper voices discovered from
the local cache (multi-speaker voices expanded per speaker) alongside `say`, speaker id dropped to
`tts:{engine}:{voice}` (rate becomes augmentation, not identity, closing the leak), and every TTS
clip gets one pitch/tempo-perturbed copy at build time. v2 feature files are untouched (frozen); the
effect is measured with v3.

### Paper maintenance (2026-09-01)

- §5 now states the deployed architecture layer by layer (stem 3×3/32 → 3 × DS block → global
  mean → Dense 23; 5 879 params, 2.07 M MACs, five TFLM builtins) and the KWT teacher (d 64,
  depth 3, 4 heads, MLP 128, 106 k params) plus the E9 loss with T = 4, α = 0.5.
- §4.4 added: split sizes (20 116 / 4 101 / 4 042 rows), per-word real-vs-TTS provenance from the
  frozen v2 features (4 real-only, 2 mixed, 15 TTS-only command words — the old "17 of 23" counted
  the 24-word vocabulary), and worked examples for clip → class, events → intent, rejections.
- Paper gate: `scripts/check-paper.sh` fails `git push` / `gh pr create` when a branch changes
  `kws_de/`, `firmware/` or a results report without touching paper.md or paper-notes.md
  (`PAPER_SKIP=1` for pure refactors).

### On-device firmware — CoreS3 dual-mode deployment (feat/cores3-firmware)

2026-09-02. The int8 command model now runs on the target hardware (M5Stack CoreS3,
ESP32-S3), pinned to ESP-IDF v5.5.5, built reproducibly in Docker (`espressif/idf:v5.5.5`),
final image 0xedf00 bytes (~950 KB, 69 % of the 3 MB app partition free). Two modes:

- **Guided recorder** — collects real word/sentence/negative takes onto flash (`/rec/spkNN/…`,
  numeric speaker ids only), end-pointed by an **energy VAD** (RMS over 20 ms frames vs an
  adaptive noise floor, 2-frame open / 500 ms trailing close) that replaces esp-sr's AFE VAD:
  no model partition, no flash cost, and host-testable. Pulled over **USB mass storage** with
  `scripts/pull-recordings.sh`. This is the collection path for the v3 real-speech dataset the
  paper's §4.4 provenance table is waiting on.
- **Recogniser** — the same MFCC front-end as `kws_de.features` (the C port matches Python to
  **1.83e-4** max abs error, host-checked against committed test vectors) feeding the int8 model
  under TFLite-Micro with a `MicroMutableOpResolver<7>` holding exactly the five builtins the
  export uses (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX — plus RESHAPE, ADD),
  arena 139 264 B in PSRAM, then the same `KeywordStream` detector as the host. Confirms the
  op-set gate and the front-end parity claims the paper makes are real on-device, not just in
  simulation. Device throughput (`infer_ms`, real `arena_used`) still to be logged on hardware.

Reproducibility: config-derived C headers are checked current by `kws-fwgen --check` (CI
`gen-fresh` gate) — structure byte-exact, float tables (mel/DCT/window, TV_MFCC) within a
tolerance, because those are computed through numpy/scipy kernels whose SIMD (CPU-feature)
and BLAS (reduction-order) paths are not bit-reproducible across machines; model headers via
`kws-export --v2 --firmware`. Follow-up: switch the `--firmware` INT8 calibration from
`features_v2_train[:200]` to `export.balanced_calibration` (now available post-merge; the
distill run showed balanced calibration recovers ~1.1 pt isolated / ~4 pt catalog) and re-export.

**First on-device bring-up (2026-09-02, fix/cores3-psram-quad).** Flashed to a real CoreS3
over the network (esptool, `write_flash @flash_args`). Two `sdkconfig.defaults` corrections the
compile-only reviews could not catch: the CoreS3 uses **quad** SPI PSRAM, not octal
(`CONFIG_SPIRAM_MODE_OCT` boot-looped on `octal_psram: chip not connected`; `..._QUAD` →
`Found 8MB PSRAM, memory test OK`), and `CONFIG_TINYUSB_MSC_BUFSIZE` must be ≥ the 4096-byte WL
sector (default 512 aborted `storage_mount`). After both, it boots clean and the guided recorder
runs end to end — energy VAD end-points speech and writes `/rec/spk01/<word>/001.wav` (observed
for `kueche`, `waermer`, `licht`). Known minor: the first save formats the FAT and blocks CPU0
long enough to trip the idle-task watchdog once (WDT-panic off, non-fatal, never recurs). Lesson
for the paper's "reproducible on-device" claim: board-specific memory config is the real
bring-up cost, invisible to a host build.

**Recorder UX + recogniser bring-up (2026-09-02, fix/cores3-recorder-ux).** Live testing on
the device drove a round of recorder fixes (umlaut font subset, layout fit, two reads per
word for later misread review, paced get-ready/between-read beats, a single colour-coded
"SPEAK NOW" pill instead of full-screen tinting, mic gain +6 dB) — reviewed by streaming the
LVGL framebuffer over the serial console as RLE-packed base64 (a gated debug tool). The
recorded set passed a technical QC (21/21 words, 16 kHz mono 16-bit, 0.8–1.3 s, non-silent,
non-clipped; levels ~−18 dBFS peak before the gain bump).

The recogniser's first on-device run produced near-uniform outputs (top-1 ≈ 0.12, never
firing). Systematic debugging isolated it to the **saved `command.keras` being a
mode-collapsed artifact** — 0.3 % accuracy on its own training set, ~3 classes predicted for
everything — not to the firmware: the C MFCC matches Python to 5e-4 and a fresh model on the
same data reaches 65 % val in 4 epochs. Retrained (87 % train / **74.8 % INT8 test**) and
re-exported; on-device the model is now confident (0.5–0.8) and the detector fires. Two
guards so it cannot recur silently: `kws-export --v2 --firmware` runs a **model-health gate**
(≥50 % held-out accuracy, ≥10 predicted classes) before writing the device header, and a
pure regression test covers both failure modes in CI.

Residual finding worth a paper sentence: with a healthy model, real mic speech of a command
word is still classified **`_unknown_`** (0.7–0.8) — the TTS-dominated v2 training set does
not generalise to the real microphone, which is precisely the gap the recorder collects data
to close (v3). Perf: the naive one-shot DFT cost ~1.2 s/inference on the S3 (an integer
modulo per multiply plus a `-Og` build); an incremental twiddle index + `-O2` fixes it, and
the recogniser was moved below the LVGL task priority so touch stays responsive during
inference. Firmware headers now carry Doxygen docs; requirements are traced to tests with
sphinx-needs (see `docs/sphinx`).

**Streaming front-end (2026-09-02, perf/streaming-mfcc).** The recogniser recomputed all 49
MFCC frames of the trailing second every step; it now keeps a persistent 49-frame log-mel ring
and pushes only the frames that arrived since the last step. Measured on the CoreS3 (`-O2`):
**1001 ms → 173 ms per step** (5.8×). The residual is the naive 480-point DFT at ~12 ms per
frame — with a ~270 ms loop period that is still ~14 new frames per step — so the next lever is
an exact 480-point mixed-radix FFT (kissfft, being vendored for the wake-word front-end), which
should bring a step to ~10 ms and make the on-device recogniser genuinely real-time.

**Data provenance housekeeping (2026-09-02).** All datasets and models now live under one
`KWS_DATA_ROOT` on the external SSD, shared by every worktree, with immutable per-version
snapshots in `archive/<version>/` (v2 = the frozen 20 116 / 4 101 / 4 042 set + manifest +
the models and E9/E10 report). The paper's provenance table regenerates from a snapshot, and
the device-recording ingest gets a canonical home (`data/recordings/`) for the v3 build.

**Remote-controllable USB mode (2026-09-02, feat/usb-cdc-console).** The serial console
(`mode`/`status`) used to go dark the moment the device entered USB mode: TinyUSB's MSC device
takes the USB PHY, and the console's own port rides that same PHY, so it vanished along with
it — the automated data-ingest loop had no way to leave USB mode again except a physical touch
on the screen. Fixed by making the USB device composite: MSC ("KWSREC") plus a CDC-ACM serial
port, with stdio redirected onto the CDC port for the duration of USB mode
(`firmware/main/usb_drive.c`) and restored on exit. One real bug surfaced building it: the
console task's `fgets(stdin)` used a blocking UART read, so a mode switch triggered from a
different task (e.g. tapping the menu) while the console task sat blocked in that read could
leave it parked forever, deaf to the new CDC port — fixed by making stdin non-blocking
(`O_NONBLOCK`), matching how TinyUSB's own CDC read already behaves, so the console task never
blocks past one 20 ms poll tick on either side of the switch (`firmware/main/console.c`).

**Console input root cause (2026-09-03).** The serial console had only ever worked by
accident: the CoreS3's USB-C is the ESP32-S3's own USB-Serial-JTAG peripheral (no UART bridge),
IDF mirrors stdout onto it as the *secondary* console, but `stdin` stays on the unconnected
UART0. Making stdin non-blocking for the CDC hand-over then broke the accidental path
completely (`fgets` dropped every partial line). The console now reads the USB-Serial-JTAG
driver with a bounded wait (and the CDC-ACM port in USB mode) and assembles lines itself;
verified on the device, and a reusable host-side helper that opens the port with DTR/RTS low
(a careless open resets the chip) replaced the ad-hoc `cat`/`echo` capture.

**Wake model v4 on the device (2026-09-03).** Retrained the "Hey Bus" model with TTS hard
negatives (near-misses, the command vocabulary, everyday sentences), reverb augmentation and
multi-voice positives (the mls checkpoint's speakers plus the project's other German Piper and
macOS voices; two Piper voices held out for the probe): 9,000 + 9,000 clips, 20k steps, 58,080 B.
Host probe: "hey bus" fires in 3 of 4 probe voices (v1: 1 of 4), but one seen voice still peaks
0.99 on "licht küche an". On the device with a real speaker the 2 s peak trace reads 0.83-0.99 on
"Hey Bus" (v1: 0.13) and <= 0.44 on silence/room noise, so the gate moved from 0.99 to 0.85 (x2
consecutive steps, 1.5 s refractory); a synthetic clip played through a laptop speaker fires 3/3.
Real "Hey Bus" takes from the recording session are the next positives; false-accept rate on
real speech is still unmeasured.

**Leaving USB mode (2026-09-03).** First fully remote ingest: `mode usb` over the JTAG console,
the CDC-ACM port appears next to the mounted drive, pull, `mode menu` over CDC — and then no USB
device at all on the host: once TinyUSB releases the PHY the USB-Serial-JTAG peripheral does not
re-enumerate until a physical re-plug. Leaving USB mode now restarts the chip (menu in ~2 s,
console back), which is what the command means anyway; a PHY re-attach for the JTAG peripheral
is the cleaner upgrade.

**Wake model round 5: real positives, user-customised by design (2026-09-03).** Ten real "Hey Bus"
takes (two sessions of a main user, pulled and QC-approved by the remote loop) were added to the
round-4 recipe as their own feature set (sampling weight 5). Through the firmware int8 feature
path at the device gate (0.85 × 2 steps): round 4 fires on 4 of 10 real takes, round 5 on **10 of
10** (peak 0.996 on every clip); a variant trained on one session only fires 5 of 5 on the other,
unseen session. The TTS non-wake worst peak fell 0.988 → 0.758. The price is generic-voice
margin: a synthetic Piper "hey bus" clip played through a laptop speaker drops from 3 of 3 fires
(0.96–0.99) to 0 of 3 (0.59–0.64). This is the intended trade: the wake model is customised to
the device's main users, the same "user-customised, in-training" policy the command model
follows, and each new speaker's five takes go through the same loop.

**Is a bigger wake model free? (2026-09-03, negative result).** Hypothesis from the runtime audit:
the streaming model is overhead-bound (45 compute ops for 24,736 MACs, 3 ms per step), so more
capacity should cost nothing. Three variants trained on the identical round-5 data: wide (channels
×1.5: 47,856 MACs, 90,224 B), deep (+1 block, kernel 25: 30,432 MACs, 71,304 B), both (59,472
MACs, 111,928 B). Compute is indeed nearly free (same 45 ops for wide), but the arena grows 49 →
82 / 61 / 106 KB, and everything above the ~66 KB of internal SRAM the wake model may take falls
back to PSRAM, which gives back the 5 → 3 ms win (predicted 5.4 / 3.5 / 6.4 ms per step from the
calibrated cost model). Detection: all variants keep 10 of 10 real takes (a training-set score),
wide has the best unseen-voice margin, but false fires on 48 German non-wake clips at the device
gate rise from 2 (round 5) to 14 / 9 / 10; microWakeWord's own false-accepts-per-hour stays 0.000
for all but "both" (0.75/h at 0.85), a floor effect of its English ambient set. Kept round 5. The
missing measurement is an unseen-speaker real-take set.

**Second real session and the loop end to end (2026-09-03 evening).** A main user's guided session
(spk10, 98 sentences + 19 negatives) filled the 10 MB recordings partition four sentences short;
the remote pull recovered all 117 takes. With the recorder's per-set hangover fix, takes are
1.96–3.66 s (median 2.43 s; the first session's were cut at 0.84 s) and QC approved **116 of 117**
(first session: 65 of 208), levels median −31 dBFS, none clipped; segmentation yielded 146 word
clips and skipped 112 where Whisper's word spans did not cover every keyword (a QC gap to close).
Rebuilding v3 with all three speakers (train 32,399 rows) and QAT fine-tuning: INT8 held-out test
accuracy 0.907. On the recordings: spk02 isolated words 0.553 → 0.605; spk10 0.473 (as an unseen
speaker on the previous model) → **0.678** in training, phrases end to end 4 → 8 of 97, false
accepts 3 → 1 of 19. Phrase-level recognition on real speech remains the open problem.

### On-device wake word — isolated "Hey Bus" test mode (feat/wake-test-mode)

Added a dedicated `UI_MODE_WAKE` that runs **only** the microWakeWord streaming model, so the
wake stage can be measured on hardware without the command recogniser confounding it. The
interesting engineering point for the paper: microWakeWord's accuracy is only reproducible
on-device if the *feature front-end* matches training bit-for-bit, and that front-end is not
the librosa MFCC the command model uses — it is TFLite-Micro's fixed-point 40-channel
microfrontend (30 ms window, 10 ms step, 125–7500 Hz, PCAN on, log scaling), followed by an
integer requantisation `int8 = (v * 256 + 333) / 666 - 128` that folds training's historical
÷25.6 float scaling into the model's 0…26 → −128…127 int8 range. Rather than reimplement it,
we vendored the same C the trainer's Python bindings compile and gated it with a host parity
test: 98 × 40 int8 feature values against a `pymicro-features` golden vector, **max deviation
0 LSB (exact)**. The streaming graph itself is stateful (resource variables), so the
interpreter is created once and invoked every 3 rows (30 ms), with variables reset on mode
entry. Detection is threshold 0.99 × 2 consecutive steps + 1500 ms refractory, confirmed by a
green screen flash and a beep — the beep forced a hardware finding worth a footnote: the
CoreS3's mic and amplifier share one full-duplex I2S channel pair, so the speaker can only be
opened at the microphone's exact sample rate or capture dies.

Follow-up (same branch): the four modes (Record/Recognise/Wake/USB) were restructured behind
one selection screen — every mode's back button now returns to it instead of chaining to
Record — and the guided recorder became a single automatic session (new speaker → sentences
→ negatives → a "takes saved" summary), removing seven manual set/next/redo buttons from the
record screen. A serial console (`mode <name>`/`status` over the same USB-serial port) lets a
host script drive mode switches for unattended data-ingest runs.

**Wake model root cause (2026-09-02, on-device).** First hardware test of the isolated wake mode:
the model never fired on a real speaker (per-2 s peak probability 0.00–0.13 while saying "Hey
Bus"), although the front-end is bit-exact. A host probe through the identical int8 feature path
explains it: the model outputs ≥ 0.99 for *any* Piper sentence in its training voice ("hallo wie
geht es dir": 62 steps ≥ 0.99, "licht küche an": 73) and ≈ 0.004 for "hey bus" in unseen Piper
voices. With all positives synthetic and all negatives real recordings, the cheapest separating
feature was TTS-vs-real, not the phrase — a shortcut the held-out metrics (71.65 % recall on the
same synthetic distribution) could not reveal. Fix in progress: TTS hard negatives (near-misses,
the command vocabulary, everyday sentences) generated with the same voices, a wider speaker
spread, and reverb augmentation; the probe with unseen voices is the acceptance test.

**Real wake positives (2026-09-02).** The synthetic-only wake training is the weak link, so the
device menu gained a "Hey Bus"-only recording session (5 single-read takes per speaker, stored as
set `wake`). These real utterances enter the same ingest → QC path as the command recordings and
give the wake retrain its first in-domain positives and, held out, the first honest recall number
on real speech — the synthetic held-out metric (71.65 % recall) said nothing about the real-voice
failure.

**Sentence takes cut after the first word (2026-09-02).** QC of the first real recording session
with Whisper found sentence takes (prompts like "Licht Küche fünfundsiebzig Prozent", median
840 ms) rejected 75/102 for missing words, against word takes (median 1020 ms) mostly fine.
Energy envelopes (RMS per 100 ms) of failing sentence takes showed one ~200 ms burst — the first
word — followed by ≥ 500 ms below threshold, then the take closing; a good take of "Licht Dach
heller" showed three bursts with 200–300 ms gaps between them. Cause: the recorder's VAD closed a
take after a fixed 500 ms of trailing silence, but a natural reading pause between the words of a
longer on-screen prompt exceeds that. Fix: the trailing hangover is now per prompt set — 500 ms
for words, 1200 ms for sentences/negatives/wake (`prompt_hangover_ms` in
`firmware/main/prompts.c`, fed into `vad_reset` in `firmware/main/record.c`) — plus a false-start
filter (`vad_t.speech_total`, `MIN_SPEECH_MS` = 200 ms) that discards a take opened by a breath or
click and keeps listening instead of saving a near-empty clip.

**Recording loop (2026-09-02, feat/recording-pipeline).** The device-recording data loop is
now a repeatable pipeline: `scripts/ingest.sh` pulls a session over SSH into a stamped,
never-deleted `incoming/<stamp>/`; `kws-qc` runs an audio gate (format/duration/level) then a
Whisper large-v3 (`mlx-community/whisper-large-v3-mlx`) content gate per take, segments
approved sentence takes into 1 s word clips centred on Whisper's word spans, and writes an
idempotent `approved/` tree; `kws-dataset build --prefix features_v3` folds it into the v3
build; `kws-eval --recordings` reports two figures that are never mixed — `held-out` and
`user-customised, in-training` (speaker-level match against the training manifest). First
real run, over an early two-speaker bring-up session (208 takes, `spk01`+`spk02`, full
command vocabulary): **65/208 approved (31%), 51 word clips written, 12 word clips skipped**
(51 approved word takes, 4 sentences, 10 negatives). Rejection breakdown: 72 `missing`
(sentence token not found/out of order), 35 `wrong_word`, 34 `too_quiet`, 2 `clipped`. The
first pass over the same session approved only **48/208 (23%)** — the difference is three QC
fixes, not a looser gate: Whisper writes the light levels as numerals ("50" for "fünfzig"),
it glues keywords into one token ("Lichtdach" for "Licht Dach"), and a single hallucinated
2-letter keyword ("An den fahren wir los" heard for "wann fahren wir los") was rejecting
clean negatives. A fourth fix — requiring a short keyword to match as a whole token — moved
the count back down from 69 to 65 by removing false approvals. Most remaining rejects are
content mismatches, not audio-quality failures — several transcripts ("Vielen Dank.",
"Test.") show these were early/placeholder takes rather than genuine misreads, so the
approval rate here is not yet a QC-strictness signal; a clean recording session is needed
before this loop's numbers say anything about QC threshold tuning. `scripts/data-loop.sh`
chains ingest → QC → build → train → export (model-health gate) → evals behind one command,
stopping at the first failing stage.

The domain gap this loop exists to close, in one number: the **stock v2 command model**
(TTS-dominated training set, no device recordings) measured on this session's real voices as
held-out data — **isolated-word accuracy 0.19** (`spk01`, 16 clips) and **0.27** (`spk02`,
45 clips), **0 of 5 phrases** correct end to end, and **0 false accepts on 6 negatives**. A
model reporting ~0.9 on its own held-out MSWC/TTS split recognises roughly a quarter of what
the real microphone hears.

**Exact 480-point FFT in the MFCC front end (2026-09-03).** The streaming command recogniser
ran at **164–181 ms per step** on the CoreS3, and the front end, not the model, was the cost:
`firmware/main/mfcc.c` computed each frame's 480-bin spectrum as a naive DFT — 241 bins ×
480 samples ≈ 116k multiply-adds per frame, ~8.5 ms of the step per new frame. 480 = 2^5·3·5
is not a power of two, which is why the DFT was there in the first place; it is, however, an
exact kissfft mixed radix (`kiss_fftr` at nfft = 480 factors its 240-point complex half
transform as 4,4,3,5, every stage a dedicated butterfly). The kissfft already vendored for
the wake front end now serves the command front end too, through a small C-linkage shim
(`firmware/main/mfcc_fft.cc`); the tempting alternative — zero-padding to 512 — was rejected
because it changes the bin spacing and therefore the mel energies the models were trained on.
Measured, same device, same firmware otherwise: step **164–181 ms → 82–85 ms**, and per new
frame **8.5 ms → 3.0 ms** (fitting step time against the 9–15 frames each step consumes).
Features did not move: host max |Δ| against the Python reference is **5.4e-4** absolute
(1.3e-6 of the reference peak) both before and after — the residual is float32-vs-float64
accumulation in the log/DCT stage, not the transform — and the int8 tensor actually fed to
the command model is **identical (0 LSB)** to the one quantised from the Python features, a
new assertion in `firmware/test/test_mfcc.c`. The wake path is untouched (5 ms/step before and
after); it runs the TFLM microfrontend, not this code. What the FFT does *not* explain is the
~54 ms fixed cost per step that the same fit exposes, independent of frame count — that is
TFLM `Invoke`, and the next note takes it apart.

**TFLM arenas in internal RAM (2026-09-03).** With the front end no longer dominant, the
recogniser step decomposes as **52–53 ms `Invoke` + ~30 ms front end**, and both TFLM tensor
arenas were being allocated `MALLOC_CAP_SPIRAM`. TFLM touches its arena on every operator, so
arena placement is the lever on `Invoke`: internal SRAM is a direct access, PSRAM goes over
the cached octal bus. `arena_alloc` (`firmware/main/arena.h`) now asks for
`MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT` first and falls back to PSRAM with a `WARN` line,
logging free internal RAM either way. The budget, measured at boot rather than assumed: the
S3's 512 KB of SRAM leaves **148,895 B free internal** by the time the models start, after
the IDF, LVGL and the audio ring. Both arenas do not fit — the wake arena is 49,152 B and the
command arena, as `kws-export` generates it, is 139,264 B. So the rule the task set applies:
the **wake model, which runs continuously, gets internal RAM** (free internal 132,063 →
**82,907 B**, comfortably above the 64 KB floor the UI and audio ring need) and the command
arena stays in PSRAM, which the boot log now says out loud instead of leaving it to be
guessed. Result: **wake 5 → 3 ms/step (−40 %)**; the command step is unchanged at 82–85 ms
with `Invoke` at 52–53 ms.

Worth recording because it is the obvious next optimisation and it is *not* blocked by the
hardware: the command model's `Invoke` only ever uses **55,024 B of its 139,264 B arena**
(TFLM's own `arena_used_bytes`). A right-sized arena would fit internal RAM with room to
spare — but 148,895 − ~60,000 − 49,152 ≈ 40 KB free internal, under the 64 KB floor, so it
trades the recogniser's latency against headroom for the UI, and the export step would have
to emit an arena size derived from the measured need rather than the current fixed margin.
That is a deliberate decision about the floor, not a free win, so it is left for its own
change.

**Quantisation-aware training, `--qat` (2026-09-03).** `tensorflow-model-optimization` (tfmot)
only wraps `tf.keras` models built under Keras 2; TF 2.18's default `tf.keras` is Keras 3, which
tfmot's `QuantizeWrapperV2` cannot wrap. Fix: `kws-train --qat` and `kws-export --qat` re-exec
themselves once under `TF_USE_LEGACY_KERAS=1` (the `tf_keras` shim) before TensorFlow is imported
anywhere in the process — the env var has to be set pre-import, so this has to happen at module
load, not inside `main()`. `kws-train --qat` loads the existing float `command_v3.keras`'s weights
(`model.load_weights` round-trips across the Keras 2/3 boundary even though a full `load_model`
does not — confirmed empirically, not assumed) into a freshly-built architecture under the legacy
runtime, wraps it with `tfmot.quantization.keras.quantize_model` (per-tensor fake-quant on every
activation, per-channel on conv/dense kernels), and fine-tunes 10 epochs at `Adam(1e-5)`. The
QAT model is saved as a SavedModel dir (`command_v3_qat/`), not `.keras` — reloading a
`quantize_model`-wrapped model from the `.keras` zip format hit a real, reproduced tfmot/Keras-3
variable-naming bug (`Layer 'quantize_layer' expected 5 variables, but received 3`) on the exact
same architecture that round-trips cleanly through `save_format="tf"`. `kws-export --qat` reloads
it under `tfmot.quantization.keras.quantize_scope()` and converts with the same `to_int8_tflite`
PTQ conversion already uses — the QAT model's baked-in fake-quant ranges do the work, no separate
code path needed for the TFLite conversion itself.

Same architecture (`build_dscnn`, 23 classes), same `features_v3` data, same `features_v3_test`
held-out split:

| Model | Test accuracy | Size |
|---|---|---|
| Float (`command_v3.keras`) | 89.4 % | 178 142 B |
| INT8 PTQ (`command_v3.tflite`, today's baseline) | 88.0 % | 18 296 B |
| INT8 QAT (`command_v3_qat.tflite`, 10 fine-tune epochs) | **91.2 %** | 17 880 B |

QAT recovers all of PTQ's 1.4-point INT8 accuracy loss and adds another 1.8 points on top of
the *float* model — the fake-quant fine-tune found a better minimum for the quantised graph,
not just a less-lossy one. Real-recordings isolated-word accuracy (`kws-eval --recordings`,
same two device speakers, `user-customised, in-training`, same manifest) moves the same
direction, not just the synthetic test set:

| Speaker | Words (n) | PTQ acc | QAT acc | Negatives (n) | PTQ FA rate | QAT FA rate |
|---|---|---|---|---|---|---|
| spk01 | 13 | 0.538 | **0.615** | 0 | n/a | n/a |
| spk02 | 38 | 0.553 | **0.737** | 10 | 0.000 | 0.000 |

False-accept rate is unchanged (0/10, both models) and isolated-word accuracy improves for both
speakers — QAT does not trade detection performance for the quantisation-error recovery; it
improves both. `export.assert_model_healthy` (50 % accuracy floor, ≥ 10 predicted classes) passes
for both INT8 models with all 23 classes represented in predictions. Held-out phrase accuracy
(4 clips, 1 speaker) is 0/4 for both models — too small an n to read anything into; the isolated-
word and false-accept figures above are the ones with enough clips to mean something.

**DS-CNN width sweep (2026-09-03).** `build_dscnn` gained a `width` parameter (default 32,
the shipped size) on every conv/depthwise-separable-block channel count, plumbed through
`kws-train --width` and `kws-export --width` (non-default widths get a `_w<N>` export-name
suffix, e.g. `command_v3_w16_qat.tflite`); `kws-export --stats` prints params + MACs
(`kws_de.budgets.estimate_macs`, already used by `kws-benchmark`) for a loaded model.
Widths 24 and 16 were trained on `features_v3` with the same recipe as the width-32 QAT
baseline above (40 epochs, `--qat --qat-epochs 10`); width 12 was skipped per the stopping
rule below since width 16 already missed by a wide margin. Distillation from the width-32
model was skipped: `kws_de.distill.distill()` only supports a KWT teacher for the fixed-width
DS-CNN student (no `width` param on the student side), a different use case than a same-
architecture narrower student — not worth threading through for widths that fail on their
own. MACs/params are architecture-only (independent of trained weights), computed directly
from `build_dscnn`; the recordings figures reuse the same two device speakers as the
baseline table, `user-customised, in-training`, isolated-word accuracy:

| Width | INT8 test acc | Params | MACs | Size | spk01 acc (n=13) | spk02 acc (n=38) | False accepts (n=10) |
|---|---|---|---|---|---|---|---|
| 32 (baseline) | **91.2 %** | 5,879 | 2,070,496 | 17,880 B | **0.615** | **0.737** | 0 |
| 24 | 88.7 % | 3,839 | 1,270,632 | 14,528 B | 0.462 | 0.605 | 0 |
| 16 | 84.7 % | 2,183 | 658,928 | 11,528 B | 0.385 | 0.500 | 0 |
| 12 | skipped (16 already missed the recommendation bar) | 1,499 | 423,636 | — | — | — | — |

Recommendation: **keep width 32.** Width 24 already misses the ≤ 1.0-point INT8-test-accuracy
bar (91.2 % → 88.7 %, a 2.5-point drop) and both narrower widths lose isolated-word accuracy on
*both* real speakers versus the baseline — narrowing the channel count trades real-voice
recognition, not just a synthetic-test-set fraction of a point. `export.assert_model_healthy`
still passed for every exported width; the export health gate itself was not touched.

### 2026-09-03 — recording storage: microSD instead of a 12 MB flash partition

The CoreS3 recorder wrote to the internal wear-levelled FAT partition, 12 MB — about
one guided session (~9.5 MB of takes; a full sentence+negative set has already run it dry),
so every session needed a USB pull before the next speaker could sit down. Recordings now
go to a microSD when one is present (`storage_root()`), which turns that ceiling into
hours: a 32 GB card holds ~47 000 takes, i.e. ~3 000x the flash budget, and the session
cadence stops being limited by storage at all. Flash stays the fallback so a card-less
device (and CI) behaves exactly as before. Practical note for data collection: cards are
the weak link — the first card tried acknowledged every write and persisted none, which is
why the mount now ends with a write-and-read-back probe before the card is trusted.

### E12 — generated inference vs. the TFLM interpreter (wake, 2026-09-04, measured on the CoreS3)

The wake model no longer runs through `tflite::MicroInterpreter`. `kws-codegen` emits the
graph as a flat C function (`firmware/main/gen/wake_infer.c`) that calls esp-nn's ESP32-S3
kernels directly, with the streaming ring buffers as plain static arrays;
`CONFIG_KWS_INFER_GENERATED` picks the path, and in the default build the interpreter is
compiled *out*: no `MicroInterpreter`, no resource variables, no 40 KB tensor arena. Both
builds measured on the device in one session, same configuration otherwise, two minutes of
the 2 s peak trace each (medians over the trace windows):

| | TFLM interpreter | generated (esp-nn) |
|---|---|---|
| wake step | 1891 µs | **1281 µs** (−32 %) |
| model evaluation alone | 1735 µs | **1220 µs** (−30 %) |
| within-window spread | ±502 µs | **±151 µs** |
| model memory | 40,960 B heap arena (31,388 B used) + 1 KB variable arena | **15,680 B arena + 4,200 B ring state, all `.bss`** |
| free internal RAM once wake is up | 58,511 B | **81,371 B** (+22,860 B) |
| app image | 1,165,872 B | **1,098,992 B** (−66,880 B: no interpreter, no kernel set) |
| output on live device audio | `parity: out byte generated 71, interpreter 71` | identical |

The memory row is what changes the deployment shape: the tensor arena is not *also*
allocated, it is gone, and the 22.9 KB that frees is internal SRAM — the scarce kind. (It did
not buy the command model a seat: its 65,536 B arena needs one contiguous block and the
largest is 31,744 B, so the recogniser still runs from PSRAM.) `CONFIG_KWS_INFER_PARITY_LOG=y`
re-links the interpreter and re-allocates the arena — that is the developer-verification
build, not the shipped one.

**Why it is faster is not "better kernels" — they are the same esp-nn kernels.** The kernel
timers say so: in the interpreter run, conv + depthwise + FC is ~1,090 µs of the 1,735 µs
`Invoke`, and the remaining ~640 µs is per-op dispatch, resource-variable bookkeeping and the
reference-C glue ops (`CONCATENATION`, `STRIDED_SLICE`, `QUANTIZE`, `LOGISTIC`). The generated
function keeps the ~1,090 µs of kernels, replaces the glue with `memcpy`/`memmove` on the rings
and a 256-entry LUT, and lands at 1,220 µs total. **The interpreter's overhead was a third of
the wake inference**, and the variance collapses with it: no allocator and no per-step tensor
bookkeeping competing with the LVGL task.

That also settles what is left of the spec's "wake step well under 1 ms" target: it is **not**
met at 1.28 ms, and code generation cannot close the gap — ~1.1 ms of the 1.22 ms is esp-nn
kernel time for *this* model, so the remaining lever is model size (channels, layers), not the
inference runtime. Recorded as open rather than quietly restated.

Bit-exactness is the point, and it is checked at three levels: `wake smoke: 0/64 steps differ`
(synthetic vectors, model-free, runs in CI), `wake parity: 0/635 steps differ (11 clips,
4200 B state)` (the ten approved "Hey Bus" takes, needs the data root), and on the device
itself once per mode entry on live microphone features. The generated arena's esp-nn scratch
block is sized by a Python port of `esp_nn_get_conv_scratch_size_esp32s3`, emitted into the
header as `WAKE_INFER_SCRATCH_BYTES`; the firmware asks the real function on the real chip at
boot, gets 15,552 B — exactly what the port reserved — and refuses to run the generated path
if it ever comes back larger, because that failure mode is a silent overrun into the ring
state rather than a crash.

### E13 — generated inference for the command model (2026-09-04, measured on the CoreS3)

The same treatment for the 23-class DS-CNN, and with it the interpreter leaves the firmware
entirely: `firmware/main/gen/command_infer.c` is 10 straight-line esp-nn calls (CONV_2D, then
three DEPTHWISE_CONV_2D / 1x1 CONV_2D pairs, MEAN, FULLY_CONNECTED, SOFTMAX), no ring state,
one static arena. Both builds measured on the device in one session, ~100 s of recognition
each, medians over the ~5 s trace lines with the cold first one dropped:

| | TFLM interpreter | generated, arena PSRAM (default) | generated, arena internal |
|---|---|---|---|
| recognise step | 46.0 ms | **33.0 ms** (−28 %) | **31.0 ms** (−33 %) |
| model evaluation alone | 41,710 µs | **28,726 µs** (−31 %) | **26,983 µs** (−35 %) |
| arena / state | 65,536 B in PSRAM (54,824 B used) | 51,248 B in PSRAM, 0 B state | 51,248 B in internal `.bss`, 0 B state |
| free internal at recogniser start | 36,231 B | **59,679 B** | 8,431 B |
| every app_main task created? | yes | yes | **no** — record's 8 KB stack fails |
| recognise task stack high-water | 6,368 B of 10,240 | 6,516 B of 10,240 | 6,436 B of 10,240 |
| app image | 1,165,696 B (as built at `78fa92c`) | **1,001,616 B** | 1,001,632 B |
| `selftest int8 out:` (23 bytes, golden vector) | `-128,…,-36,…,0,-94,…,-127,…` | byte-identical | byte-identical |
| live `parity:` line (PARITY_LOG=y) | — | **`parity: 0/23 output bytes differ`** | build too tight to run |

**Same story as E12, at ten times the scale.** The interpreter's own kernel timers attribute
28.4 ms of its 41.7 ms `Invoke` to conv + depthwise + FC + softmax and ~13.0 ms to the
`rest` column — dispatch, tensor bookkeeping, the reference-C `MEAN`. The generated function
keeps the kernels and deletes the 13 ms. So the spec's "command Invoke at least 2x faster" is
**not** met: 1.45x with the shipping arena placement (1.55x with the arena internal) is what
removing all interpreter overhead is worth here, because unlike the wake model this one is
genuinely arithmetic-bound (49x10x32 activations through three depthwise blocks). Recorded as
open, like E12's sub-1 ms target; the lever left is the model, not the runtime.

**Arena placement turned out to be a memory question, not a speed one.** The generated arena is
one static array, so where it lands is settled at link time — no allocation, no contiguity
requirement — which made internal SRAM look free. It is not: 51,248 B is more internal memory
than this board has spare, and taking it left 8,431 B at recogniser start. Task stacks must be
internal (`CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY` is off, and `SPIRAM_MALLOC_ALWAYSINTERNAL`
only redirects `malloc`), so the *next* task `app_main` creates simply did not exist:

```text
E (27435) record: record task (8192 B stack) not created: free internal 8035, largest block 7680
```

Record mode would have sat in `REC_IDLE` for ever, queueing into a queue nobody drained. It was
invisible because every `xTaskCreatePinnedToCore` in the firmware dropped its return value;
they all go through one checked helper now (`firmware/main/task.h`), which is how the line
above exists at all. The fix proper is a Kconfig choice defaulting to PSRAM: 1.7 ms of the
~28.7 ms evaluation buys back 51 KB of the scarcest memory on the board, and the interpreter
it is being compared against ran its own arena from PSRAM anyway. Internal stays as the
opt-in for measurement builds. **The general lesson is worth keeping for the paper: moving a
model from an interpreter to generated code converts a heap allocation into a linker
placement, and a linker placement has no failure path — it succeeds and something else
starves.**

Bit-exactness on the device is now checked two ways rather than one. `CONFIG_KWS_INFER_PARITY_LOG=y`
fits once the arena is in PSRAM, and logs `parity: 0/23 output bytes differ` on live microphone
features on the first step after entering the mode (that build is still tight enough to lose
the record task — logged, and acceptable for a verification build). Independently, the
`selftest int8 out:` line prints all 23 output bytes for the golden MFCC vector on *every*
build, and is byte-identical between the interpreter and generated builds. Off-device:
`command smoke: 0/368 bytes differ` (16 synthetic windows, model-free, runs in CI) and
`command parity: 0/1564 bytes differ (68 clips, arena 51248 B)` (4 synthetic vectors plus 64
real test-split windows, needs the data root).

**A trap found on the way.** `models/command_v3_qat.tflite` is *not* the model the firmware
runs. A retrain rewrites the `.tflite` without touching `firmware/main/gen/model_data.h`, and
the two had already diverged (17,912 B / `f985f282` on disk against the 17,880 B / `fc36da9f`
the device's `KWS_MODEL_ID` names — different weights, not a re-serialisation). Generating
from the `.tflite` would have shipped a generated path computing a different model from the
one `model_config.h`'s quantisation constants describe, with nothing failing loudly. So
`kws-codegen` reads the embedded C array directly (`codegen.model_bytes` accepts a
`model_data.h`), which makes the model the device runs the single source of truth — and, as a
side effect, puts the command model's byte-exact freshness check inside CI, where the wake
model's could never go because `models/` is not in the repository.

### E14 — one esp-nn scratch region for both models (2026-09-04, measured on the CoreS3)

Review finding on E12/E13's arrangement, and a real latent bug: each generated model carved
its esp-nn scratch out of the end of its own arena, on the theory that a crossed pointer would
at worst read the other model's scratch. Scratch is a **write** target, and the two reserves
differ by 4,336 B. esp-nn's kernels reach it through file-static globals — one per kernel
family for the whole image, not one per model — so in assist mode, where the wake task
(priority 3) preempts the recogniser (priority 2) on the same core, the command model's
depthwise could run with the wake arena's pointer and write 4,336 B past the end of a
15,680 B array in internal `.bss`. Silent, heap-adjacent corruption; small per step, and
assist mode runs both models continuously.

Fixed in two halves. The generator now emits one shared region, `kws_infer_scratch`, sized to
the widest op of *any* shipped model (19,888 B), 16-byte aligned, in internal `.bss`, pointed
at once per inference entry rather than once per kernel; the arenas hold activations only.
The firmware serialises the two evaluations on one mutex (`firmware/main/infer_lock.h`) — the
models contend only inside an assist window, and the wait is bounded by one command inference,
which the wake task absorbs because it reads from an audio ring holding a second of history.
Re-measured, same session, same method as E12/E13:

| | E12/E13 (scratch inside each arena) | E14 (one shared region) |
|---|---|---|
| wake arena / state / scratch | 15,680 + 4,200 B (scratch inside the arena) | **128 + 4,200 B, + 19,888 B shared** |
| command arena / state / scratch | 51,248 B PSRAM (scratch inside) | **31,360 B PSRAM, + 19,888 B shared internal** |
| wake step | 1,281 µs | **1,250 µs** |
| wake evaluation | 1,220 µs | 1,248 µs |
| wake within-window spread | ±151 µs | **±97 µs** |
| recognise step (arena in PSRAM) | 33.0 ms | **31 ms** |
| command evaluation (arena in PSRAM) | 28,726 µs | **27,283 µs** |
| command evaluation (arena internal) | 26,983 µs | 26,981 µs |
| free internal, recogniser start, arena internal | 8,431 B (record task lost) | **23,879 B** (every task created) |
| free internal, wake up | 81,371 B | 77,291 B (−4,080) |
| free internal, recogniser start | 59,679 B | 55,239 B (−4,440) |
| app image | 1,001,616 B | 1,002,560 B (+944: the scratch-query functions) |
| `selftest int8 out:` | `-128,…,-36,…,0,-94,…,-127,…` | byte-identical |

**The interesting row is the command model's.** Its scratch used to live in the PSRAM arena;
sharing put it in internal RAM, and that alone recovers nearly all of E13's arena-placement
gap for 4,336 B of internal RAM instead of 51,248 B. The Kconfig choice that was worth
1.7 ms is now worth **0.3 ms** (27,283 → 26,981 µs, both re-measured), because the part of the
working set that wanted the fast memory is already in it: the kernels hit scratch on every
output row and the activations far less often. The choice also stopped being a starvation
risk — with only 31,360 B of activations to place, the internal build leaves 23,879 B free at
recogniser start and every `app_main` task is created, where E13's 51,248 B left 8,431 B and
lost the record task. PSRAM stays the default anyway: 31 KB of the scarcest memory for 0.3 ms
is still a poor trade, and the headroom is what keeps the next task from hitting that line.
E13's conclusion stands but sharpens: the placement question is not "arena in PSRAM or not",
it is "which part of the working set is worth internal SRAM" — and answering it per-part
bought both the speed and the memory.

The wake numbers moved within the noise of the trace (the step is a mean per 2 s window, the
evaluation figure a single sample from it); the fix adds one mutex pair and three pointer
stores per step. Bit-exactness is unchanged at every level: `wake smoke: 0/64`,
`command smoke: 0/368`, `wake parity: 0/635`, `command parity: 0/1564`, and the device's
always-on `selftest int8 out:` line byte for byte.

Two smaller things from the same review, both about guards that checked one half of a
symmetric pair. The boot scratch guard's dimensions used to be a hand copy of the generated C
into `wake.cc`/`recognise.cc`; a regenerated model would have left it querying the *previous*
geometry, getting an answer that fits, and passing — the exact failure it exists to prevent,
with a green boot log in front of it. The generator now emits `<model>_infer_scratch_query()`
from the same dimensions it emits the kernels from, and the firmware calls that; the device
answers 15,552 B and 19,888 B, exactly what the Python port reserved. And the model-stamp
drift check covered the command pair only, though the wake model is the one whose `.tflite`
CI can never see; it now loops over both.

### E12/E13 memory rows, superseded

Every arena figure in E12 and E13 predates E14: the esp-nn scratch was inside those arenas
and is now a separate shared region. The step and evaluation timings in E12/E13 stand as
measured; E14's table gives the current ones.

### E15 — deploying the session-2 command export (2026-09-04, measured on the CoreS3)

E13 closed on a trap rather than a fix: `models/command_v3_qat.tflite` was a newer export than
the model the firmware embedded, so the device had been running the *previous* training loop's
command model since the session-2 rebuild. This entry closes it. The deploy is three
regenerations and a flash — `kws-export --firmware --qat --prefix features_v3 --model
command_v3.keras` rewrites `gen/model_data.h` + `gen/model_config.h` and the `KWS_MODEL_ID`
stamp, then `kws-codegen firmware/main/gen/model_data.h --name command` regenerates
`gen/command_infer.{c,h}` + `gen/command_smoke_vectors.h` **from the newly embedded bytes**,
which is what makes the generated path and the stamp the same object by construction (E13).

| | old (was on the device) | new (deployed) |
|---|---|---|
| `KWS_MODEL_ID` | `command_v3_qat.tflite@fc36da9f 2026-09-03` | `command_v3_qat.tflite@f985f282 2026-09-04` |
| size | 17,880 B | 17,912 B |
| INT8 held-out test accuracy | 91.2 % | **90.7 %** |
| training speakers | spk01, spk02 | spk01, spk02, spk10 |
| input scale / zero point | 3.17866826 / 71 | 3.18189716 / 71 |
| generated ops / arena / scratch | 10 / 31,360 B / 19,888 B | unchanged |

**One incidental confirmation worth keeping.** Re-running the export reproduced
`f985f282` byte for byte from the QAT SavedModel — the TFLite conversion is deterministic
across runs on this host, so the bytes flashed are exactly the bytes evaluated below, not a
re-conversion that happens to be close.

**The real-voice comparison, both models on the same clips.** The deciding measurement is not
the held-out test accuracy in the table above — that went *down* — but `kws_de.eval`'s
`eval_recordings` over the full QC-approved set (197 word clips, 101 phrases, 29 negatives,
three speakers), run twice: once on the bytes parsed straight out of the old
`gen/model_data.h`, once on the new export. Same clips, same feature front-end, same
`manifest_v3_qat.json` for the in-training/held-out labelling, so only the weights differ:

| speaker | words n | old acc | new acc | phrases n | old intent | new intent | neg n | old FA | new FA |
|---|---|---|---|---|---|---|---|---|---|
| spk01 | 13 | **0.615** | 0.538 | 0 | — | — | 0 | — | — |
| spk02 | 38 | **0.737** | 0.605 | 4 | 0.000 | 0.000 | 10 | 0.000 | 0.000 |
| spk10 | 146 | 0.479 | **0.678** | 97 | 0.062 | **0.082** | 19 | 0.158 | **0.053** |
| **all** | **197** | 0.538 | **0.655** | **101** | 6/101 | **8/101** | **29** | 3/29 | **1/29** |

Deployed on that table, and the argument is worth stating precisely because the first two rows
argue against it. `spk01` and `spk02` were in training for *both* models and both lose ground —
8 and 13 points. `spk10` was an unseen voice for the old model and is in training for the new
one, so its 20-point gain is partly a training-set score and must not be quoted as
generalisation. What is *not* discountable is the aggregate over every clip either model has
ever been asked about (0.538 → 0.655) and the false-accept rate on negatives (3/29 → 1/29),
which is the safety-side metric and improves by two-thirds. Phrase intent is 6/101 → 8/101:
better, and still the open problem it has been since v3.

**Two lessons for the paper, both about which number you look at.** The synthetic held-out
split and the real-voice set moved in *opposite* directions here — 91.2 → 90.7 % against
0.538 → 0.655 — which is the strongest instance yet of E2/E3's theme that a TTS-dominated test
split is not a proxy for a microphone. And the per-speaker regressions are the honest cost of
personalisation at fixed capacity: 5,879 parameters split three ways instead of two. The
session-2 note above records this change as "spk02 0.553 → 0.605", which compares the new model
against the *PTQ* baseline rather than against the QAT model actually deployed; against the
deployed one it is 0.737 → 0.605, a regression. Comparing a new model against whichever
previous number flatters it is an easy mistake to make twice, and the fix is mechanical —
evaluate the bytes in `gen/model_data.h`, because that is the only artefact that is definitely
what the device was running.

**Everything downstream regenerated and checked.** `kws-fwgen --check firmware/main/gen` and
`kws-codegen … --name command --check firmware/main/gen` are both clean, and E13's drift
warning ("`command_v3_qat.tflite` has been re-exported since the firmware headers were
written") is gone — the stamp, the embedded array and the `.tflite` are one model again. The
wake pair was deliberately not touched and is byte-identical. Host smoke parity on the
regenerated vectors: `command smoke: 0/368 bytes differ` — same 368 bytes (16 synthetic
windows x 23 classes, a shape not a weight count), different expected values, zero differing.
`wake smoke: 0/64` unchanged.

Regenerating `docs/sphinx/_generated/command.dot` turned up a third stale artefact nobody was
watching: its caption read "11 ops … 17,880 B" while the graph it captions draws ten nodes and
`kws-codegen` has reported ten ops for both models. The node structure was already right, so
only the summary line was wrong — a caption that had not been regenerated since a model two
retrains back. Same class of bug as the model drift itself, and it survived because a caption
is not checked by anything. `kws-fwgen --check` and `kws-codegen --check` cover the headers;
the diagram caption has no equivalent, which is worth either a check or a note in the docs.

**On the device, same rig and method as E12–E14.** Flashed, and the boot banner now names the
new model: `models: command command_v3_qat.tflite@f985f282 2026-09-04, wake
hey_bus.tflite@dd9db24f 2026-09-03`, with `status` agreeing. The placement figures are
untouched, as expected from a same-architecture retrain — `31360 B arena (static, PSRAM) +
0 B state + 19888 B shared scratch, esp-nn scratch 19888 B queried / 19888 B reserved`,
15,552 B queried and reserved on the wake side, 55,103 B free internal at recogniser start:

| | E14 (old model) | E15 (new model) |
|---|---|---|
| recognise step, arena in PSRAM | 31 ms | **31 ms** (31/31/31/31 over four traces) |
| command evaluation | 27,283 µs | **27,108–27,451 µs** |
| wake step | 1,250 µs | 1,245–1,287 µs |
| command arena / shared scratch | 31,360 B PSRAM / 19,888 B | unchanged |
| free internal, recogniser start | 55,239 B | 55,103 B |
| app image | 1,002,560 B | 1,002,560 B |
| `selftest int8 out:` | `-128,…,-36,…,0,-94,…,-127,…` | `-124,-128,-112,…,-37,-107,…,-90,-125` |

The step time stays in the 31–33 ms band on 32 more bytes of weights, which is the expected
answer and worth having measured rather than assumed: the generated C's cost is set by the op
shapes, and the retrain changed only the values in them. The `selftest int8 out:` line is the
one row that *must* change — it is the golden MFCC vector through the deployed weights, so a
byte-identical line here would have meant the flash had not taken. In assist mode a wake fire
opens the usual one-second recogniser window (`recogniser active 250/1000 of wall, inference
70 ms per wall second`); continuous `mode recognise` reaches 1000/1000 and 239 ms per wall
second. Left in the menu on this build.

One rig note, not a firmware finding: the microSD failed its mount probe again this boot
(`failed to mount card (13)`, ~26 s of reformat attempt before the fallback), so recordings
went to the flash partition. Same defective card as E12's; it costs the boot latency the
console tooling already waits out.

### E16 — wider DS-CNN: width 40 and 48 (2026-09-04, host-only, no device work)

E15 deployed a model that lost ground on both speakers it shared with its predecessor
(spk01 0.615 → 0.538, spk02 0.737 → 0.605) while gaining on the third, and read that as
"the honest cost of personalisation at fixed capacity: 5,879 parameters split three ways
instead of two". If that reading is right, the fix is capacity. The 2026-09-03 width sweep
only went *narrower* (24, 16 — both worse on both speakers), so the hypothesis had never
been tested in the direction it predicts. Widths 40 and 48 were trained with the same
recipe as the deployed model (40 float epochs, `--qat --qat-epochs 10`) on the *same*
`features_v3` cache, no dataset rebuild — `data/manifest_v3.json`, `built_at`
`2026-09-03T17:29:02Z`, train n = 32,399 (7,263 real / 25,136 TTS), speakers spk01, spk02,
spk10 — so the only variable is the channel count. Real-voice figures are `eval_recordings`
over the full QC-approved set, the same 197 word clips / 101 phrases / 29 negatives as E15,
and the w32 column is a re-run of the deployed export in the identical invocation form
rather than a quote of E15's table:

| | w32 (deployed) | w40 | w48 |
|---|---|---|---|
| params | 5,879 | 8,303 | 11,111 |
| MACs | 2,070,496 | 3,058,520 (1.48x) | 4,234,704 (2.05x) |
| tflite bytes | 17,912 | 21,680 | 25,832 |
| INT8 held-out test acc | 90.70 % | 92.83 % | **93.59 %** |
| spk01 words (n=13) | 0.538 | 0.769 | **0.923** |
| spk02 words (n=38) | 0.605 | 0.842 | **0.895** |
| spk10 words (n=146) | 0.678 | 0.760 | **0.856** |
| all words (n=197) | 0.655 | 0.777 | **0.868** |
| phrase intent (n=101) | 8/101 (0.079) | **12/101 (0.119)** | 8/101 (0.079) |
| false accepts (n=29) | 1/29 (0.034) | **0/29** | **0/29** |
| est. command evaluation | 27.3 ms (measured, E14/E15) | ~40.3 ms | ~55.8 ms |
| est. recognise step | 31 ms (measured, E14/E15) | ~44 ms | ~59 ms |
| arena, PSRAM | 31,360 B | 39,200 B | 47,040 B |
| shared esp-nn scratch, internal | 19,888 B | **59,632 B** | 29,824 B |
| est. free internal at recogniser start | 55,103 B (measured) | **~15,359 B** | ~45,167 B |

`export.assert_model_healthy` passes for all three (23 predicted classes each). Device
figures are *estimates* scaled by MACs from E14/E15's measured w32 numbers, not
measurements — the device was not touched in this experiment.

**ETA ledger, predicted against actual.** `kws-eta predict train 1295960` before each run,
`Timed` recording the float-training phase after:

| run | predicted | actual (float phase) | full command wall |
|---|---|---|---|
| w40 | ~3.5 min (range 3.2–4.5, 5 runs) | 308.0 s (5.1 min) | 6 min 50 s |
| w48 | ~4.0 min (range 3.3–4.7, 6 runs) | 376.2 s (6.3 min) | 9 min 52 s |

Both overshoot, and the reason is a limitation of the ledger worth writing down: the
predictor's `size` is `epochs x train rows`, which carries no width term, and every run in
its history was a width-32 one. It is predicting a *different architecture's* duration from
the data volume alone. The overshoot ratios (1.47, 1.57) do not track the MAC ratios
(1.48, 2.05) either, so a width term would not be a simple multiplier — the float training
step is not purely MAC-bound.

**The interesting finding is not the accuracy, it is that width 40 is a memory trap.**
`kws-codegen` run on all three exports (regenerating w32 first as a control — it reproduced
the deployed 31,360 B arena / 19,888 B scratch exactly) reports the shared esp-nn scratch,
and w40 needs **three times** what w32 needs while w48 needs only one and a half times.
The max-scratch op is the depthwise in every case, and `_depthwise_scratch` — the
branch-for-branch port of `esp_nn_get_depthwise_conv_scratch_size_esp32s3` — branches on
`channels % 16`. 40 is 8-aligned but not 16-aligned, so it falls out of the fast path into
the generic one, which rounds the channel count up to 48 *and* allocates an output buffer
the aligned path does not need: 35,616 B of padded input plus 23,520 B of output, against
w48's 29,376 B of padded input and no output buffer. **A model that is 28 % smaller than
another in MACs asks for twice its scratch, because of an alignment branch in a kernel
library.** That matters because the shared `kws_infer_scratch` region is internal `.bss`,
sized to the widest op of any shipped model: w40 would grow it by 39,744 B and leave ~15 KB
free at recogniser start — under E14's 23,879 B "every task created" mark and close to
E13's 8,431 B, the level at which the 8 KB record task silently failed to spawn. w48 grows
it by 9,936 B and leaves ~45 KB. The arenas are PSRAM and irrelevant at these sizes.

**Reading.** E15's capacity explanation holds, and holds strongly. Width recovers both
regressed speakers and overshoots their pre-session-2 numbers — spk01 0.538 → 0.923 and
spk02 0.605 → 0.895 at w48, against the 0.615 / 0.737 the *previous* model managed with two
speakers instead of three — and spk10 rises with them rather than trading against them
(0.678 → 0.856), which is what "the model ran out of parameters" predicts and what "the
third speaker crowded the other two out" does not. Aggregate real-voice word accuracy goes
0.655 → 0.868 and the safety-side metric improves to 0/29 false accepts. The synthetic
split agrees this time (90.7 → 93.6 %), which is worth flagging precisely because E15's
headline lesson was that it disagreed: the TTS-dominated test split is not a *proxy* for a
microphone, but it is not anti-correlated either — it failed to see a data change and does
see a capacity change. The recommendation is **w48**, at 2.05x the MACs and 9,936 B more
internal SRAM, subject to one device measurement of the recognise step before flashing
(~59 ms estimated against a 100 ms step budget: real-time holds, but CPU load roughly
doubles and in assist mode the recogniser shares a core with the wake task). **w40 should
not be deployed at any accuracy** — it is beaten by w48 on every real-voice metric except
phrase intent and costs 30 KB more of the scarcest memory to be worse. If w48 measures
badly on the device, the fallback is staying at w32, not stepping down to w40.

One tooling note for whoever repeats this. `kws-eval` derives the model filename from
`--prefix` and `--qat` alone and has no `--model`, so it cannot be pointed at a
`command_v3_w<N>_qat.tflite`; the width models were scored through a throwaway driver
calling `eval_recordings` + `make_command_predict_fn` with explicit bytes, the same code
path, validated by reproducing E15's deployed-w32 row exactly. A `--model` flag is the
obvious fix if width sweeps become routine. Also, `command_v3_w48_qat_metadata.json` still
carries the stock `budgets.macs` of 3,000,000, which both new widths exceed; nothing
enforces it, but a deploy should either raise the declared budget or note the overrun.

### E17 — field capture: real interactions as training data (2026-09-04, measured on the CoreS3)

Assistent mode now optionally keeps what it hears. With the "Aufnahme" switch on,
every wake fire arms one *field take* — the pre-roll before the fire plus the command
window that fire opened — which the record task copies out of the always-on audio
ring and writes **after** the window has closed, with the recogniser already switched
off. That ordering is the whole design: a FAT write on this device costs 100–300 ms,
more than a full recognise step, so capturing during the window would have changed the
very behaviour being captured.

The take spans the window that actually happened, not a constant. A second "Hey Bus"
inside an open window extends it, and the extension has to reach the audio too, or the
device's own answer would name words that fall past the end of the WAV. So the span is
pre-roll + the real close-minus-first-fire, the *first* fire is latched (the file name,
`fire_ms` and `wake_prob` stay anchored to the phrase the audio begins with), and only
the ring caps the length — at 9.8 s, cut at the end, never at the front. A take that
did get cut carries no device prediction at all, because it can no longer say which
fires are still inside its own audio.

**Measured on the device, one session, 13 capturing windows, 13 takes, 0 dropped.**
The recognise step time inside a capturing window stayed at **38–42 ms** (5 trace
samples; invoke 33.5–34.9 ms), against **38–45 ms** over 4 samples in the same assist
mode with capture switched **off** (invoke 33.6–34.4 ms) — the capturing band sits
inside the non-capturing one, and no sample anywhere near the 100–300 ms a write costs
appears in either run. Each take's `field: saved` line landed 968–1181 ms *after* its
window's `assist: recogniser off` — never before it, and the slowest is the one
extended-window take, which is 30 % more audio to write. The write is a full second of
work that a 2.5 s window has no room for; deferring it is not an optimisation, it is
the only place it fits. The extending-window case checks out end to end: a
second console fire 1.0 s into an open window produced one file of 73,088 samples
against 56,272 for a single-window take, named after the first fire.

The trace interval is worth stating, because it bounds how hard this measurement can
be pushed: the recogniser prints its step cost every 50 steps and a window is only
about 25, so roughly every second window contributes one sample. Thirteen windows buy
five numbers. That is enough to exclude a 100–300 ms outlier, a two-order-of-magnitude
effect, and not enough to resolve a 2 ms one.

A review of that first build found the ordering claim was true of the design and not
yet of the code: a `storage_free_bytes()` floor check — an `f_getfree()` that can scan
the FAT and suspends both cores' cache — ran *before* the ring copy and before the
wait, i.e. in exactly the place the design forbids, and it also ate the 0.2 s the ring
reserves for the copy. It now runs last, immediately before the `fopen`. Two smaller
holes closed with it: the copy is clamped to the ring's write head (the span's end is
derived from the fire's ring position and the window's ms length, sampled one inference
apart, so it could sit ~32 samples in front of what had actually been written and read
the ring's previous lap into the tail), and the NVS write behind the "Aufnahme" toggle
— flash I/O on the UI task, on the one screen where windows open — is deferred to the
window's closing edge while the in-RAM flag still flips at once. Re-measured on the
device after those changes, over two capturing windows: writes landed **970 ms** and
**968 ms** after their window's `assist: recogniser off`, the one recognise step traced
inside a window read **42 ms**, and toggling the switch off and on again *while that
window was open* produced no outlier at all — the headline numbers above are unchanged,
which is the point: the fix removed a hazard the sampling was too coarse to have caught.

Then the first real pull rejected **all 60 takes as clipped**, peak −0.0/−0.1 dBFS, and
the ring copy was the obvious suspect — wrongly. The block of full-scale samples in
every take measures 996.7–1000.0 Hz and 2,416–2,455 samples long; `beep.c` plays 1 kHz
for 150 ms (2,400 samples). It is the device's own wake-confirmation tone, played at
the fire — which is the instant a take is built around — coming back into the ring at
roughly full scale, because the speaker sits centimetres from the microphone on the
same shared I2S codec. It landed 13.9–63.7 ms past the pre-roll boundary in every take
(the jitter is the LVGL repaint between the two calls), takes with two fires carry two
blocks, and with that region excluded the takes peak at 181–700 counts, below −45 dBFS.
Nothing was wrong with the copy at all: the field path faithfully recorded the
recorder. The tone is now muted while capture is armed, rather than edited out of the
WAV afterwards, so the audio and the take's own `device_words` keep describing the same
sound. Three takes on the fixed build peak at **137–434 counts (−47.6…−37.6 dBFS)** with
zero samples above 2,000, against 32,768 and 0.0 dBFS before. The lesson is the cheap
one: a self-recording device records itself, and "the copy must be corrupt" is a much
more attractive hypothesis than "we are hearing ourselves".

The pre-roll is now **1.5 s, not the 1.0 s** the measurements above were taken with. The
wake model fires ~0.2–0.3 s *after* the end of a ~0.7 s "Hey Bus", so a second of
look-back starts inside the phrase: only 3 of 11 real takes still transcribed with the
wake phrase intact. The ring cap is on the whole take, so the wider pre-roll costs
nothing but the tail of a long chain of fires. The host-side split window moved with it
(`WAKE_MAX_S` 1.3 → 1.8 s): it tests when the phrase *ends*, so it is a function of the
pre-roll and silently stops finding the phrase if the two drift apart.

The label never comes from the device. On the workstation the take is transcribed by
the same Whisper model as every guided take; a "Hey Bus" in the first 1.8 s is cut off
as a wake positive, and the remaining words are run through the *same*
`kws_de.grammar.parse` the firmware's vocabulary feeds. A valid intent becomes the
phrase label, and the take joins training exactly like a guided sentence take (phrase
clip, index row, per-word segmentation); anything that does not parse is kept as
`_unknown_` material rather than discarded — real speech that the grammar rejects is
the negative data this model is chronically short of.

One rule had to be invented for the field path, and it is the mirror image of a guided
one. The guided matcher glues *known prompt* tokens together when Whisper writes
"Lichtdach" for "Licht Dach". A field take has no prompt, so the same failure has to be
undone from the other side: a token that decomposes **completely** into a run of
vocabulary words is split back apart, and one that does not is left exactly as heard.
The bug that forced it was concrete — a real "Licht Küche an" came back as
"Lichtküche an", failed to parse, and was filed as a *negative*, which is worse than
losing the take: it teaches the model that its own command words are not commands.

What the device *did* recognise is kept beside the label and scored against it. That
gives a figure no synthetic evaluation can: **field accuracy**, the deployed model's
agreement with the transcript on real, unprompted interactions in the van. The
agreement is counted over the takes the device actually answered, not over every take
that parsed, so a ring-truncated take with no device answer cannot depress a number it
never had a chance to earn.

**First real sessions (2026-09-04, one speaker, two sessions).** The first pull was
blocked by a host-side fault, not a firmware one — a locked screen on the workstation
makes macOS eject the `KWSREC` volume ~250 ms after it attaches (`ingest.sh` now names
that cause and the recovery). The first spoken session then failed QC outright: every
take was "clipped", and the reason was the device's own wake-fire beep (1 kHz, 150 ms,
`beep.c`), recorded through the mic at full scale 14–64 ms after the fire. The fix mutes
that beep while capture is armed; the QC clip gate was also changed from "peak above
−0.5 dBFS" to "0.05 % of samples at the rail", so a one-sample click cannot discard a
take. The same session showed the 1.0 s pre-roll cutting the wake phrase: only 3 of 11
transcripts contained "Hey Bus" (the model fires ~0.2–0.3 s after a ~0.7 s phrase), so
the pre-roll is 1.5 s from here on, with `qc.WAKE_MAX_S` moved with it and the
session's real pre-roll inferred from `ms − window_ms` so older sessions are not
flagged truncated.

Figures from `kws-qc` + `kws-eval --recordings` over the two spoken sessions (the
1.0 s-pre-roll session and the 1.5 s one; 11 + 9 takes, all approved by the audio and
content gates):

| | 1.0 s pre-roll (11) | 1.5 s pre-roll (9) | both (20) |
|---|---|---|---|
| parsable intent (Whisper + grammar) | 8 | 5 | 13 (0.65) |
| wake clips ("Hey Bus" in the take) | 3 | 8 | 11 |
| unparsed, vocabulary present | 2 | 2 | 4 |
| device gave any command word | 8 | 4 | 12 |
| device–Whisper agreement over compared | 2/8 | 0/2 | 2/10 (0.20) |

The pre-roll change did what it was meant to: 3 of 11 → 8 of 9 takes carry the wake
phrase. The agreement figure is the headline: the deployed w32 command model, measured
on real Assistent-mode usage by its main user, agreed with Whisper on 2 of 10 answered
takes and answered nothing at all on 5 of the 9 takes of the second session ("Hey Bus,
Licht an" → no word fired). This is the number E16's width sweep should be read
against. The 20 takes yielded 11 wake clips, 13 phrases, 26 word clips and 3 negatives
into `approved/` (speaker spk18, held out: 0.269 word accuracy, 0 of 13 phrase intents,
0 of 3 false accepts under the same model).

Two honest caveats already visible. The agreement figure will belong to whichever model
was flashed at capture time, not to the model a later report evaluates —
`kws-eval --recordings` prints it in its own **Field** section for that reason. And a
field take is self-selected: it exists only because the wake word fired, so it measures
command accuracy *given* a successful wake, and says nothing about missed wakes, for
which no trigger exists. E21 is what answers that second caveat.

**The pre-roll was still too short, and what was wrong was a number, not a formula
(fix/field-take-offset).** At 1.5 s the wake phrase came back *clipped at the take's
first sample* in 8 of the 9 second-session takes: the transcript said "Hey Bus", but the
model that had just fired on it could not fire on the recording of it — replayed through
the firmware feature path, the round-5 model fires on only 3 of those 9 takes. The
suspects were all in the ring: `audio_write_pos()` at the fire, codec/DMA lag, which
side the copy starts on. Every one of them was innocent. The wrong number was the
model's own detection latency. Replaying the 13 approved "Hey Bus" clips spliced into
real room tone — so the front-end's noise and PCAN estimates are warm, as they are on
the device — the firmware gate (0.85 twice in a row) is met a median **1.06 s** and at
worst **1.20 s** after the phrase *ends*, not the 0.2–0.3 s `field.h`, `qc.py` and
`test_field.c` all budgeted for. The model answers one phrase with two humps: the first,
right at the phrase end, peaks around 0.87 and usually yields a single qualifying step;
the second, ~1 s later, sits at 0.996 for a third of a second and is what fires. The
0.2–0.3 s figure was the first hump.

Every field observation falls out of the corrected number. At a 1.0 s pre-roll the
phrase ends at 1.0 − 1.06 < 0, in front of the file: 8 of 11 takes carried no wake word,
and the 3 that did are the low-latency tail (0.23 s, so the phrase should end at 0.77 s;
measured 0.84 s). At 1.5 s it ends at 0.44 s and the takes measure 0.34–0.54 s. The
device's own beep is what ruled the ring out: at 14–64 ms past the pre-roll boundary in
every take of the earlier session, the fire lands exactly where the geometry says it
should, so the thing sitting a second away was the phrase, not the index.

Confirmed on the device without a human, by playing one of the speaker's own approved
wake clips through the flashing host's speaker (host-measured latency for that clip:
1.17 s), device in Assistent mode with `field on`. **Before**, 1.5 s pre-roll: fire at
0.996, phrase in the take at **0.00–0.32 s** with its onset cut off — the host
prediction (0.33 s) to within 10 ms, which incidentally says the anchor is not sliding
either, since the wake task holds 68 steps per 2 s, i.e. real time. The take itself
peaks at 0.242 through the model and does not fire. **After**, 2.5 s pre-roll: same
clip, same 1.18 s latency, phrase whole at **0.82–1.32 s** of a 5.00 s take with 0.82 s
of lead-in in front of it — and the take now *fires the model at 2.48 s*, 20 ms from the
device's own fire point. That is the property the feature exists for: a take that
reproduces the wake it was triggered by is usable as wake training data.

The fix is the budget, written down as its three terms — `FIELD_WAKE_LATENCY_MS` 1200 +
`FIELD_PHRASE_MS` 900 + `FIELD_LEAD_IN_MS` 400 = a **2.5 s** pre-roll — so the pre-roll
can no longer be set to a value that contradicts the measurement, and
`firmware/test/test_field.c` asserts that the span contains a phrase placed where those
terms put it (the assertion fails at 1.5 s). `qc.WAKE_MAX_S` moves with it, 1.8 → 2.5 s.

The lesson is E17's own, twice over. The first time the self-recording device recorded
itself; this time it was sized against a latency nobody had measured. 0.2–0.3 s was
plausible, cheap to write down, and wrong by a factor of four — and it had been copied
into three files, where each copy made the next look corroborated. The test that would
have caught it is not "the pre-roll is 1.5 s" but "the phrase the fire answers is inside
the take".

### E18 — deploying the width-48 command model (2026-09-04, measured on the CoreS3)

E16 recommended width 48 and left one thing open: the runtime was a MAC-scaled estimate, not
a measurement. This entry does the deploy, every host-side check, and the measurement — which
turns out to be the interesting part, because the estimate was wrong by 25 % in the
comfortable direction.

The deploy is E15's procedure with the width flag added. `--width` suffixes the *output*
artefacts only — the QAT SavedModel to load still has to be named in full, so `--model` and
`--width` are both required and neither implies the other:

```text
kws-export --firmware --qat --prefix features_v3 --model command_v3_w48.keras --width 48
kws-codegen firmware/main/gen/model_data.h --name command --out firmware/main/gen
```

| | old (E15, was on the device) | new (deployed) |
|---|---|---|
| `KWS_MODEL_ID` | `command_v3_qat.tflite@f985f282 2026-09-04` | `command_v3_w48_qat.tflite@8fa81d08 2026-09-04` |
| size | 17,912 B | 25,832 B |
| width / params / MACs | 32 / 5,879 / 2,070,496 | **48 / 11,111 / 4,234,704** |
| INT8 held-out test accuracy | 90.7 % | **93.6 %** |
| input scale / zero point | 3.18189716 / 71 | 3.18189716 / 71 (unchanged) |
| output scale / zero point | 3.90625e-03 / -128 | unchanged |
| classes / labels | 23 | 23, identical list |

Input and output quantisation are *bit-identical* across the two models, which is a small
piece of luck worth naming: the MFCC front-end feeds the same `(feature / 3.18189716) + 71`
and the recogniser's threshold arithmetic reads the same output scale, so nothing outside the
model had to move. The 23-class `static_assert`s in `recognise.cc`
(`COMMAND_INFER_INPUT_LEN == KWS_N_FRAMES * KWS_N_MFCC`, `COMMAND_INFER_OUTPUT_LEN ==
KWS_NUM_LABELS`, `COMMAND_INFER_STATE_BYTES == 0`) hold unchanged at 490 / 23 / 0.

As in E15 the re-export reproduced the on-disk flatbuffer byte for byte (`8fa81d08`), so the
bytes about to be flashed are exactly the bytes E16 evaluated. The wake pair was not touched
and is byte-identical by sha256 — worth checking rather than assuming, because
`kws-export --firmware` unconditionally rewrites `wake_model_data.h` / `wake_model_config.h`
from `models/hey_bus.tflite`, so a wake model with a newer mtime would silently restamp
`KWS_WAKE_MODEL_ID` inside a command-only deploy.

**The real-voice case, carried over from E16.** Both models over the same QC-approved set
(197 word clips, 101 phrases, 29 negatives, three speakers):

| | w32 (was deployed) | w48 (deployed) |
|---|---|---|
| spk01 words (n=13) | 0.538 | **0.923** |
| spk02 words (n=38) | 0.605 | **0.895** |
| spk10 words (n=146) | 0.678 | **0.856** |
| **all words (n=197)** | **0.655** | **0.868** |
| phrase intent (n=101) | 8/101 | 8/101 |
| false accepts (n=29) | 1/29 | **0/29** |

Both speakers E15 regressed are recovered and overshoot their *pre*-session-2 numbers
(0.615 / 0.737), and spk10 rises with them instead of trading against them. Phrase intent
does not move — it has been the open problem since v3 and is a decoding problem, not a
capacity one. All three speakers are in training for both models, so these are
personalisation numbers, not generalisation; the comparison across widths is fair (identical
manifest, identical clips) but the absolute values are not a held-out estimate.

**The declared MAC budget had to move.** `kws_de.config.MAX_MACS` was 3,000,000 — a spec
figure from the original plan, written into every export's `budgets.macs` sidecar and
asserted by `kws_de.budgets.check_budgets`. The deployed model spends 4,234,704. It is raised
to 5,000,000 in `kws_de/config.py` with the reason on the line above it: the binding
constraint on this model is the 100 ms recognise step it has to fit inside, and a MAC count
is only a proxy for that. Two other budgets were checked and left alone — 25,832 B against
`MAX_MODEL_BYTES` 500,000 and a 47,040 B arena against `MAX_ARENA_BYTES` 300,000, both an
order of magnitude clear. (`MAX_LATENCY_MS = 30` is dead: nothing reads it, and the real
step budget is the 100 ms one the recogniser loop enforces. Worth deleting or wiring up, not
done here.)

**Memory: the arena doubles in PSRAM, and 9,936 B of internal RAM goes.** `kws-codegen`
reports `command: 10 ops, arena 47040 B, shared esp-nn scratch 29824 B, rings 0 B`. Ten ops,
unchanged — same architecture, wider. The arena is `.ext_ram.bss` and irrelevant at this
size. The scratch is the one that costs: it is internal `.bss`, shared with the wake model
and sized to whichever asks for more, and the wake model asks for 15,552 B, so the whole
region moves with the command model:

| | E15 (w32) | E18 (w48) | delta |
|---|---|---|---|
| command arena (PSRAM `.bss`) | 31,360 B | 47,040 B | +15,680 B |
| `COMMAND_INFER_SCRATCH_BYTES` | 19,888 B | 29,824 B | +9,936 B |
| `WAKE_INFER_SCRATCH_BYTES` | 15,552 B | 15,552 B | — |
| shared `KWS_INFER_SCRATCH_BYTES` | 19,888 B | **29,824 B** | +9,936 B |
| DIRAM `.bss`, whole image | 124,296 B | 134,232 B | **+9,936 B** |
| DIRAM free, whole image | 134,525 B | 124,589 B | -9,936 B |
| app image (default config) | 1,002,560 B | 1,008,688 B | +6,128 B |
| app image (`CONFIG_KWS_INFER_GENERATED=n`) | — | 1,174,096 B | — |

Both image sizes are of the branch measured on the device, which was cut before E17's field
capture merged; rebuilt on top of that merge the default image is 1,012,080 B, the extra
3,392 B being `field.c` and its callers rather than anything in this entry. The `.bss` and
scratch figures are unaffected — the merge adds no static arrays and does not touch the
inference path.

The DIRAM row is the one that makes the argument, and it is a link-map measurement rather
than an estimate: internal `.bss` grows by *exactly* the 9,936 B the shared scratch grew, and
not one byte more. The 15,680 B the arena gained landed in PSRAM, where `idf.py size` reports
it under "Flash Data `.bss`" at 47,040 B — which is also the cheapest possible confirmation
that `KWS_INFER_COMMAND_ARENA_PSRAM` is still in force and the arena did not quietly fall
back to internal SRAM. Free internal at recogniser start should therefore land near
55,103 - 9,936 = ~45,167 B (E15 measured 55,103 B). **The device reports 45,431 B** — 264 B
above the prediction, comfortably clear of E14's 23,879 B "every task created" mark and far
from E13's 8,431 B, where the record task's 8 KB stack failed to spawn. Predicting a
device RAM figure from a host link map to within 0.6 % is worth recording as a method: the
arena and the scratch are both static arrays, so the whole placement question is answerable
before flashing, and only the residual heap use is not.

The boot guard is unchanged in shape and answers the new number. `command_infer.c` emits
`command_infer_scratch_query()` from the same dimensions it emits the kernels from, and
`recognise.cc` refuses the generated path if the chip's own
`esp_nn_get_*_scratch_size_esp32s3` answers more than `COMMAND_INFER_SCRATCH_BYTES`. Width 48
keeps the fast `channels % 16 == 0` depthwise path (E16), and the two boot lines confirm the
Python port and the real esp-nn agree on the chip:

```text
wake: inference: generated (esp-nn), 128 B arena + 4200 B state + 29824 B shared scratch,
  esp-nn scratch 15552 B queried / 15552 B reserved; TFLM not built in; free internal 67483
recognise: inference: generated (esp-nn), 47040 B arena (static, PSRAM) + 0 B state +
  29824 B shared scratch, esp-nn scratch 29824 B queried / 29824 B reserved; TFLM not built
  in; free internal 45431
```

**Runtime: the MAC-scaled estimate was 25 % pessimistic.** MAC-scaling E15's measured
27.3 ms command evaluation by the 2.05x MAC ratio predicted ~55.8 ms and a ~59 ms recognise
step. The device measures **42.2 ms and a 46–47 ms step**:

| | E15 (w32, measured) | E18 (w48, measured) |
|---|---|---|
| recognise step, arena in PSRAM | 31 ms | **46–47 ms** (47/47/47/47/46/46/46 over seven traces) |
| command evaluation | 27,108–27,451 µs | **42,175–42,425 µs** |
| front-end, per step | — | 472–485 µs over 7–8 new frames |
| wake step (assist mode) | 1,245–1,287 µs | **1,226–1,276 µs**, ±41–149 µs |
| free internal, recogniser start | 55,103 B | **45,431 B** |
| duty, continuous `mode recognise` | 1000/1000 wall, 239 ms/s | **1000/1000 wall, 320 ms/s** |
| app image | 1,002,560 B | 1,008,688 B |
| recogniser task stack free | — | 3,896 B |

Invoke time goes up **1.55x on 2.05x the MACs**, so the wider model is markedly cheaper per
MAC than the narrower one. Two things plausibly explain it and this measurement does not
separate them: 48 channels fill esp-nn's SIMD lanes where 32 leave some idle, and the
per-op fixed costs — the scratch pointer set-up, the requantisation loop preamble, the
ten-op dispatch — are constant while the arithmetic grows. It is the same lesson as E16's
alignment cliff seen from the other side: on this kernel library the channel count decides
which code path runs, and MACs are a poor predictor across paths. **MAC-scaling is a fine
way to decide whether to bother measuring, and a bad number to put in a table.**

The practical margin is what matters. The step budget is 100 ms and the step is 46–47 ms, so
the recogniser is at 47 % of real-time against 31 % before — CPU load up by half, not
doubled as feared. In continuous recognise the duty logger reads 320 ms of inference per wall
second against E15's 239 ms. The wake step is unchanged at 1,226–1,276 µs against E15's
1,245–1,287 µs, which is the confirmation worth having: the shared scratch region grew by
9,936 B for the command model's sake and the wake model, which shares it, pays nothing for
that.

**Verdict: deployed.** Step 46–47 ms against a ~65 ms acceptance bar, free internal 45,431 B
against a ~40 KB bar. Had it measured badly the fallback was staying at w32 — **not** stepping
down to w40, which E16 showed asks for 59,632 B of shared internal scratch, thirty kilobytes
more than the larger model, because 40 channels miss esp-nn's 16-alignment fast path.

**The selftest line, and an unplanned confidence measurement.** `selftest int8 out:` is the
golden MFCC vector pushed through the deployed weights, so it is the one line that *must*
change or the flash did not take:

```text
E15 (w32)  -124,-128,-112,-128,-128,-128,-112,-125,-127,-119,-37,-107,-127,-128,-90,-112,-127,-128,-128,-128,-128,-90,-125,
E18 (w48)  -124,-128,-127,-127,-128,-126,-122,-128,-127,-89,49,-127,-127,-128,-124,-126,-127,-128,-127,-128,-127,-116,-125,
```

Both models put the peak at index 10 (`auf`), which is the right answer, but the margin is
not comparable. Dequantised with the shared output scale (3.90625e-03, zero point -128) the
winning posterior goes **0.355 -> 0.691**, and every runner-up flattens toward -128: w32's
second and third places sat at -37/-90/-90, w48's at 49 then -89 and -116. On this one
vector the wider model is roughly twice as confident and much better separated, which is a
mechanism for the real-voice gains in E16's table rather than a restatement of them — the
detector thresholds and `min_consecutive` gate on exactly this margin.

**Host checks, all clean.** `kws-fwgen --check firmware/main/gen` and both `kws-codegen
--check` runs (command and wake) exit 0 with no drift warning. `make -C firmware/test`:
`command smoke: 0/368 bytes differ`, `wake smoke: 0/64 steps differ`, `host tests OK` — the
368 is unchanged because it is a shape (16 synthetic windows x 23 classes), not a weight
count, while the expected values inside `command_smoke_vectors.h` all changed. The real-clip
harness reads the embedded bytes, so it picked the new model up with no argument:
`command parity: 0/1564 bytes differ (68 clips, arena 47040 B)` — 4 synthetic vectors plus 64
real `features_v3` test-split windows, zero LSB difference against the TFLite interpreter.
Both ESP-IDF v5.5.5 Docker builds pass, default and `CONFIG_KWS_INFER_GENERATED=n`.

Two host tests had the width-32 figures written into them as literals and were updated:
`test_command_write_then_check_roundtrips` (arena and scratch in the returned info dict and
in the emitted `#define`s) and
`test_models_share_one_scratch_region_and_export_a_query_for_it` (the
`static int8_t kws_infer_scratch[19888]` fallback declaration). Both are pinning real,
deliberate constants, so hard-coding them is right and updating them by hand is the intended
cost of a deploy — but it does mean a model swap cannot be a `gen/`-only diff, which is worth
knowing before the next one.

One failure in `pytest -q` is **not** from this change:
`test_whole_wake_model_matches_the_interpreter_on_every_step` asserts `len(takes) == 10`
against `data/recordings/approved/wake`, which now holds 13 approved takes. It fails
identically on `main` with the same data root, and the fix (replay the first ten rather than
require exactly ten) is already on `feat/field-capture`. Everything else: 266 passed,
1 skipped, 1 xfailed.

**Documentation caught up in the same commit, deliberately.** E15's lesson was that stale
generated captions survive because nothing checks them, and a width change invalidates more
prose than a retrain does. `docs/sphinx/_generated/command.dot` is regenerated (its caption
now reads "10 ops, 28 tensors, 9,744 weights, 4,234,704 MACs, 0 B state, 25,832 B file"), the
per-op MAC table in `models.rst` is rewritten at width 48 and sums to 4,234,704 against
`kws_de.budgets.estimate_macs`, the upward half of the width sweep is recorded as its own
table — with the warning that it is scored against the session-2 baseline, so its width-32
row reads 0.538 / 0.605 and not the 0.615 / 0.737 of the table above it — and the four places
that quote the arena and scratch sizes as current fact (`models.rst`, `firmware.rst`,
`requirements.rst`, `tests.rst`, plus the `KWS_INFER_COMMAND_ARENA` Kconfig help) now carry
47,040 / 29,824. Where a number was a *device measurement* at width 32 — the 27.3 -> 27.0 ms
arena-placement delta, 55,239 B versus 23,879 B free — it is left alone and labelled as
measured at width 32. Those two are still w32-only figures even now: this session measured
the PSRAM configuration, which is the shipped default, and never rebuilt with the arena
internal, so there is no w48 number for that comparison and inventing one from the w48 step
time would repeat E15's mistake in a new place.

**Two rig notes, neither a firmware finding.** The assist-mode duty line reads `recogniser
active 0/1000 of wall, inference 0 ms per wall second` across the whole 45 s capture, because
no wake fire happened — the session was driven over SSH with nobody at the device to say "Hey
Bus", and the wake peaks stayed between 0.000 and 0.242 against the detector threshold. So
the *loaded* assist figure is not measured here. Scaling E15's measured window (250/1000 of
wall, 70 ms of inference per wall second at a 27.3 ms evaluation) by the new 42.2 ms gives
~108 ms per wall second, which is an estimate and is flagged as one. What the capture does
establish is the thing that could have gone wrong and did not: the wake model's own step is
unchanged while it shares a scratch region that grew by 9,936 B. And the microSD failed its
mount probe again (`sdmmc_card_init failed (0x107)`, falling back to the flash partition) —
the same defective card as E12's.

### E19 — command confirmation tone (2026-09-04, feat/command-beep)

Acceptance feedback for the user, not a measurement: a fire on a real command word (not
`_unknown_`, not `_silence_` — `stream_is_command()`, host-tested in `test_stream.c`) plays two
80 ms 1.5 kHz pips, distinct from the wake fire's single 150 ms 1 kHz tone. Two constraints
shaped it. (1) It must not reach the microphone inside an assist window, or the field-capture
takes (E17) would carry the device's own beep as training audio — so in assist mode the
recogniser only *records* that a tone is owed and the wake task plays it at the window's closing
edge. The recogniser's self-imposed deadline is not that edge: a second wake fire extends the
gate without re-arming the deadline, so keying off "recogniser inactive" would have beeped
mid-window. With capture *armed* the tone is dropped altogether, on the same `s_field.enabled`
predicate E17 already mutes the wake tone with: an armed device is the user's chosen silent
mode, and the owed flag is taken (and discarded) rather than left to sound at a later window's
close. (2) It must not lengthen an inference step, so playback is a priority-1 task woken
by a task notification; both model tasks only post. The tone reuses the existing shared duplex
codec path (`beep.c`), so nothing about the mic stream changes.

Measured on the CoreS3. The speaker opens after the mic on the same codec without error
(`beep: speaker ready (wake 1000 Hz/150 ms, confirm 2 x 1500 Hz/80 ms)`). Playing the tone in
wake mode leaves the front-end at a full 68 steps per 2 s trace window, so the concurrent write
costs the mic stream nothing, and the wake probability peaks at 0.242 against a 0.99 threshold —
the device does not hear its own beep as a wake word. Same in recognise mode: the tone played
into the live recogniser produced no fire at all, so there is no feedback loop to guard against.
An injected wake fire on silence opens and closes the usual window (2537 ms, duty 251/1000 of
wall, 70 ms per wall second) with no tone, confirming that a rejection stays silent. Measured
before this branch was rebased onto E17, so with capture *disabled* — the capture-armed mute
added in the rebase rides on the same `s_field.enabled` predicate E17 already gates the wake
tone with, and needs no separate device evidence.

### E20 — the real-clip share is a safety knob, not overfitting (wake, 2026-09-04, host-only)

Round 5 gave ten real "Hey Bus" recordings `sampling_weight: 5.0` against the TTS positives'
`2.0`. microWakeWord draws every training batch with one `random.choices` over all feature
providers (`microwakeword/data.py:540`), so that is **71 % of every positive batch coming from
ten unique recordings** — which reads like textbook overfitting, and the round was run to
correct it: cap the real share at 30 %, add hard negatives cut from the same field takes as the
positives, hold out a whole recording session. Architecture, steps, lr and augmentation
unchanged; the export is 58,080 B / 24,736 MACs in every run, identical to round 5.

Two runs, both at fixed positive total 7.0. Round 6 put the split at 4.9 / 2.1 (30 % real) and
held the negative total at 45.0 by trimming the TTS hard negatives 20 → 17 and `no_speech`
5 → 3. That confounds two changes, so round 6b restored **every** round-5a negative weight
exactly and split the positives 3.5 / 3.5 (50 % real), adding the real negatives on top. The
result is monotone and it goes the wrong way:

| real share of positives | TTS non-wake false fires / 48 | real held-out non-wake fires / 9 | held-out session fires / 3 | guided takes fire / 10 |
|---|---|---|---|---|
| 71.4 % (round 5a, installed) | **1–5** | **0/9** | 3/3 @ 0.996 | 10/10 @ 0.996 |
| 50.0 % (round 6b) | 15 | 3/9 | 3/3 @ 0.996 | 10/10 @ 0.996 |
| 30.0 % (round 6) | 33 | 3/9 | 3/3 @ 0.996 | 10/10 @ 0.996 |

Recall is pinned at the ceiling throughout — every model fires on every real positive, held-out
session included, at peak 0.996. The entire effect is on the safety side. At 30 % the model
false-fires on "hallo bus", "der bus kommt gleich", "hey du" and "wie spät ist es" across all
four probe voices, and on three ordinary commands spoken to the device in the held-out session
("Kühlschrank aus" 0.922, "Licht aus" 0.996 ×2). The round-5a TTS figure is a range because
`wake_neg_gate_probe.py` re-synthesises with Piper each run and Piper is not deterministic
(1/48 and 3/48 here; E-series `WAKE-WIDTH-REPORT` measured 2/48 and 5/48) — the 15 and 33 sit
far outside that spread.

**And the same knob sets the fire latency, pulling the other way.** Splicing each wake clip
into real room tone (3 s lead, 2.5 s tail, so the microfrontend's noise/PCAN estimates are warm)
and measuring from the phrase end — an energy endpoint on the clip, not the file end — shows
every model answering with two probability humps. At 71 % real share the first hump peaks at
0.32-0.43, below the gate, so the 0.85 x 2 rule fires on the second hump about a second later:

| real share | held-out session latency (median / max) | guided-take latency | first-hump peak (median) |
|---|---|---|---|
| 71.4 % (round 5a) | 1.13 s / 1.16 s | 0.94 s / 0.99 s | 0.316 / 0.430 |
| 50.0 % (round 6b) | 1.13 s / 1.13 s | 1.02 s / 1.04 s | 0.590 / 0.650 |
| 30.0 % (round 6) | 0.11 s / 0.11 s | 0.02 s / 0.06 s | 0.996 / 0.992 |

**Root cause, and it is data alignment, not architecture.** The ten guided takes are 1.78-1.92 s
files whose phrase ends at 0.76-1.16 s — they carry **~1.04 s of trailing silence**, against a
0.94-1.13 s measured latency. `gen_features_real.py` builds `Clips` with `remove_silence=False`,
and `truncation_strategy: truncate_start` keeps the last 1,500 ms of each 3.2 s augmentation
window, so the positive label sits about a second after the phrase ends. The model learns that
offset and applies it even to the tightly-cut field clips, which have 0.01 s of trailing silence
and still fire 1.13 s late. The TTS positives are tight, so a TTS-dominated model fires at the
phrase — which is why latency tracks the real share, and why round 6's low latency is a side
effect of leaning on TTS rather than a fix.

This is the model-side counterpart to E17. That entry measured the same ~1.17 s detection
latency from the capture side and paid for it by widening the field-take pre-roll to 2.5 s, so
the phrase would still be inside the recording by the time the fire landed. Both readings are
right and they compose: E17 makes capture correct against the model we have, and round 6c
removes the latency the pre-roll was sized to absorb. If 6c is promoted, `FIELD_LEAD_IN_MS` and
the pre-roll can shrink back toward the phrase length — but only *after* a device measurement
confirms the latency on hardware, and not in the same change that swaps the model.

**Round 6c takes both.** Round-5a positive weights (71.4 % real, the share that keeps false
wakes down), the real hard negatives, and the real positive clips silence-trimmed before feature
generation (energy endpoint, 0.25 s lead / 0.20 s tail; 1.78-1.92 s becomes 0.93-1.33 s):

| gate | round 5a | round 6c |
|---|---|---|
| held-out session fires / in-training real fires | 3/3, 10/10 @ 0.996 | 3/3, 10/10 @ 0.996 |
| real non-wake fires (9 held out + 5 in-training + 49 room) | 0 | **0** |
| fire latency, held-out / guided (median) | 1.13 s / 0.94 s | **0.08 s / -0.01 s** |
| first-hump peak (median) | 0.316 / 0.430 | **0.996 / 0.996** |
| TTS non-wake false fires / 48 | 1-4 | 9 |
| tflite bytes / MACs | 58,080 / 24,736 | 58,080 / 24,736 |

A full second of perceived wake latency removed at identical size and cost, with every
real-audio gate held. The cost is 9/48 TTS near-miss false fires against round 5a's 1-4/48,
concentrated on "hallo bus" and "der bus kommt gleich" in two Piper voices — the generic-voice
margin the 2026-09-03 decision already priced in. Nothing was promoted; that trade is the
user's call and the device check is a separate step.

**Reading.** The 71 % share was not overfitting waiting to be corrected; it was doing the
selectivity work, and this is the flip side of the 2026-09-03 decision to make the wake model
deliberately user-customised. Nine thousand Piper "hey bus" clips can be satisfied by a loose
"German speech with a stressed front syllable" feature, because the hard negatives opposing
them are Piper too; ten real recordings through the device's own microphone are a far narrower
target, and that narrowness is what rejects the near-miss family. Cutting the real weight
returns the model to round-4 behaviour, which fired on "licht küche an" at 0.988. If
two-speaker overfitting is the worry the fix is **more real speakers, not less real weight** —
a capacity/coverage answer, structurally the same conclusion E16 reached for the command model.
The other half of the policy survives intact: the real hard negatives cost nothing on any gate
in either 6b or 6c. And the latency result is the more useful one for the product — a wake word
that answers a second after you stop speaking feels broken in a way no accuracy number
captures, and it turned out to be one `remove_silence` flag and a trailing-silence trim, not a
model problem. Any future real-clip set must be checked for this before training.

**Two measurement bugs found first, both invisible until clips got short.** (1) The probe used
for round 5's "10/10 real takes fire" and for every row of the width sweep scored a directory
with one interpreter and `reset_all_variables()` between clips. That zeroes the six ring-buffer
resource variables but does not re-run the `CALL_ONCE` init subgraph, which fires once per
interpreter lifetime — so scores depended on what ran before: the same clip read 0.031, 0.316
or 0.996 across three orderings. (2) It padded with 0.5 s of leading silence against a ~1.9 s
receptive field, so a tightly-cut field clip is scored on a half-filled ring. Neither shows on
the 1.8 s guided takes of rounds 1–5, which is how both survived five rounds; both are fatal on
0.2–0.7 s field cuts. `kws_de.wake.stream_wake_probs` now builds a fresh interpreter per call
by construction and prepends `WAKE_CONTEXT_S = 2.0` s. Re-probed correctly, every round-1..5
conclusion still holds.

**The field data is mostly mislabelled, and it is a pipeline bug, not a QC judgement.** Each
approved `spk18` clip was matched back to its source take by exact sample search. Session
`…-0951` was recorded with no pre-roll — frame energy is already −20 to −32 dB in the first
100 ms, where the good session opens with ~200 ms of near-silence — so the recording starts at
or after the device's own wake detection and its eight `wake/` clips are 0.21–0.33 s tail
fragments of a ~0.7 s phrase. The installed model scores all eight at its 0.316 floor, and
scores five of the eight *full 4 s takes* at the same floor: the wake word is largely not in
the audio. Whisper still transcribes them "Hey Bus, Licht an." because it completes the
fragment from context, so the QC transcript agrees with a file that does not contain the word.
Separately, the cutter writes the **whole take** whenever the wake word is not a leading cut,
so four clips filed under `phrases/`/`negatives/` still contain "Hey Bus" — the installed model
fires 0.996 on two of them. Training on either kind is actively harmful in opposite directions:
fragments teach the model to fire on a 0.2 s syllable, and wake-bearing negatives teach it not
to wake. All twelve were excluded; two negatives were recovered by trimming the known wake span.

`scripts/wake-retrain.sh` now encodes the session-disjoint split (from each session's
`written.txt`), the feature build and the probe, with `WAKE_EXCLUDE` for known-bad clips.
ETA ledger: predicted 15.5 min, actual 16.34 min (+5 %) for round 6; predicted 15.7, actual
13.31 (−15 %) for round 6b, which had the machine to itself. One tooling defect: `with_eta.sh`
inherits `KWS_DATA_ROOT`, and a run launched without it writes its measurement to a repo-local
fallback ledger instead of the shared one — three runs had accumulated there unnoticed.

**Nothing was promoted.** The installed `hey_bus.tflite` is unchanged, `firmware/main/gen/` was
not touched, and no device work was done (the CoreS3 was in use).

**Follow-up: the cutter is fixed at the source (issue #58, `fix/qc-field-cutter`).** E20 worked
around the mislabelled clips with an exclude list; the pipeline now cannot produce them. Two
rules were separated that had been one. A `wake` clip is cut only from a *whole leading* phrase
— the take's first one or two word spans, ending within 2.5 s and at least 0.4 s long — because
a wake clip starts at sample 0, so a short one is proof the capture began after the "Hey".
Everything filed as a phrase or a negative starts after the **last** wake phrase in the take,
wherever it sits, so a take ending in "Hey Bus" or carrying a second fire cannot leak the word;
a take with nothing left after that cut, or whose transcript holds the phrase with no word span
to locate it by, is filed nowhere at all. Re-running QC on the two field sessions reproduces
E20's manual triage exactly and finds one clip more than it did: session `…-0951` yields **0**
wake clips instead of 8 fragments, the two wake-bearing negatives become one correctly trimmed
negative, the two wake-bearing phrases are dropped, and one take E20 discarded (`33-282511`,
"Hey Bus … Hey Bus, Licht aus") is *recovered* as a clean 2.21 s phrase by cutting after the
second phrase. `scripts/audit-approved.py` now audits the whole tree — format, duration band per
set, `index.csv` ↔ files both ways, `spkNN` naming, and a Whisper pass over every field-derived
phrase/negative looking for the wake regex — and reports 0 problems over 379 clips. The lesson
generalises past this bug: **per-session QC cannot see a per-tree invariant**, and the invariant
here ("no non-wake clip contains the wake word") is exactly the one whose violation is invisible
in every individual session's report.

One correction to E21 falls out of it. `qc.csv`'s `wake_clip` was "did Whisper find the phrase at
the head of the take", which was the same question as "was a wake clip written" only while every
leading phrase produced one. It no longer is: a head-cut fragment writes nothing, but the speaker
*did* wake the device with it, and counting those as "no wake clip" would have turned all eight
of session `…-0951`'s takes into false alarms. `wake_clip` now means "Whisper found the phrase in
the take", which is the question the near-miss and false-alarm counts actually need.

### E21 — the loose capture gate (2026-09-04, feat/field-loose-gate)

E17 closed on a caveat it could not fix: a field take is self-selected, because it exists
only where the wake word fired. Every clip the feature has produced so far is therefore a
*successful* wake, and the set is structurally incapable of containing the two cases the
wake model most needs — the phrase it should have woken on and did not, and the speech it
woke on and should not have. No amount of capturing at 0.85 changes that; the gate is the
sampling bias.

So while capture is armed in Assistent mode the wake gate compares against **0.60**
instead of the shipped 0.85 (`field thresh`, persisted per device). It is the same gate,
not a second detector: `WAKE_MIN_CONSECUTIVE` and the refractory period are untouched, and
`field_gate_thresh()` returns the production value everywhere else — wake mode, Assistent
mode with capture off, every other mode — and can only ever lower the bar, never raise it.
At 0.60 the set also holds the phrase that peaked at 0.7, which is the first real training
example of the failure the user actually experiences, and the non-wake speech that got
near the bar, which is real-environment negative data no synthetic corpus provides.

Nothing is lost by loosening it, because the shipped gate is reconstructed on the
workstation rather than assumed. Each take's `wake_prob` is the peak of the run that fired
(not its last step, which is a different and less informative number), and `kws-qc` re-reads
it against 0.85 into `qc.csv`'s `would_fire`. Crossed with `wake_clip` — did Whisper, never
the device, find the phrase at the head of the take — that gives the two figures directly:
a wake clip with `would_fire=0` is a **near-miss**, a take with no wake clip and
`would_fire=1` is a **false alarm**, and both are reported again against the capture
threshold itself. The trigger for missed wakes now exists; it is just deliberately
over-eager, and the report separates the two gates rather than blurring them.

The honest limit is that this widens the window rather than closing it. A phrase that
peaks below the *capture* threshold is still invisible, so the near-miss count is a lower
bound on the misses, never the miss rate — there is no threshold at which a
self-triggering recorder can observe what failed to trigger it.

### E22 — near-miss hard negatives fix the round-6c regression (wake, 2026-09-04, host-only)

Round 6c had removed ~1 s of fire latency and matched the installed round 5a on every
real-audio gate, and lost exactly one row: TTS near-misses, 9/48 against 1–4/48. That row is a
*family* — the "hey"-ish onset and the "bus"-ish nucleus in the wrong pairing ("hallo bus",
"der bus kommt gleich", "hey du") — not scattered speech, so round 6d generates the family as
hard negatives (130 Piper clips: the phrase halves alone, in near pairings, and inside everyday
sentences carrying the same syllables, four voices × two rates) in its own feature dir, added
*on top of* round 6c's weights rather than paid for out of them. Nothing else changed.

**It works, and it generalises.** Every model scored on one fixed gate set, written to disk once
and reused, because Piper is not deterministic and round 6's gate re-synthesised on every run —
round 5a scored 1/48 and 3/48 on what was meant to be one measurement.

| gate | round 5a (installed) | round 6c | **round 6d** |
|---|---|---|---|
| held-out session 0849 fires (3) | 3/3 @ 0.996 | 3/3 @ 0.996 | **3/3 @ 0.996** |
| in-training real fires (10 guided) | 10/10 @ 0.996 | 10/10 @ 0.996 | **10/10 @ 0.996** |
| spk05 set (5) | 5/5 @ 0.996 | 5/5 @ 0.996 | **5/5 @ 0.996** |
| held-out real non-wake (9) | 0/9, worst 0.758 | 0/9, worst 0.309 | **0/9, worst 0.402** |
| in-training real non-wake (5) | 0/5, worst 0.406 | 0/5, worst 0.254 | **0/5, worst 0.598** |
| room noise, in-training (40) / held out (9) | 0/40, 0/9 | 0/40, 0/9 | **0/40, 0/9** |
| TTS non-wake, **seen** voices (46 fixed clips) | 6/46 | **18/46** | **4/46** |
| TTS non-wake, **unseen** voices (36 fixed clips) | 2/36 | **11/36** | **2/36**, worst 0.938 |
| TTS unseen-voice "hey bus" (2) | 1/2 | 1/2 | **1/2** |
| fire latency, held-out / guided (median) | 1.13 / 0.94 s | 0.08 / −0.01 s | **0.14 / 0.23 s** |
| first-hump peak (median) | 0.316 / 0.430 | 0.996 / 0.996 | **0.996 / 0.996** |
| size / MACs | 58,080 B / 24,736 | 58,080 B / 24,736 | **58,080 B / 24,736** |

The negatives were generated in four voices and the improvement shows up in the other three,
which is the difference between learning the family and memorising the gate — so the gate set is
split into a seen half and an unseen half, and a round that only improves the seen half fails.
Round 6d's entire seen-voice residue is `de_DE-mls-medium`, the one voice whose *short-phrase*
output the clip quality gate flags as unintelligible (1 of 7 gate clips transcribes back, against
4–5 of 7 elsewhere) while it passes a long calibration sentence perfectly. A synthetic gate can
be limited by the synthesiser rather than by the model, and the way to see that is to transcribe
the gate clips.

**Quality-gating synthesised audio is worth doing and worth doing at two levels.** The prompt was
a device test that played English `say` voices believed to be German. A per-voice calibration
sentence (Whisper must return `de` and the sentence verbatim) catches that decisively: all seven
cached `de_DE-*` Piper voices pass at coverage 1.00 and both English `say` control voices are
rejected at `lang=en`, rendering an English sentence rather than mispronounced German. The
per-clip form of the same check, though, cannot be enforced on the material this round needs: for
**correct** German Piper output of a 0.4 s "hey du" or "hallo bus", Whisper returns `fr`, `da`,
`ja` or `zh`, and it hears "Lichtkirche" for a correct "Licht Küche" in every voice. Enforced
below ~5 tokens it deletes precisely the near-misses. So: enforce per voice always, enforce per
clip where the clip is long enough to read, and log the rest.

**The internal metric disagrees for the third round running, and this time it disagrees the other
way.** Round 6d cuts microWakeWord's own false-reject rate by twelve points at the 0.85 operating
point (0.740 vs round 5a's 0.868) and pays for that willingness with 2.06 false accepts per hour
on its English ambient set, where round 5a has 0.000. Rounds 6 and 6b showed that metric cannot
see German near-misses; this round shows it *can* see something the German gates do not, namely
willingness on hours of continuous speech. Our own real-ambient evidence is 2.9 minutes of the
device's room tone (49 takes, 0 fires, worst peak 0.273) — clean, but not five hours. It neither
vetoes round 6d nor clears it; it names what a device soak has to answer, and E21's loose capture
gate is exactly the instrument for that soak.

ETA ledger: predicted 14.9 min (range 13.3–16.7, 10 runs), actual **13.37 min** (−10.3 %, inside
the range). Full report in the training directory as `HEYBUS-R6D-REPORT.md`.

**Recommendation: deploy `hey_bus_r6d.tflite` — after a few hours of field-capture soak on the
device, not instead of one.** It is the best candidate on every measurement taken against real
audio and German speech, at identical size and MACs, and it keeps round 6c's ~0.8 s latency win.
Nothing was promoted: `hey_bus.tflite` and `firmware/main/gen/` are untouched, no firmware was
built or flashed, and no audio was played to any device or speaker in this round.

### E23 — the synthetic clips were English, and nothing checked (2026-09-04, host-only)

A device test was driven by "German" clips synthesised with macOS `say` on a host without the
German voice packs. `say` substitutes an English voice for a missing one and prints nothing, so
the clips were English throughout; the test looked like a result and measured nothing. **A human
ear was the only check in the pipeline**, and it caught this one by luck — the same clips could
as easily have gone into the training set, where nobody listens at all.

The substitution is not limited to a host with no German voices. On a Mac that *has* them,
`say -v Eddy` and `say -v Flo` — bare names straight out of this repo's `ENGINE_VOICES["say"]`
pool — produce **English** audio identical to `say -v Samantha`, because those names exist in
both languages and the German ones are listed as "Eddy (German (Germany))". Eight of the nine
names in the pool are ambiguous that way. Whisper transcribed all three as "Lichtkoschen" for
"Licht Küche an", detected language `en`; only "Anna", which exists in German alone, was German.
So every `say` clip in the v3 dataset build on this machine, except Anna's, was English.

Two fixes, both cheap, and both needed. `engine_voices("say")` now asks `say -v '?'` which
German voices are installed and uses their full names — prevention. And `kws_de.qc.tts_gate`
judges an individual clip: duration/level without a model, then one Whisper pass whose
**detected** language must be `de` (forcing `language="de"`, as recording QC does, answers "de"
for an English clip — the gate must not), plus the existing content rules, the `wake` rule for
the wake phrase and the order-tolerant `sentences` rule otherwise. `required_tokens` for
`sentences` used to return the empty list for a prompt holding no command vocabulary, which
accepted *any* transcript; it now falls back to the whole prompt.

Detection needs a record of intent, so every synthesis writes `manifest.csv`
(`file,text,voice,engine`) beside its clips, and `kws-tts-check <dir>` gates a whole directory
into `tts_check.csv`, summarised per voice — a voice that is not German is 100 % failed, which
is what makes it legible. It runs in the dataset build's TTS top-up (failures dropped and
counted, never silently kept), in `wake-retrain.sh` before feature generation, and by hand
before clips are played at the device.

Seven-clip real sample, three `say` German voices by full name, three Piper `de_DE`, one
deliberate `say -v Samantha`: the Samantha clip rejected `language:en`, all three `say` German
clips and `de_DE-thorsten-medium` accepted. Two low-quality Piper voices were also rejected and
deserved to be — `de_DE-eva_k-x_low` said "liegt an" for "Licht an", and `de_DE-kerstin-low`
produced 0.37 s that Whisper could not transcribe even with German forced. The gate is
conservative by design: a rejected clip is dropped, never repaired.

The general lesson is the one worth writing up: **every silent fallback in a data pipeline is a
labelling bug waiting to happen.** `say` substituting a voice, a voice name resolving to another
language, an ASR forced into a language it was not given — each is individually reasonable and
each destroys the one property the data is supposed to have. Synthetic data needs a machine
check for exactly the property nobody will verify by ear.

### E24 — deploying round 6d (2026-09-04, host-only)

E22 recommended `hey_bus_r6d.tflite`: best candidate on every real-audio and German-speech
gate at identical size and MACs, keeping round 6c's ~0.8 s latency win. This entry does the
deploy, host-side only — no device work, no audio played to any device or speaker.

| | round 5a (was deployed) | round 6d (deployed) |
|---|---|---|
| `KWS_WAKE_MODEL_ID` | `hey_bus.tflite@dd9db24f 2026-09-03` | `hey_bus.tflite@5fcdaf63 2026-09-04` |
| size / MACs | 58,080 B / 24,736 | 58,080 B / 24,736 (unchanged) |
| held-out session fires (3) | 3/3 @ 0.996 | 3/3 @ 0.996 |
| held-out / in-training real non-wake | 0/9, 0/5 | 0/9, 0/5 |
| TTS non-wake, seen (46) / unseen (36) | 6/46 / 2/36 | **4/46** / 2/36 |
| fire latency, held-out (median) | 1.13 s | **0.14 s** |

Full acceptance table and TTS-quality-gate method: `HEYBUS-R6D-REPORT.md` in the training
directory (round 6d = round 6c's weights plus TTS near-miss hard negatives for the
"hallo bus" / "der bus kommt gleich" family that regressed in 6c).

Since PR #48 the firmware embeds generated esp-nn C rather than the TFLM interpreter, so
"deploy" means regenerating `gen/wake_model_{data,config}.h` from the new `.tflite` and
`gen/wake_infer.{c,h}` + `gen/wake_smoke_vectors.h` from that, not touching the command
model. As E15/E18 warn, `kws-export --firmware` unconditionally rewrites the wake pair from
the fixed `models/hey_bus.tflite` path alongside a full command re-export, so this deploy
called `kws_de.export.write_wake_headers` directly instead — the wake-only half of that
function — after promoting the candidate to the canonical filename (round 5's bytes kept on
disk as `hey_bus.v5.tflite`, following the `hey_bus.v1.tflite` / `hey_bus.v4.tflite`
precedent from earlier rounds). `test_check_is_clean_against_the_committed_headers` hardcodes
`models/hey_bus.tflite` as the wake source of truth, which is what makes the promote step
necessary rather than optional: pointing the generator at `hey_bus_r6d.tflite` in place
produces byte-identical headers but fails that test.

**Command model untouched.** `gen/model_data.h`, `gen/model_config.h`, `gen/command_infer.c`,
`gen/command_infer.h` are sha256-identical before and after this deploy.

**Architecture unchanged, confirmed in `gen/wake_infer.h`:**

| | round 5 | round 6d |
|---|---|---|
| `WAKE_INFER_ARENA_BYTES` | 128 B | 128 B |
| `WAKE_INFER_STATE_BYTES` | 4,200 B | 4,200 B |
| `WAKE_INFER_SCRATCH_BYTES` | 15,552 B | 15,552 B |

**Host checks, all clean.** `kws-fwgen --check firmware/main/gen` and both `kws-codegen
--check` runs (wake and command) exit 0. `make -C firmware/test`: `wake smoke: 0/64 steps
differ`, `command smoke: 0/368 bytes differ`, `host tests OK`. `pytest -q`: 308 passed, 1
skipped, 1 xfailed — includes `test_whole_wake_model_matches_the_interpreter_on_every_step`
(`wake parity: 0/635 steps differ (11 clips, 4200 B state)`) and the four `wake-opN` layer
parity cases. `ruff check` / `ruff format --check`: clean, 93 files formatted.
`markdownlint-cli@0.42.0 --config .markdownlint.json`: clean. `sphinx-build -W --keep-going`:
clean apart from the local "doxygen XML absent" warning CI's doxygen install would clear (the
same known-environment gap E15/E18 noted). ESP-IDF v5.5.5 Docker build (default config): OK,
app image **1,020,048 B** (`0xf9090`, 68% of the 0x300000 partition free).

**Device (2026-09-04 22:55, console only).** Flashed on the CoreS3: boot banner and
`status` read `wake=hey_bus.tflite@5fcdaf63 2026-09-04`; `mode wake` trace
`step 1256 +/- 111 us (invoke 1226 us)` over 68 steps per 2 s window — identical to E18's
round-5 baseline, as the unchanged arena/state/scratch predicted; room-noise peak 0.176,
no false fire. Left in Assistent mode with field capture armed at the production gate
(0.85) for a soak. Still open: a real "Hey Bus" fire check spoken aloud — round 6d's own
gate is TTS- and held-out-recording based, not a live-microphone confirmation; the first
spoken field session on this build supplies it (`kws-qc` Field line, `wake_prob`).
- `selftest int8 out:` line changes from round 5a's bytes (expected — different weights).
- A few hours of field-capture soak per E22's recommendation, before treating this as more
  than a host-side candidate swap.

### E25 — on-device intent parse + result card (2026-09-05, feat/on-device-intent)

The demo action: the device parses a window's fired words into a device/zone/action
intent on-device, rather than only listing raw keywords, and shows the result on
screen. `firmware/main/intent.{h,c}` is a hand port of `kws_de.grammar.parse()`
(device, optional zone, exactly one action, no duplicates, no out-of-order tokens,
`_unknown_`/`_silence_` dropped like the Python filter) plus `intent_format()`
("Licht Küche → an", "Licht → fünfzig Prozent", matching `kws_de.eval.intent_text`'s
word order/level suffix). The vocabulary itself is not duplicated: `intent.c` reads
`gen/labels.h` (`KWS_LABELS`, generated from `kws_de.config.COMMAND_LABELS`) and
only encodes the label layout's structure — device/zone/action counts and each
device's allowed-action bitmask, mirroring `kws_de.config.DEVICE_ACTIONS` /
`ZONED_DEVICES` — checked at compile time (`_Static_assert` against
`KWS_NUM_LABELS`) and at test time against the Python grammar itself.

**Host parity: 19/19 cases, 0 mismatches.** `scripts/gen-intent-cases.py` runs a
fixed list of fired-word sequences — valid with/without zone, a brightness level,
missing device, missing action, wrong order (zone/device swapped), duplicate
device, duplicate action, a zone on a device that takes none, an action the
device does not support, a level with no device, two devices, an interleaved
`_unknown_` token, a lone `_unknown_`, and empty — through `kws_de.grammar.parse()`
and commits the result table as `firmware/test/intent_cases.h`.
`firmware/test/test_intent.c` runs every case through `intent_parse()` and
compares device/zone/action/valid field for field, then checks `intent_format()`
against the zoned, level and invalid cases directly. `make -C firmware/test`:
`intent parity: 0/19 cases mismatch`, `test_intent OK`, `host tests OK` (all
targets, unaffected by this change, still pass).

**Wiring (`wake.cc`, at the assist window's close, once):** the window's raw fired
words (`recognise_status_t.window_intent`, unchanged — still what `recognise.cc`
collects, and still subject to E26/#68's `ASSIST_WAKE_TAIL_MS` drop, which happens
before this parse ever sees the words) are parsed and formatted, then:

- console log: `intent: Licht Küche → an` (valid) or `intent: none (<raw words>)`
  (invalid/empty), and `status` now carries an `intent <text>` line (`"none"` for
  invalid, omitted before the first window) — cheap, one `s_lock`-guarded 64-byte
  buffer, no new task or queue;
- `field.csv`'s `device_intent` column becomes the formatted text, empty when
  invalid — `device_words` is untouched (still the raw `"<word>:<conf>"` list).
  `kws_de.qc.field_intent()` re-derives its `Intent` from `device_intent` via
  `normalise()` + `grammar.parse()`, and `normalise()`'s `[^\w\s]` filter strips
  the arrow to a space before tokenising, so the round-trip is unaffected —
  verified by hand against `qc.py`'s `field_intent`/`normalise`, not by a new
  Python-side test (out of this branch's scope: no `kws_de/*.py` file changed);
- the assist screen shows a result card for 3 s (`ui_assist_show_result()`):
  green with the formatted text when valid, grey "nicht verstanden" plus the raw
  heard words when not. The card's own 3 s lifetime runs on an LVGL timer inside
  the LVGL task; the wake task's only LVGL touch is the existing one call per
  window that already painted this screen (`ui_assist_refresh` on the same call
  site), so window close still does at most one bsp_display_lock-guarded update
  per model-task edge, unchanged from before this branch;
- the confirmation double beep is now keyed to a **valid** intent instead of any
  fired command word — `stream_is_command(fired)` still latches a per-fire flag
  in `recognise.cc` (kept, but its value is no longer read for the tone decision;
  only drained at the close edge so it never carries into the next window), and
  sits inside E26/#68's single mute predicate (`!s_field.enabled`, every mode)
  rather than a second one.

**Rebased onto E26/#68** (`fix/assist-window-tail`, merged 2026-09-05): no logic
conflict — the wake-tail drop (#68) zeroes `fired` before this branch's window-
close parse ever reads `window_intent`, so a dropped tail fire is already absent
from the words this branch parses, and the confirmation-tone `else if
(!wake_field_get())` change in `recognise.cc` and this branch's `s_cmd_fired`
comment landed on the same lines — resolved by keeping both explanations in one
comment.

**Host checks, all clean.** `make -C firmware/test`: `host tests OK` (includes
`test_intent`). `ruff check` / `ruff format --check` on the new
`scripts/gen-intent-cases.py`: clean. `markdownlint-cli@0.42.0 --config
.markdownlint.json` on the touched Markdown: clean. `sphinx-build -W
--keep-going`: clean apart from the pre-existing "doxygen XML absent" warning
(E15/E18/E24's known CI-only gap). `cc -std=c11 -Wall -Wextra -Werror` compiles
`intent.c` standalone with no warnings. ESP-IDF v5.5.5 Docker build clean, both
configs, after the rebase.

**Device (2026-09-05, CoreS3, rig per E26).** Flashed post-rebase; boot confirms
both models on the generated-inference path (`wake: inference: generated
(esp-nn) ...`, `recognise: inference: generated (esp-nn) ...`, no
`AllocateTensors failed`). All checks below with `field on thresh 0.85`, real
audio played at the mic from `bar`'s speaker — no synthetic `wakefire`, so
every window here opened on a REAL wake fire (the E26 tail-drop fires
alongside it for real, not just in a host test):

- **Silence** (real "Hey Bus" alone, no command spoken): `wake: intent: none
  ()`; card would read "nicht verstanden" with no words underneath.
- **Wake-tail drop (#68) confirmed live**, not just host-tested: nearly every
  real "Hey Bus" playback also logged `recognise: assist: dropped tail fire
  aus 0.4-0.9, 353-365 ms into the window (< 450)` — the exact artefact E26
  fixed, still present in the raw acoustic signal and still correctly
  suppressed before this branch's parser ever sees it.
- **Invalid multi-word captures correctly rejected**, e.g. `wake: intent: none
  (Licht Kühlschrank)`, `wake: intent: none (Kühlschrank)`, `wake: intent:
  none (_unknown_)` — the device word (`Licht`/`Kühlschrank`) fired reliably
  (0.47-0.99) on this rig, but a lone action word played right after it
  (`an`, `aus`, `leise`, isolated word-set clips) did not clear the command
  model's threshold in ~10 tries, including alone with no device word
  competing (`fired _unknown_ 0.55` on "an" by itself) — a real-voice recall
  gap in the deployed `command_v3_w48_qat` model for this bar-speaker-to-mic
  acoustic path, not a parsing defect: every one of these was correctly
  reported as an invalid intent, never a false valid one.
- **Valid intent, confirmed**: replaying the approved TTS clip
  `de_heybuslichtan_thorsten_medium_6db.wav` ("Hey Bus, Licht an") gave
  `recognise: fired Kühlschrank 0.66` then `recognise: fired an 0.02`
  (model confusion on the device word this one time, but a real two-fire
  sequence) and **`wake: intent: Kühlschrank → an`** — the full path exercised
  live: grammar validation, `intent_format()`'s arrow text, and the
  `field.csv` row (`field: saved /sdcard/field/spk18/116-10776.wav ...,
  intent "Kühlschrank → an"`) all matching what host parity already proved on
  19 synthetic cases. The confirmation beep did not sound, as designed
  (`field on` mutes it — #68's single predicate).
- Screenshots: not attempted — `-DKWS_UI_SCREENSHOT` only instruments
  `ui_record.c` (`firmware/main/ui/ui_record.c`); the assist screen has no
  screenshot hook in this codebase, so the log lines above are the device
  evidence.

Left in `mode assist`, `field on thresh 0.85` (confirmed via a final `status`).

**Build mishap caught and fixed before any of this:** a `firmware/sdkconfig`
left over from an earlier `CONFIG_KWS_INFER_GENERATED=n` overlay build was not
regenerated before the "default" post-rebase build, so that build silently
carried the interpreter-fallback config into the boot log's PSRAM arena size
and produced `recognise: AllocateTensors failed` on first flash. Fixed by
deleting the stale `sdkconfig` and `sdkconfig.defaults` diff and rebuilding
before any device work — see `firmware/README.md`'s existing note on deleting
`sdkconfig` after an `sdkconfig.defaults` change, which this branch did not
follow closely enough the first time.

### E26 — field-capture beep leak and the "aus" window tail (issue #64, device)

Two field-capture defects fixed together, both found and verified on the CoreS3 (branch
`fix/assist-window-tail`).

**Defect A — wake beep leaking into field takes.** `wake.cc`'s beep predicate was
`!(assist && s_field.enabled)` — muted only while field capture was on **and** the mode was
Assistent — contradicting its own comment ("silent while field capture is on"). `field.enabled`
is a device-wide toggle that outlives a mode switch, so a fire while merely testing in Wake
mode still beeps at full volume, and that tone sits in the ring exactly where the next
Assistent take's 2.5 s pre-roll (`FIELD_PREROLL_MS`) reaches back to. Real evidence:
`field/spk18/74-46922553.wav` from the session `qc/2026-09-05-1202` cited in the original
report — a 150 ms burst at 0.53-0.55 s take-relative, dominant frequency 1,060-1,280 Hz, sample
peak 29,938 (-0.8 dBFS), matching `beep.c`'s wake tone (1,000 Hz/150 ms) to spec; the take's own
arming fire sits at t = 2.5 s as always, so the burst is not this take's own tone. Fix: one
predicate for both tone call sites (`wake.cc`'s `beep_play()`, `recognise.cc`'s non-assist
`beep_double()`) — mute on `s_field.enabled` alone, in every mode.

**Defect B — command window opens on the tail of "Bus" (#64).** Round 6d's wake fire lands
~0.14 s after the phrase (E24) instead of round 5's ~1.1 s, so the recogniser's first
classification — a ~1 s retrospective slice primed from ring audio predating the window — can
still score "...Bus" itself; the field session behind #64 saw `aus` as the first device word in
12/17 real takes. Fix: `ASSIST_WAKE_TAIL_MS` (`assist_gate.h`) — a command fire this soon after
the window opens is dropped before it reaches `window_words`/`device_words` or the confirmation
tone; the window still runs the full `ASSIST_WINDOW_MS`. Host-tested directly
(`assist_gate_in_wake_tail()`, `firmware/test/test_assist_gate.c`).

**First device pass landed the constant at 300 ms** (a fire at 100 ms dropped, at 400 ms
accepted) and verified it with a same-clip before/after replay (assist, field on, thresh 0.85;
3x "Hey Bus, Licht an" + 2x "Hey Bus" alone, ~11 s apart):

| | before (main) | after (300 ms) |
|---|---|---|
| spurious word on a bare "Hey Bus" (no command spoken) | `Licht` fires 374 ms into the window | `Licht` still fires, 381-386 ms in |
| first word of a real "Licht an" take | `Kühlschrank` (343-389 ms in) then `Licht an` | `Licht an` — no spurious prefix, in 2/3 replays |

On this rig the bare-"Hey Bus" artefact's own fire lands at 374-386 ms — *above* the 300 ms
cutoff — so that value did not suppress it. The floor is the recogniser task's fixed
~150-400 ms scheduling-plus-inference latency on window entry (100 ms mandatory poll delay +
~55 ms Invoke + scheduling jitter), not the phrase itself. Real commands in the same replay
fired no earlier than 531 ms.

**Raised to 450 ms** (real commands fire >= 531 ms on this rig, so the artefact at 374-386 ms
must be inside the skip; host boundaries moved to 400 ms dropped / 500 ms accepted) and
re-verified with the identical replay:

| | before (main) | after (450 ms) |
|---|---|---|
| bare "Hey Bus" that fired (1 of 2 attempts; the other never reached `WAKE_THRESHOLD`) | `Licht` fires 374 ms in, intent `Licht` | `assist: dropped tail fire Licht 0.68, 379 ms into the window (< 450)` — intent empty, **no command word** |
| "Hey Bus, Licht an" (window 1) | `Kühlschrank` @ 389 ms, then `Licht an` | `Licht` @ 707 ms, `an` @ 1,506 ms — intent `Licht an`, no spurious prefix |
| "Hey Bus, Licht an" (window 2) | `Kühlschrank` @ 343 ms only | `Licht` @ 654 ms — intent `Licht`, no spurious prefix |
| "Hey Bus, Licht an" (window 3) | no fire | no fire (unchanged — model variance, not the gate) |

450 ms clears the observed artefact with margin on both sides (374-386 ms artefact, >=531 ms
real commands) on this rig; it should still be re-checked against the original #64 session's
own timing before calling the tail fully closed elsewhere. Defect A's specific repro path (a
Wake-mode fire bleeding into a later Assistent take) was not independently re-reproduced live —
the console tooling sends all commands before the log-capture window opens, so a mode switch
cannot be interleaved mid-playback without a second port-open resetting the chip — but the fix
is a direct, host-test-covered correction of the predicate against its own documented intent,
and the cited real evidence file's signature is unambiguous.

Also from the original evidence set: `field/spk18/74-40246202.wav` fired at 0.965 on room tone
(-40 dBFS peak). `wake_set_active(true)` resets the front-end's adaptive noise-floor/PCAN state
(`wakefront_reset()`) with no burn-in guard before the gate starts comparing against threshold,
so a fire shortly after (re-)entering Wake/Assistent mode — while that estimate is still
settling — is architecturally plausible; the pulled `sessions.csv` alone does not carry mode-
entry timestamps to confirm this specific take was that close to one, so this is flagged as a
plausible mechanism, not a confirmed root cause.

### E27 — regenerating the v3 TTS cache through the quality gate (2026-09-05, host-only)

E23 found that `raw_clips_v3.pkl`'s TTS clips predate the synthetic-clip gate (PR #61):
every one uses the legacy `kws_de.data._say_one` speaker id (`tts:<voice>:<rate>`, no
engine field — `say` was the only engine when it was written), so none of them was ever
checked to be German or to say its intended word. This entry regenerates the cache
through the gate and asks whether the result is worth training on.

**Cache characterisation and drop.** `raw_clips_v3.pkl`: 29 words, 9,000 clips, 6,622
`tts:` (all legacy `say`-format) / 2,378 real (MSWC hashes; device recordings are merged
live on every build, never baked into the cache). New `scripts/drop-tts-clips.py`
(committed, with `tests/test_drop_tts_clips.py`) drops a cache clip whose speaker id
resolves to engine `say` — the legacy 3-part id ending in a digit rate (all 6,622 here)
or an explicit `tts:say:<voice>`; other engines are kept by default, `--all-engines`
drops every `tts:` clip regardless. Backed up as `raw_clips_v3.pre-regen.pkl`, then
dropped: 9,000 -> 2,378 clips. 21 of 23 command words lost every TTS clip they had
(0 real clips left); `Außen` and `Heizung` kept a partial real base (158/300, 120/300).

**Build 1 — strict gate (default).** `kws-dataset build --cache raw_clips_v3.pkl
--prefix features_v3` regenerates through `kws_de.qc.tts_gate` unchanged: German
language detection (Whisper, `whisper_transcriber(language=None)`) plus a full
content match on every clip regardless of length. Predicted ETA was useless (the
`dataset-build` ledger had one degenerate `size=1.0` row from an earlier fallback path,
predicting "~76,663,481 min"); actual wall time ~4 h (12:43-16:40). **Aggregate gate
drop: 3,224/4,822 = 66.9%.** Four words lost every single attempt (`an`, `zu`, `heller`,
`kälter`, all 300/300); the rest ran 48-88%. `[dataset] built seed=0: train=17268,
val=3918, test=1767`.

Reading the per-word drop dicts live during the build turned up two distinct failure
modes, only one of which the gate's content match is responsible for:

- **Language misdetection on short/common German words**, independent of engine —
  `heller` failed 12x per `say` voice with `language:en` on a correctly-resolved German
  voice saying a real German word; `zu`/`kälter` failed repeatedly with `language:zh`/
  `ko`. This is a Whisper reliability limit on short audio, not a voice-resolution bug.
- **A large, noisy `piper` voice pool.** `data/piper-voices/` carries a multi-speaker
  `de_DE-mls-medium#N` model with speakers numbered into the 100s, most of which are
  simply not German (`language:` values seen across the build: en, zh, fr, ja, ko, da,
  cy, ru, ar, nl, tr, sv, pl, cs, hu, fi, es, pt, it, he, lv, yi — the gate is doing its
  job here, not misbehaving).
- **A duration floor** (`TTS_MIN_S=0.3s`) rejects `say`'s synthesis of `an` outright
  (`duration:0.28s` at every rate tried) before Whisper ever runs — no content or
  language check can rescue this one.

**Course correction: `KWS_TTS_GATE=lenient`.** The coordinator flagged mid-build that a
strict content match on a 1-2 token transcript is expected to reject good German audio
(round-6d's field report already has Whisper mishearing "Küche" as "Kirche")
more than it catches bad audio, on a build that is almost entirely single-word clips.
Added `min_tokens_for_content` to `kws_de.qc.tts_gate` (default 0, i.e. unchanged
strict behaviour): below that many heard tokens, German language detection alone
gates; at or above it, the full content match still runs. An empty transcript is never
treated as "short" — it still hits the content gate and is rejected. Wired through
`kws_de.data.tts_gate_min_tokens_for_content()` (`KWS_TTS_GATE=lenient` -> 3, anything
else -> 0) and `_tts_fill_word`. Four new tests (`tests/test_qc.py`,
`tests/test_data.py`), 100/100 passing, ruff clean.

**Build 2 — lenient gate.** Same procedure, cache reset to the clean 2,378-clip
baseline first (`cp raw_clips_v3.pre-regen.pkl raw_clips_v3.pkl && python
scripts/drop-tts-clips.py raw_clips_v3.pkl` — deterministic, reproduces the identical
baseline). **Aggregate drop: 3,122/4,822 = 64.7%** — barely better than strict's 66.9%,
confirming the per-word read above: `an` (300/300, unchanged — duration floor, gate
mode is irrelevant), `zu` (299/300, was 300/300), `heller` (299/300, was 300/300) are
essentially untouched by relaxing content matching, because their failures are
language/duration gates that `min_tokens_for_content` never reaches. `kälter` is the
one clear win (256/300, was 300/300 — 44 clips rescued; its failures included more
actual content-order mismatches than the other three). Every other word moved by low
single digits either way (noise). `[dataset] built seed=0: train=18442, val=2473,
test=2853`.

**ETA ledger.** `train`: predicted ~2.3 min (7 runs) for the strict build, actual
6m42s; predicted ~2.7 min (9 runs) for the lenient build, actual 4m44s — both overshoot
the prediction, the same gap E16 recorded (the ledger's `size` = epochs x rows carries
no width term, and its history is mostly width-32 runs).

**Real-voice comparison.** All three models scored with `eval_recordings` +
`make_command_predict_fn` on the same 197 word / 101 phrase / 29 negative approved
clips (`compare_command_models.py`, a throwaway driver in the E16 vein — `kws-eval` has
no `--model` flag) plus a fourth speaker, `spk18`, that has joined the approved tree
since E15-E18 (36 words, 17 phrases, 3 negatives, held-out):

| | deployed w48 | strict-regen w48 | lenient-regen w48 |
|---|---|---|---|
| bytes / sha256 | 25,832 / `8fa81d08` | 25,832 / `94144f90` | 25,832 / `1449b619` |
| INT8 test acc (own-era test set, not cross-comparable) | 93.59% | 76.34% | 84.23% |
| spk01 words (n=13, in-training) | 0.923 | 0.846 | 0.846 |
| spk02 words (n=38, in-training) | 0.895 | 0.895 | 0.789 |
| spk10 words (n=146, held-out) | 0.856 | 0.815 | 0.849 |
| spk18 words (n=36, held-out) | 0.333 | **0.667** | **0.667** |
| **aggregate words (n=233)** | 0.785 | **0.807** | **0.811** |
| false accepts, spk02 (n=10) | 0/10 | 0/10 | 0/10 |
| false accepts, spk10 (n=19) | 0/19 | 1/19 | 1/19 |
| false accepts, spk18 (n=3) | 0/3 | 0/3 | **1/3** |
| **false accepts total (n=32)** | **0/32** | 1/32 | 2/32 |
| phrase intent | unchanged (spk10 0.082, rest 0.000) | unchanged | unchanged |

**Decision.** Per the pre-agreed rule (deploy a candidate iff real-voice aggregate
words >= deployed's 0.785 AND false accepts no worse than deployed's 0/32 AND spk18
words > deployed's 0.333): both regenerated candidates clear the aggregate-words and
spk18 bars comfortably, driven by the same mechanism — capacity/coverage on a speaker
the deployed model never saw enough of — but **both regress the false-accept rate**
(1/32 strict, 2/32 lenient, against 0/32 deployed), which the original task brief
already named as the metric that must not get worse. **No candidate satisfies all
three conditions -> the deployed w32/w48 export stays; `firmware/main/gen/` and
`models/command_v3_w48_qat.tflite` (the actual checked-in deploy path) are untouched by
this session** — the strict/lenient exports written during evaluation went through
`kws-export` without `--firmware`, so they never touched the firmware headers, and
`models/` is gitignored regenerable build output, not a committed artefact.

**Reading.** The TTS regeneration is not wasted work — it fixes a real, silent defect
(most training-time TTS was English) and both regenerated models generalise better to
a held-out speaker the deployed model handles worst — but it is not sufficient by
itself to ship, because the new false accepts land on exactly the safety-side metric
the recipe protects hardest. The root cause is upstream of both gate settings tried
here: Whisper's language-detection reliability on short/common German words (`heller`,
`zu`, `kälter` all fail primarily on `language:`, not content) and a noisy `piper`
voice pool (`de_DE-mls-medium#N`, mostly non-German) that neither strict nor lenient
content-matching touches, plus a hard duration floor that rejects `an` outright before
any transcription runs. Fixing those — filtering `piper`'s `mls-medium` pool to its
verified-German subset, and/or a lower `TTS_MIN_S` for very short words with a
duration-aware augmentation check — is the next lever, not another gate-strictness
knob. Device flashing/measurement of either candidate was out of scope for this
host-only session and was not performed.

### E28 — voice-level TTS gate, van augmentation, decoder sweep (2026-09-06, host-only, feat/tts-voice-gate)

E27 found the per-clip content gate mostly rejects good German audio, not bad audio,
on short/single-word clips — `heller`/`zu`/`kälter` fail on Whisper language
misdetection, `an` fails a hard duration floor, and a noisy `piper` `mls-medium`
speaker pool (mostly non-German) drowns out everything else. This entry replaces the
per-clip gate with a voice-level one (`kws_de.qc.voice_gate`,
`kws_de.data.passing_voices`): one fixed calibration sentence ("Bitte schalte das
Licht in der Küche an und die Heizung im Bad aus") per voice, cached in
`tts_voice_gate.json`, ≥90% token match + detected `de` to pass; a passing voice's
clips then only need `kws_de.qc.tts_cheap_gate` (duration ≥ `TTS_MIN_S`, now 0.25s;
not silent — no per-clip Whisper). Also lands opt-in van-cabin augmentation
(`kws_de.augment.van_augment`, off by default) and `scripts/sweep-decoder.py` (a
read-only decoder threshold/hangover report, no retrain). Simplified away #71's
`min_tokens_for_content` knob and `drop-tts-clips.py --all-engines` first (neither
had a real use case once the voice gate replaced the mechanism they were patching).

**Voice-gate results.** 259 candidate voices gated once and cached: all 9 `say`
voices pass; 2/14 non-`mls` Piper voices pass (`karlsson-low`, `pavoque-low`); 40/236
`de_DE-mls-medium#N` speakers pass — confirming E27's read that the pool is mostly
non-German. **51/259 pass overall.**

Surprise: 5 of the 12 failing non-`mls` Piper voices (`thorsten-medium`, `eva_k`,
`kerstin`, `ramona`, all 8 `thorsten_emotional#N` speakers) fail not because they are
bad voices, but because Whisper transcribes the calibration sentence's "Küche" as
"Kirche" — the exact mishearing round-6d's field report already named
(`train/mww/README.md` rule 5). `thorsten-medium`'s transcript is otherwise a
verbatim match: `"...das Licht in der Kirche an und die Heizung..."`. One wrong token
out of 6 scores 0.833, under the 0.9 bar, so a demonstrably fine voice is dropped
entirely. Flagged as a follow-up (below), not fixed in this session — redoing the
~25-minute build with a different calibration sentence was judged not worth it against
the numbers already in hand.

**Drop rate, before/after.** E27: 66.9% (strict gate) / 64.7% (lenient gate) of
synthesized clips dropped, aggregated over the whole build. This build:
`raw_clips_v3.pkl` backed up to `raw_clips_v3.pre-voicegate.pkl`,
`drop-tts-clips.py` dropped 4,078 → 2,465 (say/legacy TTS out, 87 piper + 2,378 real
kept), then `kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3`
regenerated the rest. **Aggregate gate drop: 293/4,735 = 6.2%** — every dropped clip's
reason is now a plain `duration:` cheap-gate rejection, zero language/content
mismatches. Per-word: `an` 88/300 (29.3%, was 300/300 in E27), `zu` 72/299 (24.1%, was
~300/300), `heller` 0/299 (was 300/300), `kälter` 0/292 (was 44/300 partial), `Dach`
126/300 (42.0%, a new word this build actually attempted), `Küche` 6/299 (2.0%).
`[dataset] built seed=0: train=32725, val=4773, test=8901` — both splits much larger
than E27's (train 17268-18442) since `an`/`zu`/`heller`/`kälter`/`Dach` etc. now
actually receive TTS clips instead of losing every attempt.

**ETA ledger.** `train` (`--v2 --width 48 --qat --prefix features_v3`, epochs=40,
size=40×32725=1,309,000): predicted ~5.7 min (range 4.2–8.6, 10 runs), actual 8m46s
— over the range, consistent with E27's own note that the ledger's `size` carries no
width term and its history is mostly narrower runs. `kws-dataset build` itself: ~25m26s
wall time, well under the "expect < 1h" estimate.

**Decoder sweep** (`scripts/sweep-decoder.py`, committed separately, run against the
*currently deployed* model before this retrain — a read-only report, no firmware
change): 88 field takes (18 with a parsed command, 70 negative). Best over the swept
grid (threshold 0.3–0.9, hangover 0/1/2 extra confirmation frames): **threshold=0.3,
hangover=1 → 4/18 agreement, 0/70 false fires**, vs the firmware's current
threshold=0.5, hangover=1 (`KWS_THRESHOLD`/`KWS_MIN_CONSECUTIVE`) → 3/18, 0/70. A small
edge, on a small sample (18 parsed field takes) — not acted on; firmware constants are
unchanged.

**Real-voice comparison**, `scripts/compare_command_models.py` (copied from
`.worktrees/data-regen`'s throwaway E27 driver, now committed) against the deployed
w48 (`firmware/main/gen/model_data.h`) and the freshly exported candidate, same
197-word/101-phrase/29-negative approved set plus `spk18`:

| | deployed w48 | candidate w48 (voice-gate regen) |
|---|---|---|
| bytes / sha256 | 25,832 / `8fa81d08` | 25,832 / `e398048c` |
| INT8 test acc (own-era test set, not cross-comparable) | 93.59% (n=10,356) | 64.54% (n=8,901) |
| spk01 words (n=13, in-training) | 0.923 | **0.538** |
| spk02 words (n=38, in-training) | 0.895 | 0.816 |
| spk10 words (n=146, held-out) | 0.856 | 0.747 |
| spk18 words (n=36, held-out) | 0.333 | **0.806** |
| **aggregate words (n=233)** | **0.785** | 0.755 |
| false accepts, spk02 (n=10) | 0/10 | 0/10 |
| false accepts, spk10 (n=19) | 0/19 | 0/19 |
| false accepts, spk18 (n=3) | 0/3 | 0/3 |
| **false accepts total (n=32)** | **0/32** | **0/32** |
| phrase intent | spk10 0.082, rest 0.000 | unchanged |

**Decision.** Per the pre-agreed rule (deploy iff aggregate words ≥ deployed's 0.785,
false accepts no worse than 0/32, spk18 words > deployed's 0.333): the candidate
clears false-accepts (0/32, tied) and spk18 by a wide margin (0.333 → 0.806, the
speaker the deployed model handles worst), but **misses the aggregate-words bar**
(0.755 < 0.785), driven by a sharp regression on `spk01` (0.923 → 0.538, n=13 — small
sample, but the largest single move in the table and the opposite direction from
everything else). **Candidate does not satisfy all three conditions → the deployed
w48 export stays.** `firmware/main/gen/` and `models/command_v3_w48_qat.tflite` (the
checked-in deploy path) are untouched: `kws-export` ran without `--firmware`, and
`models/` is gitignored regenerable output. Current `command_v3_w48_qat.tflite`/
`.keras`/SavedModel dir backed up as `*.pre-voicegate.*` before export.

**Reading.** The voice-level gate does exactly what it was built for — it fixed the
five/six previously-zero words and cut the aggregate drop rate 10x (66.9%→6.2%) — and
the resulting model generalises dramatically better to `spk18`, the speaker the
deployed model was worst at. It does not clear the deploy bar this round because of
one speaker's regression that this session did not diagnose further (host-only,
no device access). Two concerns worth a follow-up, not resolved here: (1) the
calibration sentence's own "Küche" is exactly the word Whisper mishears elsewhere in
this codebase, so the voice gate inherits that false-negative — a sentence built from
words Whisper handles reliably (or a per-word near-miss tolerance) would likely pass
several more good voices, including `thorsten-medium`; (2) `spk01`'s regression is
worth isolating (which words, which augmentation) before the next retrain attempt.
Van-cabin augmentation (`KWS_NOISE_DIR`/`KWS_RIR_DIR`) was not enabled for this
build — the task did not call for it, and this entry's numbers reflect the voice gate
alone. Device flashing/measurement was out of scope for this host-only session and was
not performed.

### E29 — grammar rescoring at window close + wake burn-in (feat/grammar-rescoring-burnin)

Two small firmware fixes, both from device evidence: `#67`'s field test saw `an`
rarely fire even when heard clearly enough to be the second-best candidate the
moment `Licht` (or another device) landed on `_unknown_`, and a separate take
fired at wake_prob 0.965 on -58 dBFS silence 40 s after boot — the wake gate
trusting the very first post-reset steps, before `wakefront_reset()`'s cleared
noise estimate/PCAN gains had resettled.

**A — grammar rescoring.** `firmware/main/intent.c`'s `intent_rescore()`: when
the plain `intent_parse()` fails, retry once by substituting the stream
decoder's runner-up command word at each `"_unknown_"` slot, accepting the
retry only if exactly one substitution (at or above `INTENT_RESCORE_FLOOR`,
0.25) makes it valid. `firmware/main/stream.{h,c}` now keeps `last_smoothed[]`
(the moving-average vector `stream_push()` already computed but discarded) so
`firmware/main/recognise.cc` can record each fired step's runner-up alongside
`window_words` in a new `window_seconds` field
(`firmware/main/recognise.h:22`), same `"<label>:<conf>"` / `'|'` format,
positionally aligned. `firmware/main/wake.cc`'s window-close edge falls back to
`intent_rescore()` only when the plain parse is invalid, logs `intent: <text>
(rescored: X->Y)` on a successful retry, and hands `post_field_take()` the
same `text`/`rst.window_words` as before — `device_intent` carries the
rescored text, `device_words` stays the untouched raw fires.

Host test: `firmware/test/test_intent.c` adds 5 cases — a missing action and a
missing device slot, each fixed by one substitution; below-floor and
two-`_unknown_` cases both correctly refused; an already-valid parse is a
no-op. `scripts/gen-intent-cases.py`/`intent_cases.h` (the plain-parse parity
table against `kws_de.grammar.parse()`) is untouched.

**B — wake burn-in.** `firmware/main/wakefront.c`/`.h`: a `WAKEFRONT_BURN_IN_STEPS`
(33, ~1 s of 30 ms wake-model steps) counter, reset to 0 by both
`wakefront_init()` and `wakefront_reset()` and incremented (capped) by every
`wakefront_take()`; `wakefront_warm()` reports true once it reaches the cap.
Lives in `wakefront.c` rather than `wake.cc` because it is fundamentally a
front-end property (is the noise estimate/PCAN state resettled yet), which
also makes it host-testable exactly like the rest of the front-end.
`firmware/main/wake.cc`'s fire condition gains one `&& wakefront_warm()`
clause — one line, applying to every `wakefront_reset()` call site (mode
entry and the ring-overrun path both), not just mode entry. The peak trace
above the fire check is unaffected, so a burn-in "fire" that never happened
still shows up as a peak in the log.

Host test: `firmware/test/test_wakefront.c` adds a block asserting
`wakefront_warm()` is false for the first `WAKEFRONT_BURN_IN_STEPS` takes
after a reset and true from the next one on (content-agnostic — silence PCM,
since only the step count is under test).

**Host checks, all clean.** `make -C firmware/test`: `host tests OK` (12
targets, including the new `test_intent`/`test_wakefront` cases). Docker
`espressif/idf:v5.5.5 idf.py build`: clean, `kws_de_fw.bin` 0xf9eb0 B (67%
partition free). Repeated once more with a `CONFIG_KWS_INFER_GENERATED=n`
overlay in a scratch build dir (`build_overlay`, deleted after; `firmware/sdkconfig`
untouched — both gitignored) to confirm the interpreter fallback path still
builds: clean, one pre-existing unrelated warning (`recognise.cc:180: unused
variable 'use_generated'`, present on the interpreter path regardless of this
branch). `ruff check`: clean (no `kws_de/*.py` touched).

**Device verification: pending.** `bar` (the CoreS3's remote host) was
unreachable over Tailscale SSH for the whole of this session — every
`ssh bar` attempt timed out at the TCP connect stage, tried repeatedly over
more than the 10-minute budget agreed with the coordinator. Not flashed, not
run on real audio. The clips this verification needs are already staged and
gated: `uv run --no-sync kws-tts-check` on the "Hey Bus, Licht an" takes
(`eva_k`, `ramona`, `thorsten` voices, 6 dB) found `eva_k` and `thorsten` `ok`
(`de`, transcript "Hey Bus, Licht an.") and `ramona` failing content match
(transcript "Hey Bus liegt an" — Whisper mishearing "Licht" as "liegt", a real
gate rejection, not a manifest bug) — only the two `ok` clips are cleared to
play. Follow-up device session: flash, `mode assist`, confirm no fire in the
first 2 s of the wake trace (burn-in), `field on`, play the two `ok` clips
(three plays total) and confirm `intent: Licht -> an` (plain or rescored)
fires more often than `#67`'s baseline, `scripts/ingest.sh -H bar` the session
into `incoming-tts/`, leave the device in Assistent with `field on`,
`field thresh 0.85`.

### E30 — elicitation ("Situationen"): natural phrasing instead of read sentences (host-only)

Every guided sentence prompt to date is a script: the speaker reads `Licht Küche an` off
the screen, so the phrase clips it produces are read speech, not natural command speech.
Round-6d's field report (E17/E21) already measured how far that generalisation gap runs:
a command model trained only on guided (read) clips scored **0.27 intent accuracy on
real field clips** — natural "Hey Bus, ..." utterances the wake gate actually captured —
against >0.9 on its own held-out guided test set. Reading a sentence and asking for
something in your own words are different speech acts (different prosody, different word
order, filler words, self-correction), and no amount of more guided recording closes that
gap; only speech elicited the second way does.

**Design.** A third guided-recorder mode, "Situationen", next to Record (words/sentences/
negatives) and "Hey Bus aufnehmen" (the wake-only set). Each prompt is a SCENE ("Es ist
dunkel in der Küche.") or a QUESTION the device asks ("Wo soll die Heizung an?") plus the
on-screen cue "Sag es dem Bus" — never the words to say. The speaker answers however they
would naturally, wake phrase included ("Hey Bus, mach das Licht in der Küche an"). Each
prompt carries an EXPECTED INTENT (`config.SITUATIONS`: 30 (scene, intent) pairs, e.g.
`("Es ist dunkel in der Küche.", "Licht Küche an")`), not words — the recorder never tells
the speaker which tokens to hit.

**Firmware.** `PROMPT_ELICIT` (`prompts.h`/`prompts.c`): one take per prompt (a natural
answer is not read-and-redo material), a 9.8 s cap and 1200 ms hangover (an unscripted
answer runs longer than a read sentence), `_elicit_/<slug>_NNN.wav` filing under the
speaker's session dir. `record.c`'s `session.csv` row writes the EXPECTED INTENT text as
the `prompt` column (`prompt_intent()`), not the scene text (`prompt_text()`) shown on
screen — QC needs the target to score against, not the display copy. New menu entry
"Situationen", console `mode elicit`, `UI_MODE_RECORD_ELICIT` — otherwise identical to
every other guided mode (pause/resume, speaker id, storage root, USB export).

**QC (`kws_de.qc`).** An elicit take starts with the wake phrase exactly like a field
take, so it is routed through the SAME wake-split -> Whisper -> `_split_glued` ->
`grammar.parse` -> filing pipeline as `set == "field"` (wake clip, phrase + word clips, or
negative — never relabelled to the expected intent even on a match: the Whisper-derived
label is always what gets filed, because that is what the clip actually contains). Counted
separately from field takes (`n_elicit_*`, no capture-vs-production wake-gate comparison —
a guided take is not gated at all) and scored against the expected intent via a new
`expected_match` qc.csv column: `"1"`/`"0"` when the Whisper-parsed `Intent` does/doesn't
equal `field_intent(normalise(expected))`, `""` when the answer didn't parse at all (nothing
to compare). A `"0"` is not a reject — alternative phrasing of a valid command is exactly
the natural-speech data this mode exists to collect; only a genuinely unparsable answer
stays unfiled. `report.md` gets an `## Elicit` line (takes, approved, parsable,
said-what-we-expected rate over compared takes, wake clips, approved-but-unfiled), and
`kws_de.eval` an `## Elicit` section beside `## Field` (`elicit_figures`/
`render_elicit_section`, per-speaker table).

**What `expected_match` measures, and what it does not.** It is not intent-recognition
accuracy — every filed clip (matched or not) still trains/tests the model under its own
Whisper-derived label. It is a proxy for how predictably a scene elicits its intended
command: a low rate means the SITUATIONS wording is ambiguous or invites a different
phrasing than expected (a corpus-design signal), not that the recording session failed. A
model trained partly on this corpus should be evaluated the same way E17's field figures
already are — against real field clips — since that is the gap this mode exists to close;
`expected_match` on the elicitation session itself is a collection-quality check, not a
substitute for that evaluation.

Host-only session: no elicit recordings were collected. Test plan mirrors the field
one (`tests/test_qc.py` `_elicit_session`-style fixtures, `tests/test_eval_recordings.py`
`_elicit_qc_root`), plus firmware host tests (`firmware/test/test_prompts.c`) and
`kws_de.firmware_gen` coverage (`kws-fwgen --check` now also covers `KWS_ELICIT_*`).

### E31 — fixing the voice gate's own false negative, van augmentation on (2026-09-06, host-only, feat/tts-voice-gate-2)

E28 flagged two defects in its own voice-gate build without fixing them: the calibration
sentence's "Küche" is exactly the word Whisper mishears as "Kirche" elsewhere in this
codebase, wrongly failing demonstrably good voices; and van-cabin augmentation
(`KWS_NOISE_DIR`/`KWS_RIR_DIR`) was left off. This entry fixes both and reruns the
rebuild/retrain/eval E28 already did.

**Calibration sentence.** Replaced `VOICE_GATE_SENTENCE` — "...der Küche an..." — with
"Mach bitte das Licht außen an, schalte den Kühlschrank aus und stell die Heizung
wärmer", which drops "Küche" entirely while still exercising umlauts/eszett (`außen`,
`Kühlschrank`, `wärmer`) and five more command words (`Licht`, `an`, `Kühlschrank`,
`aus`, `Heizung`) — seven required vocabulary tokens. `voice_gate`'s match was also the
other half of the bug: `content_gate`'s sequential in-order matching meant a single
early mishearing zeroed out every required token *after* it in the sentence too (a
`thorsten-medium`-shaped transcript with only "Küche"→"Kirche" wrong scores 0.2, not
"5/6" — the rest of the sentence is right there in the transcript, just never reached
because the pointer never advances past the missed token). New `qc._voice_gate_score`
checks each required token's presence anywhere in the transcript (order-independent,
same exact-or-edit-distance-1 tolerance `_matches` already used), and the pass bar
moved 0.9 → 0.85 (tolerates exactly one bad token of seven). Four new/renamed tests in
`tests/test_qc.py`; `docs/sphinx/pipeline.rst` updated to match.

**Re-gating.** Deleted `tts_voice_gate.json` and re-gated all 259 candidate voices
against the new sentence/score (backed up as `tts_voice_gate.json.e28-backup`).
**252/259 pass, up from 51/259.** All 9 `say` voices still pass (unchanged). All 14
non-`mls` Piper voices now pass (was 2/14) — `thorsten-medium` is back, plus `eva_k`,
`kerstin`, `ramona`, and all 8 `thorsten_emotional#N` speakers, exactly the voices E28
named as wrongly failed. `de_DE-mls-medium#N` speakers: 229/236 pass (was 40/236). The
remaining 7 failures are real: each is missing two or more required tokens on an
actually-garbled reading (`Licht aus und anschalte` for `Licht außen an`, `Heizungwärme`
glued into one word for `Heizung wärmer`), not a mishearing artefact — the gate is
right to drop them.

**Van augmentation.** `KWS_NOISE_DIR=<mww-train>/data/fma_16k` (210 files) and
`KWS_RIR_DIR=<mww-train>/data/mit_rirs` (270 files) exported before the build; both are
plain directories of 16 kHz wavs, no code change needed (`van_augmentation_enabled()` is
already opt-in via these two env vars, per PR #73). Confirmed with the existing tests
(`tests/test_augment.py`, `tests/test_data.py`'s van tests, 4 passed) and by the row-count
delta below.

**Rebuild.** `raw_clips_v3.pre-voicegate.pkl` → `raw_clips_v3.pkl` → `drop-tts-clips.py`
(4,078 → 2,465, identical to E28's starting point) → `kws-dataset build --cache
raw_clips_v3.pkl --prefix features_v3`. **Aggregate gate drop: 331/4,735 = 7.0%**
(E28: 293/4,735 = 6.2%) — still entirely plain `duration:` cheap-gate rejections, zero
language/content mismatches; the small increase over E28 is the wider voice pool (252
vs. 51 admitted voices) drawing more short-duration attempts from the same handful of
inherently-short words (`Dach` 146/300, `an` 88/300, `zu` 76/299 — vs. E28's `Dach`
126/300, `an` 88/300, `zu` 72/299), not a regression in gate quality. `kälter` and
`Aufstelldach`/`Außen`/`Heizung`/`hundert`/the `fünf*` numbers now synthesize with
**zero** drops (was partial in E28). `[dataset] built seed=0: train=35,261, val=6,417,
test=11,507` — train grew 32,725 → 35,261 (+2,536 rows), consistent with van
augmentation's 3 extra rows (`VAN_SNRS = (0, 5, 10)` dB) per real (non-TTS) clip.

**Retrain + export.** `kws-train --v2 --prefix features_v3 --out command_v3_w48.keras
--width 48 --qat` (epochs=40, `kws-eta` predicted ~6.8 min, actual 9m32s — same
ledger-underestimate pattern E27/E28 both noted, no width term in `size`), then
`kws-export --v2 --qat --prefix features_v3 --model command_v3_w48.keras --width 48`
(no `--firmware` — evaluation only, pending the deploy decision below). Run-1's
(E28) exports backed up as `*.run1-voicegate` first (models and
`features_v3_{train,val,test}.npz`/`manifest_v3.json`).

**Real-voice comparison**, `scripts/compare_command_models.py`, deployed vs. E28's
run-1 candidate vs. this run's candidate, same 197-word/101-phrase/29-negative approved
set plus `spk18`:

| | deployed w48 | run-1 candidate (E28) | run-2 candidate (E31) |
|---|---|---|---|
| bytes / sha256 | 25,832 / `8fa81d08` | 25,832 / `e398048c` | 25,832 / `d49093a4` |
| INT8 test acc (own-era test set, not cross-comparable) | 93.59% (n=10,356) | 64.54% (n=8,901) | 65.92% (n=11,507) |
| spk01 words (n=13, in-training) | 0.923 | 0.538 | **0.769** |
| spk02 words (n=38, in-training) | 0.895 | 0.816 | 0.684 |
| spk10 words (n=146, held-out) | 0.856 | 0.747 | **0.829** |
| spk18 words (n=36, held-out) | 0.333 | 0.806 | **0.667** |
| **aggregate words (n=233)** | **0.785** | 0.755 | 0.777 |
| false accepts, spk02/spk10/spk18 | 0/10, 0/19, 0/3 | 0/10, 0/19, 0/3 | 0/10, 0/19, 0/3 |
| **false accepts total (n=32)** | **0/32** | 0/32 | 0/32 |
| phrase intent (spk10 only nonzero) | 0.082 | 0.082 (unchanged) | 0.113 |

**spk01 (n=13, in-training, one clip per word).** Deployed gets 12/13 right, missing
only `kälter`. The E31 candidate gets 10/13, missing `kälter` (unchanged — the same word
the deployed model already fails, still the thinnest-covered word in the vocabulary
even with the fixed gate) plus two new misses, `Außen` and `auf`. Neither traces to TTS
coverage: `auf` never appears in this build's "synthesizing N more" log at all — real
(MSWC/device) clips already met the 300-clip target before any TTS was drawn — and
`Außen` kept its strong real base (159/300 real, 141 TTS-topped) with **zero** cheap-gate
drops this run, so nothing noisy or `mls`-sourced fed either word. With one clip per
word, each miss is a single binary outcome in an 13-item slice; the more plausible
explanation is ordinary retrain variance now amplified by van-cabin augmentation
changing the training-noise distribution (materially different from what either the
deployed model or run-1 trained on), not a data-quality regression the calibration fix
could reach. spk02 (in-training) shows the same shape of movement, dropping from run-1's
0.816 to 0.684 — a second signal pointing at the retrain/augmentation axis rather than
the voice gate.

**Decision.** Per the same pre-agreed rule (aggregate words ≥ deployed's 0.785, false
accepts no worse than 0/32, spk18 words > deployed's 0.333): the E31 candidate ties
false accepts (0/32) and clears spk18 by a wide margin (0.333 → 0.667), but **still
misses the aggregate-words bar** (0.777 < 0.785 — short by 2 of 233 words), driven by
spk01's and spk02's in-training regressions. **Candidate does not satisfy all three
conditions → the deployed w48 export stays**, a third session in a row. `firmware/main/gen/`
and `models/command_v3_w48_qat.tflite` (the checked-in deploy path) are untouched:
`kws-export` ran without `--firmware`.

**Reading.** Both of E28's named defects are now fixed and both worked as expected in
isolation — the calibration sentence fix alone lifted the voice pass rate 51→252/259 with
no code path left to blame Whisper's own mishearing for a rejected voice, and van
augmentation trained more (and differently-perturbed) real-clip rows without breaking
anything the existing tests check. Combined, the candidate closes about 71% of run-1's
aggregate-words shortfall (0.755 → 0.777 against deployed's 0.785) while holding false
accepts and clearing spk18 — real progress — but it does not flip the deploy decision,
because a *new* pair of in-training regressions (spk01, spk02) opened up that a wider
voice pool and van-cabin noise do not obviously explain and this session did not
diagnose further (out of scope per the task brief — spk01 above is investigation, not a
fix). The next lever is almost certainly on the retrain/augmentation side, not the TTS
gate: e.g. checking whether van augmentation's SNR range or the RIR pool is too
aggressive for a model this small, or whether the run-to-run variance on a 233-clip
real-voice eval is simply too high to reliably tell a real regression from noise without
`kws-benchmark --folds` (open question, below). Device flashing/measurement was out of
scope for this host-only session and was not performed.

### E32 — per-layer profiling of the generated inference (2026-09-06, measured on the CoreS3)

E18/E29's device numbers are model-level: a 42 ms command invoke, a ~1.3 ms wake step. The
generated path calls esp-nn kernels directly (no TFLM dispatch), so `nn_timers.h`'s
per-kernel-family globals — filled only by TFLM's op wrappers — read blind on it (`conv 0, dw
0, fc 0, sm 0, pool 0, rest 42125 us`). This entry adds a per-*op* table instead of a
per-kernel-family one: `CONFIG_KWS_INFER_PROFILE` (`firmware/main/Kconfig.projbuild`) wraps
every esp-nn call `kws-codegen` emits in `gen/{wake,command}_infer.c` with a cycle-count timer
that accumulates into a per-op `{cycles, calls}` array, plus a `<name>_infer_profile_dump()`
that logs one line per op — index, type, output shape, static MACs (from the same planner
`kws-model-graph` reads), measured microseconds and MAC/us, averaged per call since the last
reset — then a total. The guard is `#if defined(CONFIG_KWS_INFER_PROFILE)`, so it is always
*emitted* (the committed `gen/` files changed once, by regeneration) and only ever *compiled*
in when the Kconfig flag is on; `kws-codegen --check` and the flag-off host suite
(`firmware/test`, 0 differing bytes on both smoke tests) are unaffected. `recognise.cc`/
`wake.cc` dump both models' tables automatically every ~50/~100 steps next to the existing
`NN_TIMERS` trace, printing a *residual* line (`invoke time − profiled total`, both sides
averaged over the same window) for whatever the table does not see; a new console command,
`profile`, dumps on demand. A host-only `test_profile` target (`firmware/test/Makefile`)
builds both generated files a second time with `-DCONFIG_KWS_INFER_PROFILE=1` and asserts
every op's call count equals the step count and resets to 0 after a dump — it pins the
wiring, not the numbers (its "microseconds" are a host `clock_gettime()` count, not the
device's cycles).

**Device tables**, `mode recognise` / `mode wake` + the `profile` console command, PSRAM
command arena (the shipped default), steady state:

```text
profile command op0   conv     10x49x48  macs=  211680  us=15118  calls=50  mac_per_us= 14
profile command op1   dw       10x49x48  macs=  211680  us= 4624  calls=50  mac_per_us= 45
profile command op2   conv     10x49x48  macs= 1128960  us= 4434  calls=50  mac_per_us=254
profile command op3   dw       10x49x48  macs=  211680  us= 4224  calls=50  mac_per_us= 50
profile command op4   conv     10x49x48  macs= 1128960  us= 4303  calls=50  mac_per_us=262
profile command op5   dw       10x49x48  macs=  211680  us= 4219  calls=50  mac_per_us= 50
profile command op6   conv     10x49x48  macs= 1128960  us= 4392  calls=50  mac_per_us=257
profile command op7   mean      1x 1x48  macs=   23520  us=  769  calls=50  mac_per_us= 30
profile command op8   fc        1x 1x23  macs=    1104  us=   99  calls=50  mac_per_us= 11
profile command op9   softmax   1x 1x23  macs=       0  us=  135  calls=50  mac_per_us=  0
profile command total us=42317 (per call, summed over 10 ops)
profile command residual 38 us/step (invoke avg 42355, profiled 42317)

profile wake   op14  conv      1x 1x32  macs=   6400  us= 558  calls=67  mac_per_us=11
profile wake   op17  dw        1x 1x32  macs=    160  us=  95  calls=67  mac_per_us= 1
profile wake   op18  conv      1x 1x64  macs=   2048  us=  75  calls=67  mac_per_us=27
profile wake   op23  dw        1x 1x64  macs=    576  us=  34  calls=67  mac_per_us=17
profile wake   op24  conv      1x 1x64  macs=   4096  us=  64  calls=67  mac_per_us=64
profile wake   op29  dw        1x 1x64  macs=    832  us=  50  calls=67  mac_per_us=17
profile wake   op30  conv      1x 1x64  macs=   4096  us=  67  calls=67  mac_per_us=61
profile wake   op32  dw        1x 1x64  macs=   1344  us=  58  calls=67  mac_per_us=23
profile wake   op33  conv      1x 1x64  macs=   4096  us=  67  calls=67  mac_per_us=61
profile wake   op36  fc        1x 1x 1  macs=   1088  us=  47  calls=67  mac_per_us=23
profile wake   op37  logistic  1x 1x 1  macs=      0  us=   1  calls=67  mac_per_us= 0
profile wake   op44  quantize  1x 1x 1  macs=      0  us=   0  calls=67  mac_per_us= 0
profile wake total us=1115 (per call, summed over 12 ops)
profile wake residual 236 us/step (invoke avg 1351, profiled 1115)
```

Command's residual is 0–2 % of its invoke (38–1,725 us measured across several dumps, against
a ~42,300–42,370 us average) — the table accounts for essentially all of it. Wake's residual
is 17–18 % (230–236 us of 1,309–1,351 us) — small in absolute terms but proportionally the
opposite of command's, because wake's per-op payloads are a few hundred to a few thousand
MACs each, so the fixed per-call cost (the `data_dims_t`/params setup between kernel calls,
the esp-nn scratch-pointer re-pointing, the ring memmove/memcpy bookkeeping) is a much larger
share of a much smaller total.

**The internal-SRAM command arena does not boot with the current (w48) model, on main,
independent of this change.** Building `KWS_INFER_COMMAND_ARENA_INTERNAL` — with
`CONFIG_KWS_INFER_PROFILE` on *and* off — crash-loops before any mode task runs:
`ESP_ERR_NO_MEM at ... console.c:186 ... usb_serial_jtag_driver_install`, logged free
internal RAM 807 B (largest block 768 B) at the point of failure. The Kconfig help text for
this choice already named the risk ("if tasks ever do stop being created, this is the first
thing to look at") from a measurement taken against a smaller arena than the deployed w48 one
(E18: internal placement now costs 15 KB more than that E16-era measurement did) — this
confirms the margin is gone, not a regression from profiling's few hundred bytes of `.bss`.
No command-model table exists for the internal placement as a result: the device cannot
reach `mode recognise` in that configuration to produce one. Filed as a concern below rather
than worked around here (out of scope for a profiling change to also fix arena sizing).

**Three levers the tables support**, ranked by expected payoff:

1. **Command op0 (the first conv, in_c=1, on the raw 10×49 MFCC window) is 5–20x less
   efficient than later same-model convs with far more MACs**: 14 MAC/us against 254–262
   MAC/us for op2/op4/op6, despite all four sharing the same 48-output-channel width. op0 is
   ~35 % of command's total invoke time on <5 % of its MACs. A single input channel cannot
   fill esp-nn's channel-packed vector path the way in_c=48 does two layers later — a stride
   change on this layer, or restructuring it (e.g. a wider first stride, or folding it into
   the depthwise that follows) is the highest-payoff lever in either table.
2. **Wake's residual (17–18 % of its invoke) is well above command's (0–2 %)** — the
   dispatch/requantisation/bookkeeping overhead the table does not see is proportionally the
   dominant cost on the wake path, because wake's per-op MAC counts are tiny (160–6,400)
   relative to the fixed per-call setup cost. Cutting call-site overhead (fusing adjacent
   small conv/dw pairs, or moving the ring memmove out of the hot path) would move wake's
   needle more than any single kernel optimisation would.
3. **Depthwise layers sit well below their paired pointwise convs' MAC/us on both models**
   (wake dw: 1–23 MAC/us vs its conv neighbours' 27–64; command dw: 45–50 vs its conv
   neighbours' 254–262) — expected in part (no channel-reduction MACs to amortise memory
   movement against) but the size of the gap suggests esp-nn's depthwise kernel is not always
   on its fastest branch for these channel widths (48 for command, 32/64 for wake). Checking
   channel counts against esp-nn's `%16==0` + 3×3 fast-path condition (`_depthwise_scratch` in
   `kws_de/codegen.py` already documents the branch) and padding to the next boundary where
   it misses is the second-highest-payoff lever.

Not implemented here — this entry is measurement only. `docs/sphinx/firmware.rst` gets a
"Profiling" subsection (how to enable/read it, what the columns mean); the flag stays off by
default and the device was rebuilt and reflashed to the default (flag-off) configuration
before ending the session (Assistent mode, `field on`, `field thresh 0.85`).

### E33 — training on the deployed model's own real false fires (wake, 2026-09-06, host-only)

E22/E24 shipped round 6d after recommending a device soak with field capture armed at the
loose gate (E21). The soak delivered exactly what it was for: one real conversation session
(spk19+spk20, nobody addressing the device) made the deployed model false-fire **16 times**
— 10 on ordinary German speech, 5 on near-silence, 1 clipped — plus 25 new real "Hey Bus"
positives. Every prior negative in this recipe was TTS or real speech cut deliberately after
a spoken "Hey Bus" (round 6's rule 3); this is the first round trained on real false fires
from a session where the device was never addressed.

Each false-fire subgroup is split 50/50 by clip (a single session cannot be session-disjoint
against itself): the train halves become two new feature dirs added *on top of* round 6d's
negative weights (weight 4.0 for real false-fire speech, 2.0 for near-silence — first
reasonable values, not a search), and the held-out halves — never trained on — become the
round's two new acceptance rows. New real positives are split 12 train / 13 held out and
folded into the same trimmed-positive dir at the same 5.0 weight as round 6d's ten, keeping
the 71.4 % real share unchanged; 3 of the 12 train-side clips trimmed to nearly their full
length instead of isolating "Hey Bus" and were dropped rather than risk the round-5/6c
label-alignment bug on clips longer than `clip_duration_ms`.

| gate | round 6d (installed) | round 7 |
|---|---|---|
| held-out session 0849 fires (3) | 3/3 @ 0.996 | 3/3 @ 0.996 |
| new real positives, held out (13) | 12/13 | **13/13** |
| held-out real non-wake (9) / in-training (5) | 0/9 (0.402) / 0/5 (0.598) | 0/9 (**0.285**) / 0/5 (**0.285**) |
| room noise, in-training (40) / held out (9) | 0/40 / 0/9 | 0/40 / 0/9 |
| **real false-fire speech, held out (5)** | n/a — this round's target | 5/5 (deployed) → **1/5** |
| **near-silence fires, held out (3)** | n/a — this round's target | 3/3 (deployed) → **1/3** |
| TTS non-wake, seen (46) / unseen (36) | 4/46 / 2/36 | 5/46 / 4/36 |
| fire latency, held-out / in-training / new-held (median) | 0.14 / 0.23 / 0.35 s | **0.02 / −0.07 / 0.04 s** |
| size / MACs | 58,080 B / 24,736 | 58,080 B / 24,736 |

**It works, at the cost the round expected.** The two target rows are the point: real
false fires that hit the deployed model 5/5 and 3/3 on their now-held-out halves — clips
round 7 never saw in any form during training — drop to 1/5 and 1/3, with recall not just
retained but improved (round 7 fixes round 6d's one held-out miss). Every real-audio
non-wake row ties or improves, and latency improves by 0.1-0.3 s across every positive set.
The residual near-silence fire is the *clipped* clip, not either near-silence one, and its
peak is bit-for-bit identical (0.996) before and after — clipping looks like a different
failure mode this recipe may not reach at all.

The cost lands entirely on the synthetic TTS gate: 1 more seen-voice and 2 more
unseen-voice fires out of 46/36 fixed clips. It is not a superset of round 6d's failure —
round 6d's entire residue was `de_DE-mls-medium`, already flagged (E22 follow-up 2) as a
poor short-phrase synthesiser, and round 7 fires on it **zero** times; the new fires are
"der bus kommt gleich" spread across three different, better-behaved voices instead. This
is the same real-audio-vs-TTS-margin trade round 5's user decision already made once
deliberately, now made again by a small amount on the far side of it.

The internal mWW ambient metric, which has disagreed with the real-audio gates in both
directions in E20 and E22, moves *with* everything else this time: faph at the 0.85 cutoff
drops from 2.062 to 0.187. Treated as supporting context, not a decision — the pattern
established in E20/E22 is that it does not reliably track what matters here.

ETA ledger note: `KWS_DATA_ROOT` was not exported for either `kws-eta` call this round, so
both landed in the repo-local fallback ledger rather than the shared one — predicted 16.1
min (4 runs), actual 17.36 min (+7.8%, just past the top of the range). Full report in the
training directory as `HEYBUS-R7-REPORT.md`.

**Recommendation: deploy `hey_bus_r7.tflite`.** It is the first model trained specifically
against the deployed model's own real-environment false fires and it cuts them sharply with
zero recall loss and improved latency, at a small, honestly-reported TTS-gate cost that
lands on voices the project would trade for real-device accuracy regardless. Nothing was
promoted: `hey_bus.tflite` and `firmware/main/gen/` are untouched, no firmware was built or
flashed, and no audio was played to any device or speaker in this round.

### E34 — deploying round 7 (2026-09-06, host-only)

Round 7 (E33, `HEYBUS-R7-REPORT.md` in the training directory) trains on the deployed
round 6d model's own real-environment false fires — the first evidence a device soak
(E21/E24) has produced — and recommends deploy. This entry does the deploy, host-side
only — no device work, no audio played to any device or speaker.

| | round 6d (was deployed) | round 7 (deployed) |
|---|---|---|
| `KWS_WAKE_MODEL_ID` | `hey_bus.tflite@5fcdaf63 2026-09-04` | `hey_bus.tflite@4aaa2f98 2026-09-06` |
| size / MACs | 58,080 B / 24,736 | 58,080 B / 24,736 (unchanged) |
| held-out session fires (3) | 3/3 @ 0.996 | 3/3 @ 0.996 |
| new real positives, held out (13) | 12/13 | **13/13** |
| held-out / in-training real non-wake, worst peak | 0/9 (0.402) / 0/5 (0.598) | 0/9 (**0.285**) / 0/5 (**0.285**) |
| real false-fire speech, held out | 5/5 @ 0.992 | **1/5** @ 0.969 |
| near-silence fires, held out | 3/3 @ 0.996 | **1/3** @ 0.996 (residual: a clipped clip) |
| TTS non-wake, seen (46) / unseen (36) | 4/46 / 2/36 | 5/46 / 4/36 (small, honest regression) |
| fire latency, held-out (median) | 0.14 s | **0.02 s** |

Full acceptance table and recipe: `HEYBUS-R7-REPORT.md` (training directory);
recommendation there is unconditional ("Deploy round 7").

Same procedure as E24's round-6d deploy. Since PR #48 the firmware embeds generated
esp-nn C rather than the TFLM interpreter, so "deploy" means regenerating
`gen/wake_model_{data,config}.h` from the new `.tflite` and `gen/wake_infer.{c,h}` +
`gen/wake_smoke_vectors.h` from that, not touching the command model. As E15/E18/E24
warn, `kws-export --firmware` unconditionally rewrites the wake pair from the fixed
`models/hey_bus.tflite` path alongside a full command re-export, so this deploy called
`kws_de.export.write_wake_headers` directly instead — after promoting the candidate to
the canonical filename (round 6d's bytes kept on disk as `hey_bus.v6d.tflite`,
following the `hey_bus.v1.tflite` / `hey_bus.v4.tflite` / `hey_bus.v5.tflite`
precedent) — then `kws-codegen --name wake`.

**Command model untouched.** `gen/model_data.h`, `gen/model_config.h`,
`gen/command_infer.c`, `gen/command_infer.h` are sha256-identical before and after
this deploy.

**Architecture unchanged, confirmed in `gen/wake_infer.h`:**

| | round 6d | round 7 |
|---|---|---|
| `WAKE_INFER_ARENA_BYTES` | 128 B | 128 B |
| `WAKE_INFER_STATE_BYTES` | 4,200 B | 4,200 B |
| `WAKE_INFER_SCRATCH_BYTES` | 15,552 B | 15,552 B |

**Host checks, all clean.** `kws-fwgen --check firmware/main/gen` and both
`kws-codegen --check` runs (wake and command) exit 0 (the pre-existing
`command_v3_w48_qat.tflite` re-export warning from `kws-fwgen --check` is unrelated to
this deploy and predates it). `make -C firmware/test`: `wake smoke: 0/64 steps
differ`, `command smoke: 0/368 bytes differ`, `host tests OK`. `pytest -q`: 367
passed, 1 skipped, 1 xfailed — includes
`test_whole_wake_model_matches_the_interpreter_on_every_step` (`wake parity: 0/635
steps differ (11 clips, 4200 B state)`) and the wake-op layer parity cases. `ruff
check` / `ruff format --check`: clean, 104 files formatted. `markdownlint-cli@0.42.0
--config .markdownlint.json`: clean. `sphinx-build -W --keep-going`: clean apart from
the local "doxygen XML absent" warning (the same known-environment gap E15/E18/E24
noted). ESP-IDF v5.5.5 Docker build (default config): OK, app image **1,025,952 B**
(`0xfa7a0`, 67% of the 0x300000 partition free; E24's 1,020,048 B included none of
E25-E32's intervening firmware work, so the +5,904 B is unrelated to this deploy — the
wake weights are the only `gen/` bytes that moved, confirmed above).

**Device (2026-09-06 19:20, console only).** Flashed on the CoreS3: `status` reads
`wake=hey_bus.tflite@4aaa2f98 2026-09-06`; `mode wake` trace `step 1277 +/- 154 us
(invoke 1165 us)` over 67 steps per 2 s window — the round-6d baseline (E24: 1,256 µs);
room-noise peaks 0.203 and 0.004 over the two windows (round 6d idled at 0.176). Left in
Assistent mode with field capture armed at 0.85 for the soak. Still open: a spoken
"Hey Bus" check and the next real session's false-fire count, which is the number E33's
1/5 predicts should fall from 16 per session.

### E35 — field-take retrain (run 3): today's spk18 session plus the grid's winning recipe (2026-09-06, host-only, feat/run3-field-takes)

A real session today (`qc/2026-09-06-1404`) added 36 word clips, 22 phrases, 25 wake
clips and 10 negatives to `approved/spk18` — natural commands ("Hey Bus, Licht
dunkler", etc.), not read sentences. `kws-dataset build` merges `approved/` on every
build, so a plain rebuild picks these up. In parallel, a sibling branch
(`exp/recipe-grid`, PR #80, **E36**) ran an 8-way width x `--real-weight` x
`--qat-epochs` grid on the *pre-rebuild* `features_v3` and found `--width 48 --qat
--qat-epochs 20 --real-weight 3` clears the standing deploy rule by a wide margin
(aggregate 0.9185, spk18 0.778, 0/32 false accepts) at identical MACs/bytes to the
shipped w48. This entry rebuilds the dataset with today's field takes included, then
trains *both* the plain default recipe and the grid's winning recipe against that
rebuild, to see whether the grid's win survives real new data.

**Correction caught mid-session.** The first rebuild omitted `KWS_NOISE_DIR`/
`KWS_RIR_DIR` (van-cabin augmentation, on since E31) — the resulting `train=32,647` was
suspiciously close to `35,261 - 2,536` (E31's augmentation row delta), confirming the
omission before any training ran on it. Redone with both env vars set
(`<mww-train>/data/fma_16k`, 210 files; `<mww-train>/data/mit_rirs`, 270 files, same
paths E31 used) and the mismatched run discarded unreported.

**Rebuild.** `kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3` (E31's
voice-gated TTS cache, unchanged). `[recordings] merged: {'Aufstelldach': 6, 'Außen':
22, 'Heizung': 12, 'Küche': 1, 'Kühlschrank': 17, 'Lesen': 12, 'Licht': 76, 'an': 26,
'auf': 3, 'aus': 19, 'dunkler': 11, 'fünfundsiebzig': 8, 'fünfundzwanzig': 8, 'fünfzig':
11, 'heller': 9, 'hundert': 13, 'kälter': 5, 'leise': 4, 'wärmer': 2, 'zu': 4,
'_unknown_': 104}`; 285 device clips moved into `train` (`--recordings-split train`,
default). `[dataset] built seed=0: train=38,646, val=6,417, test=11,291` — train grew
over E31's 35,261 by +3,385, consistent with today's added real clips each drawing the
same van-augmentation row multiplier.

**Training + export.** Both recipes trained on the same rebuilt `features_v3`:
`command_v3_w48_qat_run3.keras` (plain `kws-train --v2 --prefix features_v3 --width 48
--qat`) and `command_v3_w48_qat_run3_rw3qe20.keras` (`--qat-epochs 20 --real-weight 3`,
after merging `origin/exp/recipe-grid` into this branch to pick up the `--real-weight`
flag). Exported each into its own `models/run3-default/` / `models/run3-rw3qe20/`
directory (`kws-export --v2 --qat --width 48 --prefix features_v3`) rather than the
canonical path, so `models/command_v3_w48_qat.tflite` and `firmware/main/gen/` were
never touched — confirmed by hash before and after (`f277889b...` unchanged). The
pre-session canonical files were also backed up as `*.pre-run3` before any export ran,
belt-and-braces.

**Real-voice comparison**, `scripts/compare_command_models.py`, deployed vs. both run-3
candidates, same real-voice approved set (spk01/spk02/spk10/spk18, 233 words / 32
negatives) used by every prior entry:

| | deployed w48 | run-3 default | run-3 `rw3qe20` |
|---|---|---|---|
| bytes / sha256 | 25,832 / `8fa81d08` | 25,832 / `98c0263a` | 25,832 / `86b7105e` |
| INT8 test acc (own-era test set, not cross-comparable) | 61.78% (n=11,291) | 68.82% (n=11,291) | 69.36% (n=11,291) |
| spk01 words (n=13, in-training) | 0.923 | 0.692 | **1.000** |
| spk02 words (n=38, in-training) | 0.895 | 0.816 | **1.000** |
| spk10 words (n=146, held-out) | 0.856 | 0.842 | **0.952** |
| spk18 words (n=36, held-out\*) | 0.333 | 0.694 | **0.778** |
| **aggregate words (n=233)** | **0.785** | 0.807 | **0.936** |
| false accepts, spk02/spk10/spk18 | 0/10, 0/19, 0/3 | 0/10, **1/19**, 0/3 | 0/10, 0/19, 0/3 |
| **false accepts total (n=32)** | **0/32** | **1/32** | **0/32** |
| phrase intent (spk18/spk19/spk20 now nonzero) | 0, 0.111, 0 | 0, 0, 0 | 0.176, 0.222, 0.083 |

**\*spk18 is no longer a clean held-out figure.** `compare_command_models.py`'s default
manifest (`data/manifest_v3_qat.json`) is dated 2026-09-02 — it was never regenerated
by today's rebuild — so it still labels spk18 as `held-out` and the isolated-word count
stays at the original n=36 (today's new spk18 clips aren't indexed by this stale
manifest and so don't enter this specific word-accuracy denominator). But
`--recordings-split train` (the dataset build's default) put today's new spk18 clips
into the actual training rows regardless of what this manifest says, and both run-3
models trained on that rebuild. So spk18's jump (0.333 → 0.694 / 0.778) is now
partly-to-mostly an **in-training** result — the same speaker's voice, and likely some
of the same words, informed the model that is then scored against spk18 — not a
held-out generalisation result. Treat it as encouraging but not evidence the model
generalises to *new* unseen speakers; spk10, spk19 and spk20 (still genuinely
untouched by today's clips) are the honest held-out signal, and `rw3qe20` moves all
three up too (spk10 0.856→0.952, spk19 0.250→0.688, spk20 0.200→0.800).

**Decision, same pre-agreed rule** (aggregate ≥ deployed's 0.785, false accepts no
worse than 0/32, spk18 words > deployed's 0.333): run-3 default clears spk18 and the
aggregate bar but **introduces a new false accept** (spk10 1/19) — worse than
deployed's 0/32 — so it fails the rule outright despite the higher aggregate. **Run-3
`rw3qe20` passes all three conditions** (aggregate 0.936, 0/32 false accepts, spk18
0.778) **and has the highest aggregate of any candidate to date** — the grid's win
(E36) survives the real field-take rebuild. **This is the deploy candidate**, matching
the recipe change E36 already flagged (no architecture change, same MACs/bytes as the
shipped w48). Per the task brief for this session, deployment/flashing was explicitly
left to the coordinator: `models/command_v3_w48_qat.tflite` and `firmware/main/gen/`
remain untouched.

**Field / Elicit** (`kws-eval --recordings --prefix features_v3 --qat`, both
sections are device-Whisper agreement at *capture time* — i.e. properties of whatever
model was actually deployed when each take was recorded, not of the models compared
above, so they read identically regardless of which candidate is loaded):

Field: 125 field takes, 68 approved, 36 parsable (0.288). Against the production gate
0.85: 0 near-misses, 18 false alarms; at the capture gate 0.60: 0 near-misses, 18 false
alarms. By speaker: spk17 49 takes/0 approved; spk18 39/37 approved, 18 parsable,
agreement 0.125, 8 false alarms; spk19 24/19 approved, 9 parsable, agreement 0.500, 10
false alarms; spk20 13/12 approved, 9 parsable, agreement 0.500, 0 false alarms.

Elicit: 4 elicit takes, 4 approved, 4 parsable (1.000), 1 unparsed (vocab present);
said what the scene expected 0.667 of the 3 compared takes (spk20).

**Reading.** Two independent host-only threads converged on the same answer this round:
E36's hyperparameter search found a recipe that clears the deploy rule on old data, and
this entry confirms it still clears the rule — by an even wider margin — once real
field takes are folded in via a from-scratch rebuild. The plain-recipe retrain (no
`--real-weight`) is a useful negative control: it shows the aggregate lift is not just
"more real data helps a little," it needed the real-clip upweighting to avoid a new
false accept. `spk10`/`spk19`/`spk20` moving together under `rw3qe20` (all three
genuinely untouched by today's session) is the more convincing signal than spk18 alone.
Two open items for whoever deploys: (1) refresh `data/manifest_v3_qat.json` so future
`compare_command_models.py`/`kws-eval --qat` runs correctly re-classify spk18 as
in-training instead of quietly under-counting its held-out figure; (2) the E36 grid
never rebuilt data, so its own `w48_rw3_qe20` CSV row still reflects the pre-field-take
`features_v3` — this entry's `rw3qe20` numbers, not the grid's, are the ones that
reflect today's data. Device flashing/measurement was out of scope for this host-only
session and was not performed.

### E36 — hyper-recipe grid: width x real-clip weight x QAT epochs (2026-09-06, host-only)

E27/E28/E31 each tried a *dataset*-side fix (regenerating the TTS cache, then a voice-level
gate, then fixing the gate's own false negative + van augmentation) on top of the *same*
training recipe (`--width 48 --qat --qat-epochs 10`, epochs=40) and each time cleared the
false-accept and spk18 bars but missed the aggregate-words bar by a few points (E27
strict/lenient did clear it, but regressed false accepts instead). This entry holds the
E31 dataset fixed — `features_v3`, train=35,261, the voice-gated + van-augmented build —
and sweeps the *recipe* instead: width, a new real-clip upweighting knob, and QAT epoch
count, 8 runs total, asking whether a recipe change alone (no further data work) can
clear the deploy bar.

**New: `kws-train --real-weight N`.** No existing knob controlled the real-vs-TTS mix
seen during training (`class_weight` balances *label* frequency, not *origin*). The
feature-cache npz keeps only an `is_tts` row flag, not the original `rec:`/MSWC speaker
id, so `kws_de.train.upweight_real(X, y, is_tts, weight)` repeats every `~is_tts` row
(device recordings *and* MSWC — the same "real" population `kws_de.eval`'s
`headline_mask` already uses) `weight`-1 extra times before `class_weight` is computed;
`weight<=1` is a no-op. Wired through `--real-weight` (default 1) in both the float and
QAT-fine-tune calls. Three new tests in `tests/test_train.py` (no-op cases, real-only
repetition, no-real-rows no-op).

**Grid.** `scripts/recipe-grid.py` (committed, resumable — skips any run whose id already
has a row in `scripts/recipe-grid.csv`): width in {32, 48} x `--real-weight` in {1, 3} x
`--qat-epochs` in {10, 20}, float epochs fixed at 40 (the established recipe, E16/E28),
seed 0, all 8 runs on the identical `features_v3` train/val/test split — the only
variables are the three recipe knobs. Each candidate trains + exports into its own
`models/grid/<run>/` directory; `command_v3_w48_qat.*` and `firmware/main/gen/` are
never written by this script. Scoring reuses `scripts/compare_command_models.py`'s code
path (`eval_recordings` + `make_command_predict_fn`) over the same fixed real-voice
scoreboard prior entries used — **233 words / 32 negatives, speakers spk01/spk02/spk10/
spk18** — deliberately excluding two speakers (`spk19`, `spk20`) added to `approved/`
since E28/E31, for exact comparability with the historical table. `deployed_w48`
(`firmware/main/gen/model_data.h`, `8fa81d08`) and `run2_w48` (E31's candidate,
`d49093a4`, the currently-checked-out `models/command_v3_w48_qat.tflite`) are included
as reference rows, rescored the same way for a consistent baseline (their INT8 test
accuracy below is against the *current* `features_v3_test.npz`, not their own-era split
— not cross-comparable to earlier tables' accuracy column, same caveat E27/E28 noted).
`w48_rw1_qe10` reproduces `run2_w48` bit-for-bit (identical sha256 `d49093a4`) — training
is deterministic given the same seed/recipe/data, a useful sanity check that the grid
driver's invocation matches E31's manual one exactly.

Deploy rule, unchanged since E27 (aggregate words >= deployed's 0.785, false accepts no
worse than 0/32, spk18 words > deployed's 0.333), applied as two hard filters (false
accepts, spk18) plus the aggregate as the ranking objective. Sorted, passing rows first:

| run | width | real_weight | qat_epochs | aggregate words (n=233) | spk01 (n=13) | spk02 (n=38) | spk10 (n=146) | spk18 (n=36) | false accepts (n=32) | MACs | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **w48_rw3_qe20** | 48 | 3 | 20 | **0.9185** | 1.000 | 0.974 | 0.932 | 0.778 | **0/32** | 4,234,704 | 25,832 |
| w32_rw3_qe10 | 32 | 3 | 10 | 0.7897 | 0.538 | 0.921 | 0.815 | 0.639 | 0/32 | 2,070,496 | 17,912 |
| run2_w48 (E31) | 48 | 1 (n/a) | 10 (n/a) | 0.7768 | 0.769 | 0.684 | 0.829 | 0.667 | 0/32 | 4,234,704 | 25,832 |
| w48_rw1_qe10 | 48 | 1 | 10 | 0.7768 | 0.769 | 0.684 | 0.829 | 0.667 | 0/32 | 4,234,704 | 25,832 |
| w48_rw3_qe10 | 48 | 3 | 10 | 0.9227 | 1.000 | 0.974 | 0.932 | 0.806 | 1/32 | 4,234,704 | 25,832 |
| w32_rw3_qe20 | 32 | 3 | 20 | 0.8026 | 0.692 | 0.921 | 0.822 | 0.639 | 1/32 | 2,070,496 | 17,912 |
| deployed_w48 | 48 | n/a | n/a | 0.7854 | 0.923 | 0.895 | 0.856 | 0.333 | 0/32 | 4,234,704 | 25,832 |
| w48_rw1_qe20 | 48 | 1 | 20 | 0.7854 | 0.692 | 0.763 | 0.842 | 0.611 | 1/32 | 4,234,704 | 25,832 |
| w32_rw1_qe20 | 32 | 1 | 20 | 0.6137 | 0.462 | 0.632 | 0.651 | 0.500 | 2/32 | 2,070,496 | 17,912 |
| w32_rw1_qe10 | 32 | 1 | 10 | 0.6052 | 0.385 | 0.711 | 0.630 | 0.472 | 1/32 | 2,070,496 | 17,912 |

(`deployed_w48`'s spk18 is exactly the 0.333 baseline, not `>` it, so it fails its own
gate by definition — listed as the reference row it is, not a candidate. Rows below it
in the table fail on false accepts.)

**Real-clip weighting is the dominant lever, not width.** Every `real_weight=3` row beats
its `real_weight=1` counterpart at the same width by 0.15-0.32 aggregate points; width
alone (E16's finding) still matters but far less over this range — w48 beats w32 by
~0.1-0.15 at matched `real_weight`/`qat_epochs`. `w48_rw3_qe10` has the single highest
aggregate in the grid (0.9227, spk01 goes to a clean 13/13) but is disqualified by
exactly one false accept — `eval_recordings` traces it to `spk10` (1/19, 0.0526); the
otherwise-identical `qe20` run (10 more QAT fine-tune epochs, same float weights, same
data) removes that one false accept at a cost of only 0.0042 aggregate and 0.028 spk18 —
QAT epoch count reads here as a small false-accept/aggregate trade knob layered on top of
`real_weight`'s much bigger effect, not an independent lever. Both `w48_rw3` rows come at
**zero extra device cost** over the currently deployed w48 — same 4,234,704 MACs, same
25,832 B — this is a training-recipe change only, no architecture change.

**ETA ledger caveat.** `train_seconds_actual` (full `kws_de.train` subprocess wall time —
float train + QAT reload/fine-tune/save) is not the same quantity `kws-eta predict train`
estimates (the float-training phase alone, per `Timed("train", ...)` inside
`kws_de.train`); the ratios range 0.75x-2.9x with no clean pattern by width or
real_weight, consistent with E16/E28's standing note that the ledger's `size` (epochs x
rows) carries no width term and mixes recipes across its history. Not fixed here — a
predictor covering the QAT phase and width jointly is future work, not blocking this
grid.

**Recommendation: `w48_rw3_qe20`** (`--width 48 --qat --qat-epochs 20 --real-weight 3`,
epochs=40) is the only run that clears all three deploy-rule conditions with a
comfortable margin — aggregate 0.9185 against the 0.785 bar, spk18 0.778 against 0.333,
0/32 false accepts tied with deployed — at identical MACs/bytes to the currently deployed
model. **Not promoted**: this was a host-only recipe search per the task brief;
`firmware/main/gen/` and `models/command_v3_w48_qat.tflite` are untouched, and a device
measurement (the recognise-step latency is architecture-bound, so E16/E18's ~55.8 ms
estimate should still hold, but this was not verified on hardware) is the natural
follow-up before any deploy decision.

### E37 — deploying run-3 `rw3qe20`: field-take retrain, grid recipe (2026-09-06/07, host-only)

E35's deploy candidate promoted. `rw3qe20` (`--width 48 --qat --qat-epochs 20
--real-weight 3` on E35's field-take rebuild of `features_v3`, train=38,646) cleared
the standing deploy rule by the widest margin yet (aggregate 0.936, spk18 0.778, 0/32
false accepts) and E36 independently found the same recipe on the pre-rebuild data —
two host-only threads agreeing is what E35 flagged as the deploy candidate. This entry
does the deploy, following #54/E18's procedure: same architecture, so no docs sweep
beyond the identifying stamp and one comparison table.

**Export.** `kws-export --firmware --qat --prefix features_v3 --model
command_v3_w48_qat_run3_rw3qe20.keras --width 48`, loading the QAT SavedModel
(`command_v3_w48_qat_run3_rw3qe20_qat/`) E35 trained. `--out` was left at its default
(`$KWS_DATA_ROOT/models`), which is the canonical path this time — E35 deliberately
exported to an isolated `models/run3-rw3qe20/` directory to avoid touching canonical
files during a host-only recipe search; this entry's job is exactly to make that swap.
The pre-export canonical `command_v3_w48_qat.tflite` / `_data.h` / `_metadata.json` and
the `command_v3_w48_qat/` SavedModel directory were backed up as `*.pre-run3rw3` /
`.pre-run3rw3-dir` before the export ran. `kws-codegen firmware/main/gen/model_data.h
--name command --out firmware/main/gen` regenerated `command_infer.{c,h}` and
`command_smoke_vectors.h` from the new weights.

| | old (E35 table's "deployed w48") | new (deployed) |
|---|---|---|
| `KWS_MODEL_ID` | `command_v3_w48_qat.tflite@8fa81d08 2026-09-04` | `command_v3_w48_qat.tflite@86b7105e 2026-09-06` |
| size | 25,832 B | 25,832 B (unchanged) |
| width / params / MACs | 48 / 11,111 / 4,234,704 (unchanged) | 48 / 11,111 / 4,234,704 (unchanged) |
| INT8 test accuracy (own-era test set) | 61.78 % (n=11,291) | 69.36 % (n=11,291) |
| aggregate real-voice words (n=233) | 0.785 | **0.936** |
| false accepts (n=32) | 0/32 | 0/32 |

No architecture change — same MACs, same bytes, same quantisation shape — so
`recognise.cc`'s `static_assert`s and the arena/scratch macros are untouched by
construction, confirmed rather than assumed: `COMMAND_INFER_ARENA_BYTES` 47,040 B,
`COMMAND_INFER_SCRATCH_BYTES` 29,824 B, both identical to E18. The wake pair
(`hey_bus.tflite@4aaa2f98`, round 7, E34) is untouched — `kws-export --firmware`
rewrites the wake headers unconditionally, so this was verified by sha256 rather than
assumed: `wake_model_data.h` / `wake_model_config.h` byte-identical before and after.

**Checks.** `kws-fwgen --check firmware/main/gen` and `kws-codegen
firmware/main/gen/model_data.h --name command --check firmware/main/gen`: exit 0, no
drift warning. `make -C firmware/test`: `command smoke: 0/368 bytes differ`, `wake
smoke: 0/64 steps differ`, host tests OK. `pytest -q`'s
`test_whole_command_model_matches_the_interpreter` (which builds and runs
`test_command_parity`): `command parity: 0/1564 bytes differ` (68 real `features_v3`
clips + 4 synthetic, arena 47,040 B) — same clip count and byte total as E18, since
the model shape is unchanged. Full suite: 370 passed, 1 skipped, 1 xfailed. `ruff
check` / `ruff format --check`: clean. `markdownlint-cli@0.42.0 --config
.markdownlint.json`: clean. `sphinx-build -W --keep-going` on `docs/sphinx`: clean but
for the pre-existing "doxygen XML absent" warning (no `doxygen` binary on this
host — the same gap `docs.yml`'s CI job papers over by installing it first; not
introduced by this deploy). ESP-IDF v5.5.5 Docker, default build: total image
1,025,659 B (`.bss` 140,024 B DIRAM), unchanged architecture so no memory-shape
surprise expected and none found.

**Promotion.** Canonical `command_v3_w48_qat.{tflite,_data.h,_metadata.json}` under
`$KWS_DATA_ROOT/models` now hold the `rw3qe20` export (the `kws-export --firmware`
call above wrote them directly, since `--out` defaults to the canonical models dir);
the pre-deploy versions live alongside as `*.pre-run3rw3`. `data/manifest_v3_qat.json`
— flagged stale in E35 (dated before the field-take rebuild, so it still labelled
spk18 `held-out`) — refreshed from the current `data/manifest_v3.json` (the E35
rebuild's manifest, `built_at` 2026-09-06T18:16:55Z, train=38,646); the pre-refresh
file kept as `manifest_v3_qat.json.pre-run3rw3`. Future `compare_command_models.py` /
`kws-eval --qat` runs will now classify spk18 the way E35's prose already explained it
should be read.

**Device (2026-09-07 23:30, flashed from main after PR #82/#83/#84/#85).** Docker
default build `kws_de_fw.bin` 0xfa6f0 B; boot log `models: command
command_v3_w48_qat.tflite@86b7105e 2026-09-06, wake hey_bus.tflite@4aaa2f98`; recogniser
`47040 B arena (static, PSRAM) + 29824 B shared scratch, esp-nn scratch 29824 B queried /
29824 B reserved`, **free internal 38,055 B** at recogniser start (wake: 60,107 B) — down from
E18's 45,431 B because of the intent card, grammar rescoring and burn-in code that landed
between (#69/#72/#76), not because of this model. Continuous `mode recognise`: **step 46 /
47 ms** (`invoke 42274 us` / `42383 us`, front-end ≈ 0.49 ms over 6–7 new frames, stack
3,736 B free), duty `1000/1000 of wall, 320 ms per wall second` — identical to E18's
measurement of `8fa81d08`, as the unchanged architecture predicted. `profile` is
`CONFIG_KWS_INFER_PROFILE=n` in the default build (E32 has the per-layer profile; same
graph). Left in Assistent mode with `field on thresh 0.85`; wake step 1,259–1,265 µs,
room peaks 0.20 / 0.008. No spoken test in this session (audio embargo). E36's own
`w48_rw3_qe20` grid-CSV row and its `firmware/main/gen` untouched-hash check are unaffected
by this entry — they describe the pre-deploy state, correctly.

### E38 — pipeline hygiene: train once, deploy recipe in the loop, per-clip van aug (2026-09-07, host-only)

Tooling only — no model, dataset, or firmware artefact changed; no device. Seven
small fixes to the training/data path, each found while reading E35–E37's code paths:

1. `kws-train` trained the float model **twice** on a non-QAT run (a leftover
   `else:` retrain branch after the QAT weight-reload). Now trains exactly once; the
   QAT fine-tune starts from the in-process model, which is already Keras-2 under the
   `--qat` re-exec.
2. `scripts/data-loop.sh` trained and exported the default `w32` PTQ recipe, not the
   deployed one; its train/export pair now carries E37's `--width 48 --qat --qat-epochs 20
   --real-weight 3` (export: `--width 48 --qat`), so the loop produces
   `command_v3_w48_qat.tflite`.
3. Deleted the legacy v1 build path (`kws_de.data._build_and_split`, `kws-data --build`,
   `make_transition_windows`, and `build_dataset`'s `transition_*` inputs, none of which
   `kws-dataset build` ever used). `split_by_speaker` stays — `eval.py`/`transducer.py`
   still call it.
4. `van_augment`: `np.convolve` → `scipy.signal.fftconvolve` (same result to 1e-4,
   float32, same length; scipy was already a librosa dependency). Direct convolution of a
   16 k-sample clip with a multi-thousand-sample RIR was the slowest step of a van build.
5. Van-cabin augmentation drew **one** noise clip and **one** RIR per build and reused
   them for every real clip; now a fresh (noise, RIR) pair per real clip (wav lists
   globbed once). The manifest records `van_dirs` (the two env paths, or `null`) so a
   build without van augmentation is visible.
6. `kws-train` selects the best `val_accuracy` epoch via `ModelCheckpoint` when
   `{prefix}_val.npz` exists (the `benchmark.py` pattern), printing the epoch; falls back
   to last-epoch weights when there is no val split. QAT fine-tune unchanged.
7. `scripts/recipe-grid.py` applies the E27/E28 deploy rule per row (`passes`:
   aggregate ≥ 0.785, false accepts = 0, spk18 > 0.333) as a PASS/FAIL column plus a
   `git_sha` column; the committed CSV is back-filled (three PASS rows: `deployed_w48`,
   `w32_rw3_qe10`, `w48_rw3_qe20`; `git_sha` empty for pre-existing rows).

Checks: `pytest -q` 368 passed, 1 skipped, 1 xfailed (three transition-window tests
removed with their code, one fftconvolve-vs-`np.convolve` test added); `ruff check` /
`ruff format --check` clean; a 24-row toy `kws-train` run exercised the val-checkpoint
path. Items 5 and 6 change what the *next* build/train produces (per-clip van draws,
best-val weights), so the next retrain is not byte-comparable to E37's — expected, and
the reason none was run here.

### E39 — run 4: E38's per-clip van draws + val checkpoint vs. the deployed `rw3qe20` (2026-09-07, host-only, exp/run4-van-perclip)

E38 changed two things that alter what the *next* build/train produces without touching
the recipe: (5) a fresh (noise, RIR) pair per real clip instead of one pair per build,
and (6) `kws-train` restoring the best-`val_accuracy` epoch instead of the last. This
entry measures what they buy, replicating E35's procedure exactly — same cache, same
seed, same `--width 48 --qat --qat-epochs 20 --real-weight 3` recipe (epochs=40) — and
scoring against the deployed E37 model (`command_v3_w48_qat.tflite@86b7105e`, aggregate
0.936, 0/32, spk18 0.778). Worktree off `origin/main` at `09fffe2` (PR #83 merged);
`KWS_NOISE_DIR` (210 wavs) / `KWS_RIR_DIR` (270 wavs) set as in E31/E35 and verified by
count before the build.

**Rebuild.** E35's `features_v3_{train,val,test}.npz` + `manifest_v3.json` backed up as
`*.pre-run4` first (`manifest_v3_qat.json`, E37's refreshed copy, untouched). Then the
identical `kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3`.
`[recordings] merged: {...}` is the same 21-label dict E35 printed (285 device clips,
Licht 76 ... `_unknown_` 104). Manifest now carries `van_dirs` (both env paths).
**Wall-clock 97 s** against E35's ~25 min — item (4), `fftconvolve`, is a ~15x build
speed-up on its own.

**Row count differs from E35: `train=38,734, val=6,425, test=11,315`** vs. E35's
38,646 / 6,417 / 11,291 (+88 / +8 / +24 = +120). Diagnosed before training: the manifest
diff shows real rows identical in every split (13,774 / 1,993 / 2,163) and the delta is
entirely TTS — `[tts] Dach: 282 real clips, synthesizing 18 more` (gate dropped 4/18),
`[tts] an: 299 real clips, synthesizing 1 more`, `[tts] added: {'Dach': 14, 'an': 1}`.
15 new TTS clips x the 8-row augmentation multiplier = 120 rows (Dach 112 = 80+8+24,
an 8), so the per-clip van draws changed content only, exactly as expected, and the
multiplier is unchanged. The top-up is `build()`'s documented `_fill_with_tts`
path, which persists new clips back into the cache: `raw_clips_v3.pkl` was rewritten
(+403 KB; the pre-run4 clips are a strict subset). Not a break, but it means run 4 is
not *only* the two E38 changes — 15 extra TTS clips (0.04 % of train) ride along.

**Train.** `kws-train --v2 --prefix features_v3 --width 48 --qat --qat-epochs 20
--real-weight 3 --out command_v3_w48_qat_run4.keras` (no `--seed` flag exists; seed is
fixed at 0 inside `kws_de.train`). **Wall-clock 1,223 s** (E36's identical recipe:
1,297 s). The new checkpoint print: **`best epoch 40/40: val accuracy 0.6577`** — the
best val epoch *was* the last one, so item (6) selected exactly the weights the old
code would have saved; last-epoch val accuracy = best = 0.6577. Final float train
accuracy 0.7305, QAT final train accuracy 0.7463.

**Export**, isolated as E35 did: `kws-export --v2 --qat --width 48 --prefix features_v3
--model command_v3_w48_qat_run4.keras --out <models>/run4/` — *not* `--firmware`, which
would write `firmware/main/gen/`. Canonical `models/command_v3_w48_qat.tflite` hashed
`86b7105e` before and after; `git status` clean apart from this note and the CSV row.
Export 7 s, INT8 test accuracy 0.7158 (n=11,315, own-era test set).

**Comparison**, `scripts/compare_command_models.py --candidate <models>/run4/
command_v3_w48_qat.tflite --candidate-test-npz features_v3_test.npz --deployed-test-npz
features_v3_test.npz.pre-run4`, deployed header `86b7105e`. Note the manifest is now
E37's refreshed one, so spk18 (and spk19/spk20) are labelled in-training here, unlike
E35's table; the four-speaker scoreboard (233 words / 32 negatives) is unchanged.

| | deployed `rw3qe20` (E37) | run 4 `rw3qe20` | run 4 `rw3qe10` (step 5) |
|---|---|---|---|
| bytes / sha256 | 25,832 / `86b7105e` | 25,800 / `b8df37db` | 25,800 / `bd3ae11d` |
| INT8 test acc (own-era test set) | 69.36 % (n=11,291) | 71.58 % (n=11,315) | 71.54 % (n=11,315) |
| spk01 words (n=13) | **1.000** | 0.923 | 0.923 |
| spk02 words (n=38) | **1.000** | 0.974 | 0.974 |
| spk10 words (n=146) | **0.952** | 0.938 | 0.932 |
| spk18 words (n=36) | **0.778** | 0.750 | 0.750 |
| spk19 words (n=16, outside the 4-speaker rule) | 0.688 | **0.812** | 0.750 |
| spk20 words (n=20, outside the 4-speaker rule) | 0.800 | **0.900** | **0.950** |
| **aggregate words (n=233)** | **0.936** | 0.914 | 0.910 |
| false accepts spk02/spk10/spk18 | 0/10, 0/19, 0/3 | 0/10, **2/19**, 0/3 | 0/10, **1/19**, 0/3 |
| **false accepts total (n=32)** | **0/32** | **2/32** | **1/32** |
| phrase intent spk10/spk18/spk19/spk20 | 0.082, 0.176, 0.222, 0.083 | 0.093, 0.118, 0.333, 0.083 | 0.093, 0.118, 0.333, 0.083 |

**Deploy rule** (aggregate >= 0.785, 0 false accepts of 32, spk18 > 0.333), evaluated by
`scripts/recipe-grid.py`'s `passes()` (row `run4_w48_rw3_qe20` appended to
`scripts/recipe-grid.csv`, `git_sha 09fffe2`): **FAIL** on false accepts (2/32). Against
the deployed model specifically, run 4 is worse on every one of the rule's three
figures: aggregate -0.022 (0.936 -> 0.914, 5 fewer of 233 words), spk18 -0.028 (28/36 ->
27/36), and two new false accepts, both on spk10's negatives. **Run 4 does not beat
`86b7105e`; it should not be deployed.** The only movement in its favour is on the two
speakers the rule does not score (spk19 +0.125, spk20 +0.100) and a 2.2-point INT8
test-set gain that is not cross-comparable (different test rows). **Step 5, the `--qat-epochs 10` twin** (`command_v3_w48_qat_run4_qe10.keras`, 996 s, exported to `<models>/run4-qe10/`): the float phase reproduced bit-for-bit (same `best epoch 40/40: val accuracy 0.6577`, same 0.7305), so this row isolates QAT length on the new data. Row `run4_w48_rw3_qe10`: aggregate 0.910, spk18 0.750, **1/32** false accepts (spk10 1/19) — also **FAIL**, and also below the deployed model on all three rule figures. Shorter QAT halves the false-accept count here, the opposite direction from E36 (where `qe20` removed `qe10`'s single false accept), which reads as scoreboard noise on spk10's 19 negatives rather than a QAT-length effect.

**Reading.** Item (6) was a no-op this time (best = last epoch), so the whole delta is
per-clip van draws (+ the 15-clip TTS top-up). That delta is negative on the standing
scoreboard. One run at one seed cannot separate "per-clip draws hurt" from "the
scoreboard has noise of this size": E36 already showed a single spk10 false accept
appearing and disappearing between otherwise identical `qe10`/`qe20` runs, and 2/19 vs
0/19 on one speaker's negatives is the same order. What this entry does establish: the
E38 changes are *not* a free improvement over the deployed E37 model, and the deployed
model stays. Cheap next steps if anyone wants to pursue the aug change: (a) rerun the
build at 2-3 seeds (97 s each now) and train each, to get a variance bar for the rule's
false-accept count; (b) keep the pre-run4 npz as the deploy baseline until something
clears 0/32 again. The new npz/manifest are left in place as the current build (the
`*.pre-run4` files are E35's); the exports live under `<models>/run4/` and
`<models>/run4-qe10/`, never at the canonical path. No device, no flashing.

### E40 — run 5: seed variance of the deployed recipe + 80 epochs flat/cosine (2026-09-07, host-only, exp/run5-seeds-cosine)

E39 ended on two open questions: (a) is run 4's −0.022 aggregate / +2 false accepts
against the deployed model a real regression or single-seed noise, and (b) is the
40-epoch float phase under-trained. This entry answers both on the host, same recipe
throughout (`--width 48 --qat --qat-epochs 20 --real-weight 3`), scoring against the
deployed `command_v3_w48_qat.tflite@86b7105e` (E37: aggregate 0.936, 0/32, spk18
0.778). Worktree off `origin/main` at `625474a` (PR #84 merged); `KWS_NOISE_DIR` (210
wavs) / `KWS_RIR_DIR` (270 wavs) as in E39.

**Two small `kws-train` additions** (`kws_de/train.py`). `--seed N` (default 0) is now
passed through to `train()` and `train_qat()`, which already took `seed=` — E39 noted the
CLI had no way to set it. `--cosine` swaps the float phase's `"adam"` for
`Adam(CosineDecay(1e-3, epochs * ceil(rows / BATCH_SIZE)))` — Adam's default initial LR,
decayed to 0 over the whole run; the QAT fine-tune (`Adam(1e-5)`) is untouched. No
scheduler abstraction, no config; one test
(`test_cosine_decays_learning_rate_to_near_zero`: a 3-epoch toy run with `cosine=True`
ends with LR < 1e-4). `kws-train` also prints `epoch N: val_accuracy X` per epoch (a
`LambdaCallback` next to the E38 checkpoint) so a log answers (b) without re-running.

**Backups and restore.** Before any build: `raw_clips_v3.pkl`, `features_v3_{train,val,
test}.npz` and `manifest_v3.json` copied to `*.pre-run5` (the run-4 seed-0 build, E39).
After the last scoring, the four npz/manifest files were restored from `*.pre-run5` and
verified byte-identical (`cmp`), so the repo's baseline data is exactly E39's again. The
canonical `models/command_v3_w48_qat.tflite` hashed `86b7105e` before and after;
`firmware/main/gen/` untouched. `raw_clips_v3.pkl` is the one file left changed: the
seed-1 build's `_fill_with_tts` topped `Dach` up from 296 to 300 clips (`[tts] Dach: 296
real clips, synthesizing 4 more`, `[tts] added: {'Dach': 4}`, +134 KB), the seed-2 and
seed-0 rebuilds added nothing. Its pre-run5 copy sits alongside.

**Part A — seed variance.** `kws-dataset build --cache raw_clips_v3.pkl --prefix
features_v3 --seed N` (46–65 s each), then `kws-train --v2 --prefix features_v3 --width 48
--qat --qat-epochs 20 --real-weight 3 --seed N --out command_v3_w48_qat_run5_s{N}.keras`
(epochs 40), `kws-export --v2 --qat --width 48 --prefix features_v3 --model
command_v3_w48_qat_run5_s{N}.keras --out <models>/run5-s{N}/`, scored via
`scripts/recipe-grid.py`'s `score()`/`append_row()` (rows `run5_s{1,2}_w48_rw3_qe20`).
Seed 0 is E39's `run4_w48_rw3_qe20` row, reused. Note what the dataset seed moves: the
split is speaker-disjoint and TTS voices count as speakers, so seed 1 gives train/val/test
= 44,260 / 3,792 / 8,454 (train TTS 3,892 clips vs seed 0's 3,120) and seed 2 gives
38,884 / 8,657 / 8,964, against seed 0's 38,734 / 6,425 / 11,315. All 373 device clips of
spk01/02/10/18/19/20 are train-side in every seed. So "seed" here = split + augmentation
draws + init, and each seed's INT8 test accuracy is against its own test split (not
comparable across rows).

| | seed 0 (run 4, E39) | seed 1 | seed 2 | mean ± range/2 |
|---|---|---|---|---|
| sha256 | `b8df37db` | `0a2543e8` | `38bed244` | |
| INT8 test acc (own split) | 0.7158 | 0.5974 | 0.6817 | n/a |
| best epoch / val acc | 40/40, 0.6577 | 39/40, 0.5891 | 33/40, 0.6691 | |
| spk01 (n=13) | 0.923 | 0.923 | 0.923 | 0.923 ± 0.000 |
| spk02 (n=38) | 0.974 | 1.000 | 0.974 | 0.982 ± 0.013 |
| spk10 (n=146) | 0.938 | 0.932 | 0.938 | 0.936 ± 0.003 |
| spk18 (n=36) | 0.750 | 0.722 | 0.861 | 0.778 ± 0.069 |
| spk19 (n=16) | 0.812 | 0.812 | 0.750 | 0.792 ± 0.031 |
| spk20 (n=20) | 0.900 | 0.850 | 0.900 | 0.883 ± 0.025 |
| **aggregate (n=233)** | 0.914 | 0.910 | **0.931** | **0.919 ± 0.011** |
| false accepts (n=32) | 2 (spk10 2/19) | 1 (spk18 1/3) | **0** | 1.0, range 0–2 |
| deploy rule | FAIL | FAIL | **PASS** | |
| train wall-clock | 1,223 s | 1,317 s | 1,226 s | |

**Variance bar.** Over three seeds the aggregate spans 0.910–0.931 (range 0.021) and the
false-accept count takes every value 0, 1, 2 — on a different speaker each time. That is
exactly the size of E39's run-4-vs-deployed delta (−0.022 aggregate, +2 false accepts),
so E39's "does per-clip van aug hurt?" question cannot be answered by one seed:
run 4's shortfall is one draw from a distribution whose width is the effect being
tested. The deploy rule's ±0.02 aggregate differences and 0-vs-1 false-accept differences
between single runs are inside seed noise; a real recipe difference needs either ≥ 3
seeds per arm or a margin larger than ~0.02 / 2 FA. spk18 is the noisiest speaker
(0.722–0.861 on n=36, ±2.5 words); spk01/spk10 barely move. Seed 2 clears the rule (0/32,
spk18 0.861, aggregate 0.931) but is still 0.005 below the deployed 0.936 (one word of
233), so the standing model is not beaten. The deployed E37 model itself is one seed of
the pre-E38 build; its 0.936 / 0/32 should be read with the same ±0.011 / 0–2 bar.

**Part B — 80 epochs, flat vs cosine.** Seed-0 rebuild first (`--seed 0`, same prefix;
train/val/test 38,758 / 6,433 / 11,315 — run 4's split plus the 4 Dach TTS clips × 8
augmentation rows, so not byte-identical to run 4's npz; the `*.pre-run5` restore at the
end puts run 4's back). Then two runs, both `--seed 0 --epochs 80` on this build,
`run5_e80_flat` (constant LR) and `run5_e80_cos` (`--cosine`), exported to
`<models>/run5-e80-flat/` and `<models>/run5-e80-cos/`.

| | run 4 (40 ep, E39) | `run5_e80_flat` | `run5_e80_cos` | deployed `86b7105e` |
|---|---|---|---|---|
| sha256 | `b8df37db` | `1cf0435e` | `f7483c99` | `86b7105e` |
| INT8 test acc (own split) | 0.7158 | 0.7125 | 0.7064 | 0.6936 (n=11,291) |
| best epoch / val acc | 40/40, 0.6577 | 35/80, 0.6695 | 68/80, 0.6785 | n/a |
| float / QAT final train acc | 0.7305 / 0.7463 | 0.7696 / 0.7404 | 0.7617 / 0.7425 | |
| spk01 / spk02 / spk10 | 0.923 / 0.974 / 0.938 | 0.923 / 0.947 / 0.911 | 0.923 / 1.000 / 0.945 | 1.000 / 1.000 / 0.952 |
| spk18 / spk19 / spk20 | 0.750 / 0.812 / 0.900 | 0.778 / 0.812 / 0.950 | 0.806 / 0.688 / 0.900 | 0.778 / 0.688 / 0.800 |
| **aggregate (n=233)** | 0.914 | 0.897 | **0.931** | **0.936** |
| false accepts (n=32) | 2 (spk10) | 1 (spk10 1/19) | 1 (spk10 1/19) | **0** |
| deploy rule | FAIL | FAIL | FAIL | (reference) |
| train wall-clock | 1,223 s | 2,115 s | 2,031 s | |

Per-epoch `val_accuracy` (flat; epoch 1, then every 5th of 80): 0.085, 0.527, 0.555,
0.590, 0.632, 0.629, 0.614, 0.670 (ep 35, best), 0.638, 0.656, 0.634, 0.611, 0.650, 0.667,
0.662, 0.647, 0.659 — a plateau at 0.61–0.67 from epoch ~28 with ±0.03 epoch-to-epoch
jitter (last 20 epochs span 0.626–0.667); the best epoch (35) is inside the 40 the recipe
already runs. Cosine: 0.075, 0.553, 0.555, 0.622, 0.617, 0.635, 0.652, 0.661, 0.652,
0.668, 0.661, 0.667, 0.675, 0.671, 0.673, 0.674, 0.673 (best 68, 0.6785) — the decay
flattens the tail (last 20 epochs within 0.667–0.679) and lands
0.009 higher on val than flat's best, but the scoreboard does not follow: aggregate 0.931
equals seed 2's 40-epoch run and one spk10 false accept remains. **40 epochs is not
under-trained**: doubling the float phase costs +900 s and moves nothing outside the Part A
noise bar (flat 0.897 is the *low* end of it). `--cosine` is a harmless tail-smoother
worth keeping as a flag, not a lever.

**Answer to the report question.** No candidate both passes the deploy rule and beats
`86b7105e` on aggregate at 0 false accepts: seed 2 (0.931, 0/32) is the closest and is
one word short. The deployed model stays. What this entry does settle is the yardstick:
single-seed differences below ~0.02 aggregate / 2 false accepts are noise on this
scoreboard, which retroactively makes E36's `qe10`-vs-`qe20` false-accept flip and E39's
run-4-vs-deployed verdict both "within the bar". Anyone re-opening the per-clip-van-aug
question should run 3 seeds per arm and compare means.

**Housekeeping.** `scripts/recipe-grid.csv` gained four rows (`run5_s1_w48_rw3_qe20`,
`run5_s2_w48_rw3_qe20`, `run5_e80_flat`, `run5_e80_cos`; `git_sha` auto-filled with the
pre-commit `625474a`). Exports live under `<models>/run5-*/`, never at the canonical
path. `pytest -q tests/test_train.py` 5 passed; `ruff check` / `ruff format --check`
clean. No device, no flashing, no deploy.

### E41 — sentence-level (end-to-end phrase) performance of the deployed run-3 model (2026-09-07, host-only)

(E40 is reserved for the run-5 seed-variance entry.) Every command-model comparison since E15
has scored isolated words; phrase intent has been a footnote row ("spk10 0.082, rest 0.000").
This entry measures the deployed `rw3qe20` (`command_v3_w48_qat.tflite@86b7105e`, E37) on the
full QC-approved set through the same streaming decoder + grammar path the device runs, so
the paper can state the word→sentence gap as a number rather than an aside. Host-only, no
device, no training:

```text
KWS_DATA_ROOT=<data root> kws-eval --recordings "$KWS_DATA_ROOT/data/recordings/approved" \
  --prefix features_v3 --width 48 --qat --out "$KWS_DATA_ROOT/docs/eval-report-v3-now.md"
```

Report at `$KWS_DATA_ROOT/docs/eval-report-v3-now.md` (`.recordings.json` beside it); manifest
`data/manifest_v3_qat.json`, built 2026-09-06T18:16:55Z (E37's refreshed copy, so spk18/spk19/
spk20 are labelled in-training; phrase clips are always held-out — never trained on).

| speaker | isolated words n | acc | e2e phrases n | exact-intent acc | negatives n | false accepts |
|---|---|---|---|---|---|---|
| spk01 | 13 | 1.000 | 0 | — | 0 | — |
| spk02 | 38 | 1.000 | 4 | 0.000 | 10 | 0/10 |
| spk10 | 146 | 0.952 | 97 | 0.082 | 19 | 0/19 |
| spk18 | 36 | 0.778 | 17 | 0.176 | 3 | 0/3 |
| spk19 | 16 | 0.688 | 9 | 0.222 | 10 | 0/10 |
| spk20 | 20 | 0.800 | 12 | 0.083 | 1 | 0/1 |
| **all** | **269** | **0.911** | **139** | **≈ 0.10 (14/139)** | **43** | **0/43** |

The 269-word aggregate (0.911) is over all six speakers; the deploy rule's 233-word
four-speaker scoreboard (E35/E37) reads 0.936 on the same model — spk19/spk20 sit outside that
rule and pull the six-speaker figure down.

**Field** (device–Whisper agreement *at capture time*, i.e. of the model deployed when each take
was recorded — `8fa81d08` for every session to date, not `86b7105e`): 125 field takes, 68
approved, 36 parsable (0.288); 18 false alarms at both the production gate 0.85 and the capture
gate 0.60, 0 near-misses. spk18 39 takes / 37 approved / 18 parsable, agreement 0.125, 8 false
alarms; spk19 24 / 19 / 9, agreement 0.500, 10 false alarms; spk20 13 / 12 / 9, agreement 0.500,
0 false alarms; spk17 49 takes, 0 approved. Elicit: 4 takes, 4 approved, 4 parsable, expected-match
0.667 of 3 compared (spk20).

**Reading.** The classifier is at 0.94 on words (deploy scoreboard) and 0 false accepts in 43;
the streaming decoder over the same speakers' read sentences yields ≈ 0.10 exact intents. Every
command-model change since E15 moved phrase intent by at most a few points (spk10
0.062 → 0.082 → 0.113 → 0.082) while words went 0.538 → 0.936, and E28's read-only decoder sweep
moved field agreement 3/18 → 4/18. **The word→sentence path — segmentation, run-based decoding,
the grammar's all-or-nothing parse — is the bottleneck, not the classifier.** This is the
paper's §6.15 and the first item of its limitations. Next levers, in order: the n-best lattice
parse over existing posteriors (Open questions, below), per-word segmentation diagnostics on the
125 failing phrases, and the E8 transducer once phrase data is real rather than 392 synthetic
sentences.

### E42 — single-variable test: does a wider training time-shift (±500 ms) fix the in-context word miss? (2026-09-08, host-only, exp/shift-500)

E41's per-window diagnostic over the 139 approved phrase clips (deployed `86b7105e`, decoder
replayed at 100 ms) found that an expected word is raw top-1 in 91 % of windows when it sits where
a training clip puts it (word centre + 0.5 s = window end), 45 % at +0.3 s, 3 % at −0.4 s, 0 % at
−0.5 s. Training clips are trimmed, centred and shifted by only ±200 ms (`_random_shift`), so the
hypothesis was: the classifier has learnt "word in the middle", the streaming decoder rarely
offers that alignment, widening the shift to ±500 ms should flatten the offset curve and lift
phrase exact-intent (0.10, oracle ceiling 0.34) without hurting isolated words much. One
variable, two arms, same seed, nothing else moved.

**Code.** `kws-dataset build --shift-ms N` (default 200 = previous behaviour) is passed through
`kws_de.dataset.build → assemble → kws_de.data.build_dataset(shift_ms=)` into every
`_random_shift` call (clean, per-snr, perturbed-TTS and van rows alike) and recorded as
`"shift_ms"` in the manifest. One test (`test_random_shift_500_keeps_length_and_stays_within_bounds`).

**Procedure.** Worktree off `origin/main` at `459b05f`; `features_v3_{train,val,test}.npz`,
`manifest_v3.json`, `raw_clips_v3.pkl` backed up as `*.pre-shift` and the npz + manifest restored
(byte-identical, `cmp`) at the end; `raw_clips_v3.pkl` unchanged (`[tts] added:` empty in both
builds). `KWS_NOISE_DIR` (210) / `KWS_RIR_DIR` (270) as in E39–E40. Both arms:
`kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3 --seed 0 [--shift-ms 500]`
(48 s / 46 s, train/val/test 38,758 / 6,433 / 11,315 in both — same split, same rows, only the
shift draw differs; note val/test rows are shifted too, so INT8 test accuracy is *not*
comparable across arms), `kws-train --v2 --prefix features_v3 --width 48 --qat --qat-epochs 20
--real-weight 3 --seed 0 --out command_v3_w48_qat_shift{200,500}.keras` (1,287 s / 1,255 s),
`kws-export --v2 --qat --width 48 --prefix features_v3 --model … --out <models>/shift{200,500}/`
(canonical `command_v3_w48_qat.tflite` hashed `86b7105e` before and after; `firmware/main/gen/`
untouched), scored with `scripts/recipe-grid.py`'s `score()`/`append_row()` (rows
`shift200_w48_rw3_qe20`, `shift500_w48_rw3_qe20`), `scripts/compare_command_models.py`, and the
E41 per-window diagnostic pointed at each export.

**Control reproduces a known row.** The seed-0 build is E40 Part B's (38,758 rows) and
`best epoch 35/40: val accuracy 0.6695` is exactly `run5_e80_flat`'s best epoch, so the control
export is bit-identical to it (sha `1cf0435e`, aggregate 0.897, 1/32) — E40's "40 epochs is not
under-trained" claim confirmed the cheap way. The deployed model (`86b7105e`, E37) is a different
build era and sits at the top of the seed band; the arm-to-arm comparison is control vs treatment.

| | deployed `86b7105e` (E41) | control shift 200 | treatment shift 500 |
|---|---|---|---|
| sha256 / bytes | `86b7105e` / 25,832 | `1cf0435e` / 25,800 | `58129989` / 25,800 |
| best epoch / val acc (own val, own shift) | n/a | 35/40, 0.6695 | 35/40, 0.5548 |
| float / QAT final train acc | | 0.7330 / 0.7404 | 0.6486 / 0.6542 |
| INT8 test acc (own test rows) | 0.6936 | 0.7125 | 0.5972 |
| spk01 / spk02 / spk10 / spk18 words | 1.000 / 1.000 / 0.952 / 0.778 | 0.923 / 0.947 / 0.911 / 0.778 | 0.846 / 1.000 / 0.829 / 0.583 |
| spk19 / spk20 words (outside the rule) | 0.688 / 0.800 | 0.812 / 0.950 | 0.562 / 0.750 |
| **aggregate words (n=233)** | **0.936** | 0.897 | **0.820** |
| false accepts (n=32) | 0 | 1 (spk10 1/19) | 0 |
| deploy rule (`passes()`) | (reference) | FAIL | PASS (on the floor: 0.820 ≥ 0.785, spk18 0.583 > 0.333) |
| isolated words, all six speakers (n=269, diag) | 0.911 | 0.896 | 0.799 |
| **phrase exact-intent, unpadded (n=139)** | 0.101 (14) | 0.058 (8) | 0.094 (13) |
| phrase exact-intent, 0.7 s silence prepended | 0.40 (55) | 0.35 (49) | 0.31 (43) |
| phrase exact-intent, 0.7 s padded both ends | 0.57 (79) | 0.48 (67) | 0.37 (52) |
| oracle ceiling raw t=0.3 / t=0.5 | 0.32 / 0.29 | 0.28 / 0.24 | 0.25 / 0.20 |
| oracle ceiling smoothed t=0.3 / t=0.5 | 0.28 / 0.19 | 0.23 / 0.14 | 0.24 / 0.13 |
| argmax-only oracle | 0.29 | 0.25 | 0.25 |
| failure classes OK / MISS / SUB / INS / MULTI | 14 / 91 / 2 / 1 / 31 | 8 / 96 / 3 / 1 / 31 | 13 / 89 / 3 / 0 / 34 |
| why expected words miss: fired / never top-1 / hangover / below thr | 52 / 34 / 9 / 4 % | 50 / 39 / 6 / 5 % | 46 / 37 / 8 / 9 % |
| in-context peak raw median / run width median | 0.86 / 2 win | 0.84 / 2 win | 0.71 / 1 win |
| isolated-word stream peak raw median / run width | 0.97 / 7 win | 0.90 / 5 win | 0.86 / 6 win |

Per expected word, fired (decoder) / ever raw top-1 / isolated acc (n): Licht 40/58/0.84 (76) →
38/49/0.86 → 28/50/0.75; an 14/23/0.96 (26) → 14/23/1.00 → 18/26/0.92; aus 14/19/0.84 (19) →
14/15/0.68 → 11/17/0.47; Küche 1/1 → 1/1 → 0/1 (n=20 expected, 1 isolated clip); Dach 0/3 → 0/1
→ 0/0 (n=19, no isolated clips); Außen 15/16/0.95 (22) → 16/16/0.95 → 12/16/0.82; Lesen
15/15/1.00 (12) → 16/16/1.00 → 14/14/0.83.

**Offset curve** (guided takes, Whisper spans; cell = mean raw p(word) / share of windows where the
word is raw top-1; offset = window end − (word centre + 0.5 s), 0 = training layout):

| offset (s) | −0.5 | −0.4 | −0.3 | −0.2 | −0.1 | 0.0 | +0.1 | +0.2 | +0.3 | +0.4 | +0.5 | +0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deployed (E41) | 0.01/0 % | 0.04/3 % | 0.27/22 % | 0.56/62 % | 0.78/85 % | **0.83/91 %** | 0.74/82 % | 0.66/72 % | 0.43/45 % | 0.39/39 % | 0.33/37 % | 0.23/29 % |
| control shift 200 | 0.01/0 % | 0.02/0 % | 0.19/15 % | 0.45/55 % | 0.76/82 % | **0.83/90 %** | 0.74/81 % | 0.62/67 % | 0.41/42 % | 0.38/39 % | 0.34/38 % | 0.23/30 % |
| treatment shift 500 | 0.02/0 % | 0.09/6 % | 0.27/32 % | 0.48/54 % | 0.68/78 % | **0.69/76 %** | 0.63/69 % | 0.54/59 % | 0.37/39 % | 0.36/39 % | 0.30/37 % | 0.23/28 % |

Per word, control → treatment at the centre and at ±0.3: Licht (window ends only at ≥ +0.1
because Licht opens every clip) +0.1: 0.96/100 % → 0.81/100 %, +0.3: 0.08/5 % → 0.06/5 %; Außen
0.0: 0.83/94 % → 0.60/69 %, −0.3: −/− (n=0), +0.3: 0.24/25 % → 0.12/12 %; aus 0.0: 0.64/60 % →
0.34/30 %, +0.4: 0.96/100 % → 0.86/90 %; an 0.0: 0.84/90 % → 0.85/90 %, −0.3: 0.22/17 % → 0.55/67 %;
Lesen 0.0: 0.90/100 % → 0.64/73 %, −0.3: 0.01/0 % → 0.64/100 %, +0.3: 0.12/9 % → 0.04/0 %.

**Verdict: refuted.** The ±500 ms shift does not lift the shoulders of the offset curve; it lowers
the peak. Top-1 share at 0.0 falls 90 % → 76 % and mean p 0.83 → 0.69, while +0.3/+0.4 stay at
39–42 % and only the −0.3/−0.4 side moves at all (15 % → 32 %, 0 % → 6 % — the two words that
gained there, `an` and `Lesen`, are short words that a wider shift lets the model see with more
right-context; the left-of-centre gain is real but small). Unpadded phrase exact-intent goes 8 →
13 of 139 (0.058 → 0.094), *within* the control-vs-deployed gap (8 vs 14 on the same clips, both
±200 ms builds) and below the 0.10 it had to beat clearly; every counterfactual that removes the
alignment problem by hand gets *worse* (prepend 0.35 → 0.31, pad both ends 0.48 → 0.37), the
oracle ceilings drop (raw t=0.3 0.28 → 0.25), and the in-context peak run width shrinks to 1
window (46 % of expected words never reach a smoothed top-1 run at all, vs 43 %). The isolated
scoreboard pays for it: aggregate 0.897 → 0.820 (−0.077, 18 fewer of 233 words, seven times the
E40 seed half-range of 0.011), spk18 0.778 → 0.583, spk10 0.911 → 0.829, six-speaker isolated
0.896 → 0.799, `aus` isolated 0.68 → 0.47. `shift500` clears `passes()` only because the rule's
0.785 floor dates from the 0.785 deployed model of E27 and its false-accept count happens to be 0;
it is 0.116 below the deployed `86b7105e` on aggregate and would not be deployed under any
reading of E40's yardstick. Not deployed, not a candidate.

**Reading.** The width-48 DS-CNN at 25.8 kB does not have spare capacity to become
shift-invariant by data alone: asking it to recognise a word anywhere in the 1 s window costs
it the centred case without buying the off-centre one (val accuracy on the *same* split falls
0.6695 → 0.5548 when val rows are also shifted ±500 ms). The E41 offset curve is therefore not
mainly a training-augmentation artefact; it reads as the model's real positional resolution at
this size — a window that holds a word at its edge also holds half of the next word, and the
classifier (whose MFCC frames cover the whole second) sees a mixture it was never asked to
separate, at either shift range. The word→sentence bottleneck stays on the decoder side: the
in-context top-1 runs are 1–2 windows wide against the isolated 5–7, and E41's oracle
(0.34) is reached by lowering what the decoder needs, not by moving where the word sits. Next
levers, unchanged from E41: threshold/hangover from E28 (0.3 / 1) re-evaluated on the phrase
set, n-best over the posteriors, and a wider-context model (2 s / two-window input, or the E8
transducer) rather than more augmentation on the 1 s classifier. An intermediate shift
(±300–350 ms) is not worth a run on this evidence: the shoulder did not move at +0.3 at all.

**Housekeeping.** `scripts/recipe-grid.csv` gained `shift200_w48_rw3_qe20` (`1cf0435e`, FAIL) and
`shift500_w48_rw3_qe20` (`58129989`, PASS-on-the-floor); exports under `<models>/shift200/` and
`<models>/shift500/`, never at the canonical path; diagnostic CSVs + summaries kept in the
session scratch, not committed. No device, no flashing, no deploy.

### E43 — single-variable test: do multi-word context windows fix the in-context word miss? (2026-09-08, host-only, exp/context-mix)

E42 refuted "the model has learnt *where* the word sits". The surviving hypothesis from E41's
per-window diagnostic was about *what else* is in the window: every training row holds exactly
one word plus silence/noise, so when a neighbour word enters the 1 s window the classifier sees
a mixture it was never asked to separate and confuses the target with same-slot words
(`Licht` → `Außen`/`Lesen` in 51 % of the windows that fully contain `Licht`, 58 % for the
control), worst for the
middle word of a 3-word sentence (exact-intent 0.04 vs 0.16 for 2-word, oracle 0.11 vs 0.55).
Prediction: training on windows that contain the target *with* neighbouring words raises the
±0.3–0.4 s shoulders of the offset curve, lifts unpadded phrase exact-intent clearly above the
deployed 14/139, and keeps isolated words inside E40's seed band (0.919 ± 0.011, FA ≤ 2). One
variable, control reused from E42, same seed, nothing else moved.

**Code.** `kws-dataset build --context-mix K` (default 0 = previous behaviour). For each command
clip of the *train* split, `kws_de.data.build_dataset` appends K rows: the silence-trimmed clip
(`librosa.effects.trim(top_db=30)`, as `recordings.py`) between 1–2 other trimmed clips of the
same split (same speaker when that speaker has ≥ 2 others, otherwise any; any label incl.
`_unknown_`), 50–200 ms silent gaps, the 1 s window centred on the target (`_context_window`,
zero-padded past the sequence ends), labelled with the target word, `_random_shift`ed ±200 ms
and mixed at ONE draw from {clean, 20, 10, 0 dB} instead of the full ladder — the full ladder
(7 rows per context sequence) would have tripled the train set. The first sequence of each
clip also yields the window centred on the gap next to the target, labelled `_unknown_`, so a
word-boundary window has a class. The rows are appended AFTER the existing ones, so the
38,758 base rows and every val/test row are byte-identical to the K=0 build (test npz `cmp`
equal to E42's control; INT8 test accuracy therefore comparable). `_origin_flags` mirrors the
K + 1 rows per clip with the target clip's origin (so `--real-weight 3` triples real-target
context rows too); `"context_mix"` in the manifest. One test
(`test_context_mix_appends_target_centred_rows_and_gap_unknowns`: K=1 adds 2 rows per command
clip, base rows unchanged, flags aligned, window geometry).

**Procedure.** Worktree off `origin/main` at `4df9e85`; data backed up as `*.pre-ctxmix`, the
npz + manifest restored byte-identical at the end, `raw_clips_v3.pkl` unchanged. Same `KWS_NOISE_DIR`
(210) / `KWS_RIR_DIR` (270) as E39–E42. Control = E42's `shift200` row (`1cf0435e`, seed-0
build 38,758 rows), not retrained. Treatment: `kws-dataset build --cache raw_clips_v3.pkl
--prefix features_v3 --seed 0 --context-mix 2` (52 s; train 52,201 = 38,758 + 4,481 command
clips × (2 target + 1 gap) rows, i.e. 1.35× the control and under the 2× cap; `_unknown_`
3,689 → 8,170 rows; val/test 6,433 / 11,315 unchanged), `kws-train --v2 --prefix features_v3
--width 48 --qat --qat-epochs 20 --real-weight 3 --seed 0 --out
command_v3_w48_qat_ctxmix.keras` (1,700 s vs 1,287 s), `kws-export … --out <models>/ctxmix/`
(canonical `command_v3_w48_qat.tflite` still `86b7105e`, `firmware/main/gen/` untouched),
scored with `scripts/recipe-grid.py`'s `score()` (row `ctxmix2_w48_rw3_qe20`),
`scripts/compare_command_models.py`, and the E41/E42 per-window diagnostic.

| | deployed `86b7105e` (E41) | control shift 200 (E42) | treatment context-mix 2 |
|---|---|---|---|
| sha256 / bytes | `86b7105e` / 25,832 | `1cf0435e` / 25,800 | `8f3156e9` / 25,800 |
| train rows | (other era) | 38,758 | 52,201 |
| best epoch / val acc (same single-word val rows) | n/a | 35/40, 0.6695 | 39/40, 0.6474 |
| float / QAT final train acc | | 0.7330 / 0.7404 | 0.6680 / 0.6785 (harder rows) |
| INT8 test acc (identical test rows) | 0.7188 | 0.7125 | 0.7127 |
| spk01 / spk02 / spk10 / spk18 words | 1.000 / 1.000 / 0.952 / 0.778 | 0.923 / 0.947 / 0.911 / 0.778 | 0.923 / 1.000 / 0.932 / **0.889** |
| spk19 / spk20 words (outside the rule) | 0.688 / 0.800 | 0.812 / 0.950 | 0.812 / 0.900 |
| **aggregate words (n=233)** | **0.936** | 0.897 | **0.936** |
| false accepts (n=32) | 0 | 1 (spk10 1/19) | 1 (spk10 1/19) |
| deploy rule (`passes()`) | (reference) | FAIL (FA) | FAIL (FA) |
| isolated words, all six speakers (n=269, diag) | 0.911 | 0.896 | 0.926 |
| **phrase exact-intent, unpadded (n=139)** | 0.101 (14) | 0.058 (8) | 0.079 (11) |
| phrase exact-intent, 0.7 s silence prepended | 0.40 (55) | 0.35 (49) | 0.38 (53) |
| phrase exact-intent, 0.7 s padded both ends | 0.57 (79) | 0.48 (67) | 0.52 (72) |
| oracle ceiling raw t=0.3 / t=0.5 | 0.32 / 0.29 | 0.28 / 0.24 | 0.29 / 0.25 |
| oracle ceiling smoothed t=0.3 / t=0.5 | 0.28 / 0.19 | 0.23 / 0.14 | 0.25 / 0.14 |
| argmax-only oracle | 0.29 | 0.25 | 0.26 |
| failure classes OK / MISS / SUB / INS / MULTI | 14 / 91 / 2 / 1 / 31 | 8 / 96 / 3 / 1 / 31 | 11 / 96 / 1 / 1 / 30 |
| why expected words miss: fired / never top-1 / hangover / below thr | 52 / 34 / 9 / 4 % | 50 / 39 / 6 / 5 % | 51 / 36 / 7 / 6 % |
| in-context peak raw median / run width median | 0.86 / 2 win | 0.84 / 2 win | 0.87 / 2 win |
| isolated-word stream peak raw median / run width | 0.97 / 7 win | 0.90 / 5 win | 0.96 / 7 win |
| exact-intent 2-word (n=67) / 3-word (n=72) | 0.16 (11) / 0.04 (3) | 0.10 (7) / 0.01 (1) | 0.15 (10) / 0.01 (1) |
| oracle raw t=0.3, 2-word / 3-word | 0.55 / 0.11 | 0.51 / 0.07 | 0.54 / 0.06 |
| 3-word: first / **middle** / last word fired | 0.28 / **0.42** / 0.88 | 0.28 / **0.46** / 0.82 | 0.21 / **0.47** / 0.88 |
| 2-word: first / last word fired | 0.48 / 0.57 | 0.43 / 0.52 | 0.45 / 0.55 |

Per expected word, fired (decoder) / ever raw top-1 / isolated acc (n), deployed → control →
treatment: Licht 40/58/0.84 (76) → 38/49/0.86 → 34/50/0.92; an 14/23/0.96 (26) → 14/23/1.00 →
14/23/1.00; aus 14/19/0.84 (19) → 14/15/0.68 → 14/18/0.74; Küche 1/1 → 1/1 → 1/1 (n=20
expected, 1 isolated clip); Dach 0/3 → 0/1 → 0/0 (n=19, no isolated clips); Außen 15/16/0.95
(22) → 16/16/0.95 → 17/17/0.95; Lesen 14/15/1.00 (12) → 16/16/1.00 → 16/16/1.00.

**Offset curve** (guided takes, Whisper spans; cell = mean raw p(word) / share of windows where the
word is raw top-1; offset = window end − (word centre + 0.5 s), 0 = training layout):

| offset (s) | −0.5 | −0.4 | −0.3 | −0.2 | −0.1 | 0.0 | +0.1 | +0.2 | +0.3 | +0.4 | +0.5 | +0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deployed (E41) | 0.01/0 % | 0.04/3 % | 0.27/22 % | 0.56/62 % | 0.78/85 % | **0.83/91 %** | 0.74/82 % | 0.66/72 % | 0.43/45 % | 0.39/39 % | 0.33/37 % | 0.23/29 % |
| control shift 200 (E42) | 0.01/0 % | 0.02/0 % | 0.19/15 % | 0.45/55 % | 0.76/82 % | **0.83/90 %** | 0.74/81 % | 0.62/67 % | 0.41/42 % | 0.38/39 % | 0.34/38 % | 0.23/30 % |
| treatment context-mix 2 | 0.01/0 % | 0.02/0 % | 0.16/12 % | 0.46/52 % | 0.81/88 % | **0.85/91 %** | 0.79/86 % | 0.67/72 % | 0.42/44 % | 0.40/42 % | 0.34/36 % | 0.24/30 % |

Per word, control → treatment: Licht +0.1: 0.96/100 % → 0.96/100 %, +0.3: 0.08/5 % → 0.07/5 %;
Außen 0.0: 0.83/94 % → 0.87/94 %, +0.3: 0.24/25 % → 0.19/12 %; aus 0.0: 0.64/60 % → 0.64/70 %,
+0.4: 0.96/100 % → 0.90/100 %; an 0.0: 0.84/90 % → 0.91/100 %, −0.3: 0.22/17 % → 0.23/17 %;
Lesen 0.0: 0.90/100 % → 0.92/100 %, +0.3: 0.12/9 % → 0.21/18 %. Confusion while `Licht` is fully
inside the window (41 windows): control Außen 34 % / Lesen 24 % / Licht 5 % → treatment Außen
37 % / Lesen 24 % / Licht 5 % — unchanged. `aus` inside the window: aus 60 % → 66 %, Außen
32 % → 21 %; `Lesen`: 71 % → 85 %.

**Verdict: refuted.** Unpadded phrase exact-intent 8 → 11 of 139 (0.058 → 0.079), below the
deployed 14 it had to beat clearly, and the 3-word sentences — the specific prediction — stay
at 1/72 with the middle word firing in 47 % (control 46 %, deployed 42 %) and the first word
*less* often (28 % → 21 %). The +0.3/+0.4 s shoulders do not move (42/39 % → 44/42 %, mean p
0.41/0.38 → 0.42/0.40; the criterion was "clearly above ~40 %"); the whole curve is the
control's within ±3 points, the centre included. The `Licht → Außen/Lesen` confusion that
motivated the experiment is unchanged (58 % → 61 % of the windows holding `Licht`). Oracle ceilings
move by one clip (raw t=0.3 0.28 → 0.29), the counterfactuals by 3–5 clips (prepend 0.35 →
0.38, pad both 0.48 → 0.52), every one still below the deployed model's. What the context rows
*did* buy is isolated words: aggregate 0.897 → 0.936 (+9 of 233; spk18 0.778 → 0.889, spk10
0.911 → 0.932, spk02 0.947 → 1.000), six-speaker isolated 0.896 → 0.926 — the deployed model's
level, reached from the seed-0 build that E40 places at the bottom of its band, and the
isolated-stream run width 5 → 7 windows. The isolated criterion (≥ 0.908, FA ≤ 2) is met with
room; the two phrase criteria are not. Deploy rule: FAIL on the same single spk10 false accept
as the control (1/32), which is the rule's zero-FA clause, not the accuracy floor; on E40's
yardstick it would be a candidate for isolated words only. Not deployed.

**Reading.** Multi-word context in the training window makes the classifier a better
*single-word* classifier (it learns to ignore a neighbour that is not centred) but not a
better *streaming* one: the decoder's problem is the windows where the target is NOT centred,
and those windows now carry an explicit competing label — the gap-centred `_unknown_` rows
teach "boundary ⇒ unknown", the target-centred rows teach "centred ⇒ word", and the net effect
on a window at +0.3 s is nil. The offset curve is now flat across three independent
interventions (deployed era, ±500 ms shift, context mix): it is the model's positional
resolution at 25.8 kB, not a data artefact. With E41/E42 this closes the "fix it in the 1 s
classifier's training data" branch; the levers left are decoder-side (E28's threshold 0.3 /
hangover 1 re-evaluated on the phrase set, n-best over posteriors) or a wider-context model
(two-window input or the E8 transducer). The isolated-word gain is a separate, cheap finding
worth a seed check before it is used: a context-mix build at seeds 1/2 against E40's band
would tell whether +0.04 aggregate is real or a seed-0 coincidence.

**Housekeeping.** `scripts/recipe-grid.csv` gained `ctxmix2_w48_rw3_qe20` (`8f3156e9`, FAIL);
export under `<models>/ctxmix/`, never at the canonical path; diagnostic CSVs + summaries kept in
the session scratch, not committed. No device, no flashing, no deploy.

### E44 — firmware: recogniser cadence 146 → 100 ms, wake-tail drop made class-targeted (2026-09-08, host-only, fix/recogniser-cadence)

Two device-side reasons the host decoder numbers (E41–E43, all replayed at a 100 ms stride) never
transferred to the CoreS3, both found by reading `recognise.cc` against `kws_de/eval.py`, neither
touching the model:

**1. Cadence.** The recognise task loop was `vTaskDelay(100)` *after* ~46 ms of work, so its real
stride was ≈ 146 ms (the step trace's "front-end … over 6 new frames" at a 20 ms hop is 120–140 ms
of audio per step; 100 ms would push 5). Every decoder constant is in steps: `KWS_MIN_CONSECUTIVE`
2 needed ≈ 290 ms of stable top-1 on the device against 200 ms on the host, while E41's
in-context top-1 plateau is 1–2 windows (≈ 100–200 ms) wide — so the device drops words the host
keeps, and `KWS_SMOOTH_WIN` 3 averaged over 440 ms instead of 300. Fix: `vTaskDelayUntil` with a
fixed `pdMS_TO_TICKS(100)` period (46 ms of work fits; a step that overruns the period, e.g. a
fire opening `recognise.log` through FATFS, restarts the period from now rather than running
catch-up steps back to back over the same audio, which would feed duplicate posteriors into the
hangover). The front-end's push loop takes whatever frames arrived (no minimum; `mstate.count <
KWS_N_FRAMES` only gates the warm-up), so 5 instead of 6–7 new frames per step changes nothing
there. `wake.cc` has its own loop and is untouched. The `KWS_DUTY` line in continuous recognise
mode should now read ≈ 460 ms of inference per wall second (≈ 315 before, same 46 ms step at the
longer stride) — the paper's duty figure for the always-on baseline moves accordingly; assist
mode's per-window cost is unchanged (window length is wall time).

**2. Wake-tail drop.** `ASSIST_WAKE_TAIL_MS` 450 (#64) dropped *every* command fire in the first
450 ms of the window because the "...Bus" tail read as `aus`. Measured before changing: from the
QC outputs of all sessions (`words.csv` Whisper spans of the first vocabulary word per approved
field take; the fire's position in the take = `ms − window_ms` of `sessions.csv`, i.e. the
pre-roll; "Bus" end = the wake clip's length − `WAKE_TAIL_S`, paired via `written.txt`):

| gap | n | median | p25 | p75 | min | max | share < 450 ms | share < 300 ms |
|---|---|---|---|---|---|---|---|---|
| wake fire → first command word onset (field) | 33 | **−300 ms** | −400 | +440 | −1169 | +2780 | 25/33 (0.76) | 22/33 (0.67) |
| "Bus" end → first command word onset (field) | 24 | 200 ms | 160 | 245 | 80 | 2640 | 19/24 | 19/24 |

The first command word starts a median 300 ms *before* the wake fire lands (the wake detector
fires ≈ 500 ms after "Bus" ends on these takes, not the 140 ms the #64 comment assumed; speakers
leave 160–245 ms between "Bus" and the command). With the classifier's peak at word centre +
0.5 s (E41), a first word that starts at −300 ms peaks at ≈ +300 ms and fires at +400–450 ms — inside
the tail, and earlier still at the fixed 100 ms cadence. The blanket drop therefore discards the
first word of ~3 in 4 field interactions on the device (the host replays of E41–E43 have no such
drop, so their first-word figures are unaffected). Shortening the tail to the p25
gap is meaningless (p25 is negative) and starting the stream ring at the fire would cut the first
word's onset out of the window entirely. Chosen rule, the smallest that still blocks the ghost:
inside the tail, drop only fires whose label is `aus` (12 of 17 first words in the #64 takes) or
`_unknown_` (would otherwise become an `intent_rescore` slot); no valid intent starts with either
(`intent.c`: device word first). `assist_gate_in_wake_tail(ms, label)`, host test extended.

**Device (2026-09-08 10:10, flashed this branch's build).** Boot: same model stamps, free
internal 37,799 B (−256 B vs E37 for the tick bookkeeping). Continuous `mode recognise`:
`KWS_DUTY … recogniser active 1000/1000 of wall, inference 461 ms per wall second` (E37:
320) — ten steps per second as designed; `step 45–46 ms (front-end 511–529 us over 4–5 new
frames, invoke 42.1–42.5 ms)` — 4–5 new frames per step instead of 6–7, front-end cost
unchanged. Wake loop unaffected (`step 1316 +/- 206 us`, room peaks 0.20 / 0.00). One
thing to watch: in the quiet room the continuous recogniser now logged `fired Licht`
0.46–0.62 three times in 25 s (the E37 run at the old cadence logged `_unknown_` 0.59–0.63
three times in 20 s) — hangover 2 at 100 ms is 200 ms of stable top-1 instead of 290 ms, so
room-noise runs clear it more easily. In Assistent mode this only matters inside the
post-wake command window, where the grammar still needs a device word first; the field
session's false-alarm and agreement columns are the arbiter. Left in Assistent with `field
on thresh 0.85`; no spoken test (audio embargo).

### E45 — the word cutter never split Whisper's welded compounds: Küche 1 → 20, Dach 0 → 19 real clips (2026-09-08, host-only, fix/qc-split-glued-words)

(E44 is reserved for a parallel firmware entry.) E41's per-word diagnostic had `Küche` and `Dach`
firing 1/39 in context, and `approved/words/` held **1 Küche and 0 Dach** real clips against
76 Licht — although 28 % of the read sentences contain one of them. Cause, not a model
property: Whisper large-v3 writes "Licht Küche" as one token, and the phrase→word cutter in
`run_qc` matched the prompt's tokens against each Whisper span's *first* normalised token, so a
"Lichtküche" span matched neither `licht` nor `küche`, the cutter ran off the end of the span
list and reported the whole sentence as a segmentation gap — every word of it, `Licht`
included. The content gate had already accepted the sentence (`_token_covers` understands the
glue), and the field path had its own un-welder (`_split_glued`, E17/E25) — but only for the
*label*, never for the clip timing.

**Inventory** (every `qc/<stamp>/qc.csv`; a glued token is `Licht` welded to a zone,
case-insensitive; "segmented" = the zone word appears as its own token in that set/speaker):

| set | speaker | rows | glued rows | of which | zone segmented |
|---|---|---|---|---|---|
| sentences | spk10 | 98 | 36 | Lichtküche 16, Lichtdach 15 (2 triple-glued "Lichtdachheller/-dunkler"), Lichtlesen 5 | Außen 16, Lesen 11, Küche 0, Dach 0 |
| sentences | spk02 | 102 | 4 (+1 rejected "Lichtdachteller") | Lichtküche 2, Lichtdach 2 | 0 |
| field | spk18 | 39 | 2 | Lichtküche 2 | Dach 7, Außen/Küche/Lesen 1 each |
| field | spk20 | 13 | 1 | Lichtdach 1 | 0 |
| elicit | spk20 | 4 | 1 | Lichtküche 1 | 0 |
| field spk17/spk19, negatives, wake, guided words | | | 0 | | (guided words: Außen 5, Lesen 1) |

Whisper glued `Licht` to `Küche`/`Dach`/`Lesen` in **every** read sentence that contains them
(0 segmented); `Außen` it always separated ("Licht außen"), which is why Außen had 22 clips and
the other three had 13 between them. Four stamps carry the glue: `…09-02T16-18-05Z`,
`…09-03-1923`, `…09-04-0951`, `…09-06-1404`. The spk18 "Dach auf/zu" field takes are segmented
but unparsable (no such intent) and stay unfiled by design.

**Fix** (`kws_de.qc.word_spans`, one function). The one place every word clip takes its
timing from — guided sentence, field and elicit takes all reach the same `set == "sentences"`
branch — now yields one `(token, start, end)` per *vocabulary* token: each Whisper span is
normalised, `_split_glued` peels vocabulary words off it (only a token that decomposes
completely is split, unchanged from E25), and the span is divided among the parts **in
proportion to their letter counts** — Whisper's timestamps are per word, not per character
(`mlx_whisper.transcribe(word_timestamps=True)` gives nothing finer), so the boundary inside a
compound is an estimate: "Lichtküche" 0.20–1.00 s → licht 0.20–0.60, küche 0.60–1.00.
`segment_word` then centres its 1 s window on the part as before. No gate moved: audio,
content, wake, truncation, stub and #58 rules are untouched; the matching loop simply sees
the split tokens. Test: a "Lichtküche an." transcript files Licht, Küche and an, with the
two `words.csv` spans covering the original span end to end.

**Re-run** (`kws-qc "$KWS_DATA_ROOT/data/recordings/incoming/<stamp>"`, the four stamps only;
idempotent by construction — `_clear_stamp` removes exactly the paths in that stamp's
`written.txt` before rewriting, and `approved/*/index.csv` is filtered by the same list), then
`scripts/audit-approved.py` (with transcription):

| | Küche | Dach | Lesen | Außen | Licht | words total | of which field-derived |
|---|---|---|---|---|---|---|---|
| before | 1 | 0 | 12 | 22 | 76 | 269 | 72 |
| after | **20** | **19** | **17** | 22 | **120** | **401** | 80 |

+132 word clips; Licht gained 44 because every glued sentence had lost its `Licht` clip too.
Per stamp: `…09-02` 51 → 63 word clips (0 skipped, was 12), `…09-03-1923` 258 written / 0
skipped, `…-0951` 8, `…-1404` 44; phrases/negatives/wake counts unchanged (139/43/51).
Listen-free sanity over the 70 recovered Küche/Dach/Außen/Lesen clips: every clip is exactly
1.00 s (`config.CLIP_SAMPLES`), 0 samples at the rail, RMS no lower than −26 dBFS; derived
sub-spans 0.16–0.71 s (median 0.38). Three `Dach` sub-spans are 0.16–0.20 s — the 4-letter
share of a fast "Lichtdach" span — below the 0.2 s a spoken "Dach" should take, but the 1 s
window still holds the whole compound, so no gate trips and nothing was dropped.

The audit's one finding is **pre-existing and outside this change**:
`negatives/spk18/74-46932133_001.wav` (stamp `…09-05-1202`, not re-run) re-transcribes as
"… Hey Bus." although the whole-take transcript QC cut it by reads "Hey Bus, lieber Bus
Bananenbrot." — a Whisper whole-take vs. clip disagreement of the #58 class, to be handled
separately.

**Not done here:** no dataset rebuild, no retrain. The manifest and every command-model
figure since E15 were built from the 269-clip words tree; the coordinator schedules the
rebuild. Expected effect: Küche/Dach go from ~0 training examples to spk10-dominated 20/19,
so the E41 in-context 1/39 miss for these two words is at least partly a data gap, not only
the positional-resolution ceiling E42/E43 measured.

### E47 — word cutter: vocabulary prompt, onset/valley snap, wake-floor (2026-09-08, host-only, fix/qc-cutter-onset-prompt)

Three more cutter defects, found by a fuller inventory of `approved/words/` acoustics
(clips.csv/merged.csv over all 401 clips, retranscribed) on top of E45's fix:

1. **Whisper still glues compounds most of the time.** E45's `_QC_PROMPT` only lists the
   light-level numerals + "Prozent" + the wake word — none of `config.DEVICES`/`ZONES`/`ACTIONS` —
   so "Licht Küche" keeps arriving as one token; `_split_glued` only recovers the *label*, and the
   letter-proportional split (E45) still guesses the boundary. Tested on the 47 takes E41/E45
   flagged as glued: a vocabulary `initial_prompt` (`Hey Bus. Licht Kühlschrank Heizung.
   Aufstelldach Küche Dach. …`, built from `config.DEVICES + ZONES + ACTIONS`, 3 words/sentence so
   it stays short) un-glued 47/47 with clean word spans, against 0/47 for the old prompt.
2. **First-word start clamped to 0.** Whisper's `start` reads 0 for a real onset ≈270-290 ms in —
   26 % of sentence spans overall, 68 % of the (still-)glued ones — and `whisper_transcriber`'s
   `max(0.0, start - offset)` just hides the clamp. `qc.word_spans`' letter-proportional compound
   split inherited the same blind spot.
3. **Word windows reaching into "Hey Bus".** A field/elicit word span can start before
   `split.command_start` (Whisper's timing slop on the first post-wake word), pulling wake audio
   into the 1 s window — 17 clips.
4. No per-clip content check existed to catch any of this in `approved/` after the fact.

**Fix**, all in `kws_de/qc.py`:

- `_QC_PROMPT` now built from `config.WAKE_WORD` + `config.DEVICES + ZONES + ACTIONS` (one
  mention each, ~13 words) instead of a hand-typed subset — it cannot drift from the vocabulary.
- `energy_profile` (10 ms hop / 20 ms window RMS dBFS) + `snap_words`: every Whisper word span's
  start/end is snapped to the nearest real speech onset/offset within ±200 ms; a start of 0 is
  treated as "unknown" and matched to the first onset before the span's end, not to 0 itself. A
  span snapped to a window with **no** speech energy at all is dropped (`words.csv` line + a log
  line, not filed) — this is also the guard against a prompt-echoed "Hey Bus": an echoed phrase
  over silence has nothing to snap onto and is silently dropped rather than filed as a wake clip
  or used to set `command_start`. Onsets are consumed in time order (a `floor` that only advances)
  so two Whisper spans that are equidistant from a shared bad timestamp can't both snap to the
  *same* earlier onset — found by hand on "Lichtlesenhundertprozent" (`lesen`'s raw start briefly
  snapped backward onto `Licht`'s own onset) and fixed before it shipped.
- `word_spans` now cuts a still-glued compound at the nearest energy valley (≥ 8 dB prominence,
  80 ms margin off each edge) to the old letter-proportional cut, falling back to the
  proportional cut only when no valley is found.
- `segment_word` takes a `floor_s` (the take's `command_start` for field/elicit, 0.0 — the take's
  own first sample — for a guided sentence); the window is zero-padded left of it instead of
  reading real audio, so it can no longer reach back into the wake phrase.
- `scripts/audit-approved.py` re-transcribes every `approved/words/` clip (in addition to its
  existing field-derived phrase/negative wake-phrase check) and flags one where Whisper hears
  ≥ 2 vocabulary words in it, or the label word's own span is not within ±150 ms of the clip's
  centre; printed per source (guided/sentences/field/elicit), **report-only** — chosen over
  gating because it is a per-clip heuristic (a genuinely short, compact sentence puts two real
  command words within 150 ms of each other; see "not fixable" below), not a hard defect like
  format or a duplicate index row.

Tests (`tests/test_qc.py`): the prompt carries the wake phrase and every vocabulary word exactly
once; a synthetic take with two tone bursts ("Licht" 0.20-0.45 s, "Küche" 0.55-0.90 s, real
silence between) whose Whisper span is mocked as one glued token 0.0-0.9 s cuts two clips
centred within 30 ms of the true burst centres; a field take with `command_start` = 1.2 s and a
word span starting at 1.1 s writes a clip whose first half-second (everything before 1.2 s) is
silence. All of E45's and the pre-existing tests stay green unchanged.

**Dry run** (host-only; NOT run into `approved/` — a scratch dir, per the coordinator's
data-quality-gates policy: this fix ships without a tree re-cut, which is scheduled separately).
Re-cut the 5 stamps E45's four-stamp glue list plus `…09-05-1202` (the source of tier1's 17
wake-phrase clips) touch, with the real `mlx-community/whisper-large-v3-mlx`, same acoustics as
the audit, before (current `approved/`, E45's proportional cutter) vs after (this fix):

| | before | after |
|---|---|---|
| words written (5 stamps, non-bare) | 333 | 330-332 |
| >= 2 vocabulary words heard in the clip | 269/333 (81 %) | 305-310/330-332 (92-93 %) |
| truncated at an edge (speech at the window boundary) | 239/333 (72 %) | 233-235/330-332 (70-71 %) |
| compound label-centre offset, median (IQR) | 127 ms (52-212) | 7 ms (3-48), n dropped 93→16 |
| compound clips with a usable energy-valley boundary | 93/93 | 16/16 |

The compound-offset row is the fix's direct target and it lands hard: median error 127 ms → 7 ms.
The **n** for that row drops from 93 to 16 because the vocabulary prompt (finding 1) stops most
of these compounds from glueing at all — they no longer need a boundary cut, compound or
otherwise (e.g. "Licht Dach heller" now arrives as three separate Whisper tokens). The other two
rows (>= 2 words heard, truncated at an edge) get *worse* on the aggregate numbers, not better —
see "not fixable" below, this is expected and is the correctly-centred clip finally showing a
pre-existing property of the source audio that a badly-centred clip had been randomly masking.

**Quarantine** (`quarantine_tier1.txt`/`quarantine_tier2.txt`, 27 + 77 clips from the same
inventory), re-checked against the fix at the level the original audit measured it (the
word's own span vs. the source take's energy profile — re-transcribing the tiny 1 s output clip
in isolation turned out to be an unreliable check: Whisper gives a cropped clip noticeably
different word timings than the same audio in full-take context, confirmed by hand on several
clips whose cut is provably correct against the source profile yet "wrong" by that test):

| tier | quarantined | fixed | still bad |
|---|---|---|---|
| 1 | 27 | 27 | 0 |
| 2 | 77 | 77 | 0 |

Tier 1's 17 "wake phrase heard inside field/elicit word clip" clips all stop filing wake-tainted
audio, but not all get a clean replacement: 14 get a correctly floored, wake-free re-cut; 3 (all
stamp `…09-05-1202`, speaker spk18) get **no clip at all**. Traced by hand: the vocabulary prompt
makes Whisper hear a *second*, spurious "Hey Bus" late in those specific takes — verified against
real (non-silent) energy at that point in the source audio, so the "no speech under the span"
guard does not catch it — and `field_wake_split`'s pre-existing "start after the LAST wake
phrase" rule then treats the genuine command between the two phrases as pre-wake material and
drops it. Filing nothing is still strictly better than the wake-tainted clip these were
quarantined for (the defect is gone), but the command content is lost — a residual for the
coordinator, not folded silently into "fixed": `field_wake_split`'s last-phrase rule would need
to prefer the *first* wake phrase when Whisper hears more than one, which is out of this task's
scope.

**Not fixable by cutting:** the >= 2-vocabulary-words-heard rate rising from 81 % to 92-93 % is
multi-word content genuinely present in a fast, compact sentence ("Licht Küche an" said quickly
puts two command words within the fixed 1 s window's ± 500 ms regardless of how precisely the
window is centred) — a badly off-centre window happened to dilute this by landing partly in
dead air. Same story for "truncated at an edge" staying at ~70 %: with words this close together
in a short sentence, a correctly-centred 1 s window's own edges routinely touch the neighbour.
Neither number is a cutter defect; both are the training data's own content, now visible because
the cut is finally where it should be.

**Not done here:** no re-run over the real sessions, no writes to `approved/`, no dataset
rebuild, no retrain — a scratch-dir dry run only, per this task's scope; the coordinator
schedules the tree re-cut after a policy decision on the 3-clip wake-phrase residual above.

### E48 — split `approved/words/` into guided vs. context buckets (2026-09-08, host-only, feat/context-word-bucket)

E47's own audit numbers already said this: a "correctly centred" 1 s word window from a
sentence/field/elicit take hears a second vocabulary word 92-93 % of the time (up from 81 % on
the old cutter) — not a cutter defect, the training data's real content. 80-93 % of every
`approved/words/<label>/` clip that came from a sentence/field/elicit take (as opposed to a
dedicated single-word take) carries 2-3 vocabulary words in its 1 s window regardless of how
precisely it is centred. Only guided single-word takes are structurally clean isolated words, and
they are a small minority: 51 of 401 word clips in the tree before this change. Training treated
all 401 as equivalent "word" examples with no way to tell them apart.

**Fix.** Two buckets instead of one, split by the take's own origin (`kws_de.qc.run_qc` always
knew this — it is which of the two branches wrote the clip, not a new classifier):

- `approved/words/<label>/` — guided single-word takes only (`Take.set == "words"`). Zero
  behaviour change for this path: same directory, same filename scheme, same `_next_no`
  numbering, same bookkeeping.
- `approved/context/<label>/` (new) — every word clip `run_qc`'s "sentences" branch cuts out of
  a guided sentence *or* a field/elicit take. This branch is, by construction, the only writer of
  `words.csv` rows, so "has a `words.csv` row" and "is context-origin" were always the same fact;
  the fix is a one-line destination change (`approved / "context" / lab` instead of
  `approved / "words" / lab`) plus splitting the `words_written` counter into
  `words_guided_written` / `words_context_written` for the QC report.
- `kws_de/recordings.py`'s `load_recordings` takes a `prefix` (default `rec:`); `kws_de.data.
  merge_recordings` now folds BOTH trees into the training clip dict, tagging context clips
  `ctx:<speaker>` instead of `rec:<speaker>` — inspectable and filterable (e.g. `s.startswith
  ("ctx:")`) without a schema change, while `is_tts`/`--real-weight` are untouched (a context clip
  is `~is_tts`, exactly like a guided one, so today's training treats the two buckets identically
  until a future change opts in to weighting them apart).
- `scripts/audit-approved.py` reports `words` and `context` as two entries of its `SETS` tuple
  throughout (per-label counts, per-source counts, the word-content check) instead of one
  combined `words` count; the now-redundant `word_kind`/`word_sources` helpers (guided vs.
  sentences/field/elicit, inferred from `words.csv`'s `src` path) are deleted — which tree a clip
  physically lives in already says guided vs. context.

**Migration** (`scripts/migrate-context-words.py`, one-time, run once against the real
`$KWS_DATA_ROOT`): every `words.csv` row across all 8 QC stamps names a clip that is context-origin
by the same construction argument above, so the script does not re-cut anything — it moves each
row's `out_file` from `words/<label>/` to `context/<label>/` (located by its "approved/…" path
suffix, not the historical absolute prefix, so it also works against a scratch copy) and rewrites
that row's `out_file` plus the matching line of the stamp's `written.txt`. Dry-run rehearsed first
against a scratch copy of the real tree, then run for real:

| | moved | words/ before -> after | context/ before -> after |
|---|---|---|---|
| all labels | 350 | 401 -> 51 | 0 -> 350 |

Per label (guided remaining / context moved): Licht 4/116, an 2/29, aus 1/22, Küche 0/20,
Kühlschrank 7/10, Dach 0/19, Außen 5/17, heller 0/17, dunkler 4/14, Lesen 1/16, hundert 5/12,
fünfzig 4/12, fünfundzwanzig 2/13, fünfundsiebzig 4/9, Heizung 3/9, Aufstelldach 2/4, kälter 2/3,
zu 2/2, leise 2/2, auf 1/2, wärmer 0/2. Guided-only labels sum to 51 (matches the pre-change
count exactly); `approved/`'s total `.wav` count (634, all sets) is identical before and after,
and every moved file's sha256 was re-checked post-move against its pre-move hash — 0 mismatches.
Nothing was deleted; four labels (Dach, heller, Küche, wärmer) end up with an empty
`approved/words/` directory because every clip filed for them so far happened to be context-origin
(no dedicated single-word take of them has been recorded yet).

**New session, exercising the new cutter and the new bucket split together**
(`incoming/2026-09-08-1143`, 149 takes: spk22 98 sentences + 40 negatives, spk21 5 wake + 5
field, spk20 1 field): 138 approved, 11 rejected (2 `hey-bus` too_long, 3 `hey-bus` wrong_word —
spk21's guided wake takes came out badly on this session; 5 field takes too_quiet; 1 sentence
take a mishearing: "Lichter heller" for "Licht Dach heller"). 255 context word clips written (0
guided — this session has no bare single-word takes), 2 words skipped (one segmentation gap,
`aufstelldach-zu`, both takes). 1 of 6 field takes approved (parsable: 0; 1 false alarm at both
gates — the field speaker's utterance did not carry a recognisable command). Re-transcribing all
255 new context clips (`word_content_flags`, the same >=2-vocabulary-words check the whole-tree
audit runs): 198/255 = 77.6 % multi-word content — lower than E47's 5-stamp dry-run figure
(92-93 %) but still the clear majority, consistent with E47's "not fixable by cutting" finding:
these are read *sentences* (`Licht Dach heller`, `Aufstelldach zu`, ...), and a fast three-word
command routinely puts a second real word inside the label's fixed 1 s window no matter how the
cutter centres it.

**Whole-tree audit after the new session** (`scripts/audit-approved.py`, full run, no
`--no-transcribe`, 28.5 min: 656 word clips + 190 field-derived phrase/negative clips
re-transcribed): 0 format/duration/index/speaker-dir problems, 0 wake-phrase leaks into
phrases/negatives. Per set/source (guided session vs. field session):

| set | total | guided | field |
|---|---|---|---|
| words | 51 | 51 | - |
| context | 605 | 270 | 335 |
| phrases | 236 | 101 | 135 |
| negatives | 84 | 29 | 55 |
| wake | 51 | 10 | 41 |

`words` 51 (unchanged by this session — it has no bare single-word takes); `context` 605 (350
migrated + 255 new). Word-content check, split by bucket: **guided 47/51 flagged (92 %)**,
**context 577/605 flagged (95 %)**. The guided number is a surprise worth naming rather than
burying: it is NOT evidence the 51 guided clips are bad data, and NOT something this change
caused (the guided write path is an unmodified byte-copy of the raw take, exactly as before this
PR — only the destination directory changed for the *other* branch). It is the audit's
"offcentre" half of the check assuming a word sits within ±150 ms of a clip's own centre, which
is true by construction for a `segment_word`-cut context clip but never was for a guided clip —
that is a **raw, uncut device take** (0.5-2.0 s per `DURATION_S`, not a centred 1 s window), so
the label word lands wherever the speaker happened to say it. This is a pre-existing property of
the check (the old, pre-E48 code already bucketed "guided" separately and would have shown the
same rate), not a regression from this PR; the check would need a guided-specific centring rule
to be meaningful for that bucket, which is out of this change's scope.

**Not done here:** no dataset rebuild, no retrain (`kws_de.dataset.force_rec_to_train` and
`kws_de.manifest`'s speaker/source reporting still only recognise the `rec:` prefix, not `ctx:` —
harmless today since neither runs, but a residual for whoever does the next rebuild: context
clips would currently draw into the ordinary speaker-disjoint split rather than being forced to
train like guided device recordings are). `--real-weight` semantics are unchanged; teaching it
(or a new knob) to treat `ctx:` differently is a follow-up, not part of this change.
`kws_de.eval.eval_recordings`'s isolated-word figure still reads only `approved/words/*/*.wav`
unchanged, which is now automatically a purer guided-only isolated-word accuracy number (n
shrinks from 401 mixed clips to 51 guided ones on the real tree) rather than a mix — arguably more
correct for what "isolated" claims to measure, but the sample size drop and whether `context/`
should get its own eval bucket is the coordinator's call, not made here.

### E49 — first clean retrain after the cutter+bucket fixes: does `approved/context/` actually help? (2026-09-08, host-only, exp/run7-clean-retrain)

E45/E47 fixed the word cutter (compound-splitting, onset/valley snap, wake-floor); E48 then split
`approved/words/` into guided-only (structurally clean isolated words) vs. `approved/context/`
(word clips cut from real sentence/field/elicit takes — the majority of the tree, 605 of 656 real
word clips at E48's writeup). `data/manifest_v3_qat.json` and `data/features_v3_*.npz` predate all
three fixes, so this is the first retrain to actually exercise the corrected data. Backed up
`data/features_v3_{train,val,test}.npz`, `manifest_v3.json`, `manifest_v3_qat.json`,
`raw_clips_v3.pkl` as `*.pre-run7`; this build is the new baseline, backups are not restored.

**Residual gap confirmed and fixed.** E48 flagged that `kws_de.dataset.force_rec_to_train` and
`kws_de.manifest.build_manifest` still only match the `rec:` (guided) speaker prefix, not `ctx:`
(context, added when `merge_recordings` itself was fixed in the same PR). Context clips were
never excluded from training or from real-weighting (`is_tts` only checks the `tts:` prefix, so a
`ctx:` clip is `~is_tts` exactly like a `rec:` one) — the actual gap is narrower: `--recordings-
split train`'s promise that "every device speaker's clips go to train" only held for `rec:`
speakers. A `ctx:` speaker's clips could land in val/test via the ordinary speaker-disjoint draw
and stay there, unlike a `rec:` speaker who is always forced into train. Confirmed by an A/B
rebuild on the identical approved tree: **without** the fix, `force_rec_to_train` moves 109 clips
and `val` ends up with 12,040 rows; **with** the fix it moves 145 and `val` drops to 11,788 — a
252-row difference, i.e. 252 real context rows that used to sit in val/test now correctly train
like guided recordings do. Fix (`kws_de/dataset.py`, `kws_de/manifest.py`, ~16+11 lines incl.
docstrings): both `startswith("rec:")` checks extended to `startswith(("rec:", "ctx:"))`. New test
`test_force_rec_to_train_moves_ctx_clips_too` (`tests/test_data_v3_provenance.py`).

**Rebuild** (`kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3 --seed 0`, ~50 s):
`[recordings] merged: {..., '_unknown_': 190}` — 679 real word clips across labels (51 guided +
~628 context, the E48 count of 605 plus a handful from a session ingested since its writeup) + 190
negative windows. `[recordings] moved 145 device clips into train]`. `[dataset] built seed=0:
train=44078, val=11788, test=4206` (pre-run7 stale build: train=39682, val=6433, test=11315 — not
a clean comparison, that build predates the cutter/bucket fixes). `manifest_v3.json`: train
recording=869 (was 505 pre-run7), val/test recording=0 (context clips no longer stranded), 7
distinct device speakers now in train.

**Scoreboard** (`scripts/compare_command_models.py`, deployed `86b7105e` vs `run7_s0` `ff9915d9`,
current approved tree, default stale `manifest_v3_qat.json` — i.e. it labels held-out/in-training
per the *deployed* model's own provenance, not run7's; **not apples-to-apples** for that reason).
E48 emptied `approved/words/` for spk10 and spk18 entirely (100% of their word clips are
context-origin), so the historical 4-speaker isolated-word aggregate (spk01/spk02/spk10/spk18,
n=233) can no longer be computed — `scripts/recipe-grid.py`'s hardcoded `SPEAKERS = ("spk01",
"spk02", "spk10", "spk18")` and its `passes()` (`float(row["spk18_words"]) > 0.333`) are stale in
the same way, unfixed here (out of scope: bigger than "a few lines", the coordinator's call same
as E48 left it). The closest clean like-for-like figure is the current guided-only isolated-word
set, n=74 (spk01 13 + spk02 38, matching E48's 51, plus 23 new spk22 guided clips from a session
ingested after E48's writeup):

| | deployed `86b7105e` | run7_s0 `ff9915d9` |
|---|---|---|
| own-era INT8 test acc | 0.7188 (n=11,315) | 0.5328 (n=4,206) — sizes not comparable, the fix moved real rows train-ward |
| spk01 words (n=13, in-training) | 1.000 | 0.846 |
| spk02 words (n=38, in-training) | 1.000 | 0.974 |
| spk22 words (n=23, held-out) | 0.696 | 0.696 |
| **aggregate guided-only words (n=74)** | **0.905** | **0.865** |
| false accepts (all speakers, n=85) | 0/85 | 0/85 |
| e2e phrase exact-intent (247 clips) | 0.077 (19/247) | **0.130 (32/247)** |

Guided-only word accuracy for run7_s0 is a hair *below* deployed (spk01/spk02 both regress
slightly), but real-sentence exact-intent is up 69% relative — consistent with the working
hypothesis that context-origin training rows trade a little clean-isolated-word margin for a
better match to how the words are actually spoken.

**Sentence diag** (scratch `sentence_diag.py`, 247 approved phrase clips, deployed vs run7_s0) —
the direct test of whether the corrected clips help the words E41/E45 flagged as broken:

| | deployed | run7_s0 |
|---|---|---|
| Küche fired(decoder)/expected | 1/37 (3%) | **33/37 (89%)** |
| Küche ever top-1 | 1/37 | **33/37** |
| Dach fired(decoder)/expected | 0/34 (0%) | **26/34 (76%)** |
| Dach ever top-1 | 3/34 (9%) | **33/34 (97%)** |
| zone=Küche/Dach exact / oracle(raw t=0.3) | 1/71 (0.01) / 2/71 (0.03) | 3/71 (0.04) / **20/71 (0.28)** |
| nwords=2 exact / oracle | 16/111 (0.14) / 55/111 (0.50) | 27/111 (0.24) / 67/111 (0.60) |
| nwords=3 exact / oracle | 3/136 (0.02) / 18/136 (0.13) | 5/136 (0.04) / 34/136 (0.25) |
| argmax-only oracle (all 247) | 66/247 (0.27) | 88/247 (0.36) |

Küche and Dach essentially never fired in context for the deployed model (E41's original finding)
and now fire the great majority of the time — the clearest, largest effect in this entry, and
exactly the outcome E45/E47/E48 aimed at. Overall exact-intent and oracle ceiling both improve too
(oracle ceiling nearly doubles for the Küche/Dach zone), but stay well below 1.0 for reasons E42/E43
already established (device threshold/timing, not a data problem) and are unaffected by this run.

**Deploy decision: NOT deployed.** Against the standing rule (aggregate >= 0.785, 0 false accepts,
spk18 > 0.333): the first two clear the bar (0.865 aggregate, 0/85 FA). The third is **not
measurable** — spk18 has zero guided-only word clips left after E48's bucket split, so "spk18 word
accuracy" is undefined on the current tree, not merely low. Separately, on the one like-for-like
metric that *is* computable (guided-only isolated-word aggregate, same approved tree, both
models), run7_s0 is worse than deployed (0.865 vs 0.905) — it does not clearly beat the current
model on that axis, even though it clearly beats it on the axis this whole experiment was run to
test (sentence-level exact-intent, Küche/Dach in-context recognition). Given one rule component is
unmeasurable and the directly-comparable word-accuracy metric regresses, this does not meet the
"clearly beats the current model" bar the standing policy requires for an unattended deploy.
Canonical `command_v3_w48_qat.*` and `firmware/main/gen/` are untouched (hash-verified: canonical
`.tflite` sha256 `86b7105e...` unchanged); run7_s0's export lives only under
`$KWS_DATA_ROOT/models/run7-s0/`. This is a coordinator call: redefine what "beats deployed" means
now that spk18/spk10 have no guided data (e.g. switch the gate to sentence-level exact-intent,
which run7_s0 clearly wins), or gather guided single-word takes for spk10/spk18 so the old-style
aggregate is measurable again.

**Not done here:** no firmware export/codegen/parity/Docker verification (deploy-only steps, moot
since nothing is deployed); no fix to `scripts/recipe-grid.py`'s hardcoded speaker list or
`passes()` (same E48-caused staleness, bigger than a few lines); no change to which figure the
standing deploy rule uses (flagged above, not decided here).

## Open questions

- Grouped speaker k-fold evaluation (spec §9): single split tests few independent real voices,
  effective n ≈ (speaker, word) pairs; `kws-benchmark --folds 5` over real speakers only, TTS always
  train-side, mean ± std + per-speaker table. Build after v3 once ≥ 5 speaker groups cover every
  command.
- Probabilistic slot decoding (spec §10): detector thresholds before the grammar can weigh in;
  n-best lattice parse over the existing posteriors (≤ 8 sequences per phrase, score = ∏ probs,
  accept on tau/delta, temperature-calibrated), E11 offline re-decode of the catalog eval with
  false-accept rate on negatives as the gate; catalog DP decoding only if > 5 points remain.
