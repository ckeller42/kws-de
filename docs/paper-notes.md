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
which no trigger exists.

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

## Open questions

- Grouped speaker k-fold evaluation (spec §9): single split tests few independent real voices,
  effective n ≈ (speaker, word) pairs; `kws-benchmark --folds 5` over real speakers only, TTS always
  train-side, mean ± std + per-speaker table. Build after v3 once ≥ 5 speaker groups cover every
  command.
- Probabilistic slot decoding (spec §10): detector thresholds before the grammar can weigh in;
  n-best lattice parse over the existing posteriors (≤ 8 sequences per phrase, score = ∏ probs,
  accept on tau/delta, temperature-calibrated), E11 offline re-decode of the catalog eval with
  false-accept rate on negatives as the gate; catalog DP decoding only if > 5 points remain.
