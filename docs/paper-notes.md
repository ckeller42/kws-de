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

### E3 — v2 grounded catalog model (26 classes) + end-to-end catalog eval

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

### E5 — TTS voice diversity (planned ablation)

Hypothesis: synthetic-data quality for KWS is dominated by **voice diversity**, not
per-voice fidelity. Setup: macOS `say` only (~9 voices) vs multi-engine
(say + Piper ~7 + Parler prompt-voices, round-robin) on the 17 zero-real words.
*(numbers pending — compare catalog accuracy of both trainings)*


### E6 — sim-to-real gap (planned, needs the physical CoreS3)

Estimated vs MEASURED on-device performance (latency/arena/CPU/power) and clean-corpus vs
real-mic accuracy. All numbers so far are synthetic/clean-corpus; the CoreS3's dual-MEMS +
ES7210 + ESP-SR 2-mic AFE and the van acoustics are an unmodeled domain. Spec:
`docs/superpowers/specs/2026-09-01-on-device-hw-mic-followup-design.md`. Expected result: real
accuracy below synthetic eval; quantifying that gap is the contribution.

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
