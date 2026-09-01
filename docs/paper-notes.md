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
  This is an architecture-level fix (debounce that collapses consecutive identical
  fires, and/or training on synthesized multi-word transition audio), out of scope for
  this pass — flagged for a follow-up task rather than iterated on blindly.

### E4 — wake word, local training feasibility

microWakeWord (the ESPHome/HA wake engine) documented as Python-3.10 + Colab/GPU;
**shown to install and import in a local uv-managed 3.10 venv on Apple Silicon** (TF 2.21,
CPU/Metal). → custom "Hey Bus" trainable on a laptop. *(FA/hour numbers pending.)*

### E5 — TTS voice diversity (planned ablation)

Hypothesis: synthetic-data quality for KWS is dominated by **voice diversity**, not
per-voice fidelity. Setup: macOS `say` only (~9 voices) vs multi-engine
(say + Piper ~7 + Parler prompt-voices, round-robin) on the 17 zero-real words.
*(numbers pending — compare catalog accuracy of both trainings)*

## 4. Method details worth a figure

- Two-stage always-on architecture (LikeC4 diagrams in `docs/likec4/` — reuse for slides).
- Golden-vector MFCC contract: host (librosa) and device (esp-dsp) front-ends pinned
  bit-level by a fixed-input fixture test — kills the classic silent train/deploy mismatch.
- Budget gates as CI: "fits the MCU" as a unit test (≤500 KB / ≤3 M MACs / INT8-only ops).
- Speaker-disjoint splits everywhere; TTS clips split by voice+rate combo (a synthetic
  "speaker") so voices never straddle train/test.
- Streaming detector: posterior smoothing + threshold + refractory debounce
  (found + fixed a real off-by-one: decrement-then-check let a sustained word re-fire).

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
