# Voice-cloned TTS for training data — experiment design

**Status: FAIL (2026-09-29, paper E64) — closed; engine code removed.** Was: approved as a bounded spike (brainstorm 2026-09-29; private use, so a
non-commercial TTS licence is acceptable). **Question:** does synthesizing the
vocabulary in the *real speakers' cloned voices* beat anonymous TTS on the real
scoreboard? **Kept code if it wins:** the clone-synthesis engine + `clone:`
provenance. **Discarded if it loses:** everything but the paper entry.

## Why

Real-speaker data is the bottleneck (51 guided isolated-word clips, 3 speakers).
TTS-only bootstrap classes (light compounds, scene triggers) sit at ~0.25–0.35, and
the wake model once learned "TTS-vs-real" as its feature. Today's TTS is Piper
(`de_DE-*`, the noisy `mls#N` pool) + macOS `say`: small CPU models, strangers'
voices. thinky's RTX 3090 Ti makes zero-shot voice cloning practical: every word,
compound, scene trigger and sentence in the actual users' voices, from a few seconds
of recordings that already exist. That attacks the thin-data problem and the
TTS-vs-real gap directly, instead of adding more strangers.

## Design

### Engine

XTTS-v2 (zero-shot, German, the most proven cloner; CPML non-commercial —
acceptable for private use). F5-TTS is the fallback if XTTS-v2 quality on short
German words disappoints. Runs on thinky only (GPU); Piper stays the permissive
default engine for anonymous voices, unchanged.

### References

The guided single-word takes of the deploy-scoreboard speakers (spk01, spk02,
spk22) — the cleanest audio, and the speakers whose real figures the rule scores,
so the effect is measurable. ≥ 6 s of reference audio per speaker, concatenated
from their approved clips. Speakers are addressed only by `spkNN`.

### Synthesis + provenance

Synthesize, per cloned speaker: every `COMMAND_LABELS` word, the pending light
compounds and scene triggers (`LIGHT_COMPOUND_PROMPTS`, `SCENE_TRIGGER_PROMPTS`), and
the sentence catalog. Gate every clip with `kws-tts-check` (faster-whisper on
thinky: language ID + transcript). File under a new **`clone:`** provenance next to
`tts:`/`rec:`/`ctx:` — cloned clips are augmentation and must **never** enter
`approved/`, `approved/context/`, or the real-voice scoreboard, exactly like TTS
played through a speaker. `--real-weight` continues to upweight `rec:` only.

### Experiment

Deployed recipe (`--width 48 --qat --qat-epochs 20 --real-weight 3`, 40 epochs),
seeds 0 and 1, on thinky:

| arm | data |
|---|---|
| control | current build (Piper + `say` TTS) |
| clone | control + `clone:` clips for the 3 speakers |

Judge by the standing deploy rule on the **real** guided-only scoreboard plus false
accepts, against `86b7105e`. Two seeds because E40's seed band is ±0.011 on the
aggregate. Also read the bootstrap classes' real-clip accuracy once real "Gute
Nacht"/compound takes exist — lifting exactly those is the point of cloning.

### Pass / fail

Pass: the clone arm beats control on the guided-only aggregate in **both** seeds
with 0 false accepts, and beats `86b7105e`. Then `clone:` becomes a standard
augmentation source and the engine is documented in `docs/dev-setup.md`. Fail:
paper entry only; the engine code is removed.

## Prerequisites (done first, independent of this experiment)

- thinky's `features_v3` npz rebuilt at the current 23-class vocabulary from
  `approved/` (the SSD copy was run 8's stale 26-class build; see E61).
- `kws-train` label-range guard (E61), so a vocabulary/npz mismatch fails loudly on
  CUDA as it already did on CPU.
- The deployed recipe reproducing its Mac figures on thinky (val ≈ 0.65).

## Status (2026-09-29): FAIL

Prerequisites done (E61/E62). Spike (E63): XTTS-v2 clones the three speakers but pads one-word
texts with babble from the reference audio; first-utterance trim + strict transcript match +
several takes gave 139 usable command-word clones. Experiment (E64): clone 63/74 (0 false
accepts) and 64/74 (1) against control 65/74 (2) and 62/74 (1) on the guided-only scoreboard,
deployed `86b7105e` 67/74 (0). The clone arm does not beat control in both seeds and no run beats
the deployed model, so by the rule above the engine code and the `clone:` provenance are removed;
the paper entries E63/E64 are what is kept. The code is commit `73d1e38` in PR #115's history.
Post-mortem: `docs/superpowers/plans/2026-09-29-voice-clone-handover.md`.

Not tested, and the part of the question still open: TTS-only classes (light compounds, scene
triggers), which have no real takes to score against; the F5-TTS fallback; the owner's ear check
of the clones.

## Out of scope

Cloning speakers other than the three scoreboard speakers; using cloned audio for
the wake model; any change to the deploy rule.
