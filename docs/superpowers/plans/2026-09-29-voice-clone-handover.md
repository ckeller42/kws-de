# Voice-clone experiment — post-mortem (2026-09-29, branch `exp/voice-clone`)

Spec: `docs/superpowers/specs/2026-09-29-voice-clone-tts-design.md`. Paper: E63 (spike), E64
(experiment). **Verdict: FAIL.** The branch is docs-only; the code is commit `73d1e38` in PR #115's
history (`scripts/xtts_clone.py`, the `clone:` tree in `data.py`/`dataset.py`/`manifest.py`,
`scripts/e64-score.py`, tests).

## Result

Guided-only isolated words (n = 74), false accepts (n = 85), deployed recipe, training seeds 0/1:

| | words | false accepts |
|---|---|---|
| deployed `86b7105e` | 67 | 0 |
| control s0 / s1 | 65 / 62 | 2 / 1 |
| clone s0 / s1 | 63 / 64 | 0 / 1 |

Clone minus control is −2 and +2 clips; the two control runs differ by 3 on identical data. No
effect on the scoreboard, and nothing beats the deployed model.

## What went wrong on the way (all in E64)

- **The arms were not comparable as wired.** `clone:` rows were flagged real (`is_tts` short and
  shifted, `--real-weight` tripling the wrong rows), the clone speaker ids re-dealt the whole
  speaker split, and clones inside the label lists shifted the augmentation draws. Fixed before
  the runs; verified on the npz (val/test hashes equal, control's train rows byte-identical in the
  clone build).
- **The build command had no `--cache`.** `kws-dataset build --prefix features_v3` reads
  `raw_clips_merged.pkl` (the v2 cache, `say` voices only), not `raw_clips_v3.pkl`. E62's val
  0.883 comes from that build and is not comparable to a v3 build's 0.68; on real val rows both
  give 0.72–0.75. The experiment used `raw_clips_v3.pkl`.
- **A first session was interrupted mid-build** with the clone tree already moved aside; the
  second found `clone.off` and no partial output, and carried on from there.

## If this is picked up again

- Test what cloning was meant for: classes with TTS only (light compounds, scene triggers), once
  real takes of them exist to score against. For words the speakers have already recorded, their
  own takes (tripled by `--real-weight`) already carry what a clone adds.
- Get the owner's ear check first — still pending when the runs were made.
- Keep synthetic additions out of the split draw and out of the split's RNG stream, and check the
  npz (hashes, row identity, flag counts) before training.
- Always pass `--cache raw_clips_v3.pkl`.

## State on thinky

- Clone clips: `~/kws-data/data/recordings/clone.off/words/` (139 clips; no build reads that path).
- XTTS venv and all synthesis output: `~/xtts-spike/` (coqui-tts 0.27, `transformers<5`).
- Builds `data/features_v3_{control,clone}_*.npz`, models `models/e64/<arm>_s<seed>/` and
  `models/e64_*`, logs and `e64-scores.json` in `archive/e64-logs/`.
- `data/features_v3_*.npz` is E62's `say`-only build; rebuild with `--cache raw_clips_v3.pkl`
  before using it as a v3 baseline.
- The GPU is shared with an LLM server (E62); it was not running during these runs.
