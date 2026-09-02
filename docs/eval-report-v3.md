<!-- kws-eval:recordings:start -->

Model measured: `models/command_v3.tflite` (sha256 `1f7aca5fae6d`), evaluated 2026-09-02T21:44:43.865452+00:00.

Training manifest checked: `data/manifest_v3.json`, built 2026-09-02T21:37:45.749278+00:00.

Match is speaker-level, not per-clip: a clip counts `user-customised, in-training` if its speaker has ANY word/negative clip in this manifest's train split, including takes recorded/QC-approved after 2026-09-02T21:37:45.749278+00:00 — those clips were never actually seen by the current model. Re-run `kws-dataset build` + retrain to make the match exact again. Phrase clips are always `held-out` (never used for training).

## user-customised, in-training

61 clips across 2 speakers.

| speaker | isolated words n | acc | e2e phrases n | intent acc | negatives n | false-accept rate |
|---|---|---|---|---|---|---|
| spk01 | 13 | 0.538 | 0 | nan | 0 | nan |
| spk02 | 38 | 0.553 | 0 | nan | 10 | 0.000 |

## held-out

4 clips across 1 speakers.

| speaker | isolated words n | acc | e2e phrases n | intent acc | negatives n | false-accept rate |
|---|---|---|---|---|---|---|
| spk02 | 0 | nan | 4 | 0.000 | 0 | nan |

<!-- kws-eval:recordings:end -->
