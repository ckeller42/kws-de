<!-- kws-eval:recordings:start -->

Model measured: `models/command_v3_qat.tflite` (sha256 `f985f282edcc`), evaluated 2026-09-03T17:56:39.852366+00:00.

Training manifest checked: `data/manifest_v3_qat.json`, built 2026-09-02T21:37:45.749278+00:00.

Match is speaker-level, not per-clip: a clip counts `user-customised, in-training` if its speaker has ANY word/negative clip in this manifest's train split, including takes recorded/QC-approved after 2026-09-02T21:37:45.749278+00:00 — those clips were never actually seen by the current model. Re-run `kws-dataset build` + retrain to make the match exact again. Phrase clips are always `held-out` (never used for training).

## user-customised, in-training

61 clips across 2 speakers.

| speaker | isolated words n | acc | e2e phrases n | intent acc | negatives n | false-accept rate |
|---|---|---|---|---|---|---|
| spk01 | 13 | 0.538 | 0 | nan | 0 | nan |
| spk02 | 38 | 0.605 | 0 | nan | 10 | 0.000 |

## held-out

266 clips across 2 speakers.

| speaker | isolated words n | acc | e2e phrases n | intent acc | negatives n | false-accept rate |
|---|---|---|---|---|---|---|
| spk02 | 0 | nan | 4 | 0.000 | 0 | nan |
| spk10 | 146 | 0.678 | 97 | 0.082 | 19 | 0.053 |

<!-- kws-eval:recordings:end -->
