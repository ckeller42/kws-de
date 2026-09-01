# kws-de Evaluation Report

## Headline: real-speech accuracy (MultiNet-comparable)

Evaluated on the MSWC-validated subset only — **Licht, Kühlschrank, Heizung** + `_unknown_` + `_silence_` — restricted to test rows built from REAL MSWC speech (TTS-synthesized rows excluded, including Heizung's TTS top-up). **Camping and Wasser are excluded from this number** — see the full-model table below. n=579 held-out real-speech examples (mixed 20/10/0 dB SNR).

| Model | Accuracy |
|---|---|
| Float (keras) | 0.907 |
| **INT8 (shipped)** | **0.879** |

## Full-model snapshot (`kws_de.eval.render_report`)

# kws-de Evaluation Report

**Accuracy:** 0.919

## SNR sweep

| SNR (dB) | Accuracy |
|---|---|
| 40 | 0.932 |
| 20 | 0.952 |
| 10 | 0.935 |
| 0 | 0.846 |
(All 7 classes, INT8, command-only SNR sweep — see the full breakdown below for why this overall number mixes real and synthetic speech.)

## Full 5-word model — overall + per-command accuracy (held-out test set, mixed SNRs)

**Camping and Wasser are TTS-augmented (synthetic speech)** — Camping had only 22 real MSWC clips, Wasser had 0, so their rows below reflect synthetic-voice performance and must NOT be read as real-speech accuracy. Heizung is a real+TTS mix (120 real + 180 TTS, topped up to 300).

**Overall accuracy — float:** 0.936

**Overall accuracy — INT8 (shipped):** 0.919

| Label | Float accuracy | INT8 accuracy | Data source (clips) |
|---|---|---|---|
| Licht | 0.810 | 0.788 | real (300) |
| Kühlschrank | 0.956 | 0.933 | real (300) |
| Camping | 0.963 | 0.952 | real+TTS mix (22 real + 278 TTS) |
| Heizung | 0.926 | 0.910 | real+TTS mix (120 real + 180 TTS) |
| Wasser | 0.994 | 0.994 | TTS-only (300) |
| _unknown_ | 0.977 | 0.977 | real (600) |
| _silence_ | 0.989 | 0.920 | synthetic (noise) |

**Unknown false-accept (float):** 0.023
**Unknown false-accept (INT8):** 0.023

## Confusion matrix — INT8 model (rows=true, cols=predicted)

| true \ pred | Licht | Kühlschrank | Camping | Heizung | Wasser | _unknown_ | _silence_ |
|---|---|---|---|---|---|---|---|
| Licht | 149 | 11 | 2 | 7 | 0 | 4 | 16 |
| Kühlschrank | 2 | 126 | 1 | 2 | 0 | 2 | 2 |
| Camping | 0 | 1 | 180 | 5 | 3 | 0 | 0 |
| Heizung | 3 | 6 | 3 | 172 | 3 | 1 | 1 |
| Wasser | 0 | 0 | 0 | 1 | 173 | 0 | 0 |
| _unknown_ | 1 | 0 | 1 | 0 | 0 | 85 | 0 |
| _silence_ | 1 | 1 | 0 | 0 | 2 | 3 | 80 |

## SNR sweep — command-only accuracy (float vs INT8)

Built from ALL held-out command clips (real + TTS per the source column above), re-augmented fresh at each SNR; excludes `_unknown_`/`_silence_`.

| SNR (dB) | Float accuracy | INT8 accuracy |
|---|---|---|
| clean (~40dB) | 0.952 | 0.932 |
| 20 | 0.969 | 0.952 |
| 10 | 0.952 | 0.935 |
| 0 | 0.860 | 0.846 |

## Model budget

- Model size (tflite): 19256 bytes (budget 500000)
- MACs: 2069984 (budget 3000000)
- Full INT8: True
- Ops: CONV_2D, DELEGATE, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX
- Within all budgets: True

## MSWC real-clip counts obtained (of the 300/word target, 600 for _unknown_)

| Label | Real (MSWC) | TTS-added | Total |
|---|---|---|---|
| Licht | 300 | 0 | 300 |
| Kühlschrank | 300 | 0 | 300 |
| Camping | 22 | 278 | 300 |
| Heizung | 120 | 180 | 300 |
| Wasser | 0 | 300 | 300 |
| _unknown_ | 600 | 0 | 600 |

Noise source: ESC-50 (2000 environmental-sound clips, resampled to 16 kHz). TTS source: macOS `say`, German voices (Anna, Eddy, Flo, Grandma, Grandpa, Reed, Rocko, Sandy, Shelley) at rates 120-280 wpm, varied punctuation, used only to top up Camping/Heizung/Wasser to 300 clips since MSWC German had far fewer (or zero) real recordings of those words.

## Comparison to MultiNet

MultiNet's English command-recognition accuracy is reported at roughly 85-95% on clean speech. The comparable number here is the **headline real-speech INT8 accuracy: 0.879 (87.9%)** on Licht/Kühlschrank/Heizung + _unknown_/_silence_, real MSWC speech only, mixed 20/10/0 dB SNR (harder than MultiNet's clean-speech condition). The full 5-word model's INT8 accuracy is 0.919, but that number is inflated/deflated by two TTS-only or TTS-heavy classes (Camping, Wasser) and should not be quoted as a real-speech comparison to MultiNet.
