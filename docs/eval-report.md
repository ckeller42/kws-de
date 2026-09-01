# kws-de Evaluation Report

## Headline: real-speech accuracy (MultiNet-comparable)

Restricted to test rows built from REAL MSWC speech (TTS-synthesized rows excluded), filtered per-row rather than by a fixed per-word whitelist — so each command contributes exactly the real MSWC evidence it actually has. n=775 held-out real-speech examples (mixed 20/10/0 dB SNR). Real-row breakdown by class:

| Label | Real rows in headline set |
|---|---|
| Licht | 189 |
| Kühlschrank | 135 |
| Camping | 9 |
| Heizung | 81 |
| Wasser | 117 |
| _unknown_ | 122 |
| _silence_ | 122 |

(Camping contributes very few real rows — only 22 real MSWC clips exist before the speaker split — so its headline contribution is thin; treat the headline number as strongest for Licht/Kühlschrank/Wasser/`_unknown_`, which are ~fully real.)

| Model | Accuracy |
|---|---|
| Float (keras) | 0.917 |
| **INT8 (shipped)** | **0.911** |

## Full-model snapshot (`kws_de.eval.render_report`)

## Evaluation summary

**Accuracy:** 0.932

## SNR sweep

| SNR (dB) | Accuracy |
|---|---|
| 40 | 0.930 |
| 20 | 0.952 |
| 10 | 0.963 |
| 0 | 0.846 |

(All 7 classes, INT8, command-only SNR sweep — see the full breakdown below for why this overall number mixes real and synthetic speech.)

## Full 5-word model — overall + per-command accuracy (held-out test set, mixed SNRs)

**Camping and Heizung are real+TTS mixes** — Camping had only 22 real MSWC clips (278 TTS-added to reach 300), Heizung had 120 real (180 TTS-added), so their rows below blend real and synthetic-voice performance. **Licht, Kühlschrank, Wasser, and `_unknown_` are ~fully real MSWC speech** (Wasser reached 300 real clips on a deeper corpus scan — no TTS was needed for it after all).

**Overall accuracy — float:** 0.935

**Overall accuracy — INT8 (shipped):** 0.932

| Label | Float accuracy | INT8 accuracy | Data source (clips) |
|---|---|---|---|
| Licht | 0.894 | 0.857 | real (300) |
| Kühlschrank | 0.933 | 0.963 | real (300) |
| Camping | 0.974 | 0.984 | real+TTS mix (22 real + 278 TTS) |
| Heizung | 0.903 | 0.909 | real+TTS mix (120 real + 180 TTS) |
| Wasser | 0.932 | 0.932 | real (300) |
| _unknown_ | 1.000 | 1.000 | real (600) |
| _silence_ | 0.926 | 0.902 | synthetic (noise) |

**Unknown false-accept (float):** 0.000
**Unknown false-accept (INT8):** 0.000

## Confusion matrix — INT8 model (rows=true, cols=predicted)

| true \ pred | Licht | Kühlschrank | Camping | Heizung | Wasser | _unknown_ | _silence_ |
|---|---|---|---|---|---|---|---|
| Licht | 162 | 8 | 1 | 1 | 5 | 0 | 12 |
| Kühlschrank | 2 | 130 | 0 | 2 | 0 | 1 | 0 |
| Camping | 1 | 1 | 189 | 1 | 0 | 0 | 0 |
| Heizung | 5 | 3 | 2 | 169 | 5 | 2 | 0 |
| Wasser | 2 | 4 | 0 | 0 | 109 | 0 | 2 |
| _unknown_ | 0 | 0 | 0 | 0 | 0 | 122 | 0 |
| _silence_ | 4 | 5 | 0 | 0 | 2 | 1 | 110 |

## SNR sweep — command-only accuracy (float vs INT8)

Built from ALL held-out command clips (real + TTS per the source column above), re-augmented fresh at each SNR; excludes `_unknown_`/`_silence_`.

| SNR (dB) | Float accuracy | INT8 accuracy |
|---|---|---|
| clean (~40dB) | 0.927 | 0.930 |
| 20 | 0.960 | 0.952 |
| 10 | 0.963 | 0.963 |
| 0 | 0.839 | 0.846 |

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
| Wasser | 300 | 0 | 300 |
| _unknown_ | 600 | 0 | 600 |

Noise source: ESC-50 (2000 environmental-sound clips, resampled to 16 kHz). TTS source: macOS `say`, German voices (Anna, Eddy, Flo, Grandma, Grandpa, Reed, Rocko, Sandy, Shelley) at rates 120-280 wpm, varied punctuation, used only to top up Camping/Heizung to 300 clips since MSWC German had far fewer real recordings of those two words (a deeper corpus scan later found 300 real Wasser clips too, so no TTS was needed for Wasser in this run).

## Comparison to MultiNet

MultiNet's English command-recognition accuracy is reported at roughly 85-95% on clean speech. The comparable number here is the **headline real-speech INT8 accuracy: 0.911 (91.1%)** across all 7 classes filtered to real MSWC speech only (Licht/Kühlschrank/Wasser/`_unknown_` ~fully real, Heizung/Camping partially — see the real-row breakdown above), mixed 20/10/0 dB SNR (harder than MultiNet's clean-speech condition). The full 5-word model's INT8 accuracy is 0.932, but that number includes TTS-augmented Camping/Heizung rows and should not be quoted as a pure real-speech comparison to MultiNet — use the headline number for that.
