# kws-de v2 Evaluation Report — command catalog (Task 8)

**Method:** every entry below is a VALID intent enumerated from `config.DEVICE_ACTIONS` (+ zones for `Licht` only). For each entry, the device/zone/action WORDS are TTS-synthesized (macOS `say`, several German voices: Anna, Eddy, Flo, Grandma) and concatenated with silence gaps into one continuous utterance — the audio a streaming detector would see. That audio is run through the FULL pipeline: sliding-window `kws_de.features.mfcc` -> the trained INT8 command model -> `kws_de.stream.KeywordStream` -> `kws_de.grammar.parse` -> `Intent`, compared against the true intent. **All catalog phrases are TTS, not real recorded commands** — this measures the streaming+grammar composition end-to-end, not raw word-recognition accuracy on natural speech (see the per-word real/TTS provenance table for how much of the underlying vocabulary is real MSWC speech vs TTS-filled).

## Overall full-intent accuracy: 0.362

(152 trials = 38 catalog entries x 4 voices)

## Per-slot accuracy (clean)

- Device: 0.362
- Action: 0.362
- Zone (Licht only): 0.156

## Command catalog — per-entry full-intent accuracy

| Device | Zone | Action | Accuracy | Trials |
|---|---|---|---|---|
| Licht | - | an | 0.250 | 4 |
| Licht | Küche | an | 0.000 | 4 |
| Licht | Dach | an | 0.000 | 4 |
| Licht | Außen | an | 0.250 | 4 |
| Licht | Lesen | an | 0.250 | 4 |
| Licht | - | aus | 0.250 | 4 |
| Licht | Küche | aus | 0.250 | 4 |
| Licht | Dach | aus | 0.000 | 4 |
| Licht | Außen | aus | 0.250 | 4 |
| Licht | Lesen | aus | 0.000 | 4 |
| Licht | - | heller | 0.250 | 4 |
| Licht | Küche | heller | 0.250 | 4 |
| Licht | Dach | heller | 0.250 | 4 |
| Licht | Außen | heller | 0.250 | 4 |
| Licht | Lesen | heller | 0.250 | 4 |
| Licht | - | dunkler | 0.250 | 4 |
| Licht | Küche | dunkler | 0.000 | 4 |
| Licht | Dach | dunkler | 0.000 | 4 |
| Licht | Außen | dunkler | 0.250 | 4 |
| Licht | Lesen | dunkler | 0.250 | 4 |
| Kühlschrank | - | an | 0.250 | 4 |
| Kühlschrank | - | aus | 0.250 | 4 |
| Kühlschrank | - | leise | 0.250 | 4 |
| Heizung | - | an | 1.000 | 4 |
| Heizung | - | aus | 1.000 | 4 |
| Heizung | - | wärmer | 1.000 | 4 |
| Heizung | - | kälter | 1.000 | 4 |
| Aufstelldach | - | auf | 0.250 | 4 |
| Aufstelldach | - | zu | 0.000 | 4 |
| Campingmodus | - | an | 0.000 | 4 |
| Campingmodus | - | aus | 0.750 | 4 |
| USB | - | an | 1.000 | 4 |
| USB | - | aus | 1.000 | 4 |
| Wasser | - | an | 0.000 | 4 |
| Wasser | - | aus | 0.000 | 4 |
| Energie | - | Eco | 0.750 | 4 |
| Energie | - | Max | 0.750 | 4 |
| Energie | - | Normal | 1.000 | 4 |

## SNR sweep — overall full-intent accuracy

(2 voices per entry: Anna, Eddy)

| SNR (dB) | Full-intent accuracy |
|---|---|
| clean | 0.362 |
| 20 | 0.368 |
| 10 | 0.276 |
| 0 | 0.158 |

## Command model budget (INT8)

- Model size: 20392 bytes (budget 500000)
- Full INT8: True
- Ops: CONV_2D, DELEGATE, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX

## Wake model ("Hey Bus") budget

Not trained in this run — microWakeWord's trainer requires Piper sample generation (a separate ~cloned repo + TTS voice checkpoint, not present locally) plus several GB of pre-generated negative/ambient spectrogram feature sets from HuggingFace (`kahrendt/microwakeword`: dinner_party, dinner_party_eval, no_speech, speech). Neither was fetched in this run (out of scope for the time budget here). The local 3.10 venv with `microwakeword` installed and importable is proven (`train/mww/setup.sh`), and the runtime integration path — `kws_de.wake.WakeDetector` + `load_wake_tflite` + `kws_de.budgets.check_wake_budgets` — is unit-tested against a stand-in INT8 tflite, but no real `hey_bus.tflite` was produced.

## Vocabulary provenance (real MSWC vs TTS-added)

| Word | Real (MSWC) | TTS-added | Total |
|---|---|---|---|
| Licht | 300 | 0 | 300 |
| Kühlschrank | 300 | 0 | 300 |
| Heizung | 120 | 180 | 300 |
| Aufstelldach | 0 | 300 | 300 |
| Campingmodus | 0 | 300 | 300 |
| USB | 0 | 300 | 300 |
| Wasser | 300 | 0 | 300 |
| Energie | 0 | 300 | 300 |
| Küche | 0 | 300 | 300 |
| Dach | 0 | 300 | 300 |
| Außen | 158 | 142 | 300 |
| Lesen | 0 | 300 | 300 |
| an | 0 | 300 | 300 |
| aus | 300 | 0 | 300 |
| auf | 300 | 0 | 300 |
| zu | 0 | 300 | 300 |
| heller | 0 | 300 | 300 |
| dunkler | 0 | 300 | 300 |
| wärmer | 0 | 300 | 300 |
| kälter | 0 | 300 | 300 |
| leise | 0 | 300 | 300 |
| Eco | 0 | 300 | 300 |
| Max | 0 | 300 | 300 |
| Normal | 0 | 300 | 300 |

TTS source for both training-data fill and catalog synthesis: macOS `say`, German voices (Anna, Eddy, Flo, Grandma, Grandpa, Reed, Rocko, Sandy, Shelley) at rates 120-280 wpm.

## vs v1 / MultiNet

v1 measured word-level command accuracy on 5 devices (no grammar, no streaming composition). v2 measures a strictly harder, end-to-end task: full intent (device+zone+action) recovered from continuous synthesized speech through the streaming detector and grammar — a single wrong/missed word anywhere in the phrase fails the whole entry. The headline number above is therefore not directly comparable to v1's per-word accuracy or to MultiNet's isolated-word numbers; it is the number that matters for actually using the assistant.
