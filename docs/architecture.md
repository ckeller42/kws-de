# Architecture

German keyword spotting for the ESP32-S3, fully offline, in two stages: a tiny always-on
**wake detector** ("Hey Bus") gates a **streaming command recogniser** whose keyword events a
pure **grammar** composes into a validated intent — `Licht Küche an` →
`Intent(device=Licht, zone=Küche, action=an)`.

The algorithmic background (MFCC, DS-CNN, INT8 quantization, streaming KWS, noise
augmentation) is documented with citations in the design specs under
`docs/superpowers/specs/` in the repository.

## Training pipeline (Mac)

Everything reproducible from code: real clips from MSWC, synthetic clips from a multi-engine
German TTS stack for words MSWC lacks, SNR noise augmentation from ESC-50, a
speaker/voice-disjoint split, DS-CNN training, full-INT8 export, and resource-budget gates
that prove "fits the ESP32" without hardware.

```{likec4-view} pipeline
:title: Training pipeline
:height: 560px
```

## Deployment

The laptop trains and exports; CI re-runs tests and budget gates on every PR; the Pi edge box
holds the ESP-IDF toolchain and flashes the CoreS3; the CoreS3 runs everything offline.

```{likec4-view} deployment
:title: Deployment
:height: 480px
```

## On-device runtime (ESP32-S3)

The always-on path is only mics → AFE → MFCC → wake model (tiny, tuned for low
false-accepts/hour). The heavier command path — recogniser → `KeywordStream` (posterior
smoothing + debounce) → grammar — runs solely inside the ~3 s post-wake window.

```{likec4-view} device
:title: On-device runtime
:height: 560px
```

### Command sequence

```{likec4-view} voiceCommand
:mode: sequence
:height: 620px
```

### Training-run sequence

```{likec4-view} trainingRun
:mode: sequence
:height: 480px
```

## Command catalog

Commands are single German keywords composed by a device-specific grammar
(`device → zone? → action`, zones only for `Licht`):

| Device | Actions | Zones |
|---|---|---|
| Licht | an, aus, heller, dunkler | Küche, Dach, Außen, Lesen |
| Kühlschrank | an, aus, leise | — |
| Heizung | an, aus, wärmer, kälter | — |
| Aufstelldach | auf, zu | — |
| Campingmodus | an, aus | — |
| USB | an, aus | — |
| Wasser | an, aus | — |
| Energie | Eco, Max, Normal | — |

Invalid combinations ("Aufstelldach an", "Heizung Küche …") are rejected by the grammar, not
learned by the model.
