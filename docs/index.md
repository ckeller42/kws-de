# kws-de

German keyword-spotting models for the **ESP32-S3** — offline voice control for a camper.
A tiny INT8 DS-CNN (TFLite-Micro + ESP-NN) trained on public German speech data, plus a
wake word ("Hey Bus"), a streaming keyword detector, and a grammar that turns keyword
sequences into validated intents.

```{toctree}
:maxdepth: 2

architecture
eval-report
eval-report-v2
```

- Source & specs: <https://github.com/ckeller42/kws-de>
- v1 measured results: {doc}`eval-report` — 91.1 % real-speech INT8 accuracy
- v2 (wake + slot commands): {doc}`eval-report-v2`
