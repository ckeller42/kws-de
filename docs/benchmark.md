# KWS architecture benchmark

Phase-1 comparison of the device-runnable architecture zoo on the frozen v2 dataset (`kws_de.dataset.load_split`, `data/manifest.json` seed=0). **Isolated** = INT8 test-set word accuracy. **Catalog** = full-intent accuracy over the enumerated command catalog (`kws_de.eval.run_catalog_eval`: TTS-synthesized, clean/no noise, streaming detector + grammar parse), 3 voices (Anna, Eddy, Flo) -- reduced from the full 9-voice `_TTS_VOICES` set (and the 4-voice eval used in `docs/eval-report-v2.md`) to keep 3 architectures x ~49 catalog entries tractable; treat Catalog as indicative, not the precision-tuned number. **Params/MACs/INT8** are on-device cost (`kws_de.budgets`); **Budget** = fits the Phase-1 budgets (model <= 500,000 bytes, MACs <= 3,000,000, full INT8 I/O).

KWT (Keyword Transformer) is **reference-only**: it INT8-exports but `MultiHeadAttention`/`LayerNormalization` lower to BATCH_MATMUL/TRANSPOSE/GATHER/CONCATENATION/TILE plus float DEQUANTIZE/QUANTIZE bridges, outside the TFLM op set the other three architectures stay within -- not device-runnable, so it is excluded from this benchmark run (see `kws_de/architectures/kwt.py`, `tests/test_architectures.py::test_kwt_is_not_tflm_device_runnable`).

Config: epochs=30, seed=0, dataset manifest seed=0 (`data/manifest.json`).

| Architecture | Isolated | Catalog | Params | MACs | INT8 | Budget |
|---|---|---|---|---|---|---|
| ds_cnn | 0.834 | 0.544 | 5,879 | 2,070,496 | 20,216 | yes |
| bc_resnet | 0.773 | 0.102 | 4,919 | 1,390,584 | 31,152 | yes |
| matchboxnet | 0.903 | 0.245 | 12,957 | 467,070 | 42,840 | yes |
