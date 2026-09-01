# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Real MSWC-de + ESC-50 data pipeline (`kws_de.data._fetch_and_cache`), macOS-`say`
  TTS top-up for command words MSWC has too few/zero real clips for
  (`_fill_with_tts`), speaker/voice-disjoint train/test split (`split_by_speaker`),
  and wired `kws_de.train.main`/`kws_de.export.main`/`kws_de.eval.main` to run the
  full pipeline end to end on real data.
- `docs/eval-report.md` — real measured accuracy: headline real-speech INT8
  accuracy 87.9% on Licht/Kühlschrank/Heizung + `_unknown_`/`_silence_` (vs.
  MultiNet's ~85-95% clean-speech English baseline); full 5-word model 91.9%
  INT8 (Camping/Wasser TTS-augmented, clearly labeled as such).
- Repo scaffold with gitignored `data/`/`models/` dirs (training bytes never committed).
- Design spec `docs/superpowers/specs/2026-08-31-kws-de-design.md` — German KWS DS-CNN
  for ESP32-S3, with algorithmic documentation and cited references (§12).
- Standard tooling per repo convention: `pyproject.toml` (uv, ruff), MIT `LICENSE`,
  `.markdownlint.json`, `.github/workflows/ci.yml` (test/coverage, markdownlint, gitleaks).
