# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Repo scaffold on external SSD, symlinked to `~/src/kws-de` (spec §Storage).
- Design spec `docs/superpowers/specs/2026-08-31-kws-de-design.md` — German KWS DS-CNN
  for ESP32-S3, with algorithmic documentation and cited references (§12).
- Standard tooling per repo convention: `pyproject.toml` (uv, ruff), MIT `LICENSE`,
  `.markdownlint.json`, `.github/workflows/ci.yml` (test/coverage, markdownlint, gitleaks).
