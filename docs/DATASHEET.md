# DATASHEET.md — kws-de v2 command dataset

Follows the Datasheets-for-Datasets template (Gebru et al., 2018). Covers the
Phase-0 dataset produced by `kws-dataset build` (`kws_de/dataset.py`): frozen,
speaker-disjoint train/val/test feature tensors + `data/manifest.json`, the
input every later experiment (architecture benchmark, streaming transducer)
shares.

## Motivation

Built to train and evaluate a tiny **offline, on-device keyword-spotting
model** for German voice control of a camper van ([buspi](https://github.com/ckeller42/buspi-config)),
running INT8 on an ESP32-S3 (M5Stack CoreS3). The commands cover the
vehicle's real controllable functions — lights (with zones and brightness),
fridge, heater, and the pop-up roof — spoken in German. No cloud recognition,
no free-form natural language: the model only needs to recognize this fixed,
small vocabulary reliably, under noise, on a microcontroller.

## Composition

23-class label set (`kws_de.config.COMMAND_LABELS`): 4 devices
(`Licht`, `Kühlschrank`, `Heizung`, `Aufstelldach`), 4 light zones
(`Küche`, `Dach`, `Außen`, `Lesen`), 13 actions (`an`, `aus`, `auf`, `zu`,
`heller`, `dunkler`, `wärmer`, `kälter`, `leise`, and the four light-level
words `fünfundzwanzig`/`fünfzig`/`fünfundsiebzig`/`hundert`), plus the two
catch-all classes `_unknown_` (other German words, for negative rejection)
and `_silence_` (noise/silence, so the model doesn't fire on quiet).

Each of the 21 command words is sourced from a mix of real recorded speech
(MSWC) and TTS-synthesized speech, per word — some words are 100% real, most
are partly or wholly synthetic (MSWC has too few clips of many command
words, especially the invented/compound German ones like `Aufstelldach`
and the light-zone/level words). The exact real/TTS split per word, per
split, and per class is recorded in `data/manifest.json` (produced by a
`kws-dataset build` run, not committed to this repo as bytes — only the
manifest's counts and hashes are). A representative snapshot from the
current v2 vocabulary (`docs/eval-report-v2.md`, "Vocabulary provenance"):
of the 21 command words, 4 are 100% real MSWC clips (`Licht`, `Kühlschrank`,
`aus`, `auf`), 2 are a real/TTS mix (`Heizung`, `Außen`), and 15 are 100%
TTS — **17/21 command words carry some synthetic audio**, which is why the
dataset (23 classes counting `_unknown_`/`_silence_`) is documented
elsewhere as "17/23 words synthetic."

Noise: [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound
clips, mixed in at several SNRs (20/10/0 dB) to build noisy augmented copies
and the `_silence_` class.

## Collection

- **Real speech**: streamed from the MLCommons Multilingual Spoken Words
  Corpus (MSWC), German config (`de_wav`), via the `datasets` library
  (`kws_de.data._fetch_mswc`). Valid clips matching a target keyword are
  collected up to a per-word cap; other valid clips are pooled for
  `_unknown_`.
- **Synthetic speech**: macOS `say` and, optionally, Piper neural TTS
  (`kws_de.tts`), across multiple German voices and speaking rates, used to
  top up any word under its real-clip target (`kws_de.data._fill_with_tts`
  / `_tts_fill_word`). Each synthesized clip's "speaker id" is its
  `tts:{engine}:{voice}:{rate}` combo, so an unseen voice/rate is a distinct
  synthetic speaker for split purposes, not just a distinct utterance.
- **Noise**: ESC-50, downloaded once and cached (`kws_de.data._download_esc50`).

All fetch/synthesis code lives in `kws_de/data.py` and is versioned; no
audio bytes are committed (`data/` is gitignored). The dataset is
reproduced from code + a seed, not from checked-in files.

## Provenance

Per-word real-vs-TTS counts, and per-split (train/val/test) totals, live in
`data/manifest.json`, written by `kws-dataset build` — the verifiable
fingerprint of one build (see Splits below for how it's produced). The
manifest also records a sha256 content hash of each split's feature tensor,
so a rebuild from the same seed can be checked byte-for-byte against a
previously committed manifest.

## Licensing

- **MSWC** (real speech): CC-BY-4.0 (MLCommons).
- **macOS `say`**: system TTS, bundled with macOS; used for local synthesis
  only, not redistributed as audio.
- **Piper TTS**: neural TTS with permissively-licensed German voice models
  (see the individual voice's license on the Piper voice repository);
  used for local synthesis only.
- **ESC-50** (noise): Creative Commons licensed (see the ESC-50 repository
  for the exact CC variant and per-clip attribution).

No dataset audio is redistributed by this repository — only the code that
fetches/synthesizes it, and the small text artifacts (`manifest.json`,
this datasheet) that describe the result.

## Splits

Train / val / test, all **speaker-disjoint**: real clips are split by MSWC
`speaker_id`; synthetic clips are split by their `tts:{engine}:{voice}:{rate}`
synthetic-speaker id. No speaker (real or synthetic) appears in more than
one split (`kws_de.data.split_three_way`). A single `seed` fixes the
speaker→split assignment and is recorded in `data/manifest.json`, so the
same seed always reproduces the identical split.

Val is a first-class split (not an afterthought of train): it exists so
model selection and hyperparameter choices are made on val, never on test,
keeping the eventual test-set numbers (architecture benchmark, transducer
comparison) honest.

## Recommended uses

Training and evaluating small, on-device KWS architectures for this exact
German command vocabulary — isolated-word accuracy, streaming/catalog
full-intent accuracy, and on-device budget fit (params, MACs, INT8 size,
latency). Intended as the shared input to the Phase-1 architecture
benchmark and the Phase-2 streaming transducer described in
`docs/superpowers/specs/2026-09-01-kws-research-plan-design.md`.

## Limitations

**17 of the 21 command words (17/23 classes counting `_unknown_`/
`_silence_`) contain synthetic (TTS) audio, several of them 100% synthetic.**
This dataset is therefore **not a real-speech benchmark** — reported
accuracy numbers reflect performance on a mix of real and synthesized
German speech, not exclusively natural human speech, and TTS voices carry
their own acoustic biases (a limited set of voices/rates, no real
room noise or mic characteristics beyond the ESC-50 mixing). Evaluating on
real, recorded microphone speech captured on the actual target hardware is
explicitly out of scope for this dataset and is planned as a separate
follow-up (see `docs/superpowers/specs/2026-09-01-on-device-hw-mic-followup-design.md`).
