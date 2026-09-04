# Field capture in Assistent mode — design

Date: 2026-09-03. Status: approved in chat; implementation scheduled after the generated
inference runtime and the microSD storage work have landed (both touch the same firmware files).

## 1. Goal

Turn real usage into training data. In Assistent mode (wake-gated recogniser), after a wake fire
the device stores the audio of the interaction — the wake phrase and the 2.5 s command window —
as a *field take*, together with what the device itself recognised. Field takes go through the
existing ingest → QC path; the on-workstation Whisper model transcribes them, the command
grammar turns the transcript into the label, and the takes join training like guided takes.
The device's own prediction, kept next to the Whisper label, yields a **field accuracy** figure
for every real interaction.

Decisions taken: capture is **opt-in** (visible toggle, persisted), a take contains **wake
phrase + window**, **everything is kept** (non-parsable transcripts become negative / unknown
material), labels are **auto-derived** (Whisper + grammar), field takes are **in-training**
like guided takes, and the device prediction is **never** used as a label.

Non-goals: capturing missed wakes (no trigger exists), any cloud component, model changes.

## 2. Firmware

- **Toggle.** The Assistent screen gets an "Aufnahme" toggle (off at boot until the user turns
  it on once; the state is persisted in NVS under the existing `kws` namespace and restored at
  boot). While on, the screen shows a small "REC" badge. Console: `field on|off`, and `status`
  reports `field on|off`.
- **Capture = copy after the window, never during it.** The wake task already knows the fire
  position in the audio ring (`audio_write_pos()` at the fire). The `assist_gate` keeps
  `fire_pos`. When the window closes (2.5 s after the fire) and the toggle is on, `wake.cc`
  posts `REC_CMD_FIELD_TAKE {start = fire_pos − 1.5 s, len = 4.0 s, prob, device_intent,
  device_words}` to the recorder. The recorder copies that span out of the audio ring into its
  PSRAM take buffer and saves it with the existing `save_take()` path. No file I/O happens while
  the recogniser is active (measured in wave 2: a FAT write costs 100–300 ms on flash).
- **Audio ring length.** The ring must hold ≥ 1.5 s + 2.5 s + the recorder's worst-case latency
  to start the copy (a full recogniser step plus scheduling, < 0.2 s). `AUDIO_RING_SAMPLES` is
  checked against 4.2 s at compile time (`static_assert`) and raised if needed; the assist
  window length and the pre-roll are single constants next to it.
- **Layout and metadata.** `storage_root()/field/spkNN/<boot-ms>.wav` (16 kHz mono int16, like
  every take; speaker id = the current NVS id, no bump per interaction, so one boot of one user
  is one speaker directory) plus `storage_root()/field/spkNN/field.csv` with columns
  `file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs`. `device_intent`
  is the intent the on-device grammar composed in the window (or empty), `device_words` the
  `|`-joined fired words with confidences.
- **Storage floor.** If `storage_free_bytes()` is under the same floor the recorder uses, the
  take is dropped with one log line (`field: dropped, storage low`) and a counter in `status`.
- **Screens.** Nothing else changes for the user: green flash + beep + recognised intent as
  today. The badge is the only visible difference.

## 3. Ingest

- `scripts/pull-recordings.sh` pulls `field/` like the other sets; each `field.csv` row becomes a
  `sessions.csv` row with `set=field`, `prompt=""`, and the device columns appended
  (`fire_ms,wake_prob,device_intent,device_words`). `scripts/ingest.sh` needs no change beyond
  the column pass-through and the count verification already in place.

## 4. QC (`kws_de/qc.py`)

New content mode for `set == "field"`:

1. Transcribe the whole take (same Whisper model, prompt, padding as today).
2. **Wake part.** If the transcript's first words match the wake regex (`(hey|hej|he|hei)
   (bus|buss|bos|boss)`) and the word timestamps put them inside the first 1.3 s, cut
   `[0, end of "bus" + 0.15 s]` as a `wake` clip into `approved/wake/spkNN/` (next-free numbering,
   written.txt, as for guided wake takes).
3. **Command part.** Normalise the remaining words and run them through `grammar.parse()` (the
   same grammar the device uses, imported from `kws_de.grammar`):
   - parses to a valid intent → label = `intent_text(intent)`; the take is handled exactly like
     an approved guided *sentence* with that prompt: `approved/phrases/spkNN/…` + index row +
     word segmentation into `approved/words/`;
   - does not parse → `approved/negatives/spkNN/…` with the transcript in the index row
     (`prompt` column = transcript), so it still feeds `_unknown_` windows;
   - empty transcript / audio-gate failure → rejected as today.
4. **Provenance.** `qc.csv` gains `device_intent` and `agrees` (`1` when the device's intent
   equals the Whisper-derived intent, `0` otherwise, empty for non-parsable takes).
   `report.md` gets a "Field" section: takes, parsable, wake clips, agreement rate.

## 5. Eval (`kws_de/eval.py`)

`kws-eval --recordings` gains a **Field** section: per speaker, number of field takes, share
parsable, **device–Whisper agreement** (the field accuracy of the deployed model at the time of
capture, read from `qc.csv`, independent of the model being evaluated), and the usual
recordings figures for the field-derived phrases/words/negatives under the in-training /
held-out labels from the manifest.

## 6. Files

| Path | Change |
|---|---|
| `firmware/main/assist_gate.c/.h`, `wake.cc`, `record.c/.h`, `ui/ui_assist.c` (or the assist screen file), `console.c`, `audio.h` | toggle, fire position, `REC_CMD_FIELD_TAKE`, ring length assert, badge |
| `firmware/test/test_assist_gate.c` (+ a ring-window arithmetic test) | host tests |
| `scripts/pull-recordings.sh`, `scripts/ingest.sh` | `field/` + columns |
| `kws_de/qc.py`, `tests/test_qc.py` | field mode, wake split, grammar labels, agreement |
| `kws_de/eval.py`, `tests/test_eval_recordings.py` | Field section |
| `docs/sphinx/firmware.rst`, `pipeline.rst`, `requirements.rst`, `tests.rst`, `firmware/README.md`, `docs/paper-notes.md` | `REQ_FW_FIELD_CAPTURE` (opt-in; no I/O during the window), `REQ_PIPE_FIELD_LABELS`, docs and the first numbers |

## 7. Order and risks

1. Firmware capture + host tests + on-device check (toggle persists, take appears after an
   interaction, no change in the recogniser step time during the window — measured with the
   existing step log).
2. Ingest + QC field mode + tests; run on the first real field takes.
3. Eval section + docs + paper-notes.

Risks: ring length vs PSRAM (the ring is internal RAM today; 4 s × 16 kHz × 2 B = 128 KB — if
that does not fit internal SRAM next to the arenas, the ring moves to PSRAM, which the audio
task tolerates; measure the wake step after the move); Whisper mis-transcribing short commands
(the grammar rejects them → negatives; the agreement column makes the rate visible); privacy —
the toggle is the control, and field takes never leave the workstation.
