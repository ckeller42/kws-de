# Recording pipeline: capture → ingest → QC → data prep → train/test → evals

**Status:** design, 2026-09-02. **Depends on:** the CoreS3 firmware (`feat/wake-test-mode`
branch: selection-screen flow + `mode …` serial command), `KWS_DATA_ROOT` (PR #10).

## 1. Goal

Turn the guided recorder into a repeatable data loop: a speaker records a session on the
CoreS3; one command pulls the takes to the workstation, quality-controls them with a large
ASR model, adds the approved audio to the v3 dataset, retrains, and reports two numbers —
the standard held-out accuracy and the **user-customised** accuracy on the speaker's own
recordings (which may be in training; that overlap is the point of the step and is
labelled as such, never mixed with the held-out figure). The whole pipeline is documented
(Sphinx + paper) and its requirements are traced to tests.

Non-goals: on-device QC; speaker verification; anything cloud. The workstation
is the Apple-Silicon Mac that holds `KWS_DATA_ROOT`; the CoreS3 is attached to the second
Mac, reachable over SSH — machine names stay out of the repo (config/env only).

## 2. Data layout (under `KWS_DATA_ROOT/data/recordings/`)

```
incoming/<YYYY-MM-DD-HHMM>-spkNN/      raw pull, byte-for-byte as recorded (immutable)
  <slug>/001.wav 002.wav …             word takes (only if the words set was recorded)
  _phrase_/<slug>_001.wav …            sentence takes (the session default)
  _neg_/<slug>_001.wav …               negative-phrase takes
  session.csv                          prompt,file,ms,peak_dbfs,set,seed,ts (from the device)
qc/<same-name>/
  qc.csv                               one row per take: file,set,prompt,speaker,verdict,
                                       reason,transcript,match_score,rms_dbfs,peak_dbfs,dur_ms
  words.csv                            one row per SEGMENTED keyword: src,word,speaker,
                                       start_ms,end_ms,out_file
  report.md                            human summary: counts, rejects with reasons
approved/
  words/<word>/<spkNN>_<NNN>.wav       1 s centred keyword clips — the layout
                                       kws_de.recordings.load_recordings() already reads
  phrases/<spkNN>/<slug>_<NNN>.wav     full approved sentences (personalised e2e eval)
  negatives/<spkNN>/<slug>_<NNN>.wav   approved non-command phrases (→ _unknown_)
sessions.csv                           append-only ledger of every ingest (from pull script)
```

`incoming/` is never modified; `approved/` is regenerable from `incoming/` + `qc/`
(idempotent), so a QC rule change can be re-run over every session.

## 3. Stages

### 3.1 Remote control (firmware, done in the selection-screen work)

Serial console commands `mode menu|record|recognise|wake|usb` and `status`, echoing
`ok`/`err <reason>`. Ingest uses `mode usb` / `mode menu`. The USB drive label is `KWSREC`.

### 3.2 Ingest — `scripts/ingest.sh` (runs on the workstation)

```
ingest.sh [-H <host>] [-p /dev/cu.usbmodemNNN]
  1. ssh to the device host: write "mode usb\n" to the port; wait ≤20 s for /Volumes/KWSREC to mount
  2. ssh host: scripts/pull-recordings.sh ~/kwsrec-pull   (existing script: rsyncs spk*/,
     appends sessions.csv, moves recognise.log to logs/, ejects)
  3. rsync bar:~/kwsrec-pull/ → $KWS_DATA_ROOT/data/recordings/incoming/<stamp>-<spk>/
     (one dir per speaker found; --ignore-existing; never deletes anything anywhere)
  4. ssh to the device host: write "mode menu\n" (device back to the selection screen)
  5. print the new incoming dir(s); exit non-zero if nothing was pulled
```

Requires the remote flashing helper's conventions (device host, port auto-detect). The
device's serial link is gone while in USB mode, so `mode menu` is sent only after the
drive has been ejected and the port is back. Failure at any step leaves the device in USB
mode at worst — the script says so and how to recover (`mode menu` by hand).

### 3.3 Quality control — `kws-qc <incoming-dir>` (Python, `kws_de/qc.py`)

Model: **Whisper large-v3 via `mlx-whisper`** (Apple-Silicon native; ~3 GB, fits the
16 GB M4), German, `word_timestamps=True`. New optional-dependency group
`qc = ["mlx-whisper>=0.4"]`. The model is loaded once per run.

Per take, two gates then a verdict:

1. **Audio gate** (no model): 16 kHz mono 16-bit; duration within the set's cap
   (words/wake ≤ 4 s, phrases/negatives ≤ 6 s) and ≥ 0.3 s; RMS ≥ −45 dBFS; peak < −0.5 dBFS
   (not clipped); leading/trailing silence trimmed for the measurements only.
2. **Content gate** (Whisper): `whisper_transcriber()` biases decoding with an
   `initial_prompt` — narrow by design: only `fünfundzwanzig, fünfzig, fünfundsiebzig,
   hundert, Prozent, Hey Bus` (the words Whisper actually mangles), not the whole command
   vocabulary — a full-vocabulary prompt was tried first and caused prompt-echo
   hallucination (Whisper regurgitating chunks of the prompt on weak audio), including new
   false rejects on clean negatives; narrowing it removed that regression. Each clip is
   padded with 500 ms of silence on both sides (word timestamps shifted back by the same
   amount, clamped at 0). Transcript normalised (lower-case, umlauts kept, ß→ss and ss→ß
   both accepted, punctuation stripped, number words compared as words, "prozent" optional,
   and the numerals Whisper writes for the light levels — `25`/`50`/`75`/`100`, a trailing
   `%` dropped like "prozent" — mapped back to their German number words before matching)
   and matched to the prompt from `session.csv`:
   - words/phrases: each heard (whitespace-delimited) word either matches one required
     token outright (exact, or edit distance ≤ 1 for a token of > 5 letters over a sliding
     window of the token's length ± 1), or is the exact concatenation of two or more
     *consecutive required* tokens Whisper glued with no space (heard "Lichtdach" for
     "Licht Dach"); a short (≤5-letter) keyword never matches as a mere substring of an
     unrelated longer word ("an" inside "dank" does not match);
   - negatives: no command keyword may appear as a whole token, except a 2-letter keyword
     ("an", "zu") alone does not reject — it must appear at least twice, or be ≥ 3 letters;
   - wake ("Hey Bus" takes): the glued, lower-cased transcript must match
     `(hey|hej|he|hei)(bus|buss|bos|boss)`.
   `match_score` = fraction of required tokens found (0–1).
3. **Verdict:** `approve` if both gates pass; `reject:<reason>` otherwise (`clipped`,
   `too_quiet`, `too_short`, `too_long`, `wrong_word:<heard>`, `missing:<token>`,
   `contains_command:<token>`). Both takes of a prompt are judged independently; if one is
   approved and the other rejected the report lists the pair for a human glance.

**Segmentation (phrases only):** for each required keyword found, Whisper's word span is
extended to a 1 s window centred on the word (`kws_de.recordings.centre` semantics; the
window is padded with the recording's own neighbouring audio, zero-padded at the file
edges) and written to `approved/words/<word>/<spkNN>_<NNN>.wav` plus a `words.csv` row.
Bare word takes are copied there unchanged. Approved phrases and negatives are copied
whole to their `approved/` trees. Everything under `approved/` for that session is
regenerated on every run (delete-then-write for that session's files only).

Determinism: Whisper is run with `temperature=0` (greedy); the QC is reproducible for a
given model version, which is recorded in `report.md`.

### 3.4 Data prep — the v3 build

`kws-dataset --v3` already mines MSWC and calls `load_recordings(root, words)`; it gains
`--recordings $KWS_DATA_ROOT/data/recordings/approved/words` (default when the dir exists)
and, for `_unknown_`, windows cut from `approved/negatives/` (1 s hops, same SNR mixes as
other classes). `manifest_v3.json` records per clip: source (`mswc`/`tts`/`recording`),
speaker id, the QC session dir, and sha256 — the provenance table in the paper regenerates
from it. Speaker ids are anonymous `spkNN` ids only.

### 3.5 Train / test

Unchanged path: `kws-train --v3` (or `kws-distill --features features_v3`) → `kws-export
--v3 --firmware` through the model-health gate → device header. The held-out test split is
speaker-disjoint from nothing in particular today (single speaker); the report states the
number of distinct speakers in train and test so the reader can judge it.

### 3.6 Evaluations — `kws-eval --recordings <approved-dir> [--model …]`

Two figures, always printed together with unambiguous labels:

| Figure | Data | Label in reports |
|---|---|---|
| Held-out accuracy | `features_v3_test` | "held-out (speaker-mixed, n=…)" |
| User-customised, isolated words | `approved/words/**` per speaker | "user-customised, in-training: isolated-word acc per speaker" |
| User-customised, end-to-end | `approved/phrases/**` → MFCC → `KeywordStream` → grammar → intent | "user-customised, in-training: intent acc per speaker (n phrases)" |

Plus false-accepts on `approved/negatives/**` (any fired intent = false accept). Output:
`docs/eval-report.md` section "User-customised" (tables per speaker + per word), and the
JSON next to it. The e2e figure reuses `kws_de.eval` streaming/grammar code on real audio
instead of TTS.

### 3.7 One-shot driver — `scripts/data-loop.sh`

`ingest → kws-qc → kws-dataset --v3 → kws-train --v3 → kws-export --v3 --firmware → kws-eval
--recordings`, each step skippable by flag, stopping on the first failure, printing the two
figures at the end. Flashing is left to the remote flashing helper (manual step).

## 4. Interfaces (exact)

- `kws-qc INCOMING_DIR [--model mlx-community/whisper-large-v3-mlx] [--out QC_DIR]
  [--approved APPROVED_DIR] [--dry-run]` → exit 0 (writes qc.csv/words.csv/report.md),
  exit 2 if the incoming dir has no `session.csv`.
- `kws-eval --recordings APPROVED_DIR` → adds the user-customised section to the report.
- `scripts/ingest.sh`, `scripts/data-loop.sh` — `set -euo pipefail`, shellcheck-clean.
- `session.csv` columns are the firmware's: `prompt,file,ms,peak_dbfs,set,seed,ts`.

## 5. Error handling

Ingest: no port → clear message (USB mode / unplugged); mount timeout → leave device,
exit 3. QC: missing/unreadable WAV → `reject:unreadable`, never crashes the run; Whisper
model download failure → exit 4 with the model name. Prep/eval: approved dir absent →
skip with a warning (the v2 path keeps working).

## 6. Testing

- `tests/test_qc.py`: audio gate on synthetic WAVs (clipped/quiet/short/ok); content
  matching with a **stubbed transcriber** (dict prompt→transcript) covering word/phrase/
  negative rules, umlaut/ß/number-word normalisation, "Prozent"; segmentation centres the
  1 s window on the word span and pads at edges; re-running QC is idempotent.
- `tests/test_ingest.py`: `ingest.sh` against a fake `ssh`/`rsync` on PATH (like
  `test_pull_recordings.py`) — orders the serial commands correctly, never deletes.
- `tests/test_eval_recordings.py`: user-customised figures computed from a tiny approved
  tree with a stub model; labels present in the report.
- Whisper itself is exercised only by a marked slow test that skips when `mlx_whisper` is
  absent (CI has no Apple GPU).
- sphinx-needs: `REQ_PIPE_INGEST`, `REQ_PIPE_QC_AUDIO`, `REQ_PIPE_QC_CONTENT`,
  `REQ_PIPE_SEGMENT`, `REQ_PIPE_APPROVED_LAYOUT`, `REQ_PIPE_EVAL_LABELS` traced to the
  tests above.

## 7. Docs

- Sphinx page `docs/sphinx/pipeline.rst` (this flow, the layout, the two eval figures and
  why the in-training one is legitimate for a personalised device), linked from the index.
- `DATASHEET.md`: recordings provenance (anonymous `spkNN` ids, QC model + version, counts).
- `docs/paper-notes.md`: the loop as a method paragraph + the first real numbers.

## 8. Open items (deliberately not in scope)

Speaker-disjoint held-out split once ≥ 5 speakers exist (spec §9 of the distill design);
on-device Whisper is impossible; a GUI for reviewing rejects is a later nicety.
