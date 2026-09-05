---
name: data-audit
description: Use when checking whether recorded or synthesised (TTS) clips are safe to train on -- when to run audit-approved.py vs kws-tts-check, and what clipped/truncated/unfiled findings mean.
---

# Data audit

Two different gates, for two different sources of audio. Neither replaces
the other.

## `kws-tts-check` -- gates synthesised (TTS) clips

Run over a directory holding a `manifest.csv` (or the manifest itself), one
TTS-generation batch at a time:

```bash
uv run kws-tts-check <dir-with-manifest.csv> [--quarantine]
```

Transcribes every clip with Whisper (language auto-detected, on purpose --
the whole point is catching a clip that came out English) and writes
`tts_check.csv` beside the manifest: `file,voice,engine,ok,reason,language,
transcript`. `--quarantine` moves failing clips to `rejected/` so a later
re-run of whatever consumes the directory can't pick them up again.

This is the gate `.claude/hooks/audio-gate.sh` enforces before any
`afplay`/`say` command runs: a wav must have an `ok=1` row in a
`tts_check.csv` beside it, or the play is blocked. Run this **before**
playing a synthesised clip to a speaker or a device -- an English clip
generated because the German voice pack was missing on the synthesising
host is exactly the mistake this catches, and it already reached a device
test once.

## `scripts/audit-approved.py` -- gates the recorded, approved tree as a whole

```bash
uv run --no-sync python scripts/audit-approved.py [--no-transcribe] [<approved>]
```

`kws-qc` only ever looks at one session at a time; this looks at
`approved/` as a whole -- the check that catches the class of bug that
survives every individual QC run because no single run ever sees the whole
tree. Run it after every `kws-qc` pass, and before training or evaluating on
`approved/`. Exits non-zero on any finding, so it can gate a data pull.

Findings it can report:

- **format** -- not 16 kHz mono PCM_16, or outside its set's duration band
  (wake 0.4-2.6 s, words ~1 s, phrases 0.5-9.8 s, negatives up to 9.8 s).
- **clipped** -- too many samples parked at the rail (a 1-2 sample pop is
  fine; a sustained clip is not).
- **truncated** -- a field take the device's recording ring cut short: `ms <
  FIELD_PREROLL_MS + window_ms`. Carries no reliable device intent to agree
  or disagree with -- expected for some field takes, not itself a failure,
  but the audit counts it so the report says how many.
- **unfiled** -- approved but with nothing to compare against (e.g. a
  truncated take, or one production's recognition can't be scored against
  for another structural reason). Also expected in some numbers; a spike is
  the signal to look closer, not a fixed threshold.
- index/file mismatches, wrong speaker-directory naming (`spkNN`), and
  phrase/negative clips that still contain the wake phrase (checked against
  `qc._WAKE_RE`).

## The `incoming-tts/` rule

A session recorded by playing a TTS clip through a speaker at the device
(exercising the wake gate without a person in the room) is not a human
speaker take. It must be staged in `incoming-tts/`, never `incoming/`, and
`kws-qc` must never run on it -- filing it into `approved/` would feed
synthesised audio back into training as if it were real, which is the
"TTS-vs-real" shortcut the wake rounds exist to catch. See the
`session-pull` skill for the full pull sequence, which asks which staging
directory a session belongs in.

## Idempotency: `written.txt`

Every `kws-qc` run writes `qc/<stamp>/written.txt` -- the list of
`approved/`-relative paths that stamp wrote. Re-running QC on the same
stamp first deletes exactly those paths (and only those) before writing
again, so re-applying a changed QC rule to one session never disturbs
another stamp's or another speaker's files. A clip written before stamps
existed reports its source as `unknown` in the audit, not as a bug.
