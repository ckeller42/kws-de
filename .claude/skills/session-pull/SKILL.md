---
name: session-pull
description: Use when the user asks to pull a new recording session off the CoreS3 and get it into the dataset -- bundles ingest, staging, QC, the whole-tree audit, and the personalised eval into one sequence.
disable-model-invocation: true
---

# Session pull

The full sequence from "device has new recordings" to "eval number for this
session", in the order that avoids re-doing steps: ingest, stage, QC, audit,
eval.

Set `KWSREC_HOST` to the ssh name of the machine the CoreS3 is plugged into
before starting (never hard-code a host name in anything this skill touches).

## 1. Ingest

```bash
scripts/ingest.sh -H "$KWSREC_HOST"
```

Pulls the session from the device host into
`$KWS_DATA_ROOT/data/recordings/incoming/<stamp>/`, verifying the local copy
against the host's before trusting it. The host-side stage is never deleted
by this step, so a failure here is always recoverable.

**Screen-lock recovery**: if `ingest.sh` reports the drive did not mount
within 20 s, the host's screen is very likely locked -- loginwindow ejects a
freshly-attached USB volume the instant the screen locks. Unlock the host's
screen, then on the device do `mode menu` then `mode usb` again and re-run.

**Spotlight / manual-copy recovery**: if the ingest step fails after the
host-side pull already ran (wav-count mismatch, missing `sessions.csv`, or a
short local copy), `ingest.sh` prints the exact recovery command --
the host stage under `~/kwsrec-pull/<stamp>/` is untouched, so re-run the
`rsync` it prints by hand once the local copy is trusted.

## 2. Staging: incoming/ vs incoming-tts/

**Ask the user before proceeding**: did this session come from a person
speaking, or from a TTS clip played through a speaker at the device
(a playback/wake-gate test)?

- Human speaker -> stays in `incoming/<stamp>/` (the default from step 1).
- TTS playback -> move it to a staging directory of its own,
  `incoming-tts/<stamp>/`, and **do not run `kws-qc` on it**. Nothing under
  `approved/` may come from a loudspeaker -- filing a played-back TTS clip as
  a real speaker take is exactly the "TTS-vs-real" shortcut the wake rounds
  exist to catch, and it inflates the real-speaker share the wake recipe
  counts on. If the session is TTS-sourced, stop here after moving it; steps
  3-5 do not apply.

## 3. QC

```bash
uv run --no-sync kws-qc "$KWS_DATA_ROOT/data/recordings/incoming/<stamp>"
```

Runs the audio gate and the Whisper content gate over every take, writes
`qc/<stamp>/{qc.csv,words.csv,written.txt,report.md}`, and regenerates
`approved/` for this stamp (safe to re-run: it first undoes exactly what this
stamp wrote last time, via its own `written.txt`).

## 4. Whole-tree audit

```bash
uv run --no-sync python scripts/audit-approved.py --no-transcribe
```

Checks `approved/` as a whole, not just the session just pulled: format,
duration bands, no wake phrase leaking into phrases/negatives, `index.csv`
consistency, speaker-directory naming. Exits non-zero on any finding -- fix
before training on the result. `--no-transcribe` skips the Whisper
re-verification pass (already covered by QC's own content gate); drop the
flag for a full from-scratch check.

## 5. Field eval

```bash
uv run --no-sync kws-eval --recordings "$KWS_DATA_ROOT/data/recordings/approved" --qat
```

Reports the standard held-out figure plus the personalised figure on this
speaker's own recordings (the QC root for the **Field** section defaults to
`data/recordings/qc`, so it does not need its own flag). Look for the
**Field** line specifically -- that is the number this session pull was for.
