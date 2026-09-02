Recording pipeline
===================

How a speaker's own voice, recorded on the CoreS3 itself, becomes part of
the training set and a personalised eval number. Design:
``docs/superpowers/specs/2026-09-02-recording-pipeline-design.md``.

Overview
--------

Seven stages, from the device to a retrained model:

1. **Remote control** (firmware, already on ``main``) — serial console
   commands ``mode menu|record|recognise|wake|usb`` and ``status`` drive the
   CoreS3 without touching its screen; ``mode usb``/``mode menu`` are what
   ingest uses.
2. **Ingest** (``scripts/ingest.sh``) — pulls a recording session from the
   device host to the workstation.
3. **Quality control** (``kws-qc``) — an audio gate plus a Whisper content
   gate judge every take, and approved sentence takes are segmented into
   word clips.
4. **Data prep** (``kws-dataset build --prefix features_v3``) — folds the
   approved recordings into the v3 dataset build alongside MSWC/TTS.
5. **Train / export** (``kws-train --v2``, ``kws-export --firmware``) — the
   usual path, unchanged except for the ``--prefix``/``--out``/``--model``
   flags that point it at the v3 build.
6. **Evaluations** (``kws-eval --recordings``) — reports the standard
   held-out figure and a **user-customised** figure on the speaker's own
   recordings, always kept apart.
7. **One-shot driver** (``scripts/data-loop.sh``) — stages 2-6 chained,
   stopping at the first failure.

Data layout
-----------

Everything lives under ``$KWS_DATA_ROOT/data/recordings/``:

.. code-block:: text

   incoming/<YYYY-MM-DD-HHMM>/            one pulled session, byte-for-byte
                                           as recorded (immutable)
     spkNN/<slug>/NNN.wav                 word takes (only if recorded)
     spkNN/_phrase_/<slug>_NNN.wav        sentence takes
     spkNN/_neg_/<slug>_NNN.wav           negative-phrase takes
     sessions.csv                         speaker,pulled,prompt,file,ms,
                                           peak_dbfs,set,seed,ts (merged
                                           across every speaker in the pull)
     logs/recognise-<ts>.log              recognise-mode detection log, if any
   qc/<same-name>/
     qc.csv                               one row per take: file,set,prompt,
                                           speaker,verdict,reason,transcript,
                                           match_score,rms_dbfs,peak_dbfs,dur_ms
     words.csv                            one row per SEGMENTED keyword clip
     written.txt                          approved-relative paths this stamp
                                           wrote, for idempotent re-runs
     report.md                            counts, rejects with reasons, gaps
   approved/
     words/<label>/<spkNN>_<NNN>.wav      1 s keyword clips (bare + segmented)
     phrases/<spkNN>/<slug>_<NNN>.wav     approved full sentences
     phrases/index.csv                    file,prompt,speaker
     negatives/<spkNN>/<slug>_<NNN>.wav   approved non-command phrases
     negatives/index.csv                  file,prompt,speaker

``incoming/`` is never modified. ``approved/`` is regenerated from
``incoming/`` + the QC rules on every ``kws-qc`` run — re-running one
stamp first undoes exactly what that stamp wrote last time (via its own
``written.txt``), so a QC rule change can be re-applied to every session
without hand-editing anything, and without disturbing any other stamp's or
speaker's files.

Quality control rules
----------------------

Every take passes an **audio gate** (no model) and, if that passes, a
**content gate** (Whisper). See ``REQ_PIPE_QC_AUDIO`` and
``REQ_PIPE_QC_CONTENT`` in :doc:`requirements` for the exact requirement
text; the numbers:

- Format: 16000 Hz, mono, 16-bit PCM.
- Duration: at least 300 ms; at most 4000 ms for a word take, 6000 ms for a
  sentence or negative take.
- Level: peak below -0.5 dBFS (not clipped); RMS at or above -45 dBFS.
- Transcript normalisation: NFC, lower-cased, ``ß`` -> ``ss``, punctuation
  stripped, the filler word "prozent" dropped.
- Word match: exact for keywords of 5 letters or fewer; edit-distance-1
  forgiving above 5 letters — so "Licht" (5 letters) never accidentally
  matches a misheard "nicht", but a keyword like "Kühlschrank" tolerates one
  Whisper substitution/insertion/deletion.
- Sentence match: every required command token (device, zone, action) must
  appear in order; filler words in between are fine.
- Negative match: **no** command-vocabulary word may appear anywhere in the
  transcript; the first one found is the rejection reason.

A corrupt or unreadable WAV, or an exception from the transcriber, rejects
that one row (``unreadable: …`` / ``error: …``) — it never aborts the
batch. Approved sentence takes are segmented: for each required keyword,
Whisper's word span is turned into a 1 s window centred on its midpoint
(zero-padded at the recording's edges) and written as its own word clip,
alongside the bare word takes already at 1 s. Whisper runs at
``temperature=0`` (greedy) and its model id is recorded in ``report.md``,
so a QC run is reproducible for a given model version.

The two evaluation figures
---------------------------

``kws-eval --recordings`` always reports these separately and never
combines them into one number:

.. list-table::
   :header-rows: 1

   * - Figure
     - Data
     - Label (verbatim)
   * - Held-out
     - Everything whose speaker is **not** in the training manifest's
       ``train`` split (plus every phrase clip, always)
     - ``held-out``
   * - User-customised
     - Isolated words / end-to-end phrases / false-accepts on
       ``approved/`` whose speaker **is** in the training manifest's
       ``train`` split
     - ``user-customised, in-training``

The match is **speaker-level**, not per-clip — the training manifest
records which speakers' recordings went into a build, not which individual
files. A take recorded and QC-approved for an already-trained speaker
*after* the manifest's ``built_at`` timestamp is genuinely new data the
current model has never seen, but it is still reported under
``user-customised, in-training`` purely because its speaker id matches;
only re-running ``kws-dataset build`` and retraining makes the match exact
again. Phrase clips are always ``held-out`` — the dataset build only reads
``approved/words/`` and ``approved/negatives/``, never
``approved/phrases/``, so a phrase clip is never actually training
material regardless of its speaker.

This in-training figure is legitimate, not a leak, **because the product
is a personalised device**: a speaker's recordings exist to make the model
work better for that speaker specifically, and reporting "how well does it
now recognise the person who trained it" is the number that matters for
that promise. It answers a different question than held-out accuracy
(generalisation to unseen speakers) and must never be quoted as if it were
one — hence the always-separate sections and the explicit label on every
figure.

Running it
----------

One-shot, from ingest through evals:

.. code-block:: console

   $ export KWS_DATA_ROOT=/path/to/data-root
   $ scripts/data-loop.sh -H <device-host>

``-H`` (or ``KWSREC_HOST``) names the host the CoreS3 is plugged into over
SSH; it is never hard-coded into the repo. ``--skip-ingest`` reuses the
newest already-pulled session (or one named with ``--incoming``);
``--skip-train`` runs QC, the dataset build, and evals only; ``-n`` prints
every stage's command without running anything (dry run). Each stage's
failure stops the loop with the error from that stage.

Step by step, the same flow by hand:

.. code-block:: console

   $ scripts/ingest.sh -H <device-host>
   $ uv run --no-sync kws-qc "$KWS_DATA_ROOT/data/recordings/incoming/<stamp>"
   $ uv run --no-sync kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3
   $ uv run --no-sync kws-train --v2 --prefix features_v3 --out command_v3.keras --epochs 40
   $ uv run --no-sync kws-export --prefix features_v3 --model command_v3.keras --firmware
   $ uv run --no-sync kws-eval --recordings "$KWS_DATA_ROOT/data/recordings/approved" \
       --prefix features_v3 --out docs/eval-report-v3.md

The ``kws-dataset build`` cache (``raw_clips_v3.pkl``) must already exist —
run ``uv run --no-sync kws-data --fetch --mswc-root <mswc-de-dir>`` once
first to mine MSWC and build it; the data loop itself never fetches MSWC.
Flashing the exported model back onto the device is a manual step, covered
by the ``flashing-cores3-on-bar`` skill, not by this pipeline.

A firmware ``wake`` set (five "Hey Bus" takes per speaker,
``spkNN/hey-bus/NNN.wav``) is already recorded by the device; QC support
for it is a follow-up task, not implemented here.

Requirements
------------

.. needtable::
   :types: req
   :filter: id.startswith("REQ_PIPE")
   :columns: id, title, status
   :style: table

See :doc:`requirements` for the full requirement text and :doc:`tests` for
the tests each one traces to.
