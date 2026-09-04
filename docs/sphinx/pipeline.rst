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
   approved recordings into the v3 dataset build alongside MSWC/TTS. This
   happens on **every** build (``kws_de.data.merge_recordings``), not just
   the one that created the clip cache: QC output changes between builds,
   the cached MSWC/TTS clips do not, so each build re-reads ``approved/``
   and replaces the previous ``rec:`` clips instead of duplicating them.
   The split is speaker-disjoint by assigning each speaker to exactly one
   of train/val/test *across all labels*, so a label recorded by only one
   or two speakers can end up with no train rows, no test rows, or both —
   the fix is more speakers, not a different split. Device speakers are
   the exception: by default they all go to train (see
   `Which split the device recordings land in`_).
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
     spkNN/hey-bus/NNN.wav                wake-word ("Hey Bus") takes
     field/spkNN/<boot>-<ms>.wav           field takes, captured in Assistent
                                           mode (set=field, no prompt)
     sessions.csv                         speaker,pulled,prompt,file,ms,
                                           peak_dbfs,set,seed,ts,fire_ms,
                                           wake_prob,device_intent,
                                           device_words,window_ms (merged
                                           across every speaker in the pull;
                                           the last five are empty for a
                                           guided take)
     logs/recognise-<ts>.log              recognise-mode detection log, if any
   qc/<same-name>/
     qc.csv                               one row per take: file,set,prompt,
                                           speaker,verdict,reason,transcript,
                                           match_score,rms_dbfs,peak_dbfs,
                                           dur_ms,device_intent,agrees,
                                           truncated
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
     wake/<spkNN>/<spkNN>_<NNN>.wav       approved "Hey Bus" takes
     wake/index.csv                       file,prompt,speaker

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
- Duration: at least 300 ms; at most 4000 ms for a word or wake take,
  6000 ms for a sentence or negative take.
- Level: fewer than 0.05 % of samples at or above -0.5 dBFS (clipping parks
  many samples at the rail; a one-sample click does not); RMS at or above
  -45 dBFS.
- Transcript normalisation: NFC, lower-cased, ``ß`` -> ``ss``, punctuation
  stripped, the filler word "prozent" dropped, and the numerals Whisper
  writes for the light levels (``25``/``50``/``75``/``100``, an optional
  trailing ``%`` dropped like "prozent") mapped back onto the German number
  words ("fünfzig" etc.) before matching — evidence from the first real run
  showed Whisper large-v3 transcribing "fünfzig" as the digit "50".
  ``whisper_transcriber()`` also biases the model with an ``initial_prompt``
  — deliberately narrow: only the light-level number words, "Prozent" and
  the wake word, i.e. only the words Whisper actually mangles. An earlier
  version passed the *whole* command vocabulary, which backfired: on
  weak/ambiguous audio Whisper echoed chunks of the prompt back as the
  "transcript" instead of recognising silence, causing new false rejects on
  otherwise-clean negatives. Each clip is also padded with 500 ms of
  silence on both sides before transcribing (word timestamps are shifted
  back by the same amount, clamped at 0, so segmentation stays correct).
- Word and sentence match: each heard word (whitespace-delimited) is
  checked against the required tokens in order — either it matches one
  token outright (exact, or edit-distance-1 for a token of more than 5
  letters, over a sliding window of the token's length ± 1), or it is the
  exact concatenation of two or more *consecutive required* tokens that
  Whisper glued together with no space (heard "Lichtdach" for prompt
  "Licht Dach"). A short (5-letter-or-fewer) keyword never matches merely
  because it occurs as a substring inside an unrelated longer word — "an"
  never matches inside "dank", and "Licht" (5 letters) never fuzzy-matches
  a misheard "nicht" — while a keyword like "Kühlschrank" tolerates one
  Whisper substitution/insertion/deletion.
- Negative match: **no** command-vocabulary word may appear anywhere in the
  transcript as a whole token, with one refinement: a 2-letter keyword
  ("an", "zu") alone does not reject — it must appear at least twice, or be
  3 letters or more — since a single one-letter-off hallucination of a
  2-letter keyword (heard "An den fahren wir los" for "wann fahren wir
  los") was a false reject in the first real run.
- Wake match (the ``wake`` set, "Hey Bus" takes): the whitespace-free,
  lower-cased transcript must match
  ``(hey|hej|he|hei)(bus|buss|bos|boss)`` — loose enough for common
  mishearings, tight enough that ordinary German sentences don't
  accidentally match. Approved takes are written to
  ``approved/wake/<spkNN>/<spkNN>_<NNN>.wav`` plus a ``wake/index.csv`` row,
  idempotently per stamp like the other sets, and counted in ``report.md``.
  Wake takes are QC'd but not folded into the command dataset — the
  wake-word model is trained separately (``kws_de.wake``).

A corrupt or unreadable WAV, or an exception from the transcriber, rejects
that one row (``unreadable: …`` / ``error: …``) — it never aborts the
batch. Approved sentence takes are segmented: for each required keyword,
Whisper's word span is turned into a 1 s window centred on its midpoint
(zero-padded at the recording's edges) and written as its own word clip,
alongside the bare word takes already at 1 s. Whisper runs at
``temperature=0`` (greedy) and its model id is recorded in ``report.md``,
so a QC run is reproducible for a given model version.

A **field take** (``set=field``, captured in Assistent mode) takes a
different route through the same rules
(:need:`REQ_PIPE_FIELD_LABELS`). It has no prompt to match against, so the
content gate approves anything that transcribed to something, and its
duration cap is the firmware's ring budget (9800 ms) rather than a
sentence's 6000 ms — a window extends on every fire inside it, so a
legitimately long take is not an anomaly. The transcript is then split
around **every** "Hey Bus" in it, and the rest is run through the *same*
``kws_de.grammar.parse`` the device's vocabulary feeds.

Two separate rules govern that split, and conflating them is what put the
wake word inside ``phrases/`` and ``negatives/`` (issue #58):

- A ``wake`` clip is cut only from a **whole leading phrase** — one that is
  the take's first one or two word spans, ends within ``WAKE_MAX_S``
  (2.5 s), and is at least ``WAKE_MIN_S`` (0.4 s) long once the 0.15 s tail
  is added. A phrase said later in the take is not that take's wake fire,
  and a clip shorter than the minimum is a head-cut fragment: a wake clip
  starts at the take's first sample, so a short one means the capture began
  after the "Hey" (a session recorded with no pre-roll). A real phrase is
  0.5-0.7 s; those fragments are 0.2-0.3 s, already at full level in the
  first frame. Training a wake model on them teaches it to fire on a single
  syllable, which is the most expensive false-accept mode there is.
- Everything kept as a phrase or a negative starts after the **last** wake
  phrase in the take, wherever that sits, plus the same 0.15 s tail. A take
  that ends with "Hey Bus", or that carries a second fire inside the same
  window, therefore never yields a clip containing the wake word. If
  nothing is left after that cut — the take was only "Hey Bus", or the
  command preceded it — nothing is filed; and if Whisper returned text
  containing the phrase but no word spans to locate it by, nothing is filed
  either. An uncuttable take is dropped rather than filed poisoned.

Un-welding is the one rule the field path needs that the guided path does
not. The guided matcher can glue *known prompt* tokens together to absorb
Whisper writing "Lichtdach" for "Licht Dach"; a field take has no prompt, so
the split has to run the other way — a token that decomposes **completely**
into a run of vocabulary words is split back apart ("lichtküche" ->
"licht küche"), and one that does not is left exactly as heard, so "dank",
"anzug" and "banane" are never quietly turned into commands. Without it a
perfectly good spoken command is filed as a negative, which is worse than
losing it: it poisons the ``_unknown_`` class with the very words the model
must recognise.

A valid intent becomes the label and the take is filed as an approved
phrase — prompt, index row and word segmentation exactly as for a guided
sentence, and, like a guided sentence, the phrase clip holds only the
phrase: it is cut from the end of the wake split to the last word Whisper
heard plus 0.3 s. Streaming the whole take instead would put its pre-roll
and up to several seconds of trailing silence through the command model in
the end-to-end figure, where one spurious event scores a correct take as a
miss. The word clips still come off the full take, whose Whisper timestamps
index them. Anything else is kept as a negative with the transcript as its
prompt, with one guard: the transcript still has to pass the ordinary
negatives content gate, so speech the grammar rejected but that *does*
carry command vocabulary is left unfiled rather than entering either the
unknown class or the false-accept set. Nothing is thrown away for merely
failing to parse — unparsable real speech is precisely the ``_unknown_``
material the model needs — and the report's Field line counts what was left
unfiled, so the gap is visible rather than silent.

The device's own intent travels with the take into ``qc.csv``
(``device_intent``) next to an ``agrees`` flag, and
``kws-eval --recordings`` reports it as a separate **Field** section. It is
the accuracy of the model that was *deployed* when the recording happened —
a different model from the one being evaluated — and it is never used as a
label. The agreement rate is taken over the takes the device actually
answered: a ring-truncated take carries no device prediction at all, and
counting those as disagreements would quietly depress a figure the device
never had a chance to earn. Truncation itself is readable on the host —
``pull-recordings.sh`` carries ``window_ms`` through and QC marks the row
``truncated=1`` when the audio is shorter than the pre-roll plus that window
— so a cut take is never mistaken for one the recogniser simply ignored.
"Field takes" always means every ``set=field`` row, approved or not, with
approved reported beside it; both ``report.md`` and the eval Field section
count it that way.

Takes are captured at a **looser wake gate** than the shipped detector uses
(:need:`REQ_FW_FIELD_CAPTURE_GATE`), because a take exists only where
something fired: at 0.85 the set could contain neither a missed wake nor a
false trigger. QC's job is to put the production gate back on paper. Every
field row carries ``wake_prob`` (the device's peak at the fire),
``would_fire`` (that peak against ``qc.PROD_WAKE_THRESHOLD`` = 0.85, the one
place the shipped threshold lives on the host side) and ``wake_clip`` (did
Whisper find the phrase in the take — which is not the same question as
whether a usable wake clip could be cut from it: a head-cut fragment is no
positive, but the speaker did wake the device with it). Cross those two
booleans and the two interesting cases fall out: a **near-miss** is a heard
phrase with ``would_fire=0`` — a real "Hey Bus" the deployed detector would
have ignored — and a **false alarm** is a take with no phrase in it and
``would_fire=1``, i.e.
something production would have woken on. Both are counted again against the
capture threshold itself, so the report says what each gate would have done
rather than only what happened. Whisper, never the device, decides whether the
phrase is there — the same rule as for the label.

**Takes from playback tests are never filed.** Playing a TTS clip through a
speaker at the device is how the wake gate is exercised without a person in
the room, and the device records those fires like any other. They are not
recordings of a human speaker: filing them would feed synthesised audio back
into the training set as if it were real, which is exactly the "TTS-vs-real"
shortcut the wake rounds were built to break, and it would inflate the real
share the wake recipe counts on. Pull such a session into a staging
directory of its own and do not run ``kws-qc`` on it — nothing under
``approved/`` may come from a loudspeaker.

Auditing the whole tree
------------------------

``kws-qc`` only ever sees one session. ``scripts/audit-approved.py`` looks at
``approved/`` as a whole and exits non-zero on any finding:

.. code-block:: console

   $ uv run --no-sync python scripts/audit-approved.py [--no-transcribe]

It checks that every clip is readable 16 kHz mono PCM_16 and inside its
set's duration band (wake 0.4-2.0 s, words around 1 s, phrases 0.5-9.8 s,
negatives up to 9.8 s), that each ``index.csv`` and its directory agree in
both directions, that speaker directories are named ``spkNN``, and — by
transcribing the field-derived ``phrases``/``negatives`` clips — that none of
them contains the wake phrase. Guided clips were already matched against
their prompt at QC time and are not re-transcribed. It reports counts per
set, per speaker and per source, reading a clip's source off the QC stamps: a
session whose ``qc.csv`` holds any ``set=field`` row is a field session, and
every path in that stamp's ``written.txt`` is field-derived.

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

The ``kws-dataset build`` cache (``raw_clips_v3.pkl``) holds the MSWC/TTS
material only — the approved recordings are merged in by *every* build, never
baked into it. ``scripts/data-loop.sh`` therefore seeds it by copying
``raw_clips_merged.pkl`` (the full real+TTS cache) when it is absent; with
neither present, mine MSWC once with ``uv run --no-sync kws-data --fetch --v3
--mswc-root <mswc-de-dir>`` (the ``--v3`` flag is what selects the v3
vocabulary and the ``raw_clips_v3.pkl`` name). Never seed it from
``raw_clips_v2.pkl``: that is a 25-word subset with empty clip lists, so the
build re-synthesises ~300 TTS clips per word and the INT8 export then fails
the model-health gate.

The eval report lands at ``$KWS_DATA_ROOT/docs/eval-report-v3.md`` (paths in
the reports are always data-root-relative, never machine-local), alongside a
``.recordings.json`` sidecar carrying the same run's numbers; each run
rewrites its recordings section in place rather than appending a second copy.
Flashing the exported model back onto the device is a manual step, covered
by the remote flashing helper for the device host, not by this pipeline.

Which split the device recordings land in
-----------------------------------------

``kws-dataset build --recordings-split train|auto`` (default ``train``):

- ``train`` — every ``rec:`` speaker is forced into the train split. This is
  the personalised-device model the recorder exists for: a speaker records
  their own voice so the model learns it. With only one or two device
  speakers the global speaker-disjoint draw can otherwise put all of them in
  val/test, training on none of them.
- ``auto`` — device speakers are left to the same global speaker-disjoint
  draw as MSWC speakers, i.e. treated as ordinary held-out candidates.

Either way the eval's labelling is manifest-driven and stays honest: with
``train`` the manifest lists those speakers under ``train``, so
``kws-eval --recordings`` reports their clips as
``user-customised, in-training`` — which is exactly what they now are.

How long will it take
----------------------

Every long stage in ``scripts/data-loop.sh`` (QC, dataset build, train,
export, eval) runs through ``kws-eta run <stage> <size> -- <command>``
(``kws_de.eta``): before the command starts it prints an ETA from that
stage's recorded history on this machine, and after it finishes it appends
the measured wall time to a small ledger
(``$KWS_DATA_ROOT/data/timings.jsonl``, one JSON line per finished run) so
the next prediction improves. ``kws-train`` invoked directly also records,
via ``Timed`` in ``kws_de.train.main``.

The ledger models seconds-per-unit-of-``size`` rather than raw seconds, so
one history serves any input size: QC's ``size`` is the take count in
``sessions.csv``, dataset build's is the raw-clip cache's byte size (a cheap
stand-in for clip count -- it avoids unpickling the cache just to measure
it), train's is ``epochs × train-split rows`` (read back from the manifest
the dataset-build stage just wrote), export's is a constant ``1``, and
eval's is the approved-clip count. A prediction is the median of the last
10 same-stage, same-machine rates, with the 20th/80th-percentile rates as
its low/high band; the first run of a stage on a machine has no history, so
it prints "ETA unknown" instead of guessing.

.. code-block:: console

   $ uv run --no-sync kws-eta predict train 4920
   ETA ~6.3 min (range 5.8–7.1, from 4 runs)

For a long-running training loop that logs its own step count (e.g. an
external microWakeWord run), ``kws-eta watch <logfile> --total N --pattern
'Step #(\d+)'`` tails the log and prints ``step 8200/20000, 610 steps/min,
ETA 19.3 min`` every 30 s, alongside the ledger's prediction if ``--stage``
names one. The machine tag in every ledger row is ``KWS_HOST_TAG`` if set,
else ``platform.node()`` hashed to 8 hex characters -- never the raw
hostname, so the ledger is safe to commit or share.

Requirements
------------

.. needtable::
   :types: req
   :filter: id.startswith("REQ_PIPE")
   :columns: id, title, status
   :style: table

See :doc:`requirements` for the full requirement text and :doc:`tests` for
the tests each one traces to.
