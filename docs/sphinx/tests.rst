Tests
=====

Every test below states how it runs. See :doc:`traceability` for the
req → test coverage table and known gaps.

Host C tests (plain ``cc``, no ESP-IDF)
----------------------------------------

Run via ``make -C firmware/test`` (Mac, Pi, or CI's ``host-test`` job); each
binary is a small ``assert``-based program built by
``firmware/test/Makefile``.

.. test:: MFCC output matches the Python golden vector
   :id: TEST_MFCC_PARITY
   :status: passing
   :links: REQ_FW_MFCC_PARITY, REQ_FW_HOST_TESTS_NO_IDF

   ``firmware/test/test_mfcc.c``: runs ``mfcc_compute`` over a fixture PCM
   clip and asserts every coefficient is within ``1e-2`` of the
   ``kws_de.features.mfcc`` output for the same clip, exported as
   ``gen/test_vectors.h`` (``TV_PCM``/``TV_MFCC``).

.. test:: Streaming detector fire/gap/silence behaviour
   :id: TEST_STREAM_DETECTOR
   :status: passing
   :links: REQ_FW_DETECTOR_PARAMS, REQ_FW_HOST_TESTS_NO_IDF

   ``firmware/test/test_stream.c``: replays synthetic posterior sequences
   through ``stream_push`` and asserts the fired-event sequence (debounce,
   gap, re-fire) matches expectations.

.. test:: WAV header bytes at three sample counts
   :id: TEST_WAV_HEADER
   :status: passing
   :links: REQ_FW_RECORD_WAV_FORMAT, REQ_FW_HOST_TESTS_NO_IDF

   ``firmware/test/test_wav.c``: asserts ``wav_write_header`` produces the
   correct RIFF/fmt/data chunk sizes for 0, 16000, and 96000 samples at
   16 kHz mono 16-bit.

.. test:: Prompt shuffle determinism and coverage
   :id: TEST_PROMPTS_SHUFFLE
   :status: passing
   :links: REQ_FW_PROMPT_SHUFFLE_SEED, REQ_FW_RECORD_WAKE_SET

   ``firmware/test/test_prompts.c``: same seed twice → identical shuffled
   order; different seed → different order; every prompt index appears
   exactly once. Also asserts the wake set: 5 prompts, all "Hey Bus",
   slug "hey-bus", ``prompt_set_name(PROMPT_WAKE)`` == "wake", and
   ``prompt_takes_per_prompt`` returns 1 for the wake set vs. 2 for the
   normal sets. ``prompt_hangover_ms`` returns 500 for ``PROMPT_WORDS`` and
   1200 for sentences/negatives/wake.

.. test:: Energy VAD open/close thresholds
   :id: TEST_VAD_ENERGY
   :status: passing
   :links: REQ_FW_VAD_ENDPOINT, REQ_FW_RECORD_HANGOVER

   ``firmware/test/test_vad.c``: quiet frames never open speech; loud
   frames open after 2 consecutive frames; speech stays open through
   ``VAD_TRAILING_FRAMES - 1`` silent frames and closes exactly on the
   next. A synthetic 200 ms-burst/800 ms-silence/200 ms-burst signal stays
   one segment at the 1200 ms sentence/negs/wake hangover but splits into
   two at the 500 ms word hangover. A 40 ms transient followed by silence
   leaves ``speech_total`` under 200 ms, the false-start threshold.

.. test:: Wake front-end reproduces microWakeWord's features exactly
   :id: TEST_WAKE_FRONTEND_PARITY
   :status: passing
   :links: REQ_FW_WAKE_FRONTEND_PARITY, REQ_FW_HOST_TESTS_NO_IDF

   ``firmware/test/test_wakefront.c``: pushes ``gen/wake_test_vectors.h``'s
   1 s of synthetic PCM through ``wakefront_push`` in 10 ms strides and
   asserts all 98 x 40 int8 feature values equal ``WT_FEATURES`` — the rows
   ``pymicro_features`` produced for the same PCM under microWakeWord's
   int8 requantisation. Max deviation must be 0, not a tolerance: both
   sides run the same vendored integer C, so any difference means the
   config or the requantisation drifted. Also asserts the 3-row
   ``wakefront_take`` block the model consumes is oldest-row-first.

Python tests (pytest)
----------------------

Run via ``uv run pytest tests/`` (CI's ``test`` job in ``ci.yml``).

.. test:: Generated headers are deterministic (config-derived)
   :id: TEST_FIRMWARE_GEN_DETERMINISM
   :status: passing
   :links: REQ_FW_HEADER_DETERMINISM

   ``tests/test_firmware_gen.py``; also exercised in CI as
   ``uv run kws-fwgen --check firmware/main/gen`` (the ``gen-fresh`` job),
   which fails if the committed headers drift from a fresh run of the
   generator.

.. test:: Slug transliteration is ASCII and stable
   :id: TEST_FIRMWARE_GEN_SLUG
   :status: passing
   :links: REQ_FW_RECORD_FILENAME_SLUG

   ``tests/test_firmware_gen.py::test_slug_is_ascii_and_stable``: checks
   umlaut/ß transliteration and the ``[a-z0-9-]+`` output shape.

.. test:: Negative prompts contain no command vocabulary
   :id: TEST_FIRMWARE_GEN_NEGATIVES
   :status: passing
   :links: REQ_FW_NEGATIVE_PROMPTS

   ``tests/test_firmware_gen.py::test_negative_prompts_contain_no_command_words``.

.. test:: Prompt tables cover labels/catalog/negatives exactly once
   :id: TEST_FIRMWARE_GEN_COVERAGE
   :status: passing
   :links: REQ_FW_PROMPT_SET_COVERAGE, REQ_FW_23_CLASSES

   ``tests/test_firmware_gen.py::test_prompt_sets_cover_labels_and_catalog``.

.. test:: Exported model_config.h carries correct quantisation/arena info
   :id: TEST_EXPORT_MODEL_CONFIG
   :status: passing
   :links: REQ_FW_MFCC_QUANTIZE

   ``tests/test_export_firmware.py::test_write_model_config_reports_quant_and_arena``:
   asserts ``model_config.h`` defines input scale/zero-point, class count,
   and a 4096-aligned nonzero arena size.

.. test:: Model-health gate rejects broken models
   :id: TEST_EXPORT_MODEL_HEALTH_GATE
   :status: passing
   :links: REQ_FW_MODEL_HEALTH_GATE

   ``tests/test_export_firmware.py::test_model_health_gate_catches_broken_models``:
   regression guard for the broken-model bug — a mode-collapsed or
   random-accuracy model must be rejected by ``assert_model_healthy``
   before export.

.. test:: Python KeywordStream reference behaviour
   :id: TEST_STREAM_PY
   :status: passing
   :links: REQ_FW_DETECTOR_PARAMS

   ``tests/test_stream.py``: the host detector's reference implementation
   that ``firmware/main/stream.c`` ports; TEST_STREAM_DETECTOR checks the C
   port stays behaviourally identical.

.. test:: Python MFCC reference/shape/determinism
   :id: TEST_FEATURES_MFCC
   :status: passing
   :links: REQ_FW_MFCC_PARITY

   ``tests/test_features.py``: shape/dtype, padding/truncation, and a
   golden-vector regression test for ``kws_de.features.mfcc`` — the
   reference ``mfcc_compute`` is checked against.

.. test:: TFLM op set is exactly the gated 7 ops
   :id: TEST_TRANSDUCER_OPSET
   :status: passing
   :links: REQ_FW_TFLM_OPSET

   ``tests/test_transducer.py``: exported int8 TFLite models are checked
   against the allowed op set before ``recognise.cc``'s
   ``MicroMutableOpResolver`` registration list is trusted to cover them.

.. test:: pull-recordings.sh copies, merges, and clears a fake drive
   :id: TEST_PULL_RECORDINGS
   :status: passing
   :links: REQ_FW_USB_PULL, REQ_FW_RECORD_SESSION_CSV

   ``tests/test_pull_recordings.py``: spawns the script against a
   ``tmp_path`` laid out like the device (``spk03/licht/001.wav`` +
   ``session.csv`` + ``recognise.log``), asserts files land under
   ``data/recordings/``, ``sessions.csv`` is appended, and the source is
   emptied.

.. test:: QC audio/content gates, segmentation, and approved-tree layout
   :id: TEST_QC_RULES
   :status: passing
   :links: REQ_PIPE_QC_AUDIO, REQ_PIPE_QC_CONTENT, REQ_PIPE_SEGMENT,
           REQ_PIPE_APPROVED_LAYOUT

   ``tests/test_qc.py``: the audio gate against synthetic clipped/quiet/
   short/long/unreadable WAVs; the content gate's word/sentence/negative
   rules with a stubbed transcriber, including umlaut/ß/"Prozent"
   normalisation and the >5-letter edit-distance-1 cutoff; word
   segmentation centres the 1 s window on the word span and zero-pads at
   the edges; ``run_qc`` end to end writes the approved tree with
   collision-proof numbering and is idempotent across re-runs of the same
   stamp and across separate stamps for the same speaker.

.. test:: QC numerals, glued tokens, and short-keyword boundaries
   :id: TEST_QC_NUMERALS_GLUED
   :status: passing
   :links: REQ_PIPE_QC_CONTENT

   ``tests/test_qc.py``:
   ``test_content_gate_numerals_heard_as_digits`` (Whisper's "50" matches a
   prompt's "fünfzig"),
   ``test_content_gate_glued_keywords_still_order_sensitive_and_short_word_exact``
   ("Lichtdach" matches "Licht Dach", "Licht heller Dach" does not), and
   ``test_content_gate_short_keyword_boundary_check`` (a short keyword never
   matches inside an unrelated longer word — "an" not in "dank").

.. test:: QC negative rule for 2-letter keywords
   :id: TEST_QC_NEGATIVE_KEYWORD
   :status: passing
   :links: REQ_PIPE_QC_CONTENT

   ``tests/test_qc.py``:
   ``test_content_gate_negatives_two_letter_keyword_needs_whole_token_or_repeat``
   — one hallucinated 2-letter keyword does not reject a negative, a keyword
   of 3 letters or more does, and a 2-letter keyword heard twice does.

.. test:: QC wake set — rule, duration cap, and approved tree
   :id: TEST_QC_WAKE
   :status: passing
   :links: REQ_PIPE_QC_AUDIO, REQ_PIPE_QC_CONTENT, REQ_PIPE_APPROVED_LAYOUT

   ``tests/test_qc.py``:
   ``test_required_tokens_and_content_gate_wake`` (the
   ``(hey|hej|he|hei)(bus|buss|bos|boss)`` rule accepts "Hej Boss" and
   rejects "Hallo") and
   ``test_run_qc_writes_approved_wake_set_and_is_idempotent``
   (``approved/wake/<spkNN>/<spkNN>_<NNN>.wav`` + ``wake/index.csv``,
   counted in ``report.md``, no duplication on a re-run of the same stamp).
   The 4000 ms wake cap rides on
   ``test_audio_gate_ok_clipped_quiet_short``'s per-set cap checks.

.. test:: ingest.sh pulls a session without deleting on the remote host
   :id: TEST_INGEST
   :status: passing
   :links: REQ_PIPE_INGEST

   ``tests/test_ingest.py``: runs the script against fake ``ssh``/``scp``/
   ``rsync`` on ``PATH``, asserting the serial commands fire in order
   (usb -> pull -> rsync -> menu), the local pull is count-verified against
   the remote, and an unreachable host (ssh exit 255) exits 3.

.. test:: Recordings-based eval figures and labels
   :id: TEST_EVAL_RECORDINGS
   :status: passing
   :links: REQ_PIPE_EVAL_LABELS

   ``tests/test_eval_recordings.py``: builds a tiny approved tree with a
   stub predict function and asserts the isolated/e2e/false-accept figures
   are computed and reported under the correct ``"held-out"`` /
   ``"user-customised, in-training"`` label depending on whether the
   clip's speaker appears in a given training manifest's ``train`` split.
   ``test_eval_recordings_on_a_run_qc_tree`` runs the eval over a tree
   ``qc.run_qc`` itself produced (stub transcriber), so the index-file
   convention cannot drift between producer and consumer; the report names
   the model file it measured, data-root-relative, and a second run rewrites
   its section in place instead of appending a second copy.

.. test:: v3 dataset build reads the approved tree and records provenance
   :id: TEST_V3_PROVENANCE
   :status: passing
   :links: REQ_PIPE_APPROVED_LAYOUT, REQ_PIPE_RECORDINGS_IN_BUILD

   ``tests/test_data_v3_provenance.py``: ``recordings_root`` prefers
   ``approved/`` over the legacy layout when it exists; negative windows
   are cut at 1 s hops tagged ``rec:<spkNN>`` (and a wrong-sample-rate file
   is warned about and skipped, not silently included);
   ``merge_recordings`` folds an approved tree written after the cache into
   a cached build and does not duplicate it on a second build;
   ``force_rec_to_train`` moves device clips out of val/test; and
   ``dataset.build`` over a cache with no ``rec:`` clips produces a manifest
   whose *train* split carries the device speaker and counts it as a
   recording.

CI build/gate jobs
-------------------

.. test:: Firmware builds under the pinned IDF version
   :id: TEST_CI_FIRMWARE_BUILD
   :status: passing
   :links: REQ_FW_IDF_PIN, REQ_FW_TFLM_OPSET, REQ_FW_23_CLASSES

   ``.github/workflows/firmware.yml`` ``build`` job:
   ``espressif/esp-idf-ci-action@v1`` at ``v5.5.5``, ``idf.py build`` for
   ``esp32s3``, then a merged flashable binary is uploaded as an artifact.

.. test:: Committed generated headers match a fresh generator run
   :id: TEST_CI_GEN_FRESH
   :status: passing
   :links: REQ_FW_HEADER_DETERMINISM

   ``.github/workflows/firmware.yml`` ``gen-fresh`` job:
   ``uv run kws-fwgen --check firmware/main/gen``.

.. test:: Host C tests and shellcheck
   :id: TEST_CI_HOST_TEST
   :status: passing
   :links: REQ_FW_HOST_TESTS_NO_IDF, REQ_FW_SHELLCHECK

   ``.github/workflows/firmware.yml`` ``host-test`` job:
   ``make -C firmware/test`` then ``shellcheck scripts/*.sh``.

.. test:: No secrets/leaks in the repo
   :id: TEST_CI_GITLEAKS
   :status: passing
   :links: REQ_FW_NO_LEAKS

   ``.github/workflows/ci.yml`` ``gitleaks`` job: ``gitleaks detect
   --source . --no-git --redact``.

Manual, on-hardware checklist
-------------------------------

Hardware-coupled behaviour (audio codec, FAT storage, USB MSC, LVGL UI,
TFLM wiring) has no host equivalent; it is checked by hand against a real
CoreS3 before merging changes that touch it. See ``firmware/README.md``
"Manual test checklist".

.. test:: On-device record → USB → pull → recognise walkthrough
   :id: TEST_MANUAL_DEVICE_WALKTHROUGH
   :status: manual
   :links: REQ_FW_RECORD_TWO_TAKES, REQ_FW_RECORD_CAPS,
           REQ_FW_RECORD_SPEAKER_ID, REQ_FW_RECORD_SESSION_CSV,
           REQ_FW_USB_SINGLE_OWNER, REQ_FW_VAD_ENDPOINT,
           REQ_FW_RECORD_HANGOVER, REQ_FW_23_CLASSES, REQ_FW_RECOGNISE_LOG

   ``firmware/README.md`` "Manual test checklist": record 3 words + 1
   sentence + 1 negative → USB → pull →
   ``column -s, -t < data/recordings/sessions.csv`` lists them; toggle to
   Recognise → say "Licht" → word appears, inference < 30 ms;
   ``recognise.log`` replays through ``stream.KeywordStream`` with the same
   events. Run by hand on real M5Stack CoreS3 hardware; not automated.

.. test:: On-device "Hey Bus" wake test mode
   :id: TEST_MANUAL_WAKE_MODE
   :status: manual
   :links: REQ_FW_WAKE_DETECT, REQ_FW_WAKE_BEEP, REQ_FW_WAKE_ISOLATED

   ``firmware/README.md`` "Manual test checklist": tap **Wake** on the
   record screen → the probability updates live and stays low on silence;
   say "Hey Bus" → the screen flashes green, the speaker beeps once, and
   the fire count goes up by exactly one per utterance; other words
   (including command words like "Licht") do not fire, confirming no
   command recognition is running; tap **Menu** to leave → back in Record
   mode, ``/rec/wake.log`` carries one ``[Wake] <ms> <prob>`` line per
   fire. Run by hand on real M5Stack CoreS3 hardware; not automated.

.. test:: Selection-menu flow, guided session auto-chain, and remote control
   :id: TEST_MANUAL_MENU_FLOW
   :status: manual
   :links: REQ_FW_MENU_FLOW, REQ_FW_RECORD_SESSION, REQ_FW_RECORD_WAKE_SET,
           REQ_FW_REMOTE_MODE, REQ_FW_USB_CDC_CONSOLE

   ``firmware/README.md`` "Manual test checklist": device boots to the
   5-button menu; each of Recognition/Hey Bus/Record/Hey Bus
   aufnehmen/USB opens its screen, and that screen's back button
   (Abbrechen on Record/Hey Bus aufnehmen) returns to the menu; tapping
   **Record** bumps the speaker id and starts the sentence set, completing
   it auto-chains into the negative set without any button press, and
   completing negatives shows "Fertig - danke!" with the speaker id and a
   saved-take count, whose **Menu** button returns to the selection menu;
   aborting mid-session instead returns straight to the menu with no
   success screen. Tapping **Hey Bus aufnehmen** bumps the speaker id and
   prompts 5 single-take "Hey Bus" reads straight to the same success
   screen, with no negatives chained on. Separately, over the serial
   console: ``echo 'mode wake' > /dev/cu.usbmodemNNN`` switches the
   device to wake mode and ``echo 'status'`` reports it; ``mode
   recordwake`` then ``status`` reports the wake session's
   phase/index/speaker. Tapping **USB** (or ``mode usb``) mounts
   ``KWSREC`` and a new ``/dev/cu.usbmodemNNN`` (the CDC-ACM port) shows
   up alongside/in place of the original console port; ``echo 'status' >``
   that new port answers ``mode usb`` / ``ok``, and ``echo 'mode menu' >``
   it unmounts the drive and returns to the menu, with the original
   console port working again afterwards. Run by hand on real M5Stack
   CoreS3 hardware; not automated.

.. note::

   ``REQ_FW_RECORD_CLIP_REJECT``, ``REQ_FW_STORAGE_MIN_FREE``, and
   ``REQ_FW_DATA_NOT_COMMITTED`` have no test linked above — see
   :doc:`traceability` for why these are open gaps rather than omissions.
