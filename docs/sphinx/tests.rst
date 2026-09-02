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
   :links: REQ_FW_PROMPT_SHUFFLE_SEED

   ``firmware/test/test_prompts.c``: same seed twice → identical shuffled
   order; different seed → different order; every prompt index appears
   exactly once.

.. test:: Energy VAD open/close thresholds
   :id: TEST_VAD_ENERGY
   :status: passing
   :links: REQ_FW_VAD_ENDPOINT

   ``firmware/test/test_vad.c``: quiet frames never open speech; loud
   frames open after 2 consecutive frames; speech stays open through
   ``VAD_TRAILING_FRAMES - 1`` silent frames and closes exactly on the
   next.

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

.. test:: v3 dataset build reads the approved tree and records provenance
   :id: TEST_V3_PROVENANCE
   :status: passing
   :links: REQ_PIPE_APPROVED_LAYOUT

   ``tests/test_data_v3_provenance.py``: ``recordings_root`` prefers
   ``approved/`` over the legacy layout when it exists; negative windows
   are cut at 1 s hops tagged ``rec:<spkNN>`` (and a wrong-sample-rate file
   is warned about and skipped, not silently included).

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
           REQ_FW_23_CLASSES, REQ_FW_RECOGNISE_LOG

   ``firmware/README.md`` "Manual test checklist": record 3 words + 1
   sentence + 1 negative → USB → pull →
   ``column -s, -t < data/recordings/sessions.csv`` lists them; toggle to
   Recognise → say "Licht" → word appears, inference < 30 ms;
   ``recognise.log`` replays through ``stream.KeywordStream`` with the same
   events. Run by hand on real M5Stack CoreS3 hardware; not automated.

.. note::

   ``REQ_FW_RECORD_CLIP_REJECT``, ``REQ_FW_STORAGE_MIN_FREE``, and
   ``REQ_FW_DATA_NOT_COMMITTED`` have no test linked above — see
   :doc:`traceability` for why these are open gaps rather than omissions.
