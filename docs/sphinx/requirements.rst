Requirements
============

The CoreS3 dual-mode firmware (guided recorder + on-device recogniser)
requirements, extracted from the approved design spec
(``docs/superpowers/specs/2026-09-01-cores3-firmware-design.md``) and its
"Global Constraints". One requirement per real, testable behaviour — not
every implementation detail is a requirement.

Front end (MFCC)
-----------------

.. req:: C MFCC parity with the Python front end
   :id: REQ_FW_MFCC_PARITY
   :status: implemented

   ``mfcc_compute``/``mfcc_push_frame``/``mfcc_finish`` (``firmware/main/mfcc.c``)
   reproduce ``kws_de.features.mfcc`` bit-for-bit up to float rounding:
   16000 Hz sample rate, 480-sample analysis window, 320-sample hop, 40 mel
   bands, 10 MFCC coefficients, 49 frames per 1 s clip. All five constants
   come from the single source ``kws_de.config``, generated into
   ``firmware/main/gen/features_config.h`` — never hard-coded a second time
   in C.

.. req:: MFCC int8 quantisation matches the model's input scale/zero-point
   :id: REQ_FW_MFCC_QUANTIZE
   :status: implemented

   ``mfcc_quantize`` converts float MFCC features to int8 as
   ``q = round(x / scale) + zero_point``, clamped to ``[-128, 127]``, using
   the exact scale/zero-point ``kws-export --firmware`` wrote into
   ``model_config.h`` for the exported command model.

Detector
--------

.. req:: Streaming detector parameters match the Python reference
   :id: REQ_FW_DETECTOR_PARAMS
   :status: implemented

   ``stream_push`` (``firmware/main/stream.c``), a C port of
   ``kws_de.stream.KeywordStream``, uses ``smooth_win=3``, ``threshold=0.5``,
   ``min_consecutive=2``, ``gap_steps=2`` — the same defaults as
   ``kws_de.eval.run_catalog_eval`` — sourced from ``features_config.h``,
   never literal in ``stream.c``.

.. req:: Energy VAD opens/closes speech at the specified thresholds
   :id: REQ_FW_VAD_ENDPOINT
   :status: implemented

   ``vad_push`` (``firmware/main/vad.c``) opens speech when RMS exceeds
   ``max(noise_floor * 4, 300)`` for 2 consecutive 20 ms frames, and closes
   it after ``VAD_TRAILING_FRAMES`` = 25 consecutive frames below threshold
   (500 ms). The noise floor tracks RMS exponentially (alpha 0.05) only
   while not in speech.

Recorder
--------

.. req:: Guided recorder captures two takes per prompt
   :id: REQ_FW_RECORD_TWO_TAKES
   :status: implemented

   The recorder captures ``TAKES_PER_PROMPT`` = 2 reads of each prompt
   before advancing, so a bad first read can be reviewed/redone without
   losing the second.

.. req:: Recording time caps
   :id: REQ_FW_RECORD_CAPS
   :status: implemented

   Fixed caps on the record state machine: 300 ms pre-roll pulled from the
   always-on ring buffer, 500 ms of trailing silence closes a take, 4000 ms
   maximum word-prompt length, 6000 ms maximum sentence/negative-prompt
   length, 8000 ms no-speech timeout forces an auto-redo, and a 700 ms hold
   after a successful save before auto-advancing to the next prompt.

.. req:: Clipped takes are discarded and redone
   :id: REQ_FW_RECORD_CLIP_REJECT
   :status: implemented

   Any sample at full scale (±32767) during capture marks the take
   clipped: the level bar turns red and the take is discarded, forcing a
   redo rather than saving distorted audio.

.. req:: Saved takes are canonical 16 kHz/16-bit mono WAV
   :id: REQ_FW_RECORD_WAV_FORMAT
   :status: implemented

   ``wav_write_header`` (``firmware/main/wav.c``) writes a standard 44-byte
   RIFF/WAVE/fmt/data header for mono 16-bit PCM at the capture sample rate
   (16000 Hz); every saved take on ``/rec`` uses this exact header.

.. req:: Recording filenames are ASCII slugs
   :id: REQ_FW_RECORD_FILENAME_SLUG
   :status: implemented

   Prompt text is slugged to filesystem-safe ASCII before it is written to
   flash or a filename: German umlauts/ß transliterate (ä→ae, ß→ss),
   spaces become ``-``, output matches ``[a-z0-9-]+``. Slugs are
   precomputed on the host by ``kws_de.firmware_gen.slug`` into
   ``gen/prompts.h``; the firmware only reads them (``prompt_slug``).

.. req:: Speaker ids are numeric only
   :id: REQ_FW_RECORD_SPEAKER_ID
   :status: implemented

   Speaker ids are an NVS-backed numeric counter only (``spkNN``,
   incremented by "+ Speaker"). No speaker name, or anything else
   identifying, is ever written to the device or the repository.

.. req:: Each session appends a session.csv row per take
   :id: REQ_FW_RECORD_SESSION_CSV
   :status: implemented

   Every take appends one row (``prompt,file,ms,peak_dbfs,set,seed,ts``) to
   ``/rec/<speaker>/session.csv`` on save, giving a per-speaker audit trail
   that ``scripts/pull-recordings.sh`` merges into
   ``data/recordings/sessions.csv``.

.. req:: Prompt order is reproducible from its seed
   :id: REQ_FW_PROMPT_SHUFFLE_SEED
   :status: implemented

   ``prompt_session_init`` (``firmware/main/prompts.c``) shuffles a prompt
   set with a 32-bit xorshift PRNG seeded from a per-session seed shown on
   screen (``seed 17``); the same seed always reproduces the same order, so
   a session is fully reconstructible from ``session.csv``.

.. req:: Negative prompts contain no command vocabulary
   :id: REQ_FW_NEGATIVE_PROMPTS
   :status: implemented

   ``config.NEGATIVE_PROMPTS`` has at least 15 fixed German sentences, none
   of which contains a device, zone, or action word (case-insensitively),
   so the recorded negative class cannot accidentally teach the model a
   command word in a "wrong" context.

.. req:: Generated prompt tables cover labels and catalog exactly once
   :id: REQ_FW_PROMPT_SET_COVERAGE
   :status: implemented

   The generated ``words`` table covers exactly ``config.COMMAND_LABELS``
   minus ``_unknown_``/``_silence_``; the generated ``sentences`` table
   covers exactly the ``phrases.py`` catalog; the generated ``negatives``
   table covers exactly ``config.NEGATIVE_PROMPTS``; every prompt slug
   across all three tables is unique.

Storage / USB
-------------

.. req:: Recording stops before storage is exhausted
   :id: REQ_FW_STORAGE_MIN_FREE
   :status: implemented

   Recording is disabled with a REC_FULL banner once free space on
   ``/rec`` drops below ``STORAGE_MIN_FREE_BYTES`` = 200 KB (``storage.h``);
   the USB drive mode remains available regardless, so recordings already
   on flash can still be pulled off.

.. req:: /rec has exactly one owner at a time
   :id: REQ_FW_USB_SINGLE_OWNER
   :status: implemented

   Entering USB drive mode unmounts ``/rec`` from the app before exposing
   the partition as a USB MSC device (``usb_drive_enter``); leaving it
   stops the MSC device and remounts ``/rec`` for the app
   (``usb_drive_exit``). The app and the host PC never hold the FAT
   filesystem open at the same time.

.. req:: Host pull script copies, merges, and clears recordings
   :id: REQ_FW_USB_PULL
   :status: implemented

   ``scripts/pull-recordings.sh`` auto-detects the mounted ``KWSREC``
   volume, ``rsync -a``s ``spk*/`` into ``data/recordings/``, appends each
   ``session.csv`` (prefixed with speaker id and host date) to
   ``data/recordings/sessions.csv``, deletes the on-device copy only if
   rsync exited 0, and ejects the volume. Re-running against an
   already-emptied drive is a no-op.

Recogniser
----------

.. req:: TFLite Micro op resolver is the exact gated set
   :id: REQ_FW_TFLM_OPSET
   :status: implemented

   ``recognise.cc`` registers exactly the 7 ops gated by
   ``tests/test_transducer.py``: ``CONV_2D``, ``DEPTHWISE_CONV_2D``,
   ``FULLY_CONNECTED``, ``MEAN``, ``SOFTMAX``, ``RESHAPE``, ``ADD`` — via
   ``MicroMutableOpResolver``, never ``AllOpsResolver`` — so a model that
   needs an unlisted op fails to build rather than silently pulling in
   unaudited kernels.

.. req:: Recogniser model is the 23-class v2 command model
   :id: REQ_FW_23_CLASSES
   :status: implemented

   The on-device model is ``models/command.tflite``: int8 input
   ``[1, 49, 10, 1]``, labels = ``config.COMMAND_LABELS`` (23 classes),
   loaded via ``recognise_start``/``assert_model_healthy``-gated export.

.. req:: A broken model is rejected before it reaches firmware
   :id: REQ_FW_MODEL_HEALTH_GATE
   :status: implemented

   ``kws_de.export.assert_model_healthy`` refuses to write
   ``model_data.h``/``model_config.h`` for a mode-collapsed or
   random-accuracy model, so a training regression cannot silently ship a
   non-functional recogniser to the device.

.. req:: Recognise mode can log detections for offline comparison
   :id: REQ_FW_RECOGNISE_LOG
   :status: implemented

   Toggling "Log" in recognise mode appends ``ts,word,conf,ms`` per
   detection to ``/rec/recognise.log``; the log format is exactly what's
   needed to replay through ``kws_de.stream.KeywordStream`` on the host
   and compare event-for-event against the on-device detector.

Wake word ("Hey Bus")
---------------------

.. req:: On-device wake features match microWakeWord's training front-end
   :id: REQ_FW_WAKE_FRONTEND_PARITY
   :status: implemented

   ``firmware/main/wakefront.c`` drives the TFLite-Micro audio frontend
   vendored under ``firmware/main/microfrontend/`` with exactly the
   ``FrontendConfig`` microWakeWord trains with (16 kHz, 30 ms window,
   10 ms step, 40 channels, 125-7500 Hz, PCAN on, log scaling), and
   requantises its uint16 output to int8 with microWakeWord's own integer
   expression ``(v * 256 + 333) / 666 - 128``. The rows it emits are
   bit-identical to ``pymicro_features``' for the same PCM.

.. req:: Wake detection fires once per utterance
   :id: REQ_FW_WAKE_DETECT
   :status: implemented

   ``wake.cc`` runs the streaming ``models/hey_bus.tflite`` interpreter
   once per 3 feature rows (30 ms of audio), keeping the interpreter and
   its resource variables alive between steps and resetting them when the
   mode is entered. A detection needs ``WAKE_THRESHOLD`` (0.99) on
   ``WAKE_MIN_CONSECUTIVE`` (2) consecutive steps, after which
   ``WAKE_REFRACTORY_MS`` (1500) suppresses further fires, so one spoken
   "Hey Bus" produces exactly one fire. Each fire is logged to
   ``/rec/wake.log`` as ``[Wake] <ms> <prob>``.

.. req:: A wake detection is confirmed on screen and by ear
   :id: REQ_FW_WAKE_BEEP
   :status: implemented

   On a fire the screen background turns green for ``WAKE_FLASH_MS``
   (600 ms) and ``beep.c`` plays a 150 ms 1 kHz tone through the AW88298
   amplifier. The speaker shares one full-duplex I2S channel pair with the
   microphone, so it is opened with the mic's exact format (16 kHz,
   16-bit, 2 channels); any other rate would be rejected by
   ``esp_codec_dev`` and take capture down.

.. req:: Wake mode runs the wake model alone
   :id: REQ_FW_WAKE_ISOLATED
   :status: implemented

   ``UI_MODE_WAKE`` calls ``recognise_set_active(false)`` on entry and
   ``wake_set_active(false)`` on exit, so no command-word inference runs
   while wake mode is active and no wake inference runs outside it. The
   two tasks share priority 3 on core 1 and only one is ever enabled, so
   what the wake screen reports is the wake model's behaviour and nothing
   else's.

Selection menu and remote control
----------------------------------

.. req:: Every mode is reached from, and returns to, one selection menu
   :id: REQ_FW_MENU_FLOW
   :status: implemented

   ``app_main`` boots into ``UI_MODE_MENU`` (``ui_show_menu()``), a 2x2
   grid of four buttons — Recognition, Hey Bus, Record, USB — each a
   direct ``app_set_mode()`` call. Every other screen's back/abort button
   (record, recognise, wake, USB) calls ``app_set_mode(UI_MODE_MENU)``,
   so the menu is the only hub: no mode links directly to another mode.

.. req:: Entering Record always starts a fresh guided session
   :id: REQ_FW_RECORD_SESSION
   :status: implemented

   ``app_set_mode(UI_MODE_RECORD)`` posts ``REC_CMD_START_SESSION``,
   which bumps the speaker id (``nvs_bump_speaker``) and starts the
   sentence set. On ``REC_DONE`` the recorder auto-chains: sentences
   completing re-seeds the negative set and continues; negatives
   completing sets ``REC_SESSION_DONE`` (with a running
   ``saved_takes`` count in ``record_status_t``), which
   ``ui_record_refresh()`` turns into the success screen
   (``ui_show_success``). Aborting mid-session (Abbrechen) instead posts
   ``REC_CMD_PAUSE`` and returns straight to the menu — no success
   screen. ``PROMPT_WORDS`` remains in the code but is not reachable from
   this flow.

.. req:: The device accepts remote mode/status commands over the serial console
   :id: REQ_FW_REMOTE_MODE
   :status: implemented

   ``console.c`` reads newline-terminated commands from ``stdin`` (the
   UART console) in a low-priority task: ``mode
   menu|record|recognise|wake|usb`` calls ``app_set_mode()``; ``status``
   reports the current mode and, in record mode, the recorder's
   phase/index/count/speaker. Every command ends with an ``ok`` or ``err
   <reason>`` line, so a host script driving the device (e.g. ``echo
   'mode usb' > /dev/cu.usbmodemNNN``) can tell when a command finished.

Build / CI
----------

.. req:: IDF version is pinned identically in three places
   :id: REQ_FW_IDF_PIN
   :status: implemented

   The ESP-IDF tag (``v5.5.5``) is pinned in ``firmware/README.md``,
   ``firmware/CMakeLists.txt`` (build-time version check), and
   ``.github/workflows/firmware.yml``; the three must agree.

.. req:: Config-derived generated headers are deterministic
   :id: REQ_FW_HEADER_DETERMINISM
   :status: implemented

   ``firmware/main/gen/{labels,prompts,features_config,test_vectors}.h``
   are produced by ``kws-fwgen``/``kws-export --firmware`` from
   ``kws_de.config``/``kws_de.phrases`` and the exported model; running the
   generator again reproduces the committed files (within float
   last-digit tolerance for BLAS/SIMD-derived tables). ``kws-fwgen
   --check`` fails CI if they've drifted from the generator.

.. req:: Pure-logic C units build and test without ESP-IDF
   :id: REQ_FW_HOST_TESTS_NO_IDF
   :status: implemented

   ``mfcc.c``, ``stream.c``, ``wav.c``, ``prompts.c``, ``vad.c``, and
   ``wakefront.c`` (with the vendored microfrontend) have no ESP-IDF
   dependency and build/run as host binaries with plain ``cc``/``c++``
   via ``firmware/test/Makefile`` (``make -C firmware/test``), on Mac, the
   Pi, and CI, with no Docker/IDF toolchain needed.

.. req:: Shell scripts pass shellcheck
   :id: REQ_FW_SHELLCHECK
   :status: implemented

   ``scripts/pull-recordings.sh`` and ``scripts/flash.sh`` pass
   ``shellcheck`` in CI.

Data hygiene
------------

.. req:: No speaker names, machine paths, or provenance leak into the repo
   :id: REQ_FW_NO_LEAKS
   :status: implemented

   No speaker names, machine-specific paths, or vehicle/brand/decompiled-app
   provenance appear anywhere in the committed repository; speaker ids are
   ``spkNN`` only.

.. req:: Large/generated artifacts are never committed
   :id: REQ_FW_DATA_NOT_COMMITTED
   :status: implemented

   ``data/``, ``models/``, ``firmware/build/``,
   ``firmware/managed_components/``, ``firmware/sdkconfig``, and
   ``firmware/dependencies.lock`` are gitignored and never committed;
   ``firmware/main/gen/*.h`` (the config/model-derived headers) ARE
   committed, since CI builds without training.
