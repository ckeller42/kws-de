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
   it after ``v->trailing_frames`` consecutive frames below threshold, set
   per call by ``vad_reset()`` (``VAD_TRAILING_FRAMES`` = 25 frames / 500 ms
   is the default). The noise floor tracks RMS exponentially (alpha 0.05)
   only while not in speech. See :need:`REQ_FW_RECORD_HANGOVER` for the
   per-prompt-set hangover the recorder feeds in.

Recorder
--------

.. req:: Guided recorder captures two takes per prompt
   :id: REQ_FW_RECORD_TWO_TAKES
   :status: implemented

   ``prompt_takes_per_prompt`` (``firmware/main/prompts.c``) returns 2 reads
   per prompt for the word/sentence/negative sets before advancing, so a bad
   first read can be reviewed/redone without losing the second. (The
   "Hey Bus" wake set is the one exception — see
   :need:`REQ_FW_RECORD_WAKE_SET`.)

.. req:: Recording time caps
   :id: REQ_FW_RECORD_CAPS
   :status: implemented

   Fixed caps on the record state machine: 300 ms pre-roll pulled from the
   always-on ring buffer, trailing silence closes a take per
   :need:`REQ_FW_RECORD_HANGOVER`, 4000 ms maximum word-prompt length,
   6000 ms maximum sentence/negative-prompt length, 8000 ms no-speech
   timeout forces an auto-redo, and a 700 ms hold after a successful save
   before auto-advancing to the next prompt.

.. req:: Trailing-silence hangover is per prompt set; false starts are discarded
   :id: REQ_FW_RECORD_HANGOVER
   :status: implemented

   First real recording session, QC'd with Whisper: sentence takes (median
   840 ms) were rejected 75/102 for missing words, vs. word takes (median
   1020 ms) mostly fine. Cause: a natural reading pause between the words
   of a longer on-screen prompt exceeds the fixed 500 ms trailing-silence
   hangover, so ``capture_one`` (``firmware/main/record.c``) closed the
   take after the first word. Fix, two parts:

   1. ``prompt_hangover_ms`` (``firmware/main/prompts.c``) returns 500 ms
      for ``PROMPT_WORDS`` and 1200 ms for sentences/negatives/wake;
      ``capture_one`` passes ``prompt_hangover_ms(set) / 20`` as the
      trailing-frame count to ``vad_reset`` (see :need:`REQ_FW_VAD_ENDPOINT`)
      instead of the fixed ``VAD_TRAILING_FRAMES``.
   2. False-start filter: ``vad_t.speech_total`` counts every frame above
      threshold since the last ``vad_reset``. If a take closes with less
      than ``MIN_SPEECH_MS`` (200 ms) of total speech frames — a breath or
      click before the speaker starts — ``capture_one`` discards it and
      keeps listening in the same call, so the 8 s ``NO_SPEECH_MS`` timeout
      (:need:`REQ_FW_RECORD_CAPS`) keeps running from the original start
      instead of restarting.

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

.. req:: Guided recorder can capture a wake-word-only session
   :id: REQ_FW_RECORD_WAKE_SET
   :status: implemented

   The generated ``PROMPT_WAKE`` set is ``config.WAKE_WORD`` ("Hey Bus")
   repeated ``config.WAKE_PROMPT_REPEATS`` (5) times, set name ``wake``
   (``prompt_set_name``); ``prompt_takes_per_prompt(PROMPT_WAKE)`` returns 1
   (not 2), so a session prompts exactly 5 single-take "Hey Bus" reads
   before finishing straight to ``REC_SESSION_DONE`` — no
   sentence/negative sets chained on, unlike the normal guided session.
   Takes save under ``spkNN/hey-bus/NNN.wav``, same as the isolated-word
   set, and each ``session.csv`` row uses the same
   ``prompt,file,ms,peak_dbfs,set,seed,ts`` shape with ``set`` = ``wake``.

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

.. req:: The remote console survives USB mode
   :id: REQ_FW_USB_CDC_CONSOLE
   :status: implemented

   ``usb_drive_enter``/``usb_drive_exit`` (``firmware/main/usb_drive.c``)
   bring up TinyUSB as a composite MSC + CDC-ACM device and move stdio onto
   the CDC-ACM port for the duration of USB mode (``esp_tusb_init_console``/
   ``esp_tusb_deinit_console``), instead of only the MSC device that used to
   take the USB PHY and silently drop the console's own port with it.
   ``console.c``'s stdin is opened ``O_NONBLOCK`` so the console task can
   never be left blocked in the old port's read call across the switch,
   whichever task triggers it (see :need:`REQ_FW_REMOTE_MODE`).

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

   ``app_main`` boots into ``UI_MODE_MENU`` (``ui_show_menu()``), a column
   of five buttons — Recognition, Hey Bus, Record, Hey Bus aufnehmen, USB —
   each a direct ``app_set_mode()`` call. Every other screen's back/abort
   button (record, record-wake, recognise, wake, USB) calls
   ``app_set_mode(UI_MODE_MENU)``, so the menu is the only hub: no mode
   links directly to another mode.

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
   this flow. ``app_set_mode(UI_MODE_RECORD_WAKE)`` posts
   ``REC_CMD_START_WAKE_SESSION`` instead, the "Hey Bus"-only variant — see
   :need:`REQ_FW_RECORD_WAKE_SET`.

.. req:: The device accepts remote mode/status commands over the serial console
   :id: REQ_FW_REMOTE_MODE
   :status: implemented

   ``console.c`` reads newline-terminated commands from ``stdin`` (the
   UART console) in a low-priority task: ``mode
   menu|record|recordwake|recognise|wake|usb`` calls ``app_set_mode()``;
   ``status`` reports the current mode and, in record/record-wake mode,
   the recorder's phase/index/count/speaker. Every command ends with an
   ``ok`` or ``err <reason>`` line, so a host script driving the device
   (e.g. ``echo 'mode usb' > /dev/cu.usbmodemNNN``) can tell when a
   command finished.

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

Recording pipeline (host workflow)
-----------------------------------

The workstation-side loop that turns a pulled CoreS3 session into
QC-approved, dataset-ready audio: :doc:`pipeline` covers the full flow.

.. req:: Ingest never deletes on the remote host
   :id: REQ_PIPE_INGEST
   :status: implemented

   ``scripts/ingest.sh`` runs its steps in exactly this order: (1) send
   ``mode usb`` over the CoreS3's serial port and wait for the ``KWSREC``
   volume to mount, (2) run ``scripts/pull-recordings.sh`` on the remote
   host into a stamped ``~/kwsrec-pull/<stamp>/`` staging directory, (3)
   ``rsync`` that staging directory to
   ``$KWS_DATA_ROOT/data/recordings/incoming/<stamp>/`` with
   ``--ignore-existing``, verifying the local ``.wav``/``sessions.csv`` row
   count matches the remote before declaring success, (4) send ``mode
   menu``. Nothing on the remote host is ever deleted — the stamped staging
   directory is a safety copy, pruned by hand once trusted. An unreachable
   host (ssh exit 255) exits 3.

.. req:: QC audio gate thresholds
   :id: REQ_PIPE_QC_AUDIO
   :status: implemented

   ``kws_de.qc.audio_gate`` rejects a take unless it is 16000 Hz mono
   16-bit PCM, at least 300 ms long, no longer than its set's cap (4000 ms
   for a ``words`` or ``wake`` take, 6000 ms for a ``sentences`` or
   ``negatives`` take), peak level below -0.5 dBFS (not clipped), and RMS
   at or above -45 dBFS. A corrupt or unreadable WAV is rejected
   (``unreadable: <exception>``), never a crash.

.. req:: QC content gate rules and text normalisation
   :id: REQ_PIPE_QC_CONTENT
   :status: implemented

   Whisper's transcript is normalised (``kws_de.qc.normalise``: NFC,
   lower-cased, ``ß`` -> ``ss``, punctuation stripped, the filler word
   "prozent" dropped, and the numerals Whisper writes for the light levels
   — ``25``/``50``/``75``/``100`` — mapped back onto the German number
   words) before matching. A word take approves if the single keyword is
   heard, exact match required at 5 letters or fewer and edit-distance-1
   forgiving above 5 letters (so "Licht" never matches "nicht"), and a
   short keyword never matches merely because it is a substring of a longer
   heard word. A sentence take approves if every required command token
   (device/zone/action) appears in order, filler in between allowed; a
   heard word may also be the exact concatenation of two or more
   *consecutive required* tokens Whisper glued together ("Lichtdach" for
   "Licht Dach"), which stays order-sensitive. A negative take approves
   unless a command-vocabulary word is heard as a whole token, with one
   refinement: a 2-letter keyword ("an", "zu") rejects only if it occurs at
   least twice — a keyword of 3 letters or more rejects on a single
   occurrence — because one hallucinated 2-letter token was a false reject
   on real audio. The word found is reported as ``contains_command:<word>``.
   A ``wake`` take ("Hey Bus") approves if the whitespace-free transcript
   matches ``(hey|hej|he|hei)(bus|buss|bos|boss)``.

.. req:: Word segmentation window
   :id: REQ_PIPE_SEGMENT
   :status: implemented

   ``kws_de.qc.segment_word`` cuts a 1 s window (``config.CLIP_SAMPLES``)
   centred on the midpoint of Whisper's word span, zero-padded where the
   window runs past the start or end of the source recording.

.. req:: Approved-tree layout is regenerable and speaker ids are numeric
   :id: REQ_PIPE_APPROVED_LAYOUT
   :status: implemented

   Approved clips land at ``approved/words/<label>/<spkNN>_<NNN>.wav``,
   ``approved/phrases/<spkNN>/<slug>_<NNN>.wav`` (+ ``phrases/index.csv``),
   ``approved/negatives/<spkNN>/<slug>_<NNN>.wav`` (+
   ``negatives/index.csv``), and ``approved/wake/<spkNN>/<spkNN>_<NNN>.wav``
   (+ ``wake/index.csv``); ``<NNN>`` is the next-free number in that
   directory, independent of the source take number. Every ``index.csv``
   holds ``file,prompt,speaker`` with ``file`` relative to ``approved/``
   (it carries its own ``phrases/``/``negatives/``/``wake/`` segment), which
   is what ``kws_de.eval.eval_recordings`` reads back. Speaker ids are
   anonymous ``spkNN`` ids, never a name. Re-running QC on the same
   ``incoming/<stamp>`` first undoes exactly what that stamp wrote last
   time (via ``qc/<stamp>/written.txt``), so the approved tree is fully
   regenerable from ``incoming/`` + the QC rules, and never touches another
   stamp's or another speaker's files.

.. req:: Approved recordings enter every dataset build
   :id: REQ_PIPE_RECORDINGS_IN_BUILD
   :status: implemented

   ``kws_de.data.merge_recordings`` folds the QC-approved tree into the
   cached clip dict on every ``kws-dataset build`` (and every
   ``kws-data --build``), not only on the build that created the cache:
   approved word clips become clips of speaker ``rec:<spkNN>`` for their
   label, approved negatives become ``_unknown_`` material via
   ``negative_windows``. Previous ``rec:`` entries are dropped first, so a
   rebuild replaces rather than duplicates them, and the recordings are
   never written into the cache pickle. ``--recordings-split train`` (the
   default) then forces every ``rec:`` speaker into the train split — the
   personalised, user-customised model — while ``auto`` leaves them to the
   global speaker-disjoint draw. The manifest records the outcome per split
   (``sources`` counts by origin plus the device ``speakers``), which is
   what the eval's labelling reads.

.. req:: Recordings-based eval never mixes held-out and in-training figures
   :id: REQ_PIPE_EVAL_LABELS
   :status: implemented

   ``kws_de.eval.eval_recordings`` labels every approved-recording figure
   with one of exactly two strings, verbatim: ``"held-out"`` or
   ``"user-customised, in-training"`` (a speaker-level match against the
   training manifest's ``train`` split). The two are always reported as
   separate sections, never combined into one number.

Training (host workflow)
-------------------------

.. req:: Quantisation-aware training is available for the command model
   :id: REQ_MODEL_QAT
   :status: implemented

   ``kws-train --qat`` fine-tunes the (built-or-loaded) float command model
   with ``tensorflow_model_optimization.quantization.keras.quantize_model``
   (per-tensor fake-quant on every activation, per-channel on conv/dense
   kernels) for ``--qat-epochs`` (default 10) epochs at a low learning rate,
   and saves the result as a SavedModel dir named ``<out>_qat`` alongside
   the plain float ``<out>.keras`` — the PTQ path's own artefact is
   untouched, so the two can be compared on the same architecture and data.
   ``kws-export --qat`` loads that model back (under the same
   ``TF_USE_LEGACY_KERAS=1`` runtime tfmot requires, via a module-level
   re-exec in both CLIs before TensorFlow is imported) and converts it to
   INT8 TFLite using its baked-in fake-quant ranges, writing
   ``command<suffix>_qat.tflite`` next to the plain PTQ export.
