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

.. req:: MFCC front end uses an exact 480-point FFT
   :id: REQ_FW_FRONTEND_FFT
   :status: implemented

   The per-frame spectrum is a real FFT over the vendored kissfft
   (``firmware/main/mfcc_fft.cc``, ``kiss_fftr`` at nfft = 480 = 2^5*3*5, an
   exact mixed radix), never a zero-padded 512-point transform: padding
   changes the bin spacing and hence the mel energies the models were
   trained on. Feature parity with the Python front end
   (:need:`REQ_FW_MFCC_PARITY`) is unaffected by the transform — max
   absolute deviation 5.4e-4, and 0 LSB on the int8 tensor fed to the
   command model. The measured front-end cost on the CoreS3 is **0.46 ms per
   new frame**, against 8.5 ms for the naive DFT this replaced: the exact
   FFT took it to 3.0 ms, quad-I/O flash to 2.0 ms, and the banded mel
   filterbank (:need:`REQ_FW_MEL_BANDED`) to 0.46 ms.

.. req:: Mel filterbank ships as non-zero bands, not a dense matrix
   :id: REQ_FW_MEL_BANDED
   :status: implemented

   ``kws-fwgen`` emits the mel filterbank as ``KWS_MEL_START``,
   ``KWS_MEL_LEN`` and 459 concatenated weights rather than a dense
   ``[40][241]`` matrix, and ``mfcc.c`` walks the bands. A triangular filter
   is non-zero over 4-33 of the 241 bins, so the dense form was 95% exact
   zeros — 38.5 KB of flash rodata read in full for every frame. Only exact
   zeros are dropped and the surviving terms accumulate in the same order,
   so the mel energies are bit-identical, not approximated;
   ``firmware_gen.mel_bands`` refuses to compress a filter whose non-zeros
   are not one contiguous run. Measured: front end 2.0 -> 0.46 ms per frame.

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
   (16000 Hz); every saved take uses this exact header.

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
   ``<root>/<speaker>/session.csv`` on save (``<root>`` = ``storage_root()``),
   giving a per-speaker audit trail
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

.. req:: Recordings go to a microSD card when one is usable
   :id: REQ_FW_STORAGE_SD
   :status: implemented

   At boot ``storage_mount()`` (``firmware/main/storage.c``) tries the
   microSD slot. A card that mounts — or that carries no filesystem and can
   be reformatted FAT on the spot, ``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL`` —
   becomes the recording volume: ``storage_root()`` returns ``/sdcard``,
   the recorder, ``session.csv``, ``wake.log`` and ``recognise.log`` are
   written under it, USB-drive mode exports the card, and ``status``
   answers ``storage sd <free>/<total> MB``. The internal partition is
   never formatted.

   **Fallback to internal flash is unconditional and needs no user
   action.** With no card in the slot, with a card that cannot be mounted
   even after the one-time format attempt, or with one that mounts but
   fails the write-and-read-back probe, the device behaves exactly as it
   did before microSD support: ``storage_root()`` is ``/rec`` on the
   internal 12 MB wear-levelled FAT partition, recordings and logs are
   written there, USB-drive mode exports that partition, and ``status``
   answers ``storage flash <free>/<total> KB``. This is not an error
   state — it is the normal configuration for a device with no card, and
   the only path CI can build and exercise.

   A card pulled while the device runs is reported by ``storage_recheck()``
   on the next mode entry; ``storage_free_bytes()`` then reads 0, so the
   recorder refuses each take as REC_FULL instead of writing into a dead
   mount or crashing. Making the card the recording volume again needs a
   restart.

   Verified on real hardware both ways
   (:need:`TEST_MANUAL_STORAGE_FALLBACK`). The volume choice is one line of
   BSP/IDF-dependent C with no host-buildable part, so it has no host test.

.. req:: Recording stops before storage is exhausted
   :id: REQ_FW_STORAGE_MIN_FREE
   :status: implemented

   Recording is disabled with a REC_FULL banner once free space on the
   recording volume (``storage_root()``: the microSD if one is in use, the
   flash partition otherwise) drops below ``STORAGE_MIN_FREE_BYTES`` =
   200 KB (``storage.h``); the USB drive mode remains available regardless,
   so recordings already saved can still be pulled off.

.. req:: The recording volume has exactly one owner at a time
   :id: REQ_FW_USB_SINGLE_OWNER
   :status: implemented

   Entering USB drive mode unmounts the recording volume from the app
   before exposing it as a USB MSC device (``usb_drive_enter``); leaving it
   stops the MSC device and remounts the volume for the app
   (``usb_drive_exit``). This holds for either medium — the microSD or the
   flash partition — because both are mounted through the same esp_tinyusb
   media handle (:need:`REQ_FW_STORAGE_SD`). The app and the host PC never hold the FAT
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

.. req:: TFLM arenas prefer internal SRAM, wake model first
   :id: REQ_FW_ARENA_PLACEMENT
   :status: implemented

   ``arena_alloc`` (``firmware/main/arena.h``) allocates every TFLM tensor
   arena with ``MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT`` and falls back to
   PSRAM with a ``WARN`` line, logging
   ``heap_caps_get_free_size(MALLOC_CAP_INTERNAL)`` before and after either
   way — arena placement, not the model, is what sets ``Invoke`` time, and
   the log has to say which heap was used rather than leaving it to be
   guessed. At least 24 KB of internal RAM stays free after the arenas, for
   the UI, the audio ring and both model tasks' 16 KB stacks.

   Only one arena fits: the largest contiguous DRAM region is about 76 KB
   against a 64 KB command arena and a 40 KB wake arena, whichever order
   they are requested in. ``main.c`` therefore starts the wake task **first**
   so the choice is deliberate — the wake model is the one that runs
   continuously, and with a 64 KB data cache it gains far more from internal
   SRAM (2.6x a step) than the recogniser loses by going to PSRAM (5%).
   Measured on the CoreS3: 116,875 B free internal when the models start,
   47,539 B free after both arenas; wake step 4.91 -> 1.90 ms, recogniser
   step 43 -> 46 ms.

   Arena sizes are the device's measured ``arena_used_bytes()`` plus
   headroom (``kws_de.export``), not the desktop interpreter's
   sum-of-tensors, which overestimates TFLM's planner by about 2.5x.

.. req:: TFLite Micro op resolver is the exact gated set
   :id: REQ_FW_TFLM_OPSET
   :status: implemented

   ``recognise.cc`` registers exactly the 7 ops gated by
   ``tests/test_transducer.py``: ``CONV_2D``, ``DEPTHWISE_CONV_2D``,
   ``FULLY_CONNECTED``, ``MEAN``, ``SOFTMAX``, ``RESHAPE``, ``ADD`` — via
   ``MicroMutableOpResolver``, never ``AllOpsResolver`` — so a model that
   needs an unlisted op fails to build rather than silently pulling in
   unaudited kernels.

.. req:: Generated inference is bit-exact with the interpreter
   :id: REQ_FW_INFER_GENERATED
   :status: implemented

   With ``CONFIG_KWS_INFER_GENERATED=y`` both shipped models run as C generated
   by ``kws-codegen`` (``firmware/main/gen/wake_infer.c``,
   ``firmware/main/gen/command_infer.c``) calling esp-nn's ESP32-S3 kernels
   directly, not through ``tflite::MicroInterpreter``. Every output byte is
   identical to the interpreter's for the same
   input: requantisation multipliers and shifts are prepared with TFLM's own
   ``QuantizeMultiplier`` integer math, activation ranges with TFLM's own
   rounding, and ``LOGISTIC`` is a 256-entry table read out of the reference
   kernel itself. The generated footprint replaces, not supplements, the TFLM
   arena in the default build. Wake: a 128 B transient arena plus 4,200 B of
   persistent ring state in ``.bss``, against the 40,960 B arena the
   interpreter allocates. Command: a 31,360 B transient arena and no state at
   all, against the 65,536 B arena the interpreter allocates. Both then work in
   one shared 19,888 B esp-nn scratch region, sized for the widest op of either
   — esp-nn's kernels reach their scratch through file-static globals, one per
   kernel family for the whole image, so a region per model would be handed to
   the other model's kernels, which *write* into it. The two evaluations are
   serialised on one mutex (``firmware/main/infer_lock.h``); they contend only
   inside an assist window, and the wait is bounded by one command inference.
   The generated arena is one static array, so its placement is a
   link-time choice (Kconfig ``KWS_INFER_COMMAND_ARENA``) rather than an
   allocation: PSRAM by default, internal SRAM as an opt-in. The scratch region
   stays internal either way, which is where nearly all of that choice's
   ~1.7 ms lived — the choice now covers 31,360 B of activations and is worth
   27.3 -> 27.0 ms, measured, with 55,239 B free at recogniser start against
   23,879 B. PSRAM stays the default because 31 KB of the scarcest memory on
   the board is a poor trade for 0.3 ms; when that same choice still moved all
   51,248 B it left 8,431 B free and the record task's stack, which must be
   internal, was then never created (:need:`REQ_FW_ARENA_PLACEMENT`).
   Neither reserve is taken on trust: at boot the firmware calls
   ``<model>_infer_scratch_query()``, generated beside the kernels from the
   same dimensions, which asks the chip's own
   ``esp_nn_get_*_scratch_size_esp32s3`` for every op of that model that takes
   scratch, and refuses to run the generated path if the answer exceeds
   ``WAKE_INFER_SCRATCH_BYTES`` / ``COMMAND_INFER_SCRATCH_BYTES`` from the
   generated header. Generating the query rather than hand-copying the
   dimensions is what stops a regenerated model leaving the guard asking about
   an op that is no longer there, since the failure it guards against is a
   silent overrun out of the shared scratch region.

.. req:: TFLite Micro stays as a build-time fallback
   :id: REQ_FW_INFER_FALLBACK
   :status: implemented

   Both inference paths live in one firmware family; ``menuconfig``'s
   ``CONFIG_KWS_INFER_GENERATED`` (``firmware/main/Kconfig.projbuild``) picks
   one and the boot log prints which is active. A model that uses an op the
   generator refuses is a loud generation-time error naming the op and tensor,
   and the interpreter build still runs it. The default build compiles the
   interpreter *out*: no ``MicroInterpreter``, no resource variables and
   neither tensor arena, which is the memory the generated path is there to
   save. Enabling ``CONFIG_KWS_INFER_PARITY_LOG`` builds it back in for both
   models and logs ``parity: 0/23 output bytes differ`` on live device audio
   once per mode entry; that build is tight enough that the record task is not
   created (logged, not silent), so it verifies rather than ships. The
   interpreter is a fallback for a model the generator refuses, not a
   configuration the device has room to run alongside the generated one. Both
   configurations are built in CI — the ``firmware-build`` job runs
   ``idf.py build`` a second time with ``CONFIG_KWS_INFER_GENERATED=n`` — so
   the fallback cannot rot behind an edit that only compiles under the
   default. The gating semantics around the model are untouched either
   way: threshold, consecutive-step count and refractory period
   (:need:`REQ_FW_WAKE_DETECT`) see the same probability byte.

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
   detection to ``recognise.log`` on the recording volume; the log format is exactly what's
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

   ``wake.cc`` runs the streaming ``models/hey_bus.tflite`` graph once per 3
   feature rows (30 ms of audio) — by default on the generated path
   (:need:`REQ_FW_INFER_GENERATED`), otherwise through the interpreter — and
   keeps the streaming state (the generated ring buffers, or the interpreter's
   resource variables) alive between steps, resetting it when the mode is
   entered. A detection needs ``WAKE_THRESHOLD`` (0.85) on
   ``WAKE_MIN_CONSECUTIVE`` (2) consecutive steps, after which
   ``WAKE_REFRACTORY_MS`` (1500) suppresses further fires, so one spoken
   "Hey Bus" produces exactly one fire. Each fire is logged to
   ``wake.log`` on the recording volume as ``[Wake] <ms> <prob>``.

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

.. req:: Assist mode gates the recogniser behind a wake fire
   :id: REQ_FW_ASSIST_GATE
   :status: implemented

   ``UI_MODE_ASSIST`` is the deployment shape the always-on recognise mode
   only measures: the wake model runs continuously at ~1.9 ms per 30 ms of
   audio, and a wake fire opens a 2.5 s window
   (``ASSIST_WINDOW_MS``) in which the command recogniser runs. A fire
   inside an open window extends it rather than opening a second — one
   interaction, not two.

   The window is decided by ``firmware/main/assist_gate.c``, which is pure
   C over caller-supplied milliseconds (no clock, no globals, no FreeRTOS)
   and is host-tested by ``firmware/test/test_assist_gate.c``, including the
   32-bit millisecond wrap. The **deadline is enforced by the recognise task
   itself** (``recognise_listen_for``), not only by the wake task that
   opened it: the recogniser is the expensive task, and a window that could
   only be closed by another task let it starve its own off switch — with
   both model tasks on core 0 the window never closed and the task watchdog
   fired. The recognise task also runs one priority below the wake task for
   the same reason.

   Duty is reported in the log once per 10 s, in both modes and in the same
   format (``KWS_DUTY``): the fraction of wall time the recogniser was
   active, and the measured inference CPU per wall second. Measured on the
   CoreS3, one interaction per 10 s: assist 253/1000 of wall and 97 ms of
   inference per wall second, against 1000/1000 and 315 ms/s for the
   always-on recognise mode.

.. req:: Recordings-based eval never mixes held-out and in-training figures
   :id: REQ_PIPE_EVAL_LABELS
   :status: implemented

   ``kws_de.eval.eval_recordings`` labels every approved-recording figure
   with one of exactly two strings, verbatim: ``"held-out"`` or
   ``"user-customised, in-training"`` (a speaker-level match against the
   training manifest's ``train`` split). The two are always reported as
   separate sections, never combined into one number.

.. req:: Every long pipeline stage prints an ETA that improves from history
   :id: REQ_PIPE_ETA
   :status: implemented

   ``kws_de.eta`` keeps a JSON-lines ledger (one record per finished stage:
   stage, size, seconds, an 8-hex-char hashed machine tag, timestamp, note)
   and predicts a stage's duration as the median of its last 10 per-unit
   (seconds/size) rates on the current machine, scaled by the requested
   size, with the 20th/80th-percentile rates as a low/high band; a stage
   with no history yet reports "ETA unknown" instead of guessing. Every
   stage in ``scripts/data-loop.sh`` (QC, dataset build, train, export,
   eval) and a direct ``kws-train`` invocation record their measured wall
   time, so predictions improve run over run. A run that raises is recorded
   with note ``"failed"`` and excluded from future predictions. The machine
   tag is never the raw hostname, so the ledger is safe to commit or share.

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
