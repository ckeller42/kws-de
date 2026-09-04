Firmware (M5Stack CoreS3)
===========================

The device as it runs today: what's on the boot menu, what each mode's
screen shows and how you get back out of it, the guided recording session,
the serial console protocol, USB-drive mode, and the on-device
measurements. Build/flash instructions and the full manual test checklist
stay in ``firmware/README.md``; this page is the "what it does and what it
measures" companion, cross-referenced to the :doc:`requirements` it
implements. Architecture diagrams and the paper live on the project's
`published docs <https://ckeller42.github.io/kws-de/sphinx/>`_.

Boot menu
---------

The device boots into a dark-theme selection screen: a small "kws-de"
title over a column of five buttons, each a direct
``app_set_mode()`` call (:need:`REQ_FW_MENU_FLOW`):

- **Recognition** ("Recognition demo") — the on-device command recogniser.
- **Hey Bus** ("Hey Bus demo") — the wake-word test mode, wake model only.
- **Record** — the guided recorder: sentences, then negatives.
- **Hey Bus aufnehmen** — the wake-word-only recording session (5 takes).
- **USB** — exposes recordings over USB mass storage.

Every mode's own back/abort button returns to this menu; no mode links
directly to another. **Not yet on the menu:** an "Assistent" mode that
would run the wake model continuously and hand off to the command
recogniser on a "Hey Bus" fire — the always-on two-stage pipeline the
architecture is designed around (see the project paper's architecture
section). Today Wake and Recognition only run in isolation from each other
(:need:`REQ_FW_WAKE_ISOLATED`); a combined always-listening mode is planned
but not implemented.

Modes
-----

Recognition demo
~~~~~~~~~~~~~~~~~

Mic -> MFCC -> the int8 command model -> the same streaming detector as
``kws_de.stream`` (:need:`REQ_FW_23_CLASSES`, :need:`REQ_FW_DETECTOR_PARAMS`).
The screen shows the last fired word, confidence, inference time, model arena
bytes used, and a running fired count; every firing also
appends to ``recognise.log`` on the recording volume
(:need:`REQ_FW_RECOGNISE_LOG`), replayable
through ``kws_de.stream.KeywordStream`` on the host. Back path: the
screen's own back button returns to the menu.

The command model, like the wake model, runs on the generated inference path
by default (:need:`REQ_FW_INFER_GENERATED`); the default build carries no
interpreter at all. Measured on the CoreS3 in one session, ~100 s of
recognition per path, medians over the ~5 s trace lines with the cold first
one dropped: **step 46 -> 33 ms** and **model evaluation 41.7 -> 28.7 ms**
(-31 %). The ~13 ms saved is almost exactly the ``rest`` column of the
interpreter's own trace (dispatch, tensor bookkeeping and reference-C glue):
the esp-nn kernels are the same on both paths, so the generated code deleted
the overhead rather than making the arithmetic faster. The spec's "at least
2x" target is therefore **not** met at 1.45x — the remaining 28 ms is kernel
time for this model's 49x10x32 activations, which is a model-size question,
not a code-generation one.

Where the arena lives is a Kconfig choice (``KWS_INFER_COMMAND_ARENA``), and
the default is PSRAM. The generated arena is one static array, so unlike the
interpreter's heap allocation its placement is settled at link time; putting
it in internal ``.bss`` is measurably faster — **28.7 -> 27.0 ms**, and the
step drops 33 -> 31 ms — but 51,248 B is more internal SRAM than this device
has to give. Internal placement leaves 8,431 B free at recogniser start, and
the record task's 8 KB stack (task stacks must be internal and cannot move to
PSRAM) then fails to be created:

.. code-block:: text

   E (27435) record: record task (8192 B stack) not created: free internal 8035, largest block 7680

With the arena in PSRAM the same figure is **59,679 B** and every task starts.
So 1.7 ms buys back 51 KB of the scarcest memory on the board; internal is
kept as the opt-in for measurement builds, where that 1.7 ms is the point. The
interpreter's own arena never fit internal either (``command arena 65536 B
does not fit internal RAM (free 47019, largest block 31744) — using PSRAM``),
so the comparison above is PSRAM against PSRAM.

The boot line reads ``inference: generated (esp-nn), 51248 B arena (static,
PSRAM) + 0 B state, esp-nn scratch 19888 B queried / 19888 B reserved; TFLM
not built in; free internal <n>`` — it names the placement that was actually
linked, and the scratch figure is the larger of what the chip's own
``esp_nn_get_depthwise_conv_scratch_size`` and ``esp_nn_get_conv_scratch_size``
answer for this model's widest ops, checked against what the generator
reserved. Building with ``CONFIG_KWS_INFER_PARITY_LOG=y`` runs both paths on
the same live features once per mode entry and logs
``parity: 0/23 output bytes differ``; that build keeps both interpreters and is
tight enough that the record task does not start (it says so in the log), so
it is for verification, not for shipping.

Hey Bus demo (wake test mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs *only* the microWakeWord streaming model
(:need:`REQ_FW_WAKE_ISOLATED`) — the command recogniser is switched off for
as long as this mode is active. Shows "Hey Bus?", the live wake
probability, and a fire counter. A fire turns the background green for
600 ms and plays a 150 ms 1 kHz tone (:need:`REQ_FW_WAKE_BEEP`), and appends
``[Wake] <ms> <prob>`` to ``wake.log`` on the recording volume. Back path: back button ->
menu.

The model itself runs on the generated inference path by default
(:need:`REQ_FW_INFER_GENERATED`), and the default build carries no interpreter
at all. Measured on the CoreS3 in one session, same build otherwise, two
minutes of the peak trace per path: **wake step 1.89 -> 1.28 ms** (mean
of the 2 s trace windows), within-window spread ±502 -> ±151 µs, and
**free internal RAM after the wake model is set up 58,511 -> 81,371 B** —
the generated path's 15,680 B arena and 4,200 B of ring state are ``.bss``,
and the interpreter's 40,960 B tensor arena is never allocated. The boot line
reads ``inference: generated (esp-nn), 15680 B arena + 4200 B state, esp-nn
conv scratch 15552 B queried / 15552 B reserved; TFLM not built in; free
internal <n>``: the queried figure is what the chip's own
``esp_nn_get_conv_scratch_size_esp32s3`` answers for this model's widest
convolution, checked against what the generator reserved — a larger answer
makes the firmware refuse the generated path rather than let the kernels
overrun the arena into the ring state.

``CONFIG_KWS_INFER_PARITY_LOG=y`` builds the interpreter back in as a
reference (and re-allocates its arena): on entering the mode both paths run on
the same live features and both answers are logged — ``parity: out byte
generated 71, interpreter 71``, identical, as the host tests require. That one
step also carries the extra ``Invoke``, so the first trace window after a mode
entry reads a few ms high. Two more things when reading the trace on the
generated path: the per-kernel esp-nn timers belong to esp-tflite-micro's
wrappers, which the generated code does not go through, so they read zero and
its cost lands in the residual; and the spec's "well under 1 ms" target is not
reached — of the ~1.2 ms, ~1.1 ms is esp-nn kernel time for this model, so
what remains is a model-size question, not a code-generation one.

Record
~~~~~~~

Starts a fresh guided session (see "Guided recording session" below). Back
path: **Abbrechen** aborts and returns straight to the menu with no
success screen; completing the session shows the success screen, whose
**Menu** button returns to the menu.

Hey Bus aufnehmen
~~~~~~~~~~~~~~~~~~

The wake-word-only variant of the guided recorder
(:need:`REQ_FW_RECORD_WAKE_SET`): bumps the speaker id the same way Record
does, then prompts 5 single-take reads of "Hey Bus" straight to the
success screen — no sentence/negative sets chained on. Back path: same as
Record (Abbrechen -> menu, success screen's Menu -> menu).

USB
~~~

Exposes the recording volume as a ``KWSREC`` FAT drive over USB mass storage
(:need:`REQ_FW_USB_SINGLE_OWNER`), see "USB-drive mode" below. Back path:
sending ``mode menu`` (button or serial command) unmounts the drive and
returns to the menu, restarting the chip in the process (see "Leaving USB
mode" below).

Guided recording session
-------------------------

Tapping **Record** starts a fresh session (:need:`REQ_FW_RECORD_SESSION`):

1. Bumps the speaker id (``spk01``, ``spk02``, ... — numeric only,
   :need:`REQ_FW_RECORD_SPEAKER_ID`).
2. Records the **sentence set**, each prompt read **2 times**
   (:need:`REQ_FW_RECORD_TWO_TAKES`) for wrong-read review.
3. Auto-chains into the **negative set** with no button press, same
   two-reads-per-prompt shape.
4. A success screen shows "Fertig - danke!", the speaker id, and the
   number of takes saved this session; **Menu** returns to the selection
   menu.

The wake-word session (**Hey Bus aufnehmen**) is the one exception to the
two-takes/sentences-then-negatives shape: 5 single-take reads of the wake
word only, straight to the same success screen
(:need:`REQ_FW_RECORD_WAKE_SET`).

Every take appends one row (``prompt,file,ms,peak_dbfs,set,seed,ts``) to
``<root>/<speaker>/session.csv`` (:need:`REQ_FW_RECORD_SESSION_CSV`), where
``<root>`` is the recording volume chosen at boot (see "Storage" below);
the row's ``file`` column stays relative to that root either way. Trailing
silence closes a take after a per-prompt-set hangover — 500 ms for words,
1200 ms for sentences/negatives/wake — and a false-start filter discards a
take opened by a breath or click and keeps listening
(:need:`REQ_FW_RECORD_HANGOVER`). A clipped take is discarded and redone
(:need:`REQ_FW_RECORD_CLIP_REJECT`); a corrupted or too-short take is
rejected too. Aborting with **Abbrechen** at any point returns straight to
the menu with no success screen.

Storage
---------

Recordings land on a **microSD card when one is usable, and on the internal
flash partition otherwise** (:need:`REQ_FW_STORAGE_SD`). ``storage.c``
makes that choice once at boot and hides it behind ``storage_root()``: the
recorder, ``session.csv``, ``wake.log`` and ``recognise.log`` all build
their paths from it, so nothing else in the firmware knows which medium is
underneath. Both media are registered with esp_tinyusb and mounted through
it, which is what lets USB-drive mode export exactly the volume the
recorder writes to.

The card is worth having because of the size gap: the flash ``storage``
partition is 12 MB — about one guided session — while a card holds hours.
A card that carries no filesystem is formatted FAT once at first use
(``CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL``), which erases whatever was on it;
the internal partition is never formatted. The FAT label is forced to
``KWSREC`` on both media at mount, so the host mounts the same name
whichever volume is live and ``scripts/pull-recordings.sh`` needs no
configuration.

Falling back to flash is the *normal* path, not a failure mode: with no
card, with a card that cannot be mounted even after the format attempt, or
with one that mounts but fails the write-and-read-back probe (a dying or
counterfeit card can acknowledge writes and silently keep the old
sectors), the device records to ``/rec`` exactly as it did before microSD
support, with no user action. ``status`` reports which volume is live —
``storage sd <free>/<total> MB`` or ``storage flash <free>/<total> KB``.
A card pulled at runtime is logged by ``storage_recheck()`` on the next
mode entry, and the recorder then refuses takes (REC_FULL) instead of
writing into a dead mount.

Serial console protocol
-------------------------

The console port accepts newline-terminated commands
(:need:`REQ_FW_REMOTE_MODE`):

.. code-block:: text

   mode menu|record|recordwake|recognise|wake|assist|usb
   status
   wakefire

- ``mode <name>`` switches the app mode, same as tapping the matching
  menu/back button.
- ``status`` prints the current mode, the model stamps as
  ``models command=<id> wake=<id>``, the live recording volume as
  ``storage sd <free>/<total> MB`` or ``storage flash <free>/<total> KB``
  (:need:`REQ_FW_STORAGE_SD`), and in record/record-wake mode also the
  recorder's phase/index/count/speaker.
- ``wakefire`` injects one synthetic wake fire down the same path as a real
  one — a measurement hook for the assist-mode duty cycle
  (:need:`REQ_FW_ASSIST_GATE`), not a feature.

A **model stamp** is ``<file name>@<first 8 hex of the tflite's sha256>
<date>``, for example ``command_v3_qat.tflite@fc36da9f 2026-09-03``.
``kws-export --firmware`` generates it into ``gen/model_config.h`` and
``gen/wake_model_config.h`` alongside ``KWS_MODEL_BYTES`` /
``KWS_WAKE_MODEL_BYTES``, and the firmware logs both once at boot
(``main: models: command <id>, wake <id>``). A firmware image outlives the
checkout that built it, so this is what makes "which models is this device
running" answerable from the device rather than by rebuilding and comparing
bytes. The digest is over the exact flatbuffer that becomes the C array, so
the stamp changes when and only when the model does.

- Every command ends with ``ok`` or ``err <reason>`` on its own line, so a
  host script can tell when it finished.

That port is the ESP32-S3's own **USB-Serial-JTAG** peripheral, not a UART
bridge — see "Hardware facts" below for what that means for opening it.

USB-drive mode
---------------

Entering USB mode unmounts the recording volume from the app and exposes
it — the microSD or the flash partition, whichever is live — as a
composite USB device: the ``KWSREC`` mass-storage drive plus a
second, CDC-ACM serial port that takes over the console for as long as
USB mode is active (:need:`REQ_FW_USB_CDC_CONSOLE`). The normal console
port (USB-Serial-JTAG) goes dark the moment TinyUSB takes the PHY, same as
it always has; the CDC-ACM port is what ``mode``/``status`` reach while
the drive is mounted, and it enumerates as a *different* device node than
the normal console port.

**Leaving USB mode restarts the chip.** Once TinyUSB releases the PHY, the
USB-Serial-JTAG peripheral does not re-enumerate on its own — there is no
re-attach without a physical re-plug — so ``mode menu`` (button or serial)
now restarts the device instead: the menu comes back in about 2 seconds
and the normal console port returns with it. This is what leaving USB mode
means on this hardware today; a PHY re-attach for the JTAG peripheral
without a restart is a possible cleaner upgrade, not yet done.

Host side: ``scripts/pull-recordings.sh`` rsyncs each ``spk*/`` directory,
merges every speaker's ``session.csv`` into a top-level
``sessions.csv``, moves ``recognise.log``, empties the device, and ejects
the drive (:need:`REQ_FW_USB_PULL`).

On-device measurements
------------------------

As of main commit ``a6c584d`` (2026-09-03), measured on a real CoreS3:

.. list-table::
   :header-rows: 1

   * - Measurement
     - Value
     - Requirement
   * - Command recogniser step (9-10 frames)
     - 82-85 ms (was 164-181 ms before the exact 480-point FFT)
     - :need:`REQ_FW_FRONTEND_FFT`
   * - Command model ``Invoke`` (TFLM, PSRAM arena)
     - 52-53 ms of the step above
     - :need:`REQ_FW_ARENA_PLACEMENT`
   * - Command front end, per new frame
     - 3.0 ms (was 8.5 ms with the naive DFT)
     - :need:`REQ_FW_FRONTEND_FFT`
   * - Wake step, internal-SRAM arena
     - 3 ms (was 5 ms in PSRAM, a 40% cut)
     - :need:`REQ_FW_ARENA_PLACEMENT`
   * - Free internal RAM when the models start
     - 148,895 B
     - :need:`REQ_FW_ARENA_PLACEMENT`
   * - Free internal RAM after the wake arena (49,152 B) is placed
     - 82,907 B
     - :need:`REQ_FW_ARENA_PLACEMENT`
   * - Command arena actually used vs. allocated (PSRAM)
     - 55,024 B of 139,264 B
     - :need:`REQ_FW_ARENA_PLACEMENT`
   * - MFCC feature deviation, C vs. Python reference
     - 5.4e-4 max abs (1.3e-6 of reference peak)
     - :need:`REQ_FW_MFCC_PARITY`, :need:`REQ_FW_FRONTEND_FFT`
   * - Quantised int8 tensor fed to the command model, C vs. Python
     - 0 LSB (identical)
     - :need:`REQ_FW_MFCC_QUANTIZE`
   * - Wake front-end features, C vs. ``pymicro_features`` golden vector
     - 0 LSB (exact, 98x40 values)
     - :need:`REQ_FW_WAKE_FRONTEND_PARITY`

Screenshot mechanism
----------------------

Built with ``-DKWS_UI_SCREENSHOT=1``, the firmware streams each UI-state
change over the console as ``[SHOT w h RLE16 n]`` + base64-encoded,
RLE-packed RGB565 rows + ``[/SHOT]``; a host-side decoder turns the frames
into PNGs. This is the mechanism, not yet the images — real screenshots of
every screen (menu, guided recording, Hey Bus recording, Hey Bus test,
recognition, USB mode, the success screen) are tracked in
`issue #20 <https://github.com/ckeller42/kws-de/issues/20>`_ and not yet
committed under ``docs/sphinx/_static/screens/``.

Hardware facts that matter
-----------------------------

- The CoreS3's USB-C port **is** the ESP32-S3's own **USB-Serial-JTAG**
  peripheral — there is no separate UART-to-USB bridge chip, and UART0 is
  not wired to anything reachable. IDF mirrors ``stdout`` onto it as a
  secondary console, but the console firmware reads commands directly from
  the USB-Serial-JTAG driver (with a bounded wait) rather than relying on
  ``stdin``, and assembles lines itself.
- **Opening the port resets the chip.** The port's DTR/RTS toggle on open
  is what triggers the reset (same mechanism used for flashing); a host
  tool must hold both lines low before/while opening (e.g. pyserial with
  ``dtr = rts = False`` before ``open()``) and wait for the boot log to
  pass before sending commands. A reusable host-side helper doing exactly
  this replaced ad-hoc ``cat``/``echo`` capture.
- The mic and the speaker (AW88298 amp) share **one full-duplex I2S
  channel pair**: the speaker can only be opened at the microphone's exact
  sample rate (16 kHz, 16-bit, 2 channel) or capture dies
  (:need:`REQ_FW_WAKE_BEEP`).

See ``firmware/README.md`` for build/flash instructions and the full
on-hardware manual test checklist, and :doc:`traceability` for how every
requirement above is (or isn't yet) covered by a test.
