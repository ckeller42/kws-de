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
The screen shows the last fired word, confidence, inference time, TFLite
Micro arena bytes used, and a running fired count; every firing also
appends to ``/rec/recognise.log`` (:need:`REQ_FW_RECOGNISE_LOG`), replayable
through ``kws_de.stream.KeywordStream`` on the host. Back path: the
screen's own back button returns to the menu.

Hey Bus demo (wake test mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs *only* the microWakeWord streaming model
(:need:`REQ_FW_WAKE_ISOLATED`) — the command recogniser is switched off for
as long as this mode is active. Shows "Hey Bus?", the live wake
probability, and a fire counter. A fire turns the background green for
600 ms and plays a 150 ms 1 kHz tone (:need:`REQ_FW_WAKE_BEEP`), and appends
``[Wake] <ms> <prob>`` to ``/rec/wake.log``. Back path: back button ->
menu.

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

Exposes ``/rec`` as a ``KWSREC`` FAT drive over USB mass storage
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
``/rec/<speaker>/session.csv`` (:need:`REQ_FW_RECORD_SESSION_CSV`). Trailing
silence closes a take after a per-prompt-set hangover — 500 ms for words,
1200 ms for sentences/negatives/wake — and a false-start filter discards a
take opened by a breath or click and keeps listening
(:need:`REQ_FW_RECORD_HANGOVER`). A clipped take is discarded and redone
(:need:`REQ_FW_RECORD_CLIP_REJECT`); a corrupted or too-short take is
rejected too. Aborting with **Abbrechen** at any point returns straight to
the menu with no success screen.

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
  ``models command=<id> wake=<id>``, and in record/record-wake mode also
  the recorder's phase/index/count/speaker.
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

Entering USB mode unmounts ``/rec`` from the app and exposes the partition
as a composite USB device: the ``KWSREC`` mass-storage drive plus a
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
