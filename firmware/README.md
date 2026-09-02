# kws-de firmware (M5Stack CoreS3)

## What it does

Multi-mode ESP-IDF app on the M5Stack CoreS3. It boots to a 5-button
selection menu; every mode's back button returns to it (see "Modes and the
selection screen" below).

- **Record mode** — a guided speech recorder. Entering it starts a fresh
  session for a new speaker id: the sentence set, then the negative set,
  captured onto the device's flash with the built-in VAD and logged to a
  `session.csv` alongside the WAVs.
- **Record-wake mode** ("Hey Bus aufnehmen") — a "Hey Bus"-only guided
  session: same recorder, same speaker-id bump and `session.csv` row shape,
  but it prompts `WAKE_PROMPT_REPEATS` (5) single-take reads of the wake
  word and finishes straight to the success screen — no sentence/negative
  sets chained on. Collects real wake-word positives for `models/hey_bus.tflite`.
- **Recognise mode** — an on-device keyword recogniser: mic → MFCC → the
  int8 TFLite Micro model → the same streaming detector logic as
  `kws_de.stream`, shown live on the LCD and logged to flash.
- **Wake mode** — a "Hey Bus" wake-word test mode. It runs *only* the
  microWakeWord streaming model, showing its live probability and flashing
  the screen green plus a beep on every detection.
- **USB mode** — exposes the on-flash recordings and the recognise log as a
  `KWSREC` FAT drive over USB so a host can pull them off. The device
  presents as a composite USB device in this mode: the `KWSREC` MSC drive
  plus a second serial port (CDC-ACM) that keeps the console reachable - see
  "Serial commands" below.

A device on the same USB serial port also accepts a small set of remote
commands (mode switching, status) — see "Serial commands" below.

## Build

Docker (no local IDF needed):

```bash
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:v5.5.5 idf.py build
```

Native, with ESP-IDF already installed and exported:

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
```

Host-only unit tests (`mfcc`, `stream`, `wav`, `prompts`, `vad`,
`wakefront`; needs `cc`/`c++` and nothing else): `make -C firmware/test`.

The firmware is pinned to **ESP-IDF v5.5.5** (a newer esp_tinyusb needs an
`esp_vfs_fat_register` signature that only lands in v6.x — untested here).
The pin is kept equal in three places: the version-check warning in
`firmware/CMakeLists.txt`, `esp_idf_version` in
`.github/workflows/firmware.yml`, and this README — update all three
together.

## Flash

CI builds every push/PR and uploads a merged, flashable image as an
artifact named `kws-de-fw-<sha>`. Download it, then:

```bash
scripts/flash.sh kws-de-fw-<sha>.bin            # port auto-detected
scripts/flash.sh kws-de-fw-<sha>.bin /dev/ttyACM0
```

It needs `esptool.py` (`pip install esptool`, or an ESP-IDF venv) and holds
no state — safe to run from any machine with the device plugged in.

From a local build instead, flash straight from the container:

```bash
docker run --rm -v "$PWD/firmware:/project" -w /project --device=/dev/ttyACM0 \
  espressif/idf:v5.5.5 idf.py -p /dev/ttyACM0 flash
```

## Modes and the selection screen

The device boots into a dark-theme selection menu: a small "kws-de" title
over a column of five big buttons — **Recognition**, **Hey Bus**,
**Record**, **Hey Bus aufnehmen**, **USB** — each switching straight to
that mode. Every mode's own back/abort button returns to this menu; no mode
links directly to another mode. `app_set_mode()` (`firmware/main/main.c`)
is the only place that suspends/resumes the consumer task for the mode
being left/entered.

## Record-mode walkthrough

Tapping **Record** on the menu starts a fresh guided session: it bumps the
speaker id (`spk01`, `spk02`, ...) and records the **sentence set**; when
that completes it auto-chains straight into the **negative set** with no
button press. The screen shows only the guided prompt, the status pill
(SPEAK NOW / get ready / ...), a progress line (`<set> <n>/<N> - read
<r>/2`), the input-level bar, and one **Abbrechen** button that aborts the
session and returns to the menu (isolated-word prompts exist in the code
but are not part of this flow).

- Each prompt is read twice (`read 1/2`, `read 2/2`) for wrong-read review.
- Each speaker directory gets a `session.csv`
  (`prompt,file,ms,peak_dbfs,set,seed,ts`) with one row per saved take —
  the ground truth for what was recorded, in what order, and how loud.
- Recording caps: sentences and negatives 6 s, with a 300 ms pre-roll
  before speech onset and the VAD closing the take 500 ms after speech
  ends.
- Clipping (any sample hits full scale) or 8 s of silence auto-fails the
  take and prompts a redo instead of saving a bad WAV.
- When the negative set also completes, a success screen shows "Fertig -
  danke!", the speaker id, and how many takes were saved this session; its
  **Menu** button returns to the selection menu. Aborting instead returns
  straight to the menu — no success screen.

**Hey Bus aufnehmen** on the menu runs the same guided-recorder machinery
in a "Hey Bus"-only variant: it bumps the speaker id the same way, then
prompts 5 single-take (not doubled) reads of the wake word straight to the
success screen — no sentence/negative sets chained on. Takes land under
`spkNN/hey-bus/NNN.wav`, and each `session.csv` row has `set` = `wake`
(same `prompt,file,ms,peak_dbfs,set,seed,ts` shape as every other row).

## Pull recordings

Tap **USB** on the menu to expose `KWSREC`, then on the host:

```bash
scripts/pull-recordings.sh                      # default: data/recordings
KWSREC_MOUNT=/path/to/KWSREC scripts/pull-recordings.sh
KWSREC_NO_EJECT=1 scripts/pull-recordings.sh    # keep the drive mounted
```

It rsyncs each `spk*/` directory into `data/recordings/spkNN/...`,
appends every speaker's `session.csv` rows into a top-level
`data/recordings/sessions.csv` (prefixed with speaker id and pull
timestamp), moves `recognise.log` into `data/recordings/logs/`, empties
the device, and ejects the drive. Tap **Menu** on the device once the pull
finishes.

## Recognise mode

The screen shows the last fired word plus live stats: confidence,
inference time in ms, TFLite Micro arena bytes used, and a running fired
count. Every firing is also appended to `/rec/recognise.log` on the
device as `[Log] <ms> <word> <conf>` (`<ms>` is milliseconds since boot)
— pulled off with the recordings and replayable through
`kws_de.stream.KeywordStream` on the host.

## Wake test mode

Tap **Hey Bus** on the menu. The screen shows "Hey Bus?", the live wake
probability, and a fire counter; the command recogniser is switched off
while this mode is active, so what you see is the wake model alone.

- Audio path: mic → the TFLite-Micro audio front-end vendored under
  `firmware/main/microfrontend/` (`wakefront.c`) → 40 int8 features per
  10 ms → `models/hey_bus.tflite`, a *streaming* model invoked once per 3
  feature rows (every 30 ms) with its state kept alive between calls.
- The front-end parameters and the int8 requantisation are microWakeWord's
  own; `firmware/test/test_wakefront.c` asserts the C rows are bit-identical
  to `pymicro_features`' for a golden PCM vector.
- Detection: probability ≥ 0.99 on 2 consecutive steps, then 1500 ms of
  refractory so one spoken "Hey Bus" fires exactly once. All three are
  `#define`d tunables at the top of `firmware/main/wake.h`.
- On a fire the background goes green for 600 ms, the speaker plays a
  150 ms 1 kHz tone, and a line `[Wake] <ms> <prob>` is appended to
  `/rec/wake.log` (pulled off with the recordings in USB mode).
- The speaker and the microphone share one full-duplex I2S channel pair, so
  `beep.c` opens the speaker with the mic's exact format (16 kHz, 16-bit,
  2 channels). A different rate would be rejected by `esp_codec_dev` and
  would take capture down with it.

## Regenerating headers

- After changing labels, prompts, MFCC config, or test vectors:
  `uv run kws-fwgen` — regenerates
  `firmware/main/gen/{labels,prompts,features_config,test_vectors}.h`.
  CI's `gen-fresh` job diffs these against a fresh run and fails on drift,
  so always commit the regenerated headers together with the config
  change.
- After retraining or re-exporting the model:
  `uv run kws-export --v2 --firmware` — regenerates
  `firmware/main/gen/{model_data,model_config}.h` and, when
  `models/hey_bus.tflite` is present,
  `firmware/main/gen/{wake_model_data,wake_model_config}.h`. These are
  committed as-is (not diff-gated) since they carry the trained weights.
- The wake front-end golden vector (`gen/wake_test_vectors.h`) comes from
  `kws-fwgen` and needs the `wake` extra
  (`uv sync --extra dev --extra tts --extra docs --extra wake`); without
  `pymicro-features` installed the generator leaves the committed header
  alone and `--check` skips it.

## Serial commands

The console port (the same USB-serial port used for flashing/monitor,
e.g. `/dev/cu.usbmodemNNN`) also accepts newline-terminated commands,
letting a laptop drive the device — the automated data-ingest workflow
uses this to switch the device into USB mode without touching the screen.
That port is the ESP32-S3's own USB-Serial-JTAG peripheral (the CoreS3 has
no UART bridge; UART0 is not connected), so the firmware reads commands
from the USB-Serial-JTAG driver, not from `stdin`. Opening the port toggles
DTR/RTS, which resets the chip: use a serial tool that holds both lines low
(e.g. pyserial with `dtr = rts = False` before `open()`), wait for the boot
log to pass, then send:

```text
mode usb
status
```

- `mode menu|record|recordwake|recognise|wake|usb` — switches the app
  mode, same as tapping the matching menu/back button.
- `status` — prints the current mode, and in record/record-wake mode also
  the recorder's phase/index/count/speaker.

Every command ends with `ok` or `err <reason>` on its own line so a host
script can tell when it finished. Implemented in `firmware/main/console.c`.

**During USB mode**, the console moves to a second, CDC-ACM serial port
that TinyUSB adds alongside the `KWSREC` mass-storage device
(`firmware/main/usb_drive.c`) - the normal console port disappears for as
long as TinyUSB owns the USB PHY, same as it always has, but this new port
takes over so `mode`/`status` still work while the drive is mounted. It
enumerates as a *different* device node from the normal console port (a new
`/dev/cu.usbmodemNNN`, not the one flashing/monitor uses) once the device
is in USB mode; watch `ls /dev/cu.usbmodem*` before/after tapping **USB** or
sending `mode usb` to see it appear. `echo 'mode menu' >
/dev/cu.usbmodemNNN` (the CDC port) leaves USB mode and remounts the drive
for the app - the normal console port returns once that command completes.
Flashing still needs the original JTAG/console port, not the CDC one (it
only exists while USB mode is active).

## Manual test checklist

On-device manual checklist: from the menu, tap **Record** → the session
starts at a new speaker id and the sentence set; complete it (or let it
run) → it auto-chains into the negative set with no button press →
completing that shows "Fertig - danke!" with the speaker id and a
saved-take count; tap **Menu** → back at the selection screen. Aborting
mid-session with **Abbrechen** instead returns straight to the menu with
no success screen. Tap **USB** → pull → `column -s, -t <
data/recordings/sessions.csv` lists the session; tap **Recognition** →
say "Licht" → word appears, inference < 30 ms; `recognise.log` replays
through `stream.KeywordStream` with the same events.

Wake mode: tap **Hey Bus** → the probability updates live and stays low
on silence; say "Hey Bus" → the screen flashes green, the speaker beeps
once, and the fire count rises by exactly one per utterance; say "Licht"
and a few other words → no fire (nothing but the wake model is running);
tap **Menu** → `/rec/wake.log` has one `[Wake] <ms> <prob>` line per fire.

Serial commands: with the device connected over USB serial,
`echo 'mode wake' > /dev/cu.usbmodemNNN` switches the screen to wake mode
and `echo 'status' > /dev/cu.usbmodemNNN` reports `mode wake` followed by
`ok`; `echo 'mode record' > ...` then `status` reports the recorder's
phase/index/speaker.

USB mode's CDC console: tap **USB** on the menu (or send `mode usb`) → the
`KWSREC` drive mounts on the host and a new `/dev/cu.usbmodemNNN` appears
alongside (or in place of) the original console port; `echo 'status' >
<the new port>` answers `mode usb` / `ok`; `echo 'mode menu' > <the new
port>` unmounts the drive and returns the device to the selection menu, and
the original console port comes back (`echo 'status' > <original port>`
answers again).

## Pi note

A Raspberry Pi with ESP-IDF at `~/esp/esp-idf` can build natively:

```bash
git -C ~/esp/esp-idf checkout v5.5.5
~/esp/esp-idf/install.sh esp32s3
. ~/esp/esp-idf/export.sh
cd firmware && idf.py build
```

This checks out the pinned tag for this build without touching any other
IDF checkout already in use on the same Pi.
