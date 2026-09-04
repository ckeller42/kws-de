# kws-de firmware (M5Stack CoreS3)

## What it does

Multi-mode ESP-IDF app on the M5Stack CoreS3. It boots to a 5-button
selection menu; every mode's back button returns to it (see "Modes and the
selection screen" below).

- **Record mode** — a guided speech recorder. Entering it starts a fresh
  session for a new speaker id: the sentence set, then the negative set,
  captured onto the microSD (or the device's flash, if no card is in the
  slot) with the built-in VAD and logged to a `session.csv` alongside the
  WAVs.
- **Record-wake mode** ("Hey Bus aufnehmen") — a "Hey Bus"-only guided
  session: same recorder, same speaker-id bump and `session.csv` row shape,
  but it prompts `WAKE_PROMPT_REPEATS` (5) single-take reads of the wake
  word and finishes straight to the success screen — no sentence/negative
  sets chained on. Collects real wake-word positives for `models/hey_bus.tflite`.
- **Recognise mode** — an on-device keyword recogniser: mic → MFCC → the
  int8 TFLite Micro model → the same streaming detector logic as
  `kws_de.stream`, shown live on the LCD and logged to the recording
  volume.
- **Wake mode** — a "Hey Bus" wake-word test mode. It runs *only* the
  microWakeWord streaming model, showing its live probability and flashing
  the screen green plus a beep on every detection.
- **USB mode** — exposes the recordings and the recognise log as a
  `KWSREC` FAT drive over USB so a host can pull them off (the microSD
  card if one is in use, the internal flash partition otherwise — see
  "Where recordings are stored"). The device presents as a composite USB
  device in this mode: the `KWSREC` MSC drive plus a second serial port
  (CDC-ACM) that keeps the console reachable - see "Serial commands" below.

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

## Where recordings are stored

Two possible volumes, chosen once at boot by `storage_mount()`
(`firmware/main/storage.c`). Everything the device writes — the WAVs,
`session.csv`, `wake.log`, `recognise.log` — goes under `storage_root()`,
and USB mode exports whichever volume that is.

| | mount point | size | holds |
|---|---|---|---|
| microSD in the slot | `/sdcard` | the card | hours of sessions |
| no card (fallback) | `/rec` | 12 MB partition | about one guided session |

- **microSD is preferred and needs no setup.** A card that mounts is used
  as-is. A card that probes but carries no filesystem is formatted FAT
  once, on the spot, and then used
  (`CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL` in `sdkconfig.defaults`) — so a
  blank or foreign-formatted card just works, at the cost of one long
  boot (~25 s for a 64 GB card). **Anything already on such a card is
  erased**, so use a card you are happy to hand over to the device. The
  internal flash partition is never formatted.
- **A card that can never be formatted pays that ~25 s on every boot**
  before the fallback takes over, so a boot that is suddenly half a minute
  slower means the card is write-protected, dying, or counterfeit — take it
  out. An empty slot costs nothing.
- **The volume label is forced to `KWSREC`** on both media at every mount,
  whatever the card arrived with, so the host always mounts the same name
  and `scripts/pull-recordings.sh` finds it without configuration.
- **Fallback to flash is automatic and is not an error.** With no card, an
  unmountable card, or a card that fails the write-and-read-back probe at
  mount (a dying or counterfeit card can report writes as successful and
  silently keep the old sectors), the device records to `/rec` exactly as
  it did before microSD support. The boot log says which volume won, and
  `status` over the serial console answers `storage sd <free>/<total> MB`
  or `storage flash <free>/<total> KB`.
- **Pulling the card while the device runs** is reported on the next mode
  entry and makes the recorder refuse takes (REC_FULL) rather than write
  into a dead mount. Restart to pick a volume again.

## Pull recordings

Tap **USB** on the menu to expose `KWSREC` (the card if one is in use, the
flash partition otherwise), then on the host:

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
count. Every firing is also appended to `recognise.log` on the recording
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
  `wake.log` on the recording volume (pulled off with the recordings in
  USB mode).
- The speaker and the microphone share one full-duplex I2S channel pair, so
  `beep.c` opens the speaker with the mic's exact format (16 kHz, 16-bit,
  2 channels). A different rate would be rejected by `esp_codec_dev` and
  would take capture down with it.
- A recognised **command** (a fire on a real command word — `_unknown_` and
  `_silence_` stay silent) is confirmed with two 80 ms 1.5 kHz pips, higher
  and shorter than the wake tone so the two are never confused. In assist
  mode the pips wait until the window has closed, so the speaker cannot be
  heard in the window's own audio; playback runs on a low-priority task, so
  no inference step ever waits on the codec.
- **Both tones are muted while field capture is armed.** An armed device is a
  device the user has asked to be quiet, and neither tone may ever end up
  inside a take. A command still recognised under capture simply passes
  silently — the owed tone is dropped, not deferred to a later window.

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

- `mode menu|record|recordwake|recognise|wake|assist|usb` — switches the app
  mode, same as tapping the matching menu/back button.
- `status` — prints the current mode; the model stamps as
  `models command=<id> wake=<id>`; and in record/record-wake mode also
  the recorder's phase/index/count/speaker.
- `wakefire` — injects one synthetic wake fire down the same path as a real
  one (gate, beep, log, UI). A measurement hook for the assist-mode duty
  cycle, which cannot be exercised without fires.
- `beep` — plays the command-confirmation tone once, so the speaker path can
  be checked without speaking to the device.

A model stamp is `<file name>@<first 8 hex of the tflite's sha256> <date>`,
e.g. `command_v3_qat.tflite@fc36da9f 2026-09-03`. It is generated into
`gen/model_config.h` / `gen/wake_model_config.h` by `kws-export --firmware`
and also logged once at boot (`main: models: command <id>, wake <id>`), so a
flashed device can say which models it carries without rebuilding.

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

## Storage

The recordings partition (`storage`, FAT with wear levelling) is 12 MB of the 16 MB
flash: about one full guided session (≈ 120 takes ≈ 9.5 MB) plus wake sessions. Pull
after every session — the ingest script does — or the recorder stops with "flash
full" (nothing is lost; takes already on the flash stay until pulled). Changing the
partition table erases the partition on the next flash, so pull first.

## Manual test checklist

Playing clips at the device (`afplay` a synthesised set at the microphone
instead of speaking): the clips must pass `uv run --no-sync kws-tts-check
<dir>` — exit 0, no failures — **before** any of them is played. macOS `say`
falls back to an English voice when the German one is missing or the name is
ambiguous, silently, and a test driven by English "German" clips measures
nothing while looking like a result; it has happened (paper notes E23).
`kws_de.tts` writes the `manifest.csv` the check reads, beside the clips.

Storage: boot with the microSD slot **empty** → the boot log reports no
usable card and mounts the flash partition, `status` answers `storage
flash <free>/<total> KB`, a session writes `spkNN/session.csv` and its
WAVs under `/rec`, and `mode usb` mounts `KWSREC` on the host with those
files on it. Boot with a **card inserted** → the log prints the card's
name, type and size (a blank or unformatted card is formatted FAT once
first), `status` answers `storage sd <free>/<total> MB`, a session writes
under the card root, `mode usb` mounts `KWSREC` showing the card, and
`scripts/ingest.sh -H <device-host>` pulls from the card. **Pull the card
while the device runs** → the next mode entry logs that the volume stopped
responding and takes are refused (REC_FULL), with no crash.

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
tap **Menu** → `wake.log` on the recording volume has one
`[Wake] <ms> <prob>` line per fire.

Field capture (Assistent mode): after a fresh flash, tap **Assistent** → the
"Aufnahme" switch is off and no "REC" badge is shown; `status` over the console
answers `field off thresh 0.60 takes 0 dropped 0`. Turn the switch on → the badge appears;
reset the device and re-open Assistent → the switch is still on (the toggle is
in NVS; `takes` re-zeroes, it counts since boot). Send `field off` / `field on`
over the console instead → the badge follows that too, so the screen can never
disagree with what is actually being recorded. Say "Hey Bus" and then a command:
the screen behaves exactly as before (green flash, recognised word) — but
silently, since capture is armed and both tones are muted.
`status` now reports `field on thresh 0.60 takes 1 dropped 0`, and in USB-drive mode the
drive holds `field/<spkNN>/<boot>-<ms>.wav` plus a `field.csv` row for it. Say
"Hey Bus" a **second** time while the window is still open → still exactly one
take, named after the *first* fire, with a larger `window_ms` and a longer WAV
than a single-window take. In the serial log, each `field: saved` line lands
after its window's `assist: recogniser off`, and the `recognise` step times
inside the window stay in their usual range — a 100–300 ms outlier would mean a
file was written while the recogniser ran. Toggling `field off` then `field on`
over the console *while a window is open* must not produce an outlier either:
the toggle takes effect at once, but the NVS write behind it is deferred to the
window's closing edge. Capture threshold: `field thresh 0.6` answers `ok`,
`status` echoes `thresh 0.60` and the badge reads `REC 0.60` (plain `REC` again
at `field thresh 0.85`, which is the production gate); `field thresh 0.2` and
`field thresh abc` answer `err thresh must be 0.30..0.85` and change nothing;
the value survives a reset. With it set, a "Hey Bus" said quietly enough to peak
below 0.85 must still fire and still produce a take — the near-miss the shipped
gate would have dropped — while the same phrase in Assistent mode with capture
**off** must not fire at all.

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
