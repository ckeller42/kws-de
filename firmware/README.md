# kws-de firmware (M5Stack CoreS3)

## What it does

Dual-mode ESP-IDF app on the M5Stack CoreS3:

- **Record mode** — a guided speech recorder. It walks through a prompt set
  (word / sentence / negative), captures each take onto the device's flash
  with the built-in VAD, and writes a `session.csv` alongside the WAVs.
- **Recognise mode** — an on-device keyword recogniser: mic → MFCC → the
  int8 TFLite Micro model → the same streaming detector logic as
  `kws_de.stream`, shown live on the LCD and logged to flash.
- **USB mode** — exposes the on-flash recordings and the recognise log as a
  `KWSREC` FAT drive over USB so a host can pull them off.

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

Host-only unit tests (`mfcc`, `stream`, `wav`, `prompts`; needs `cc` and
nothing else): `make -C firmware/test`.

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

## Record-mode walkthrough

The screen shows the active set, a seed, the running index (`n/count`),
the current speaker id, and the prompt text, with a live input-level bar.

- **W / S / N** buttons pick the prompt set (words / sentences /
  negatives); switching draws a fresh shuffled order from a new seed.
- **Redo** re-captures the current prompt; **Skip** and **Next** both move
  to the next prompt without re-recording (Skip gives up on a bad prompt,
  Next is the normal move-on action after a take saves).
- **+Spk** starts a new speaker id (`spk01`, `spk02`, ...) and re-seeds the
  current set.
- Each speaker directory gets a `session.csv`
  (`prompt,file,ms,peak_dbfs,set,seed,ts`) with one row per saved take —
  the ground truth for what was recorded, in what order, and how loud.
- Recording caps: words 4 s, sentences and negatives 6 s, with a 300 ms
  pre-roll before speech onset and the VAD closing the take 500 ms after
  speech ends.
- Clipping (any sample hits full scale) or 8 s of silence auto-fails the
  take and prompts a redo instead of saving a bad WAV.

## Pull recordings

Tap **USB** in record mode to expose `KWSREC`, then on the host:

```bash
scripts/pull-recordings.sh                      # default: data/recordings
KWSREC_MOUNT=/path/to/KWSREC scripts/pull-recordings.sh
KWSREC_NO_EJECT=1 scripts/pull-recordings.sh    # keep the drive mounted
```

It rsyncs each `spk*/` directory into `data/recordings/spkNN/...`,
appends every speaker's `session.csv` rows into a top-level
`data/recordings/sessions.csv` (prefixed with speaker id and pull
timestamp), moves `recognise.log` into `data/recordings/logs/`, empties
the device, and ejects the drive. Tap **Back** on the device once the pull
finishes.

## Recognise mode

The screen shows the last fired word plus live stats: confidence,
inference time in ms, TFLite Micro arena bytes used, and a running fired
count. Every firing is also appended to `/rec/recognise.log` on the
device as `[Log] <ms> <word> <conf>` (`<ms>` is milliseconds since boot)
— pulled off with the recordings and replayable through
`kws_de.stream.KeywordStream` on the host.

## Regenerating headers

- After changing labels, prompts, MFCC config, or test vectors:
  `uv run kws-fwgen` — regenerates
  `firmware/main/gen/{labels,prompts,features_config,test_vectors}.h`.
  CI's `gen-fresh` job diffs these against a fresh run and fails on drift,
  so always commit the regenerated headers together with the config
  change.
- After retraining or re-exporting the model:
  `uv run kws-export --v2 --firmware` — regenerates
  `firmware/main/gen/{model_data,model_config}.h`. These are committed
  as-is (not diff-gated) since they carry the trained weights.

## Manual test checklist

On-device manual checklist: record 3 words + 1 sentence + 1 negative →
USB → pull → `column -s, -t < data/recordings/sessions.csv` lists them;
toggle to Recognise → say "Licht" → word appears, inference < 30 ms;
`recognise.log` replays through `stream.KeywordStream` with the same events.

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
