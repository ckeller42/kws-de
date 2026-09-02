# kws-de firmware (M5Stack CoreS3)

Dual-mode firmware: guided speech recorder (words / sentences / negatives → WAV on flash → USB drive) and an on-device keyword recogniser fed by `kws-export`.

## ESP-IDF version

Pinned to **v5.5.5**. The same string lives in `CMakeLists.txt` and `.github/workflows/firmware.yml`; change all three together.

## Build (no local IDF needed)

```bash
docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:v5.5.5 \
  bash -c 'idf.py set-target esp32s3 && idf.py build'
```

With a native IDF install (e.g. on the Pi): `cd firmware && idf.py set-target esp32s3 && idf.py build`.

## Host unit tests

```bash
make -C firmware/test
```

Pure-C units only (`mfcc`, `stream`, `wav`, `prompts`); needs `cc` and nothing else.

## Flash

See `scripts/flash.sh` (any host with `esptool.py`, no IDF).
