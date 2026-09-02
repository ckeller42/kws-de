#!/usr/bin/env bash
# Flash a merged CI/CD image onto a CoreS3. Usage: scripts/flash.sh [image.bin] [port]
# Needs esptool (pip install esptool, or the ESP-IDF venv). Hold BOOT/reset if the port is not found.
set -euo pipefail
# shellcheck disable=SC2012 # ls -t is simplest way to pick the newest match; filenames here are plain shas/dev nodes
img=${1:-$(ls -t kws-de-fw-*.bin 2>/dev/null | head -1)}
[[ -n ${img:-} && -f $img ]] || { echo "usage: $0 [kws-de-fw-<sha>.bin] [port]" >&2; exit 1; }
# shellcheck disable=SC2012
port=${2:-$(ls /dev/cu.usbmodem* /dev/ttyACM* 2>/dev/null | head -1)}
[[ -n ${port:-} ]] || { echo "no serial port found (/dev/cu.usbmodem* or /dev/ttyACM*)" >&2; exit 1; }
esptool.py --chip esp32s3 --port "$port" --baud 921600 write_flash 0x0 "$img"
echo "flashed $img → $port"
