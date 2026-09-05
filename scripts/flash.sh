#!/usr/bin/env bash
# Flash a merged CI/CD image onto a CoreS3. Usage: scripts/flash.sh [image.bin] [port]
# Needs esptool (pip install esptool, or the ESP-IDF venv). Hold BOOT/reset if the port is not found.
# Env: KWSREC_HOST  ssh host the CoreS3 is plugged into (flash over ssh instead of locally);
#      KWSREC_PORT  serial port on that host (default: autodetect /dev/cu.usbmodem* there)
set -euo pipefail
# shellcheck disable=SC2012 # ls -t is simplest way to pick the newest match; filenames here are plain shas/dev nodes
img=${1:-$(ls -t kws-de-fw-*.bin 2>/dev/null | head -1)}
[[ -n ${img:-} && -f $img ]] || { echo "usage: $0 [kws-de-fw-<sha>.bin] [port]" >&2; exit 1; }

if [[ -n ${KWSREC_HOST:-} ]]; then
  host=$KWSREC_HOST
  port=${2:-${KWSREC_PORT:-}}
  if [[ -z $port ]]; then
    port=$(ssh "$host" 'ls /dev/cu.usbmodem* 2>/dev/null | head -1' || true)
    [[ -n $port ]] || {
      echo "no /dev/cu.usbmodem* on $host — device unplugged, or in USB-drive mode (send 'mode menu' first)" >&2
      exit 1
    }
  fi
  remote="/tmp/$(basename "$img")"
  scp -q "$img" "$host:$remote"
  # shellcheck disable=SC2029 # $port/$remote expand client-side on purpose
  ssh "$host" "esptool.py --chip esp32s3 --port '$port' --baud 921600 write_flash 0x0 '$remote'"
  echo "flashed $img → $host:$port"
else
  # shellcheck disable=SC2012
  port=${2:-$(ls /dev/cu.usbmodem* /dev/ttyACM* 2>/dev/null | head -1)}
  [[ -n ${port:-} ]] || { echo "no serial port found (/dev/cu.usbmodem* or /dev/ttyACM*)" >&2; exit 1; }
  esptool.py --chip esp32s3 --port "$port" --baud 921600 write_flash 0x0 "$img"
  echo "flashed $img → $port"
fi
