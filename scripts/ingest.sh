#!/usr/bin/env bash
# Pull a CoreS3 recording session from the device host into the workstation's
# data root: put the device in USB-drive mode over its serial console, run the
# host-side pull script, rsync the result here, return the device to the menu.
# Usage: ingest.sh -H host [-p /dev/cu.usbmodemNNN] [-d recordings_root] [-n]
# Env:   KWSREC_HOST_PYTHON  python on the host with pyserial (default python3)
#   host: ssh name of the machine the CoreS3 is plugged into (never hard-coded here)
set -euo pipefail

host=${KWSREC_HOST:-}; port=""; root=${KWS_DATA_ROOT:+$KWS_DATA_ROOT/data/recordings}; dry=0
while getopts "H:p:d:n" o; do case $o in H) host=$OPTARG;; p) port=$OPTARG;; d) root=$OPTARG;; n) dry=1;; *) exit 2;; esac; done
[[ -n $host ]] || { echo "usage: $0 -H host [-p port] [-d root] [-n]  (or set KWSREC_HOST)" >&2; exit 2; }
[[ -n $root ]] || { echo "set KWS_DATA_ROOT or pass -d" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
run() { if (( dry )); then echo "+ $*"; else "$@"; fi; }

# Run a read-only command on the host; leaves its stdout in $reply. Called as a plain
# statement (never inside "$(...)") so its `exit` really terminates the script:
# distinguishes "ssh itself failed" (exit 255: unreachable/auth, fatal) from "command
# found nothing" (any other status, e.g. ls on a dir that doesn't exist yet — normal
# while polling for the device/drive to show up).
ssh_probe() {
  local status
  # shellcheck disable=SC2029 # $1 is the caller-built remote command; client-side expansion is intended
  if reply=$(ssh "$host" "$1" 2>/dev/null); then status=0; else status=$?; fi
  if (( status == 255 )); then
    echo "cannot reach host: $host (ssh exit 255)" >&2
    exit 3
  fi
}

# Retry a probe up to 20x1s until it prints something; leaves it in $reply.
# Returns 1 (reply empty) on timeout.
wait_for() {
  for _ in $(seq 1 20); do
    ssh_probe "$1"
    [[ -n $reply ]] && return 0
    sleep 1
  done
  reply=""
  return 1
}


# Send one console line over the device's serial port on the host. The port is the
# ESP32-S3's USB-Serial-JTAG: a shell redirect (`printf > port`) toggles DTR/RTS on
# open and resets the chip, losing the command. pyserial with both lines held low
# does not. KWSREC_HOST_PYTHON names a python on the host that has pyserial
# (default: python3).
serial_send() {
  local py=${KWSREC_HOST_PYTHON:-python3}
  # shellcheck disable=SC2029 # $port/$1 expand client-side on purpose
  run ssh "$host" "$py - <<'PYEOF'
import serial, time
s = serial.Serial(); s.port = '$port'; s.baudrate = 115200; s.timeout = 1
s.dtr = False; s.rts = False; s.open(); s.dtr = False; s.rts = False
time.sleep(2.5); s.read(1 << 20)
s.write(b'$1\\n'); s.flush(); time.sleep(0.8)
try:
    s.read(1 << 16); s.close()
except Exception:
    pass  # the port vanishes when the device switches USB mode: expected
PYEOF"
}

if [[ -z $port ]]; then
  ssh_probe 'ls /dev/cu.usbmodem* 2>/dev/null | head -1'
  port=$reply
  [[ -n $port ]] || { echo "no /dev/cu.usbmodem* on $host — device unplugged or already in USB-drive mode" >&2; exit 1; }
fi
# 1. device -> USB drive mode (serial link disappears while the drive is exported)
ssh_probe 'ls -d /Volumes/KWSREC 2>/dev/null'
if [[ -n $reply ]]; then
  echo "KWSREC already mounted on $host — device is in USB-drive mode, not switching"
else
  serial_send "mode usb"
fi
wait_for 'ls /Volumes/KWSREC 2>/dev/null' || {
  echo "KWSREC did not mount on $host within 20 s" >&2
  echo "likely cause: the host's screen is locked, so loginwindow ejects the drive right after attach" >&2
  echo "recovery: unlock the host's screen, then on the device do 'mode menu' then 'mode usb' again" >&2
  exit 3
}
# 2. copy the pull script over (the host need not carry a kws-de checkout) and run it
#    there into a stamped stage dir. Never wipe the host stage ourselves — it is the
#    only surviving copy of the device's data until step 4 verifies the local rsync.
#    Old stamped dirs under ~/kwsrec-pull/ can be pruned by hand once you trust the pull.
stamp=$(date +%Y-%m-%d-%H%M); dest="$root/incoming/$stamp"
# shellcheck disable=SC2088 # tilde is meant to expand on the remote host, not here
hostdir="~/kwsrec-pull/$stamp"
run scp -q "$script_dir/pull-recordings.sh" "$host:~/pull-recordings.sh"
run ssh "$host" "bash ~/pull-recordings.sh $hostdir"
# 3. bring it here, never deleting anything on either side
mkdir -p "$dest"
run rsync -a --ignore-existing "$host:$hostdir/" "$dest/"
# 4. verify the local copy actually matches what's on the host before trusting it
local_wavs=$(find "$dest" -name '*.wav' | wc -l | tr -d ' ')
ssh_probe "find $hostdir -name '*.wav' 2>/dev/null | wc -l"
remote_wavs=$(printf '%s' "$reply" | tr -d ' ')
if [[ $local_wavs != "$remote_wavs" ]]; then
  echo "wav count mismatch: local=$local_wavs remote=$remote_wavs — host copy kept at $host:$hostdir" >&2
  exit 1
fi
[[ -f "$dest/sessions.csv" ]] || {
  echo "no sessions.csv in $dest — host copy kept at $host:$hostdir" >&2
  exit 1
}
session_rows=$(($(wc -l < "$dest/sessions.csv") - 1))
if [[ $session_rows != "$local_wavs" ]]; then
  echo "sessions.csv has $session_rows rows but $local_wavs wavs — host copy kept at $host:$hostdir" >&2
  exit 1
fi
(( local_wavs > 0 )) || { echo "nothing pulled into $dest" >&2; exit 1; }
# 5. device back to the selection screen (port is back once the drive is ejected)
if wait_for 'ls /dev/cu.usbmodem* 2>/dev/null | head -1'; then
  port=$reply
else
  echo "warning: no /dev/cu.usbmodem* on $host within 20 s — still using '$port'" >&2
fi
serial_send "mode menu" || echo "warning: could not send 'mode menu' — tap Menu on the device" >&2
echo "ingested $local_wavs takes -> $dest"
