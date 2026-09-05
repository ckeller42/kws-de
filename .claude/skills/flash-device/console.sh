#!/usr/bin/env bash
# Talk to the CoreS3's serial console on the host it is plugged into: send
# commands, capture the reply and N more seconds of log. Opens the port with
# DTR/RTS held low -- the S3's USB-Serial-JTAG resets the chip on a careless
# open (a plain `cat`/`stty`/default-pyserial open leaves it silent forever).
# Usage: console.sh [-H host] [-p port] [-l SECONDS] [-w WAIT] cmd
#   -H  ssh host the CoreS3 is plugged into (default: $KWSREC_HOST)
#   -p  serial port on that host (default: $KWSREC_PORT, else autodetect)
# Env: KWSREC_HOST_PYTHON  python on the host with pyserial (default: python3)
#   -l  seconds of log to capture after the command (default 5)
#   -w  seconds to wait after open for the boot log to pass (default 3)
#   cmd exactly ONE console line, e.g. status | 'mode wake' | 'mode menu'
# Output: everything the device printed, screen-frame base64 stripped.
#
# Pass only one command per invocation: opening the port resets the chip, so
# chaining two console.sh calls in one shell invocation delivers them only
# ~30 ms apart -- long before the first one's reset settles.
set -euo pipefail
host=${KWSREC_HOST:-}; port=${KWSREC_PORT:-}; logsec=5; wait=3
while getopts "H:p:l:w:" o; do case $o in H) host=$OPTARG;; p) port=$OPTARG;; l) logsec=$OPTARG;; w) wait=$OPTARG;; *) exit 2;; esac; done
shift $((OPTIND-1))
[[ -n $host ]] || { echo "usage: $0 -H host [-p port] [-l secs] [-w secs] cmd  (or set KWSREC_HOST)" >&2; exit 2; }
(( $# == 1 )) || { echo "pass exactly one console command per invocation (opening the port resets the chip)" >&2; exit 2; }
cmd=$1

if [[ -z $port ]]; then
  port=$(ssh "$host" 'ls /dev/cu.usbmodem* 2>/dev/null | head -1' || true)
  [[ -n $port ]] || { echo "no /dev/cu.usbmodem* on $host -- unplugged, or a CDC port from USB-drive mode (ls /dev/cu.usbmodem* to check; 'mode menu' over the CDC port leaves USB mode)" >&2; exit 1; }
fi

py=${KWSREC_HOST_PYTHON:-python3}
# shellcheck disable=SC2029 # $py/$port/$cmd/$wait/$logsec expand client-side on purpose
ssh "$host" "$py - <<'EOF'
import re, sys, time
import serial
s = serial.Serial(); s.port = '$port'; s.baudrate = 115200; s.timeout = 1
s.dtr = False; s.rts = False; s.open(); s.dtr = False; s.rts = False
time.sleep($wait); s.read(1 << 20)                     # let the reset's boot log pass
out = b''
s.write(('$cmd' + '\n').encode()); s.flush(); time.sleep(0.8); out += s.read(1 << 16)
end = time.time() + $logsec
while time.time() < end: out += s.read(8192)
for line in out.decode('utf-8', 'replace').splitlines():
    if line.startswith('[SHOT') or line.startswith('[/SHOT') or re.fullmatch(r'[A-Za-z0-9+/=]{60,}', line): continue
    print(line)
EOF"
