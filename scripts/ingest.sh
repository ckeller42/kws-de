#!/usr/bin/env bash
# Pull a CoreS3 recording session from the device host into the workstation's
# data root: put the device in USB-drive mode over its serial console, run the
# host-side pull script, rsync the result here, return the device to the menu.
# Usage: ingest.sh -H host [-p /dev/cu.usbmodemNNN] [-d recordings_root] [-n]
#   host: ssh name of the machine the CoreS3 is plugged into (never hard-coded here)
set -euo pipefail

host=${KWSREC_HOST:-}; port=""; root=${KWS_DATA_ROOT:+$KWS_DATA_ROOT/data/recordings}; dry=0
while getopts "H:p:d:n" o; do case $o in H) host=$OPTARG;; p) port=$OPTARG;; d) root=$OPTARG;; n) dry=1;; *) exit 2;; esac; done
[[ -n $host ]] || { echo "usage: $0 -H host [-p port] [-d root] [-n]  (or set KWSREC_HOST)" >&2; exit 2; }
[[ -n $root ]] || { echo "set KWS_DATA_ROOT or pass -d" >&2; exit 2; }
run() { if (( dry )); then echo "+ $*"; else "$@"; fi; }

if [[ -z $port ]]; then
  # shellcheck disable=SC2029 # expand $port client-side; it names the file on the remote we write to
  port=$(ssh "$host" 'ls /dev/cu.usbmodem* 2>/dev/null | head -1' || true)
  [[ -n $port ]] || { echo "no /dev/cu.usbmodem* on $host — device unplugged or already in USB-drive mode" >&2; exit 1; }
fi
# 1. device -> USB drive mode (serial link disappears while the drive is exported)
# shellcheck disable=SC2029
run ssh "$host" "printf 'mode usb\n' > '$port'"
for _ in $(seq 1 20); do
  if [[ -n $(ssh "$host" 'ls /Volumes/KWSREC 2>/dev/null' || true) ]]; then break; fi
  sleep 1
done
[[ -n $(ssh "$host" 'ls /Volumes/KWSREC 2>/dev/null' || true) ]] || { echo "KWSREC did not mount on $host within 20 s" >&2; exit 3; }
# 2. copy the pull script over (the host need not carry a kws-de checkout) and run it there
run scp -q scripts/pull-recordings.sh "$host:~/pull-recordings.sh"
run ssh "$host" 'rm -rf ~/kwsrec-pull && bash ~/pull-recordings.sh ~/kwsrec-pull'
# 3. bring it here, never deleting anything on either side
stamp=$(date +%Y-%m-%d-%H%M); dest="$root/incoming/$stamp"
mkdir -p "$dest"
run rsync -a --ignore-existing "$host:~/kwsrec-pull/" "$dest/"
# 4. device back to the selection screen (port is back once the drive is ejected)
for _ in $(seq 1 20); do
  p=$(ssh "$host" 'ls /dev/cu.usbmodem* 2>/dev/null | head -1' || true); [[ -n $p ]] && { port=$p; break; }; sleep 1
done
# shellcheck disable=SC2029
run ssh "$host" "printf 'mode menu\n' > '$port'" || echo "warning: could not send 'mode menu' — tap Menu on the device" >&2
n=$(find "$dest" -name '*.wav' | wc -l | tr -d ' ')
(( n > 0 )) || { echo "nothing pulled into $dest" >&2; exit 1; }
echo "ingested $n takes -> $dest"
