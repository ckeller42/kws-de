#!/usr/bin/env bash
# Pull recordings off a CoreS3 in USB-drive mode into data/recordings/ and clear the device.
# Usage: scripts/pull-recordings.sh [DEST]   (default data/recordings)
# Env:   KWSREC_MOUNT=/path  override auto-detect;  KWSREC_NO_EJECT=1  skip eject (tests)
set -euo pipefail

dest=${1:-data/recordings}
mnt=${KWSREC_MOUNT:-}
if [[ -z $mnt ]]; then
  for c in /Volumes/KWSREC /media/*/KWSREC /run/media/*/KWSREC; do
    [[ -d $c ]] && mnt=$c && break
  done
fi
if [[ -z $mnt || ! -d $mnt ]]; then
  echo "no KWSREC drive mounted (put the device in USB mode, or set KWSREC_MOUNT)" >&2
  exit 1
fi

mkdir -p "$dest/logs"
pulled=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sessions="$dest/sessions.csv"
[[ -f $sessions ]] || echo "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts" > "$sessions"

shopt -s nullglob
for spk in "$mnt"/spk*/; do
  spk=${spk%/}; name=$(basename "$spk")
  rsync -a --exclude session.csv "$spk/" "$dest/$name/"
  if [[ -f $spk/session.csv ]]; then
    tail -n +2 "$spk/session.csv" | sed "s/^/$name,$pulled,/" >> "$sessions"
  fi
  rm -rf "$spk"
  echo "pulled $name"
done
if [[ -f $mnt/recognise.log ]]; then
  mv "$mnt/recognise.log" "$dest/logs/recognise-${pulled//:/-}.log"
fi

if [[ -z ${KWSREC_NO_EJECT:-} ]]; then
  if command -v diskutil >/dev/null; then diskutil eject "$mnt" >/dev/null
  elif command -v udisksctl >/dev/null; then udisksctl unmount -b "$(findmnt -n -o SOURCE "$mnt")" >/dev/null
  fi
fi
echo "done → $dest"
