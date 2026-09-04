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
[[ -f $sessions ]] || echo "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,device_intent,device_words,window_ms" > "$sessions"

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

# Field takes (Assistent mode, opt-in): same shape as a guided take plus what
# the device itself recognised. field.csv is folded into sessions.csv rather
# than copied, so QC reads one file for every set.
for spk in "$mnt"/field/spk*/; do
  spk=${spk%/}; name=$(basename "$spk")
  rsync -a --exclude field.csv "$spk/" "$dest/field/$name/"
  if [[ -f $spk/field.csv ]]; then
    # field.csv: file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs
    # sessions:  speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,device_intent,device_words,window_ms
    # window_ms ($6) rides along last: together with ms ($7) it is what lets QC
    # mark a take the ring cut short (ms < FIELD_PREROLL_MS + window_ms).
    tail -n +2 "$spk/field.csv" | awk -F, -v OFS=, -v s="$name" -v p="$pulled" \
      '{print s, p, "", "field/" s "/" $1, $7, $8, "field", "", $2, $2, $3, $4, $5, $6}' >> "$sessions"
  fi
  rm -rf "$spk"
  echo "pulled field/$name"
done
rmdir "$mnt/field" 2>/dev/null || true

if [[ -f $mnt/recognise.log ]]; then
  mv "$mnt/recognise.log" "$dest/logs/recognise-${pulled//:/-}.log"
fi

if [[ -z ${KWSREC_NO_EJECT:-} ]]; then
  if command -v diskutil >/dev/null; then diskutil eject "$mnt" >/dev/null
  elif command -v udisksctl >/dev/null; then udisksctl unmount -b "$(findmnt -n -o SOURCE "$mnt")" >/dev/null
  fi
fi
echo "done → $dest"
