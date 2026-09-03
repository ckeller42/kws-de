#!/usr/bin/env bash
# The whole data loop, end to end: ingest -> QC -> v3 dataset build -> train ->
# export (model-health gate) -> evals. Stops at the first failing stage.
# Usage: data-loop.sh -H host [--skip-ingest] [--skip-train] [--incoming DIR]
#                      [--prefix PREFIX] [-n]
#   -H host        ssh name of the machine the CoreS3 is plugged into (see
#                  scripts/ingest.sh); required unless --skip-ingest is given
#                  (or set KWSREC_HOST)
#   --skip-ingest  reuse an already-pulled session instead of running ingest.sh
#   --skip-train   run QC + dataset build + evals only; skip train/export
#   --incoming DIR use this incoming session (default: the newest one, or the
#                  one ingest.sh just pulled)
#   --prefix P     npz/model/manifest prefix (default: features_v3)
#   -n             dry run: print the commands each stage would run, run none
set -euo pipefail
cd "$(dirname "$0")/.."   # scripts/ingest.sh and `uv run` resolve against the repo root

host=${KWSREC_HOST:-}
prefix=features_v3
skip_ingest=0
skip_train=0
incoming=""
dry=0
while [[ $# -gt 0 ]]; do
  case $1 in
    -H) host=$2; shift 2 ;;
    --skip-ingest) skip_ingest=1; shift ;;
    --skip-train) skip_train=1; shift ;;
    --incoming) incoming=$2; shift 2 ;;
    --prefix) prefix=$2; shift 2 ;;
    -n) dry=1; shift ;;
    *)
      echo "usage: $0 -H host [--skip-ingest] [--skip-train] [--incoming DIR] [--prefix PREFIX] [-n]" >&2
      exit 2
      ;;
  esac
done
: "${KWS_DATA_ROOT:?set KWS_DATA_ROOT}"
rec="$KWS_DATA_ROOT/data/recordings"

run() { if (( dry )); then echo "+ $*"; else "$@"; fi; }
stage() { echo "== $* =="; }

if (( ! skip_ingest )); then
  [[ -n $host ]] || {
    echo "-H host required for ingest (or set KWSREC_HOST, or pass --skip-ingest)" >&2
    exit 2
  }
  stage "ingest ($host)"
  if (( dry )); then
    echo "+ scripts/ingest.sh -H $host"
  else
    incoming=$(scripts/ingest.sh -H "$host" | tail -1 | awk '{print $NF}')
  fi
fi
if [[ -z $incoming ]]; then
  incoming=$(find "$rec/incoming" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
  [[ -n $incoming ]] || {
    echo "no incoming session under $rec/incoming and none was pulled" >&2
    exit 1
  }
fi

# Every long stage below runs through `kws-eta run <stage> <size> -- ...`: it prints an
# ETA from this stage's recorded history before the command starts, then records the
# actual wall time after. `size` is whatever cheap-to-compute number scales with the
# stage's work (see each stage's comment) -- never something that itself requires doing
# the work first.
eta() { run uv run --no-sync kws-eta run "$1" "$2" -- uv run --no-sync "${@:3}"; }

stage "QC $incoming"
# size = number of takes in this session (sessions.csv has one header + one row/take).
qc_size=$(($(wc -l < "$incoming/sessions.csv" 2>/dev/null || echo 1) - 1))
(( qc_size > 0 )) || qc_size=1
eta qc "$qc_size" kws-qc "$incoming"

stage "dataset build ($prefix)"
# The v3 cache holds the MSWC/TTS material only; the approved recordings are merged in
# by every build (kws_de.data.merge_recordings), so seeding it from the full real+TTS
# cache is enough. NEVER seed from raw_clips_v2.pkl: it is a 25-word subset with empty
# lists, which makes the build re-synthesise ~300 TTS clips per word (87% TTS) and the
# INT8 model then fails the export health gate.
cache="$KWS_DATA_ROOT/data/raw_clips_v3.pkl"
if [[ ! -f $cache ]]; then
  seed="$KWS_DATA_ROOT/data/raw_clips_merged.pkl"
  [[ -f $seed ]] || {
    echo "no $cache and no $seed to seed it from — mine MSWC once first:" >&2
    echo "  uv run --no-sync kws-data --fetch --v3 --mswc-root <mswc-de-dir>" >&2
    exit 1
  }
  echo "seeding $(basename "$cache") from $(basename "$seed") (full real+TTS cache)"
  run cp "$seed" "$cache"
fi
# size = the cache pickle's byte size -- a cheap stand-in for clip count that doesn't
# require unpickling it (the pickle's own load, inside the build, is the real work).
build_size=$(wc -c < "$cache" 2>/dev/null | tr -d ' ' || echo 1)
eta "dataset-build" "$build_size" kws-dataset build --cache raw_clips_v3.pkl --prefix "$prefix"

epochs=40
if (( ! skip_train )); then
  stage "train"
  # size = epochs * train-split row count, read back from the manifest the dataset-build
  # stage above just wrote (falls back to epochs alone if that manifest is missing, e.g.
  # a dry run).
  manifest="$KWS_DATA_ROOT/data/manifest${prefix#features}.json"
  n_train=$(uv run --no-sync python -c '
import json, sys
try:
    print(json.load(open(sys.argv[1]))["splits"]["train"]["n"])
except (OSError, KeyError):
    print(0)
' "$manifest" 2>/dev/null || echo 0)
  train_size=$(( epochs * n_train ))
  (( train_size > 0 )) || train_size=$epochs
  eta train "$train_size" \
    kws-train --v2 --prefix "$prefix" --out command_v3.keras --epochs "$epochs"
  stage "export (health gate)"
  eta export 1 kws-export --prefix "$prefix" --model command_v3.keras --firmware
fi

stage "evals"
# size = number of approved clips this eval scores.
eval_size=$(find "$rec/approved" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
(( eval_size > 0 )) || eval_size=1
eta eval "$eval_size" \
  kws-eval --recordings "$rec/approved" --prefix "$prefix" --out docs/eval-report-v3.md

echo "done: held-out + user-customised figures in \$KWS_DATA_ROOT/docs/eval-report-v3.md. Flash with your flash script for the device host."
