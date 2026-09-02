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

stage "QC $incoming"
run uv run --no-sync kws-qc "$incoming"

stage "dataset build ($prefix)"
run uv run --no-sync kws-dataset build --cache raw_clips_v3.pkl --prefix "$prefix"

if (( ! skip_train )); then
  stage "train"
  run uv run --no-sync kws-train --v2 --prefix "$prefix" --out command_v3.keras --epochs 40
  stage "export (health gate)"
  run uv run --no-sync kws-export --prefix "$prefix" --model command_v3.keras --firmware
fi

stage "evals"
run uv run --no-sync kws-eval --recordings "$rec/approved" --prefix "$prefix" --out docs/eval-report-v3.md

echo "done: held-out figures + user-customised section in docs/eval-report-v3.md. Flash with your flash script for the device host."
