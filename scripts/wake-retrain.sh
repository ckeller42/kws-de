#!/usr/bin/env bash
# Retrain the "Hey Bus" microWakeWord model per the recipe in train/mww/README.md.
#
# microWakeWord is python-3.10-only and its datasets run to tens of gigabytes, so it
# lives in its own directory outside this repo (train/mww/setup.sh creates it). This
# script drives that directory: it stages a session-disjoint train/held-out split out
# of $KWS_DATA_ROOT, builds the real-audio feature sets, trains, and probes the result
# against the currently installed model. Nothing is written inside the repo and no
# model is installed — the candidate is left in $KWS_DATA_ROOT/models/ under its round
# name for a human to compare and promote.
#
# Usage: scripts/wake-retrain.sh [--stage-only] [--skip-train]
#   --stage-only  stage the split and stop (inspect what would be trained on)
#   --skip-train  re-export and re-probe an already-trained round
#
# Environment:
#   KWS_DATA_ROOT (required) shared data root — recordings in, candidate model out
#   MWW_DIR       (required) the microWakeWord training directory: holds .venv,
#                 gen_features_real.py, with_eta.sh and training_parameters_<round>.yaml
#   WAKE_ROUND    round name; names the work dir, config and candidate (default r6)
#   WAKE_HOLDOUT  space-separated recording-session stamps kept out of train AND
#                 validation (default: the newest session that produced wake clips)
#   WAKE_SIL_DIR  directory of speech-free field takes to use as room-noise negatives
#   WAKE_EXCLUDE  space-separated basenames of approved clips to drop from training
#                 entirely. The two failures it existed for -- phrase/negative clips
#                 still containing "Hey Bus", and "wake" clips holding only the tail
#                 fragment of the phrase -- are fixed in kws_de.qc (#58) and checked by
#                 scripts/audit-approved.py, so a clean tree needs no exclusions. See
#                 train/mww/README.md.
#   WAKE_TTS_DIRS space-separated synthetic-clip directories to run kws-tts-check over
#                 before feature generation (relative to MWW_DIR unless absolute;
#                 default generated_samples_v3/{positives,negatives}). Each needs the
#                 manifest.csv kws_de.tts writes; a directory without one cannot be
#                 vouched for and fails the check. Set to " " to skip.
#   WAKE_BASELINE model to compare against (default $KWS_DATA_ROOT/models/hey_bus.tflite)
set -euo pipefail

: "${KWS_DATA_ROOT:?set KWS_DATA_ROOT}"
: "${MWW_DIR:?set MWW_DIR to the microWakeWord training directory}"
round=${WAKE_ROUND:-r6}
baseline=${WAKE_BASELINE:-$KWS_DATA_ROOT/models/hey_bus.tflite}
repo=$(cd "$(dirname "$0")/.." && pwd)

stage_only=0
skip_train=0
for arg in "$@"; do
  case $arg in
    --stage-only) stage_only=1 ;;
    --skip-train) skip_train=1 ;;
    *) echo "usage: $0 [--stage-only] [--skip-train]" >&2; exit 2 ;;
  esac
done

rec=$KWS_DATA_ROOT/data/recordings
work=$MWW_DIR/${round}_data
[ -d "$MWW_DIR/.venv" ] || { echo "no .venv in MWW_DIR — run train/mww/setup.sh" >&2; exit 1; }
[ -d "$rec/approved" ] || { echo "no approved recordings under $rec" >&2; exit 1; }

# A retrain has to be judged on a session it has never seen. Splitting one session at
# random measures memorisation, not generalisation: every clip in it shares a room, a
# mic gain and a firmware build. So the held-out unit is the session.
holdout=${WAKE_HOLDOUT:-}
if [ -z "$holdout" ]; then
  for stamp in $(find "$rec/qc" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort -r); do
    if grep -q '^wake/' "$rec/qc/$stamp/written.txt" 2>/dev/null; then holdout=$stamp; break; fi
  done
fi
[ -n "$holdout" ] || { echo "no session under $rec/qc produced wake clips" >&2; exit 1; }
echo "== round $round | held-out session(s): $holdout"

# Each session's written.txt lists the approved files it produced, so the held-out file
# set is exactly the union of those lines — no filename convention to keep in sync.
rm -rf "$work"
mkdir -p "$work"/{pos_train,pos_hold,neg_train,neg_hold,sil_train}
held=$work/holdout.txt
: > "$held"
for stamp in $holdout; do
  [ -f "$rec/qc/$stamp/written.txt" ] || { echo "unknown session $stamp" >&2; exit 1; }
  grep -E '^(wake|phrases|negatives)/' "$rec/qc/$stamp/written.txt" >> "$held"
done
sort -u -o "$held" "$held"

# The in-training hard negatives are the FIELD takes' own non-wake speech — same
# voice, same microphone, same room as the positives (train/mww/README.md rule 3).
# The guided phrase/negative takes are real speech too, but they are studio-style
# prompted reads from other speakers, so mixing them in silently changes what "real
# hard negative" means between rounds. A session whose qc.csv holds a set=field row
# is a field session, and every path in its written.txt is field-derived.
field=$work/field.txt
: > "$field"
for stamp in "$rec"/qc/*/; do
  if ! [ -f "$stamp/qc.csv" ] || ! [ -f "$stamp/written.txt" ]; then continue; fi
  if awk -F, 'NR > 1 && $2 == "field" { found = 1 } END { exit !found }' "$stamp/qc.csv"; then
    grep -E '^(phrases|negatives)/' "$stamp/written.txt" >> "$field" || true
  fi
done
sort -u -o "$field" "$field"

for rel in $(cd "$rec/approved" && find wake phrases negatives -name '*.wav' | sort); do
  [ "${rel%%/*}" = wake ] && kind=pos || kind=neg
  if grep -qxF "$rel" "$held"; then
    dest=${kind}_hold
  else
    case " ${WAKE_EXCLUDE:-} " in *" $(basename "$rel") "*) continue ;; esac
    if [ "$kind" = neg ] && ! grep -qxF "$rel" "$field"; then continue; fi
    dest=${kind}_train
  fi
  ln -sf "$rec/approved/$rel" "$work/$dest/"
done

# Speech-free takes are the device's own room noise: negatives at no labelling cost.
if [ -n "${WAKE_SIL_DIR:-}" ]; then
  find "$WAKE_SIL_DIR" -name '*.wav' -exec ln -sf {} "$work/sil_train/" \;
fi

for dir in "$work"/*/; do
  printf '   %-10s %4s clips\n' "$(basename "$dir")" "$(find "$dir" -name '*.wav' | wc -l | tr -d ' ')"
done
[ "$stage_only" = 1 ] && exit 0

cd "$MWW_DIR"
if [ "$skip_train" = 0 ]; then
  # The synthetic positives/negatives are most of the training set and nobody ever
  # listens to them: a voice that turned out not to be German would train the wake word
  # on the wrong language. Gate them before they become features — kws-tts-check exits
  # non-zero on any failing clip, and set -e stops the round here.
  # shellcheck disable=SC2086 # the list is space-separated by contract
  for d in ${WAKE_TTS_DIRS:-generated_samples_v3/positives generated_samples_v3/negatives}; do
    case $d in /*) ;; *) d=$MWW_DIR/$d ;; esac
    [ -d "$d" ] || continue
    echo "== tts-check $d"
    (cd "$repo" && uv run --no-sync kws-tts-check "$d")
  done

  # Real audio gets its own feature dirs so its share of a batch is a number in the
  # training config, not an accident of how many clips happen to exist.
  echo "== features"
  .venv/bin/python gen_features_real.py "${round}_neg_speech" "$work/neg_train" 0.2 55
  [ -n "$(ls -A "$work/sil_train")" ] &&
    .venv/bin/python gen_features_real.py "${round}_neg_silence" "$work/sil_train" 0.1 5

  echo "== train"
  ./with_eta.sh 20000 -- .venv/bin/python -m microwakeword.model_train_eval \
    --training_config="training_parameters_$round.yaml" \
    --train 1 --test_tflite_streaming_quantized 1 --use_weights best_weights mixednet \
    --pointwise_filters "64,64,64,64" --residual_connection "0,0,0,0" \
    --repeat_in_block "1,1,1,1" --mixconv_kernel_sizes "[5],[9],[13],[21]" \
    --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3
fi

candidate=$KWS_DATA_ROOT/models/hey_bus_$round.tflite
cp "trained_models/hey_bus_$round/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite" \
   "$candidate"
echo "== candidate $candidate ($(wc -c < "$candidate" | tr -d ' ') bytes)"

echo "== probe at the device gate (0.85 x 2), baseline first"
for set_dir in pos_hold pos_train neg_hold neg_train sil_train; do
  [ -n "$(ls -A "$work/$set_dir" 2>/dev/null)" ] || continue
  for model in "$baseline" "$candidate"; do
    printf '%-10s %-24s ' "$set_dir" "$(basename "$model")"
    (cd "$repo" && uv run --no-sync python scripts/wake_probe.py "$model" "$work/$set_dir" | tail -1)
  done
done
