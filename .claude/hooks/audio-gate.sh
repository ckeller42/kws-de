#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): gate `afplay`/`say` (locally or over
# `ssh host ...`) so an unchecked or non-German TTS clip never reaches a
# speaker or a device test again -- an English "German" clip already did
# once. Exit 2 blocks the call and explains why; anything else passes.
# Bypass: put KWS_AUDIO_GATE=0 in front of the command.
set -uo pipefail
input=$(cat)
# Cheap substring pre-check on the raw JSON before spawning python3 for every
# Bash call -- only afplay/say invocations need the full parse.
case "$input" in *afplay*|*"say "*) ;; *) exit 0 ;; esac
case "$input" in *"KWS_AUDIO_GATE=0"*) exit 0 ;; esac

# shellcheck disable=SC2016 # single-quoted python source below, no shell expansion intended
msg=$(printf '%s' "$input" | python3 -c '
import csv, json, os, re, sys

cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
if "afplay" not in cmd and "say " not in cmd:
    sys.exit(0)

tts_dir = os.environ.get("KWS_TTS_DIR", "")
problems = []

# say -v VOICE: the voice must be a German one, blocked outright otherwise --
# macOS `say` silently falls back to an English voice when the German pack is
# not installed on whatever host runs the command.
m = re.search(r"say\s+-v\s+(\"[^\"]*\"|\x27[^\x27]*\x27|\S+)", cmd)
if m:
    voice = m.group(1).strip("\"\x27")
    if "(German" not in voice:
        problems.append(f"say -v {voice!r}: not a German voice (name must contain \"(German\") -- blocked outright")


def check_wav(raw_path: str) -> None:
    path = raw_path.strip("\"\x27")
    local = os.path.expanduser(path)
    d = os.path.dirname(local) or "."
    if not os.path.isdir(d):
        # Cannot resolve this locally (e.g. a remote-only path like
        # ~/kws-walk/x.wav on the other end of an ssh command) -- fall back to
        # the local staging dir, matched by filename.
        if tts_dir and os.path.isdir(tts_dir):
            d = tts_dir
            local = os.path.join(tts_dir, os.path.basename(path))
        else:
            problems.append(
                f"{path}: cannot resolve locally -- set KWS_TTS_DIR to the local TTS "
                f"staging dir, or bypass with KWS_AUDIO_GATE=0 in front of the command"
            )
            return
    csv_path = os.path.join(d, "tts_check.csv")
    name = os.path.basename(local)
    if not os.path.isfile(csv_path):
        problems.append(f"{path}: no tts_check.csv in {d} -- run kws-tts-check first")
        return
    with open(csv_path, newline="") as fh:
        rows = {r["file"]: r for r in csv.DictReader(fh)}
    row = rows.get(name)
    if row is None:
        problems.append(f"{path}: not in {csv_path} -- run kws-tts-check first")
    elif row.get("ok", "0") not in ("1", "True", "true"):
        reason = row.get("reason") or "?"
        problems.append(f"{path}: failed kws-tts-check ({reason}) -- do not play it")


for wav in re.findall(r"\"[^\"]+\.wav\"|\x27[^\x27]+\.wav\x27|[^\s\"\x27]+\.wav", cmd):
    check_wav(wav)

print("\n".join(problems))
' 2>/dev/null)

[ -z "$msg" ] && exit 0
echo "$msg" >&2
exit 2
