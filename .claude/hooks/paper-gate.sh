#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): before `gh pr create`, require the branch
# to carry its paper update (scripts/check-paper.sh). Exit 2 blocks the call and
# feeds the reason back to Claude. Anything else passes through.
set -uo pipefail
input=$(cat)
# Cheap substring pre-check on the raw JSON before spawning python3 for every Bash call —
# only "gh pr create" invocations need the full parse.
case "$input" in *"gh pr create"*) ;; *) exit 0 ;; esac
cmd=$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null) || exit 0
case "$cmd" in *"gh pr create"*) ;; *) exit 0 ;; esac
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
"$root/scripts/check-paper.sh" && exit 0
exit 2
