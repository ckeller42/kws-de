#!/usr/bin/env bash
# Claude Code PostToolUse hook (Edit|Write|MultiEdit): lint the single file just
# edited so a lint break surfaces in-session, not at commit/push/CI. Exit 2 feeds
# the findings back to Claude to fix before moving on; clean or non-lintable -> 0.
set -uo pipefail
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

# Repo-relative path (markdownlint's ignore logic rejects paths outside cwd);
# skip anything not under this repo.
# Script via -c (not a heredoc) so stdin stays the hook's JSON, not the script.
f=$(python3 -c '
import json, os, sys
p = json.load(sys.stdin).get("tool_input", {}).get("file_path", "")
if not p:
    sys.exit(0)
rp = os.path.relpath(os.path.abspath(p), sys.argv[1])
print("" if rp.startswith("..") else rp)
' "$root" 2>/dev/null) || exit 0
[ -z "$f" ] && exit 0

case "$f" in
  *.py)
    out=$(uv run ruff check "$f" 2>&1) && exit 0
    { echo "ruff findings in $f:"; echo "$out"; } >&2; exit 2 ;;
  *docs/superpowers/*.md) exit 0 ;;  # excluded from lint (illustrative fences)
  *.md)
    out=$(npx --yes markdownlint-cli@0.42.0 --config .markdownlint.json "$f" 2>&1) && exit 0
    { echo "markdownlint findings in $f:"; echo "$out"; } >&2; exit 2 ;;
esac
exit 0
