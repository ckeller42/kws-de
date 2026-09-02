#!/usr/bin/env bash
# The paper travels with the code: a branch that changes the model, data,
# evaluation, or firmware must also touch docs/paper.md or docs/paper-notes.md.
# Compares the current branch against its merge-base with origin/main. Used by
# .githooks/pre-push and the Claude PreToolUse hook on `gh pr create`.
# Bypass (deliberately, e.g. pure refactors): PAPER_SKIP=1.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
[ -n "${PAPER_SKIP:-}" ] && exit 0

base=$(git merge-base HEAD origin/main 2>/dev/null) || exit 0   # no origin/main: nothing to compare
changed=$(git diff --name-only "$base" HEAD)
echo "$changed" | grep -qE '^(kws_de/|firmware/|docs/[a-z0-9-]+-report\.md)' || exit 0
echo "$changed" | grep -qE '^docs/(paper|paper-notes)\.md$' && exit 0

cat >&2 <<EOF
[check-paper] this branch changes code/results but not the paper:
$(echo "$changed" | grep -E '^(kws_de/|firmware/|docs/[a-z0-9-]+-report\.md)' | sed 's/^/  /')
Update docs/paper.md and/or docs/paper-notes.md (results, method, or a paper-notes log
entry), or set PAPER_SKIP=1 for a change with no paper-visible effect.
EOF
exit 1
