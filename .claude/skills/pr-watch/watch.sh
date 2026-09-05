#!/usr/bin/env bash
# Poll a PR until it merges cleanly, or stop and explain why it can't.
# Usage: watch.sh PR_NUMBER [interval_seconds]
# Run from inside the repo (a DIRTY check needs a local git worktree).
set -euo pipefail
pr=${1:?usage: watch.sh PR_NUMBER [interval_seconds]}
interval=${2:-30}

base=$(gh pr view "$pr" --json baseRefName -q .baseRefName)
if [[ $base != main ]]; then
  echo "PR #$pr bases onto '$base', not main -- refusing to auto-merge a stacked PR (merge the base branch into main first)" >&2
  exit 2
fi

while true; do
  state=$(gh pr view "$pr" --json state -q .state)
  case "$state" in
    MERGED) echo "PR #$pr merged"; exit 0 ;;
    CLOSED) echo "PR #$pr closed without merging" >&2; exit 1 ;;
  esac

  mss=$(gh pr view "$pr" --json mergeStateStatus -q .mergeStateStatus)
  case "$mss" in
    BEHIND)
      echo "PR #$pr behind main -- updating branch"
      gh pr update-branch "$pr" >/dev/null
      ;;
    CLEAN)
      echo "PR #$pr clean -- merging"
      exec gh pr merge "$pr" --merge
      ;;
    DIRTY)
      echo "PR #$pr has merge conflicts with main:" >&2
      branch=$(gh pr view "$pr" --json headRefName -q .headRefName)
      git fetch -q origin main "$branch"
      tmp=$(mktemp -d)
      git worktree add -q --detach "$tmp" "origin/$branch" >/dev/null 2>&1
      ( cd "$tmp" && git merge -q --no-commit --no-ff origin/main >/dev/null 2>&1
        git diff --name-only --diff-filter=U )  >&2
      git worktree remove --force "$tmp" >/dev/null 2>&1
      exit 2
      ;;
    UNSTABLE)
      echo "PR #$pr has failing checks:" >&2
      gh pr checks "$pr" 2>/dev/null | grep -iE 'fail' >&2 || gh pr checks "$pr" >&2
      exit 2
      ;;
    *)
      ;;
  esac
  sleep "$interval"
done
