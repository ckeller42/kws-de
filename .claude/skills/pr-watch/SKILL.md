---
name: pr-watch
description: Use when the user asks to watch, babysit, or wait on an open kws-de pull request until it merges -- replaces re-typing the poll/update-branch/merge loop by hand each time.
disable-model-invocation: true
---

# PR watch

Polls one PR until it merges, and stops with a clear reason the moment it
can't proceed on its own.

## Usage

```bash
.claude/skills/pr-watch/watch.sh PR_NUMBER [interval_seconds]
```

Run from inside the repo (a DIRTY result needs a local git worktree to find
the conflicting files). Default poll interval is 30 s.

## What it does each poll

1. Refuses to start at all unless the PR's base branch is `main` -- this
   tool merges into main; a PR stacked onto another feature branch needs
   that branch merged first, by hand.
2. `state == MERGED` -> prints it and exits 0. `CLOSED` -> exits 1.
3. `mergeStateStatus`:
   - `BEHIND` -> runs `gh pr update-branch`, keeps polling.
   - `CLEAN` -> runs `gh pr merge --merge` and exits 0.
   - `DIRTY` -> does a scratch `git worktree` merge of the PR branch against
     `origin/main`, prints the conflicting files, and exits 2. Resolve them
     on the PR branch and re-run the skill.
   - `UNSTABLE` -> prints the failing check names from `gh pr checks` and
     exits 2. Wait for CI to fix itself or fix it, then re-run.

## When to use this instead of polling by hand

Any time you'd otherwise write a `while true; gh pr view ...; sleep 30; done`
loop to babysit a PR to merge. This is that loop, with the guardrails this
repo has learned to want: never auto-merge a stacked PR, never merge past a
real conflict or a red CI run.
