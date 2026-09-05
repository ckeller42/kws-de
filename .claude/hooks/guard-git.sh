#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): block the git/gh footguns that are cheap
# to make and expensive to undo -- staging everything blind, skipping hooks,
# force-pushing without a lease, and stacking a PR onto a non-main base by
# accident. Exit 2 blocks the call and explains why; anything else passes.
set -uo pipefail
input=$(cat)
# Cheap substring pre-check on the raw JSON before spawning python3 for every
# Bash call -- only these four shapes need the full parse.
case "$input" in
  *"git add"*|*"git commit"*|*"git push"*|*"gh pr create"*) ;;
  *) exit 0 ;;
esac

msg=$(printf '%s' "$input" | python3 -c '
import json, re, shlex, sys

cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
# Split on shell sequencing/piping so each git/gh invocation in a chained
# command is checked on its own -- not a real shell parse, good enough here.
parts = re.split(r"&&|\|\||;|\|", cmd)
problems = []

for part in parts:
    try:
        toks = shlex.split(part)
    except ValueError:
        toks = part.split()
    if len(toks) < 2 or toks[0] not in ("git", "gh"):
        continue
    p = part.strip()

    if toks[0] == "git" and toks[1] == "add":
        args = toks[2:]
        if "-A" in args or "--all" in args or "." in args:
            problems.append(f"{p!r}: stages everything blind -- add specific files by name")

    if toks[0] == "git" and "commit" in toks and "--no-verify" in toks:
        problems.append(f"{p!r}: --no-verify skips commit hooks -- fix the hook failure instead")

    if toks[0] == "git" and "push" in toks:
        if "--no-verify" in toks:
            problems.append(f"{p!r}: --no-verify skips pre-push hooks -- fix the hook failure instead")
        has_force = any(t == "--force" or t == "-f" for t in toks)
        has_lease = any(t == "--force-with-lease" or t.startswith("--force-with-lease=") for t in toks)
        if has_force and not has_lease:
            problems.append(f"{p!r}: --force without --force-with-lease can clobber someone else'\''s push")

    if toks[0] == "gh" and toks[1:3] == ["pr", "create"]:
        base = None
        for i, t in enumerate(toks):
            if t == "--base" and i + 1 < len(toks):
                base = toks[i + 1]
            elif t.startswith("--base="):
                base = t.split("=", 1)[1]
        if base != "main":
            problems.append(
                f"{p!r}: pass --base main explicitly (this repo default is main; "
                f"checking gh repo view --json defaultBranchRef here is too slow to be worth it -- "
                f"if this really is a stacked PR onto a feature branch, merge that branch to main first)"
            )

print("\n".join(problems))
' 2>/dev/null)

[ -z "$msg" ] && exit 0
echo "$msg" >&2
exit 2
