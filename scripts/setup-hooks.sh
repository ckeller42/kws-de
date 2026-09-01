#!/usr/bin/env bash
# One-time per clone: point git at the committed hooks in .githooks/.
# (Committed hooks can't self-activate -- git only runs .git/hooks or the
# core.hooksPath you set here.)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true
echo "core.hooksPath -> .githooks  (pre-commit: ruff+markdownlint+gitleaks; pre-push: pytest)"
