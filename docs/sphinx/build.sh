#!/usr/bin/env bash
# Build the kws-de firmware requirements-traceability + API docs.
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/) with the `docs` extra synced:
#       uv sync --extra docs
#   - doxygen, to include the C API reference page (api.rst); optional —
#     without it the build still succeeds, just without api.rst:
#       macOS:  brew install doxygen
#       Debian: apt-get install -y doxygen
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root

if command -v doxygen >/dev/null 2>&1; then
    (cd firmware && doxygen Doxyfile)
else
    echo "build.sh: doxygen not found — building without the API page" >&2
fi

uv run sphinx-build -b html docs/sphinx docs/sphinx/_build/html
echo "build.sh: docs/sphinx/_build/html/index.html"
