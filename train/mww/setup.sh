#!/usr/bin/env bash
# Provision the local Python 3.10 environment for microWakeWord training.
# microWakeWord is 3.10-only, so it lives in its own venv, separate from the 3.11 kws-de package.
# Verified working on Apple Silicon (CPU/Metal); GPU not required.
set -euo pipefail
cd "$(dirname "$0")"

# uv fetches Python 3.10 on demand. --seed gives the venv pip (microWakeWord installs via pip/git).
uv venv --python 3.10 --seed .venv

# microWakeWord is not on PyPI — install from git.
./.venv/bin/pip install "git+https://github.com/kahrendt/microWakeWord"

# Sanity check.
./.venv/bin/python -c "import microwakeword; print('microwakeword OK:', microwakeword.__file__)"

echo
echo "Ready. Next: run the microWakeWord trainer for 'Hey Bus' (Piper positives + ambient/negatives),"
echo "then copy the exported model to ../../models/hey_bus.tflite (gitignored)."
