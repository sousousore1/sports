#!/usr/bin/env bash
# Idempotent bootstrap for the Roboflow `sports` repository.
# Creates an isolated virtualenv, installs the `sports` package and the
# soccer example dependencies (ultralytics/torch, gdown).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# The default base image ships CPython 3.12 but not the `ensurepip`/`venv`
# stdlib support package, which is required to create a virtualenv.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r examples/soccer/requirements.txt

echo "sports environment ready. Activate with: source .venv/bin/activate"
