#!/bin/bash
# Runs "Build Token Tables.ipynb" end to end from the command line, without
# needing JupyterLab open — e.g. after a DICES database correction, to
# regenerate the token tables with the fix picked up.
#
# Executes into a scratch copy rather than in place, so the run's side
# effects (regenerated token CSVs, etc.) happen but the committed notebook's
# cell outputs aren't touched, and re-running doesn't create git diffs.
set -euo pipefail
cd "$(dirname "$0")"

./venv/bin/jupyter nbconvert --to notebook --execute --output-dir /tmp --output "build-token-tables-run.ipynb" "Build Token Tables.ipynb"
