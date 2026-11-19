#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

if [[ -n "${PYTHON:-}" ]]; then
    interpreter="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    interpreter="python3"
else
    interpreter="python"
fi

exec "$interpreter" run_app.py "$@"
