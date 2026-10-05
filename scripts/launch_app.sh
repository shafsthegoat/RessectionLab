#!/bin/sh
# Launch an existing package, or run the installed development app.
set -eu
ressection_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ressection_binary="$ressection_root/dist/RessectionLab.app/Contents/MacOS/RessectionLab"
if [ "${1:-}" = "--source" ]; then
    shift
elif [ -x "$ressection_binary" ]; then
    exec "$ressection_binary" "$@"
fi
ressection_python="$ressection_root/.venv/bin/python"
if [ ! -x "$ressection_python" ]; then
    printf '%s\n' 'No built app or development environment found. Follow docs/packaging.md.' >&2
    exit 1
fi
export PYTHONPATH="$ressection_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ressection_python" -c 'from resectionlab.app import main; raise SystemExit(main() or 0)' "$@"
