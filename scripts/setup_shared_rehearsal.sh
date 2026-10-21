#!/bin/sh
# Explicit dependency installation for a collaborator's Mac.
set -eu
rehearsal_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rehearsal_python=${REHEARSAL_PYTHON:-python3.12}
if [ "$(uname -s)" != Darwin ]; then
  echo 'This setup recipe targets macOS; other platforms have not been qualified for this handoff.' >&2
  exit 1
fi
command -v "$rehearsal_python" >/dev/null 2>&1 || { echo 'Install Python 3.12, or set REHEARSAL_PYTHON to its executable.' >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo 'Install Node.js 22.17 or newer.' >&2; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo 'Install pnpm 11, then rerun this script.' >&2; exit 1; }
"$rehearsal_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python3.12 for this recipe"'
node -e 'const [major,minor]=process.versions.node.split(".").map(Number);if(major<22||(major===22&&minor<17))throw Error("Node22.17+ required")'
if [ ! -d "$rehearsal_root/.venv" ]; then "$rehearsal_python" -m venv "$rehearsal_root/.venv"; fi
"$rehearsal_root/.venv/bin/python" -m pip install -r "$rehearsal_root/requirements-lock.txt"
"$rehearsal_root/.venv/bin/python" -m pip install -e "$rehearsal_root" --no-deps
pnpm --dir "$rehearsal_root/desktop" install --frozen-lockfile
echo 'Dependencies installed. Run .venv/bin/python scripts/verify_shared_rehearsal.py, then scripts/open_shared_rehearsal.sh.'
