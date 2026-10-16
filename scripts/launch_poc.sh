#!/bin/sh
# Launch the actual local Electron app. Reopening reads saved masks; no model runs.
set -eu
POC_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
POC_BUNDLED="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies"
if [ -x "$POC_BUNDLED/node/bin/node" ]; then
  PATH="$POC_BUNDLED/node/bin:$POC_BUNDLED/bin/fallback:$PATH"
  export PATH
fi
if ! command -v node >/dev/null 2>&1; then
  echo 'Node.js is required. See docs/poc-a-workbench.md for setup.' >&2
  exit 1
fi
if [ ! -x "$POC_ROOT/.venv/bin/python" ] || [ ! -x "$POC_ROOT/desktop/node_modules/.bin/electron" ]; then
  echo 'The local app dependencies are missing. See docs/poc-a-workbench.md for setup.' >&2
  exit 1
fi
cd "$POC_ROOT/desktop"
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vite/bin/vite.js build
exec node node_modules/electron/cli.js . --force-renderer-accessibility "$@"
