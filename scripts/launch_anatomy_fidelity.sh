#!/bin/sh
# Existing local acquired studies; launch never runs inference or training.
set -eu
DEMO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "${1:-mri}" in
  mri) DEMO_FLAG=--anatomy; DEMO_PATH="$DEMO_ROOT/outputs/anatomy-integration-v3/sub06-refined/patient-anatomy.json" ;;
  cta) DEMO_FLAG=--anatomy; DEMO_PATH="$DEMO_ROOT/outputs/poc-angio-fidelity-v2/native-cta/patient-anatomy.json" ;;
  rehearsal) DEMO_FLAG=--case; DEMO_PATH="$DEMO_ROOT/outputs/anatomy-integration-v3/desktop-integration/rehearsal-reopened.ressectionlab" ;;
  *) echo 'Usage: launch_anatomy_fidelity.sh [mri|cta|rehearsal]' >&2; exit 2 ;;
esac
if [ ! -f "$DEMO_PATH" ]; then
  echo "Local source-bound demonstration is unavailable: $DEMO_PATH" >&2
  exit 1
fi
exec "$DEMO_ROOT/scripts/launch_poc.sh" "$DEMO_FLAG" "$DEMO_PATH"
