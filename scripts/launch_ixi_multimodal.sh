#!/bin/sh
# Reopen cached IXI anatomy or the healthy research simulator; no acquisition.
set -eu
IXI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "${1:-anatomy}" in
  anatomy) IXI_FLAG=--anatomy; IXI_PATH="$IXI_ROOT/testcases/environments/IXI265-multimodal/native/patient-anatomy.json" ;;
  simulator) IXI_FLAG=--case; IXI_PATH="$IXI_ROOT/testcases/environments/IXI265-multimodal/simulator/patient.ressectionlab" ;;
  *) echo 'Usage: launch_ixi_multimodal.sh [anatomy|simulator]' >&2; exit 2 ;;
esac
if [ ! -f "$IXI_PATH" ]; then
  echo "Prepared local IXI study is unavailable: $IXI_PATH" >&2
  exit 1
fi
exec "$IXI_ROOT/scripts/launch_poc.sh" "$IXI_FLAG" "$IXI_PATH"
