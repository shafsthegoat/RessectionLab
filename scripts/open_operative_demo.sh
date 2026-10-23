#!/bin/sh
# Open the saved patient rescue comparison; no inference or training.
set -eu
demo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
demo_case="$demo_root/testcases/outputs/product-first-20261023/rescue-v2/rescue-strategies.ressectionlab"
if [ ! -f "$demo_case" ]; then
  echo 'Prepare the rescue comparison first; see docs/operative-strategies.md.' >&2
  exit 1
fi
demo_native="$demo_root/testcases/outputs/rl-readiness-20261022"
if [ -f "$demo_native/idle-integer-native-production-build/build-receipt.json" ]; then
  export RESSECTIONLAB_IDLE_THERMAL_BUILD="$demo_native/idle-integer-native-production-build/build-receipt.json"
fi
if [ -f "$demo_native/prefix-json-canonical-native-production-build/build-receipt.json" ]; then
  export RESSECTIONLAB_PREFIX_JSON_BUILD="$demo_native/prefix-json-canonical-native-production-build/build-receipt.json"
fi
export RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592
export RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES=8589934592
exec "$demo_root/scripts/launch_poc.sh" --case "$demo_case" "$@"
