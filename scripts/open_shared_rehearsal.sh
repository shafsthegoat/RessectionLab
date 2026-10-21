#!/bin/sh
# Portable entry point for the published October 20 rehearsal. No inference.
set -eu
rehearsal_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rehearsal_display=--secondary-display
case "${1-}" in
  --allow-primary-display) rehearsal_display=; shift ;;
  --help|-h)
    echo 'Usage: scripts/open_shared_rehearsal.sh [--allow-primary-display]'
    echo 'Default: require a secondary monitor. Use the option on your own single-monitor Mac.'
    exit 0 ;;
esac
if [ "$#" -ne 0 ]; then echo 'Unknown argument; use --help.' >&2; exit 2; fi
rehearsal_case="$rehearsal_root/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
if [ ! -f "$rehearsal_case" ]; then
  echo 'The published rehearsal is missing. Clone the codex/poc-first-rehearsal branch with its tracked testcases files.' >&2
  exit 1
fi
export RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592
export RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES=8589934592
if [ -n "$rehearsal_display" ]; then
  exec "$rehearsal_root/scripts/launch_poc.sh" "$rehearsal_display" --case "$rehearsal_case"
else
  exec "$rehearsal_root/scripts/launch_poc.sh" --case "$rehearsal_case"
fi
