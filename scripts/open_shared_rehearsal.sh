#!/bin/sh
# Portable entry point for the published October 20 rehearsal. No inference.
set -eu
rehearsal_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rehearsal_bundle_root=${RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT:-$rehearsal_root}
rehearsal_display=--secondary-display
while [ "$#" -gt 0 ]; do
  case "$1" in
    --allow-primary-display) rehearsal_display=; shift ;;
    --bundle-root)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then echo '--bundle-root requires a directory.' >&2; exit 2; fi
      rehearsal_bundle_root=$2; shift 2 ;;
    --help|-h)
      echo 'Usage: scripts/open_shared_rehearsal.sh [--bundle-root /absolute/restored-root] [--allow-primary-display]'
      echo 'Bundle root defaults to RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT or the checkout.'
      echo 'Default: require a secondary monitor. Use the display option on your own single-monitor Mac.'
      exit 0 ;;
    *) echo 'Unknown argument; use --help.' >&2; exit 2 ;;
  esac
done
if [ ! -d "$rehearsal_bundle_root" ]; then
  echo "Rehearsal bundle root does not exist: $rehearsal_bundle_root" >&2; exit 1
fi
rehearsal_bundle_root=$(CDPATH= cd -- "$rehearsal_bundle_root" && pwd)
rehearsal_case="$rehearsal_bundle_root/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
if [ ! -f "$rehearsal_case" ]; then
  echo 'The optional historical rehearsal is missing. Restore the archive and select --bundle-root or RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT. See scripts/rehearsal_support/README.md.' >&2
  exit 1
fi
export RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT="$rehearsal_bundle_root"
export RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592
export RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES=8589934592
if [ -n "$rehearsal_display" ]; then
  exec "$rehearsal_root/scripts/launch_poc.sh" "$rehearsal_display" --case "$rehearsal_case"
else
  exec "$rehearsal_root/scripts/launch_poc.sh" --case "$rehearsal_case"
fi
