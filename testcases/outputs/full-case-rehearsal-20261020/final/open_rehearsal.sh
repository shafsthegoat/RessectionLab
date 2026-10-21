#!/bin/sh
set -eu
rehearsal_root=/Users/shaf/coding_projects/RessectionLab
export RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592
export RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES=8589934592
exec "$rehearsal_root/scripts/launch_poc.sh" --secondary-display --case "$rehearsal_root/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
