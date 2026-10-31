# RessectionLab data rebuild

This data-building cycle uses the existing primary checkout and native workspace
format. Bulk sources, derived volumes, review images, packages and migration
archives live on the external data root. No RL training or remote publication is
part of this cycle. All existing person roles and exposure records remain binding.

Set `RESSECTIONLAB_DATA_ROOT` to the mounted directory before running preparation
or starting the desktop app. The current local destination is
`/Volumes/shafrir/old coding projects/RessectionLabData`.
The volume identity marker prevents accidental use of a replacement volume.
Preparation refuses absent mounts and reserves the larger of 50 GiB and 10% of
the volume. All temporary imaging work belongs in its `staging/` directory.

The resumable work ledger is `catalog/work-ledger.json` under that root; inventory,
source qualification, case manifests, review decisions and release evidence are
linked from that ledger. Independent review is AI engineering review, not clinical
or neuroradiologist certification. Pending cases do not enter the accepted catalog.

The case contract is a versioned `metadata.data_case` extension in the existing
native format, preserving source identity, family role, component evidence,
observations, synthesis recipe, capabilities and review references. Original
observations, source annotations, estimates and synthetic components retain
separate evidence labels. All generated descendants inherit their host role.

Local cleanup follows copy, checksum verification, dependency/reference review,
production reopening, then removal of only verified redundant copies. Unresolved
dependencies stay local. Migration receipts contain restoration paths and hashes.
