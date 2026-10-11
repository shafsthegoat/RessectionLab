# Shared local storage limit

The October 11 supplement caps additional resident growth at 10 GiB and retains
at least 50 GiB free. One owner coordinates allocations; one new dataset pilot
may be materialized at a time. These experiments reuse existing read-only data,
with no new dataset, dependency or model download. No cleanup or deletion occurs.

The baseline project allocation is 387.513 GiB. The recorded milestone ending
09:45:42 UTC measures 15.773 MiB growth; the later free-space check at 09:46:17 UTC
reports about 332.934 GiB free, or 332.708 GiB after outstanding reservations.
Reservations are 96 MiB for the next tool-switch comparison, 128 MiB for complete
paged route planning and 8 MiB for compact root artifacts. These are bounds, not
targets or scientific execution permission. The original two completed experiment
reservations are closed against their exact terminal receipts.

The guard is advisory and checks declared project growth and free space at
allocation boundaries. It counts data, outputs, code/artifact copies, dependencies,
project caches and Git. Per-scope highwater accounting does not reclaim budget
from observed decreases. OS/home changes are not attributed; volume free-space
checks still apply. APFS allocated blocks are an estimate, not unique physical
extents. Transient or concurrent growth between scans may be missed. Worker
output caps remain required, and scientific compute is serialized separately.

The live ledger remains under `build/storage-extension-v1`. Original commands
are preserved; subsequent closure uses `storage_guard_v2.py`, which accepts the
two established terminal-receipt cleanup field names while requiring confirmed
termination and no surviving process. The initial naming refusal is retained.
Eleven original controls and six schema controls pass. Do not reset the baseline
or replace the live ledger from this archive; do not run full project scans during
timed scientific work. These are dated snapshots, not continuously current usage.

At the subsequent 09:52:26 UTC milestone, growth is 24.414 MiB and free space
is 332.917 GiB. The completed switch-guard reservation is closed; 128 MiB for
paged-route preparation and 16 MiB for archives remain reserved. See the exact
`milestone-closure-02.json` snapshot; no new dataset was acquired.

The 10:11:48 UTC snapshot closes the paged-route and fixed-four runs. Recorded
growth is 60.340 MiB, free space 332.854 GiB, and free after reservations
332.713 GiB. A single h48 continuation is reserved at 128 MiB for preparation;
root archive allowance stays 16 MiB. Later archive copies are not included in
that dated scan. See `milestone-handoff-03.json` and `milestone-handoff-04.json`.
