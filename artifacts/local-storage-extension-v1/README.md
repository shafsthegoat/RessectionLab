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

At 10:30:13 UTC, the h48 closure measures **118.352 MiB** additional project growth and **332.808 GiB** free. The root archive reservation increases to 24 MiB before copying the compact result; source preparation is separately reserved. See `milestone-closure-05.json`. This dated measurement precedes the new archive and further source preparation; no dataset download or deletion occurred.

At 10:42:00 UTC, public-bound-v1 closure measures **143.953 MiB** additional growth and **332.791 GiB** free. Archives occupy 17.656 of 24 MiB before copying this result. See `milestone-closure-06.json`. No acquisition or deletion occurred; a separately reserved local-query successor reuses the same five cases.

The 10:56:34 UTC snapshot closes the local-window comparison. Measured growth
is 168.211 MiB with 332.760 GiB free (332.698 GiB after remaining reservations).
Residual-blocker preparation is reserved at 32 MiB and compact root archives at
32 MiB. The subsequent v2 compact archive is not included in this dated scan.
See `milestone-closure-07.json`; no new data or dependencies were acquired.

The 11:02:32 UTC snapshot closes the residual-blocker diagnostic. Growth is
189.438 MiB, free space 332.713 GiB, and root archive allowance remains32 MiB.
The later 2.03 MiB compact result archive, Git changes and separate1 MiB
source-only preparation reservation are not included in this dated scan.
See `milestone-closure-08.json`; no new dataset or dependency was acquired.

At 11:26:06 UTC, complete-corridor closure measures **204.852 MiB** additional growth and **332.677 GiB** free. Its 37 MiB reservation is closed; root archives have 16.848 MiB headroom within a 48 MiB allowance before the new compact result archive. The 8.17 MB generated selection trace remains local and ignored; compact independent projections are committed. See `milestone-closure-09.json`. No download or dependency was added.

At 11:39:12 UTC, recursive-corridor closure measures **214.773 MiB** additional growth and **332.653 GiB** free (332.605 GiB after remaining reservations). The 37 MiB runtime reservation is closed. Root archive headroom is 15.688 MiB within 48 MiB before the compact 489 KB result archive; the 8.53 MB full generated trace stays local. See `milestone-closure-10.json`. No acquisition, dependency or deletion occurred.

At 11:57:49 UTC, focused-corridor closure measures **226.730 MiB** additional growth and **332.649 GiB** free (332.601 GiB after reservations). Its 37 MiB reservation is closed with no overrun. Root archives have 14.168 MiB headroom within 48 MiB before the next compact 622 KB result archive. The 9.61 MB full trace remains local. See `milestone-closure-11.json`; no acquisition or dependency was added.

At 12:19:35 UTC, preparation-choice closure measures **247.461 MiB** additional growth and **332.632 GiB** free (332.583 GiB after reservations). The 37 MiB runtime reservation closes with no overrun. Root archives have 13.004 MiB headroom before the later approximately 0.88 MiB evidence archive, Git and source writes. The two full generated traces remain local. See `milestone-closure-12.json`; no acquisition or dependency was added.

At 12:45:19 UTC, the combined goal-burden/generic-grid milestone measures **252.277 MiB** additional growth and **332.584 GiB** free (332.532 GiB after reservations). The generic runtime is closed; root archive headroom is10.184 MiB before the later0.39 MB evidence archive and source/Git writes. An HBE source allocation exceeded its1 MiB block-accounted allowance by24,576 bytes; the refusal was preserved and the allowance increased to2 MiB using the same scan. A2 MiB cumulative-cut source reservation is also prepared. See `milestone-closure-13.json`. No acquisition, deletion or new dependency occurred.

At13:03:11 UTC, the union comparison closes with **254.680 MiB** measured additional growth and **332.571 GiB** free. Free after reservations is330.520 GiB, including a separately prepared2049 MiB specimen-solver output allowance. Source preparation remains separate. Archive headroom is8.941 MiB before the later compact union result, source and Git writes. See `milestone-closure-14.json`; the solver products were not yet created. A later generated mechanics fixture exceeded its source allowance; the original failed products are retained and the owner separately amends that allowance. This dated snapshot does not include those later writes.


October 11, 13:36:46 UTC — completed mechanics and cutting-cache iteration: [final-disk-snapshot-02.json](final-disk-snapshot-02.json) records 1,749,950,464 bytes (1.630 GiB) additional allocated growth, 331.204 GiB free, and 331.148 GiB free after outstanding reservations. Both limits hold, no overruns are recorded, and all native outputs/failures remain. This includes the compact result archives and plot/font cache. Snapshot/status/Git writes after the timestamp are not included; no recursive follow-up scan is needed. No new dataset, weights or dependencies were acquired.

October 11, 14:01:03 UTC — [partial-evacuation closure](milestone-closure-16.json)
records **1,755,901,952 bytes (1.635 GiB)** cumulative additional allocated growth
and **331.228 GiB free** (331.173 GiB after remaining reservations). The iteration
adds **5.676 MiB**, including canonical files, caches, compact archive and Git
growth, within its closed 24 MiB allowance. The build stage retains 3.680 MiB;
root archive headroom is 4.461 MiB within 48 MiB. All original outputs remain.
No acquisition, dependency or deletion occurred. Both global storage limits pass.
Snapshot copies, this note and subsequent Git writes are outside the dated scan;
no recursive scan is required. Allocation, source-begin and terminal bindings
are retained beside the snapshot.

October 11, 14:42:15 UTC — [exposure closure](milestone-closure-17.json) records
**1,768,136,704 bytes (1.647 GiB)** cumulative additional allocated growth and
**331.193 GiB free**. The exposure build stage retains **7.016/16 MiB**, and its
compact archive occupies **2.652/3 MiB**. No acquisition, new dependency or
deletion occurred. The original failed supervisor remains failed; storage
closure certifies no scientific or clinical outcome.

The scan also records aggregate root source/archive/Git growth exceeding its
internal 48 MiB allowance by **296 KiB**. This original record is preserved.
The [additive amendment](root-bookkeeping-amendment-01.json) allocates 52 MiB
within the unchanged user 10 GiB total, preserving the original origin and
growth high-water mark. Root took over the single storage-owner role and began
the already reserved 4 MiB opening-task source controls; full comparison release
remained separate. Fresh free space after reservations is above 331 GiB.
Copies, this note and later Git/source writes are outside the dated scan.

October 11, 15:02:13 UTC — [opening comparison closure](milestone-closure-18.json)
records **1,769,844,736 bytes (1.648 GiB)** cumulative additional allocated growth
and **331.133 GiB free**. Both paired workers and saved audits are terminal; the
source/results stage retains **0.539/4 MiB**. Global and reservation limits pass.
The existing 52 MiB root source/archive/Git allowance has 2.621 MiB measured
headroom. A separate 8 MiB continuation-value stage reserves source preparation
only (384 KiB shared initial ceiling); no learning execution follows from that
reservation. No dataset or dependency acquisition occurred. Two unused pytest
temporary convenience links were removed after tests; their targets/results remain.
Later copies, source edits and Git writes are outside this dated scan.
