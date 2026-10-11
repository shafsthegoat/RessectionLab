# Initial corridor burden is discriminative, but remains an unvalidated proxy

On the existing SELECT013 initial state, the new ordering finds one suction goal
with **83 estimated preparation cells**, versus **93** for the actual retained
central goal. The central goal ranks ninth among 26; the next best score is 86.
The 10.75% estimated reduction is not measured tissue removal, fewer required
actions, or a completed route. No action was executed.

The helper uses current public nominal occupancy, existing tool geometry and
existing goal rays. It subtracts optimistic whole-cell active-tip containment
from the full shaft supercover. It preserves pre-entry geometry, reports crop
and unknown-domain limitations, and changes no state or catalog. Earlier
trajectory outcomes are excluded from the ranking. All goals retain unassessed
proximal geometry outside the image and nine declared external-workspace cells.

The estimate takes **0.101 s**, including provider enumeration, with 52 capsule
cover and 26 containment calls. Including construction, the diagnostic costs
117 native previews, **4.384 s** parent time, **546.8 MB** sampled memory and
149,121 output bytes. There are zero ranking previews, transitions or learned
operations. Nine generated controls pass; an independent saved audit passes
597 scalar/hash/identity/accounting checks. It does not independently recompute
the geometry. [Detailed review](postrun/REPORT.txt).

Retain the helper as an experimental planning input. Before running another
complete route, test the actual generic tools under fixed physical geometry at
different voxel spacings, grid shifts and rotations. The older resolution
control used different tools and already exposed grid sensitivity. This
grid-dependent ranking must not be presented as patient benefit or mechanical
validation. Patient splits and held evaluation remain unchanged.

Runtime HEAD: `e4c4105c36974538297f2109292c5d4993bf344d`. The initial metadata
freeze raced a denominator correction and was preserved before any acquired
execution; one acquired attempt then completed cleanly. Generated-control
failure and repair records are also retained. [Evidence index](evidence-index.json).
No data or dependencies were acquired. Storage was last measured at 247.461 MiB
growth before subsequent compact archives, source and Git writes.
