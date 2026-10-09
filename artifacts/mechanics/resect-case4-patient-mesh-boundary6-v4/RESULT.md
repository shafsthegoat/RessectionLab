# RESECT Case4 boundary-6 geometry attempt

The separately released, single Gmsh attempt **failed the unchanged 2 mm
bidirectional surface-fidelity gate**. It returned 19,070 nodes and 10,512
straight-reference tet10 elements. The sampled source-to-mesh and
mesh-to-source maximum distances were 3.869 and 2.264 mm; the corresponding
full-surface upper bounds were 4.576 and 3.254 mm. Both sampled distances
already exceed 2 mm, so this is a geometric rejection, not merely a loose
covering bound. Total enclosed volume differed by 0.315%, which does not
establish local shape agreement.

The one native generation took 17.63 seconds under the 180-second cap and
reached 591 MB sampled process-group RSS under the 3 GiB cap. It did not time
out or exceed output limits. All recorded inputs stayed unchanged and the
saved candidate-file hashes checked. There was no retry, MRI re-extraction,
landmark or during-US access, solver call, training update, or clinical
admission. The rejected candidate is retained in the ignored local output
directory for audit; [summary.json](summary.json) records selected receipt
values and hashes without committing patient geometry. An
[independent saved-output audit](independent-review.md) verified the frozen
source and context hashes, diagnostic arrays and mesh volume, and approved
publication of this negative result. It did not recompute the expensive
directed surface distances from scratch.

Boundary refinement improved the two sampled distances from the earlier
12/24 mm graded attempt (5.460 and 3.834 mm) but did not reach the gate.
It also increased the returned mesh from 5,223 nodes/2,761 tet10 to
19,070 nodes/10,512 tet10. The conservative three-array skyline storage
projection rose from 2.95 GB to 39.28 GB before other solver allocations;
this is a projection, not a measured solve. Blind further refinement is not
currently justified. Localized boundary errors and a memory-feasible solver
representation must be understood before another frozen mesh candidate.

Case4 is a **DEVELOPMENT** case: its comparison landmark outcomes were
previously revealed. The input surface is an estimated T1-derived envelope
with known inferior/cerebellar exclusions and no anatomical acceptance.
This result tests geometry against that retained surface. It does not test
tissue deformation, tool interaction, cutting force, injury, or planning
benefit. Any physical-fidelity claim still needs measured displacement or
force data appropriate to the specific interaction and an independent
held-out case.
