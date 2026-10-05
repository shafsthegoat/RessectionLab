# Independent saved audit of the curvature attempt

The saved records are internally consistent. The attempt remains **failed/incomplete**, with no returned volume mesh and no geometry-fidelity result. This audit grants no retry or solver authorization.

I independently rehashed the 12 compact records, all five raw files and their lossless copies, all 27 released input bindings, and both earlier attempts against their frozen output indices. The ten archived source files exactly match commit `b8de4416e777b11d31f1b75adbf164522a92eaa3`. The release, preflight, worker and terminal records agree. Anatomy containers were hashed as opaque bytes; no images, landmarks or arrays were decoded.

The supervisor records **180.028101 s**, exit **−9**, and the exact reason `supervision_exception`. Its final `ps` sampling call raised `TimeoutExpired` with **0.011743584 s** allotted from the remaining budget. The observed 28.1 ms beyond 180 s includes supervision and cleanup overhead. The peak sampled group RSS was **590,708,736 bytes**; unobserved peaks between samples remain possible. The saved cleanup error is null and the postflight records no surviving group members.

The native log contains one 1D meshing phase, completed in **170.079 s**, followed by entry into 2D surface meshing. It records no completed 2D phase or 3D phase. One generation was charged before entry; the worker's `null` generation count and both `running` checkpoints are preserved interrupted state. Terminal acceptance and supervision correctly report failure.

The fixed 2 mm surface and 3% volume gates are unchanged but were not evaluated for a returned mesh. Counts, Jacobians, topology, overlap, volume error and surface error are unassessed. There is no complete or partial diagnostic packet. Five raw files total **22,993 bytes**, including the terminal parent receipt, within the declared 32 MiB / 8 MiB per file / 32-file budget.

This is a runtime-feasibility failure near the fixed time allowance, not a geometric-fidelity rejection. The stage observations identify where the recorded time went; they do not establish isolated causality or a controlled performance comparison. No Gmsh, VTK distance query, mesher, solver, training, or prior attempt was replayed. No reporting blocker remains.
