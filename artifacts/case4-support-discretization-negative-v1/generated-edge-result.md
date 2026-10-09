# Generated boundary control, first attempt

The first generated oblique/nonidentity-affine control failed one overly exact
test expectation: it asserted 25 SciPy `mode="constant"` versus ANTs mismatches,
but observed **45** on this runtime. The stronger voxelwise check using
`mode="grid-constant"` passed before that assertion. The difference reflects
additional generated face/corner boundary samples under the oblique physical
frame; the exact old mismatch count is not the tested contract. This was a
generated 5³ volume only, not a Case4 replay. The revised assertion requires a
nonzero old mismatch and exact grid-constant/ANTs voxelwise agreement. The
initial failure is retained here rather than silently rewritten.

After that one assertion was narrowed, the same pinned isolated runtime ran
both generated tests successfully: **2 passed in 0.018 seconds**. The controls
include an oblique NIfTI frame, nonidentity ITK affine translation, and
half-voxel boundary samples. `grid-constant` matched ANTs voxelwise in these
generated cases; this is not yet proof that it explains the saved Case4 hold.
