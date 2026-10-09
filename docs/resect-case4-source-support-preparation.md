# Case4 source-only observation-support preparation

**Status: HOLD.** This is a non-executable, synthetic-tested numerical helper
and prospective input/rule manifest. It has not opened or assessed Case4 image,
mask, surface or landmark arrays. It does not release patient support execution.
The current Case4 receipts do not accept local T1–FLAIR or MRI–before-US anatomy
and do not accept the unchanged estimated T1 envelope as a local retained-tissue
domain. A qualified independent reviewer must decide both before any source
coordinate is passed to a support checker. The fixed 15-pair registration fit,
header overlap, before-US grid containment and six-B spatial rank do not fill
that gap.

The pure helper in `scripts/mechanics_case4_source_support.py` treats a binary
native mask as the union of affine-transformed full voxel cells. Its signed
distance is the Euclidean distance to exposed voxel faces in physical RAS mm,
positive inside, negative outside. It uses the complete sform linear map rather
than replacing the small observed obliquity with an orthogonal grid. A zero or
near-zero signed distance is ambiguous and blocks support admission. Native T1
full-cell bounds are checked separately from membership in the binary mask.

The fixed observation kernel is `max(0, 1 − r / 5 mm)`. The helper integrates it
over only the unchanged mask, using native voxel-cell midpoint rules at four
and eight nodes per axis plus a separate eight-node Gauss–Legendre rule. The
voxel Jacobian supplies physical mm³. It reports geometric ball-intersection
volume (mm³), weighted mass (mm³), weighted mass divided by the analytic full
**tent-weighted** mass `π(5 mm)³/3`, weighted first moment about the source (mm⁴), and
centroid offset (mm). The predeclared mass discrepancy is below 1% and
centroid discrepancy below 0.1 mm for both comparisons with the finer midpoint
result. These are engineering tolerances, not clinical thresholds or rigorous
integration error bounds. Any failure is HOLD; no point is moved and no mask is
expanded. Connected support is estimated from positive quadrature mass in
six-neighbour native voxels. A future independent check must review support
classification against the retained surface, especially near narrow features.
The numerical observation operator already requires weighted support of at
least `1e-4` of the full tent mass; positive but smaller mass is unsupported.
This threshold protects the operator from near-zero denominators and says
nothing about a clinically adequate amount of tissue. An exact exposed-face
count above two million blocks construction before face arrays or a spatial
index are allocated, so a highly fragmented mask cannot silently overrun the
prospective 2 GiB sampled memory cap. Before collecting candidate voxels for
one source point, the helper also rejects a kernel bounding box larger than
100,000 native cells or more than four million planned quadrature evaluations
at the selected order. These caps prevent dense fine grids from exhausting
memory or silently creating an unbounded integration job. A future patient run
must still profile actual peak memory and wall time; these counts are
preparatory bounds, not proof that the process caps will pass.

The partition is frozen at B rows 1, 14, 8, 17, 19, 7 and V rows 2, 3, 4, 5,
6, 9, 10, 11, 12, 13, 15, 16, 18. Every B centre must be strictly inside with
converged mass above the frozen numerical support floor, one containing connected support component, mask and
surface agreement, and rank six for the averaged rigid modes. The V coverage
table always has denominator 13. If any V source is unsupported, the original
all-13 RMS endpoint is not estimable; a supported subset could only be
descriptive. The helper accepts neither V destinations nor observed motion.
Case4's V outcomes were exposed in earlier development comparisons, so a later
complete result would still be retrospective engineering evidence.

The manifest `manifests/experiments/resect-case4-source-support-preparation-v1.json`
pins the original T1, FLAIR, before-US, unchanged mask/surface, source-only
partition and baseline-fit receipt by path and SHA256. It contains no patient
coordinates or destination values. A future isolated release must additionally
bind exact committed source, verify those hashes and the accepted transform,
restrict source parsing to the first triplet, and enforce the proposed 120 s,
2 GiB sampled RSS, one-thread, no-retry and output caps. Row-level coordinates,
image overlays and numerical details stay in ignored restricted local storage;
portable status stays aggregate and marked `review_pending` until independent
checks finish. Neither this preparation nor geometric support validates contact
forces, cutting, retraction, injury, clinical outcomes or deployment planning.
