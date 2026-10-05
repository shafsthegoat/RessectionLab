# Synthetic spatial descriptor probe

This isolated research proposal targets the specific actor alias in
[the axis learner review](native-axis-learner-review.md). It adds no production
input profile, policy, adapter change or patient experiment. No training is
authorized by this probe.

The actor descriptor contains six physical coordinates: entry and tip, each
expressed relative to the declared access center. The inward access normal is
the third axis. Project the first usable declared source-affine axis onto the
access plane for the first tangent; their cross product supplies the second.
All coordinates are millimetres. STOP has six zeros, retaining the existing
STOP indicator outside this proposal. Candidate rows keep their supplied order.

The critic proposal contains eight values: actual cavity volume and centroid,
then residual source-target volume and centroid. Volumes use source-cell count
times the affine determinant, and centroids use source-cell centers in the same
access frame. Empty regions have zero volume and zero centroid. These are
observable cavity/source-label descriptors; no latent anatomy is supplied.
Physical units are explicit, and no scaling choice is inferred from this test.

The executable probe reconstructs the established 7×7×8 synthetic tissue fixture
and its two certified fine-tool cuts. It compares their existing feature rows,
the proposed coordinates, and their exact two-cut returns by trying every final
legal action including STOP. It also compares the existing and proposed critic
summaries after the first cuts. No neural policy is constructed or evaluated.

Separate algebraic tests transform the saved coordinates, access and affine
together under translation, proper rigid rotation and RAS-to-LPS conversion.
They verify descriptor invariance on the regular fixture and candidate-row
permutation equivariance; they provide no unconditional conditioning guarantee.
These tests concern the descriptor transformation; they do not establish frame
invariance of every geometry operation in the planner.

Two limits are deliberate. Different cavity masks can have identical volumes
and first moments and therefore identical proposed summaries. The saved abstract
occupancy counterexample does not assert those masks are reachable by the native
tools. Also, changing source-axis ordering while holding physical points fixed
changes the declared tangent basis. The descriptor is not invariant to arbitrary
source reindexing. A future cross-source experiment would need an explicitly
shared tangent frame or a tested canonical source-grid convention.

Distinguishing one known alias establishes additional information, not that a
network learns to use it or that the full state is Markov. Geometry, connectivity,
topology and interactions among candidates remain compressed or absent. The eight
summary values also omit partial-contact history, current tool and remaining
action budget unless those are retained separately. They are not a complete
replacement for the existing state. Any
learned-policy comparison must be separately declared and approved after this
probe, with frozen geometry/reward and paired initialization; neither a new
patient run nor an extension of the current RAW pilot follows from these files.

## Measured synthetic result

The isolated probe completed in **0.363 seconds**, using nine native transitions,
zero gradients, zero evaluated policies and no patient data. All eight focused
tests passed in 0.06 seconds. Numerical libraries were limited to one thread.
The actual two-cut enumeration reproduced returns **39.43 and 39.68** after
identical first rewards of **4.33** and identical existing 15-feature action rows.

The new action rows are `(0,0,0,0,0,6.5)` and `(1,0,0,1,0,6.5)` mm. After the
cuts, the existing six state features remain identical. The proposed summaries
distinguish cavity centroids at tangent coordinate 0 and 1 mm; their residual
target centroids are 0 and −0.041667 mm. Both cavities contain 7 mm³ and both
residual targets contain 120 mm³. These measurements establish distinction for
this fixture only.

Translation, RAS-to-LPS and row permutation had zero recorded descriptor error.
Proper rotation with translation had maximum candidate error
`1.11e-15` mm and zero state error. The abstract different-mask counterexample
still gives exactly equal eight-value summaries. Reordering the declared source
axes changes candidate coordinates by as much as 1 mm in this example, as
expected for the declared basis. Neither limitation is hidden by the positive
alias result.

## Later independent finding: tangent conditioning

The [independent review](../artifacts/native-spatial-feature-independent-v1/README.md)
preserves a counterexample outside the regular fixture. With identity source
affine and inward normal `(1, 1e-8, 0)`, the first tangent lies at the `1e-8`
cutoff. Floating-point rounding under a proper coordinate rotation changes the
chosen source axis from 1 to 0, changing the descriptor by
**2.00000000034 mm**. A separate perturbation of a near-parallel normal by `4e-8`
also produces about 2 mm coordinate sensitivity; that comparison changes the
normal and is not a same-geometry invariance test. Exact inputs and results are
in [conditioning-counterexample.json](../artifacts/native-spatial-feature-independent-v1/conditioning-counterexample.json).

The original recorded errors remain valid for their regular, well-conditioned
fixture. Numerical invariance near the axis-selection cutoff is not established
and fails in this counterexample. A future descriptor requires an explicit
tangent convention and conditioning rule. The independent cell-based reward
check also identifies the 0.25 return difference as new partial-contact cost
(66 versus 41 newly contacted normal cells on the final cuts), reinforcing the
need to retain contact memory separately. This later finding changes neither
the frozen prototype nor its original execution receipts or source archive.

Exact source bytes, commands, hashes and raw values are under
[`artifacts/native-spatial-feature-probe-v1`](../artifacts/native-spatial-feature-probe-v1/README.md).
Implementation: `research/native_spatial_features.py` and
`scripts/probe_native_spatial_features.py`; checks:
`tests/test_native_spatial_features.py`. Production 15+6 profiles, pilot/cache
declarations and public results are unchanged. Before a learned comparison,
declare the cross-source tangent convention and input units, then separate
actor-coordinate and critic-summary interventions so their effects can be
attributed. This result itself authorizes no training.
