# Independent synthetic spatial-feature review

Seven independent checks passed in 0.61 seconds with numerical libraries
limited to one thread. They cover hand-calculated physical volumes and centroids
on a sheared, left-handed source grid; a general three-dimensional proper
rotation; normal and source-column scaling; empty versus nonempty regions whose
centroid is the access origin; a conditioning counterexample; all 37 archived
source-file hashes; and independent reward arithmetic from actual source-cell
sets. The final test invocation executed eight tiny synthetic native
transitions, with no policy, gradient, optimizer, or patient run.

The original fixture result is reproduced. Each first cut removes five target
and two normal cells and newly contacts 25 normal cells. Its reward is
`5 - 0.2*2 - 0.05*0.2*25 - 0.02 = 4.33`. Each best second cut removes 40 target
and 21 normal cells, but newly contacts 66 versus 41 normal cells. Including the
action and tool-change costs gives second rewards of 35.10 and 35.35 and totals
of 39.43 and 39.68. The 0.25 difference therefore comes from the declared cost of
new partial normal-tissue contact. These are conditional optima for the two
specified first cuts and the finite final action inventories.

`conditioning-counterexample.json` retains an additional limitation. With
identity source affine and inward normal `(1, 1e-8, 0)`, the first candidate
tangent is at the declared threshold. A numerically proper rotation
(orthogonality error 2.22e-16) changes the chosen source tangent axis from 1 to 0
through rounding, changing a point's descriptor by approximately 2 mm.
Separately, changing a near-parallel normal by only 4e-8 reverses tangent
coordinates by approximately 2 mm. The latter comparison changes the normal;
it is a sensitivity counterexample, not a same-geometry invariance test.

This does not invalidate the recorded invariance checks for the regular saved
fixture. It prevents extending them to an unconditional numerical invariance or
conditioning guarantee. Future cross-source work needs an explicitly frozen
tangent convention and conditioning rule. The proposed eight summaries also
omit partial-contact history, current tool and remaining action budget unless
those are retained separately, in addition to the already documented loss of
geometry and topology. The original abstract summary collision and source-axis
reindexing limitations remain valid. No change to the archived numerical
prototype, production profile, adapter, pilot, or cache follows from this review.

Reproduce the conditioning evidence from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python artifacts/native-spatial-feature-independent-v1/reproduce_conditioning.py
```

Independent tests are in `tests/test_native_spatial_features_adversarial.py`.
The exact source hashes and observed test invocation are in `review.json`.
