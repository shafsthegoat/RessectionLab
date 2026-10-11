# Experimental cumulative cutting coverage

The native simulator now offers an explicit opt-in cumulative cutting model.
The default fingerprint, histories, state hashes and metrics retain the original
behavior. Fixed affine subcells can accumulate certificates from executed active
aspiration sweeps; each subcell must lie wholly inside one capsule. Merely touching
a cell, covering its eight outer corners with different capsules, or predicting a
future cut is insufficient. Whole cells remain occupied and receive zero removal
credit until all subcells are certified and connected to the existing cavity.

This assumes that certified active subcells are cut and evacuated. It is an
idealized geometric model, not measured tissue mechanics. Partial cells must be
exposed before the interval. Shaft clearance uses the prior cavity. Every newly
completed cell is removed immediately; probes cannot bank cutting coverage, and
failed or capped previews discard their temporary coverage. Coverage-only strokes
are explicitly identified, with zero removed volume and ordinary movement costs.

The independent checker has separate containment arithmetic and reconstructs
coverage from executed witnesses. It requires an explicit matching specification;
the default checker refuses the new footprint. Review caught and repaired a
potential delayed-removal loophole before the experiment. Resident cells and
witness work are bounded. The checker's leaf-test accounting verifies a lower
bound and the declared cap; unsuccessful producer tests are not independently
reconstructed from successful witnesses.

Validation: 131 generated/canonical controls pass, including immutable clones,
transaction rollback, temporal ordering, oblique frames, proof tampering and exact
disabled-mode parity against the pinned prior source. All ten experiment-runner
controls also pass. Two initial comparison failures were only live tuple versus
saved JSON list representations; all numeric/state/history values were identical.
The normalized wire comparison now passes. A broader first test invocation also
encountered four historical fixtures calling a deliberately closed implicit
learning pathway; those errors are preserved, and that pathway was not reopened.

The matched 18-row experiment is the next step. Source and unit-control success
does not demonstrate better routes, faster planning, patient transfer, physical
cutting fidelity or RL improvement. Initial outer-voxelization errors and partial
target-volume under-credit may remain; the original baseline is retained.
