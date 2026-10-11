# Shared Ogden shape: tested implementation, specimen fit still unmeasured

The mechanics code now accepts an explicit shared Ogden exponent in the FEBio
deck and independently reconstructed principal-stretch energy. This adds a
physically interpretable shape parameter to investigate the existing
compression/tension mismatch. The default exponent 2 preserves the previous
energy arithmetic, deck bytes and metadata exactly. Geometry, bonding,
compressibility ratio and measurement partitions remain unchanged.

**132 canonical generated controls pass in 0.56 s**, including the new 55-case
suite and adjacent physics, mesh and readout regressions. They test energy
derivatives against independent analytic stress, small-strain modulus, rigid
states, repeated stretches, nonuniform cell-volume averaging, invalid inputs,
and old-model compatibility. Exact old sources are tracked with the tests; no
ignored historical directory is required. These are numerical implementation
checks, not a nonlinear specimen solve or physical validation.

The initial isolated 55-test run passed its assertions but failed an overly
strict runner restriction on ordinary macOS metadata. The exact metadata
allowance was repaired; the second run passed and terminated cleanly. Both
receipts are retained. No mathematical change was made for that repair.

The [measurement-reference trace](origin-reference-v1/SOURCE-TRACE.txt) uses only
the same exposed specimen records and creator metadata. These are processed
loading/unloading averages, not a chronological initial ramp. Four small
negative tension forces are 1.6–5.1% of the tensile peak, compared with the old
model's weighted RMS error of 128% of that peak. Neither branch contains exactly
zero displacement; this is consistent with its even resampling grid and does
not prove an absent instrument reference. Tare, filtering and noise uncertainty
remain unquantified. All points and signs are retained; no offset is fitted.

The implementation supports the next bounded shared-exponent/modulus
DEVELOPMENT comparison. It does not yet demonstrate lower measured error.
Any such fit must solve the bonded specimen, verify the changed model's energy,
balance and refinement behavior, and retain the original baseline and all
residuals. Homogeneous analytic controls cannot substitute for those solves.
No patient material parameters, cutting response, neurological injury threshold,
held-donor gain, RL update or clinical conclusion follows.

[Source proposal](SOURCE-READY.txt) records the formula and prospective limits;
its historical “tests unrun” status predates the receipts here. No native fit or
solver execution was performed. [Evidence index](evidence-index.json) separates
the exact staged source from the portable canonical test change. All work reused
existing dependencies and observations.
