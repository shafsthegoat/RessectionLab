# Explicit native-axis route validation

Development geometry experiment on UCSF-PDGM-0004, October 4, 2026. This is an
annotation-assisted research scenario with an estimated skull-stripped MRI
support envelope. The opening is hypothetical; no cortical or clinical safety
claim follows from the geometric checks.

The original 54 physical routes and original generic tool dimensions remain
unchanged. A separate named source-grid-axis scenario uses entry
`[-181.5, 85, 105]`, target `[-162, 85, 105]`, and an access disk of radius 6 mm.
The two additional tool profiles retain their pre-existing native-model
dimensions. Neither is a replacement definition for the generic aspirator.

| Explicit tool profile | Active tip radius | Shaft radius | Contained target cells | Contained normal cells | Independent audit |
| --- | ---: | ---: | ---: | ---: | --- |
| native-fine-aspiration | 1.25 mm | 0.45 mm | 19 mm³ | 1 mm³ | Passed |
| native-wide-aspiration | 2.25 mm | 1.1 mm | 174 mm³ | 11 mm³ | Passed |

The source grid is 1 mm isotropic. All counted tissue consists of fully contained,
connected source cells. Partial contacts remain occupied and are reported
separately. Whole-tool hard-exclusion checks and swept-shaft checks against the
prior cavity remain active. Each history was frozen before the independent
audit. No policy updates or clinical probabilities were produced.

`report.json` contains the separately hashed search, frozen source-module
checksums, and independent results. The two `*-candidate.json` files hold the
full microstep histories. `source-snapshot/` and the retained runner reproduce
the completed experiment. The bounded run took 6.58 seconds on this machine.

`regression-analysis.json` confirms that the optional explicit-target API
preserves the current default planner's complete candidate records and model
hash. All original 54 tool/entry/target tuples also match the earlier screen.
Their IDs changed from that older screen because a separate structural-evidence
fix added review status and a cortical-access prohibition to the hashed support
receipt. The old receipts remain valid records of their original model.

`exact-factory-preflight.json` is a subsequent check using the separately
retained `native_simulation-selected-ray.py`. It confirms exact entry, target,
and tool preservation in the corrected selected-route factory. The original
packaged Route02A now has its actual ray represented and still has zero legal
cutting actions because its shaft contacts remaining tissue. Each new native
profile has one legal initial action. This check took no environment steps and
performed no training.

The first runner invocation aborted on an incorrect module import before any
experiment. Its snapshot and failure note are retained in the sibling
`native-route-alternatives-v1` directory. After the completed run, the helper
also gained an explicit rejection for overlapping target compartments, matching
the native simulation's existing requirement; this case has no overlaps.
