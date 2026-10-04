RESECT access-contract review
=============================

The reviewed prospective design is ready for root commitment. This is a static contract review, not permission to acquire or reveal patient measurements and not a passing parser, solver or clinical experiment.

The original acquisition manifest remains unchanged at `4d1d5123572370e8af06730f17ea993813053dfc`. Its general calibration wording permits an empty calibration set; the specific experiment explicitly uses exactly six B observations and all remaining rows as V, with no C. The design document was an untracked draft at review; its exact SHA256 is recorded in `verification.json`.

Metadata fixity and the six-file size total were checked using saved metadata only. No patient payload, coordinate, image, model experiment or test was opened or run. Independent parser controls are proposed separately and remain pending the owner source freeze. Workflow persistence, real frame/alignment verification and domain/physics validation remain explicit gates.
