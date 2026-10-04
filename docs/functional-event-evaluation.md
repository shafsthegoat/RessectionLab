# Whole-tool functional sensitivity events

`functional_events.py` connects the existing frozen candidate evaluator to raw
`FunctionalEvidence` fields. It evaluates a supplied-map encounter, not loss of
patient function. Population priors remain unreviewed and uncalibrated.

For each route or independently certified native history, exact segment/box
distance intersects the complete active and shaft swept capsules with the
declared tissue cells. The union is deduplicated across the entire sequence;
external tool in air contributes no anatomical exposure. A touched cell's whole
volume is a conservative exposure surrogate, not an exact intersection volume,
and never counts as removed tissue. Continuous axial microsteps collapse to one
equivalent macro sweep. Reorientation and nonaxial motion fail explicitly.

Each world applies one shared inverse rigid transform to motor and language
maps. Linear sampling at the touched native cell centres defines a
piecewise-constant source-grid model. Every interpolation contributor with
nonzero weight must have coverage. The event occurs if any covered touched cell
is at or above the declared map threshold. A known hit remains true despite other
unknown cells; absence becomes unknown when any touched cell lacks coverage.
Zero supplied-map support does not mean absent patient function.

Reports preserve counts, total denominator, unknown worlds, Wilson intervals
conditional on the fixed generator, and mean/upper-tail CVaR of the map-weighted
contact surrogate. Incomplete coverage suppresses the total surrogate. Unknown
event outcomes suppress frequency; known positive hits can still establish
events despite partial coverage. Known-support lower bounds and unassessed
volume remain separate.
No sum of voxel values is presented as a probability. Independent geometry
failures remain rejected candidates without reassuring functional numbers.

Before freezing candidates, bind `functional_evidence_hash`,
`functional_event_config`, `tissue_support_hash`, and a `functional_footprint_hashes`
mapping in the decision model's geometry. Each footprint binds its full cell
content, source frame, sweep geometry, candidate and case hash. The existing
evaluator enforces unseen evaluation worlds and rejects changes after freeze.
Numeric transform uniqueness treats positive and negative zero as identical;
zero perturbations therefore cannot become robustness evidence through seeds.

The first owner check found and corrected that signed-zero counting bug. A
Gaussian analytical control correctly lost coverage on a large displacement;
the full-coverage interval control now uses bounded uniform translations.
Seventeen owner controls and thirty-eight existing evaluation checks passed.
These are software controls, not training data or a real-case experiment.
