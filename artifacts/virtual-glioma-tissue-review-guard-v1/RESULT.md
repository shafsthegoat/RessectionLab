# Strict growth inputs require a review of the exact tissue pair

Real-source `GM_WM_FRACTIONS` admission now requires an explicit
`fk_tissue_pair_review`. Adding tissue-class names or a blanket acceptance flag
to unreviewed intensity estimates cannot admit them. No review or acceptance
was created for IXI265, and no new acquired-image fitting or real-host growth
was run.

The review binds patient/case identity, acquired-image and frame hashes, source
references, brain envelope, exact GM/WM masks, classification support, UNKNOWN
coverage and each layer's source/method provenance. Source and run-receipt
hashes must occur in attached provenance. The named reviewer, timezone-aware
timestamp, decision and rationale have the limited scope
`research_categorical_GM_WM_support_for_FK`. The software checks this attestation's
binding; it does not establish reviewer expertise, clinical clearance, normal
host status or physical tissue fractions.

Known support and UNKNOWN must partition the source brain; GM/WM are disjoint
and inside known support. On the simulation grid, nearest-neighbor known/UNKNOWN
coverage independently constrains the domain, with mapped hashes retained.
Linear tissue interpolation cannot fill an UNKNOWN cell. The resulting weights
are geometric interpolation of reviewed categorical classes, not physical PVE.

The generated exemption requires every condition together: the synthetic flag,
exclusively simulated source references, generated health-evidence status and
both tissue layers' generated roles. Uniform diffusion and genuine generated
strict qualification/reconstruction match the preserved pre-change source
exactly on three fixed fixtures. Historical records are not rewritten; an old
unreviewed real strict receipt cannot gain renewed execution authority.

The [owner controls](controls.json), SHA256
`adac0b1c1b775238c50b6d5c2b8d89981406ea18db7fcbe07ea9f2d65e2174c7`,
record 63 passing focused/regression tests. Three warnings in the unchanged
seed-placement test remain documented; that test passes. The final source is
SHA256 `c726a61516a68809f45ace18cf51f7bf547d40816be81cb0cb47aa7e3ef2a03d`.

The [independent audit](../virtual-glioma-tissue-review-independent-v1/independent-qa.json),
SHA256 `b28fea7d5f90d46a4c2e5ae538ea9628374cffe50442bf25c1f2f3232bbd4526`,
passes 96 checks, including stale evidence, generic flags, protected-role
rejection, all 32 exemption combinations and three shifted/oblique/reflected
coverage mappings. Its generated witness independently reproduced the
[initial UNKNOWN interpolation finding](../virtual-glioma-tissue-review-independent-v1/initial-coverage-finding.json)
before correction. That record retains the intermediate source hash and exact
witness; the intermediate source snapshot was not saved. The final source and
pre-change source are preserved separately. No patient payload or solver is
opened by the independent audit.

A [separate IBSR20 prospectus](../ibsr20-component-prospectus-v1/PROTOCOL.md)
fixes a possible external tissue-component comparison and its scoring before
label access. Independent review finds no concrete protocol error. Both image
archive routes require login; additional terms, use eligibility, exact transfer
bounds and decoding/adapter preparation remain unresolved. No acquisition,
model run, cohort role or anatomy admission follows from that document.
