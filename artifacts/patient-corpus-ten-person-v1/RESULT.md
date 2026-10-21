# Ten-person source catalog and native case discovery

This separate `existing-nine-plus-ixi265-v1` release adds the already materialized
IXI265 acquired T1/MRA candidate to the nine existing qualified source identities.
It acquires zero new people, counts zero generated variants as people, grants no
training admission and accepts no anatomy as truth. The original nine-person
catalog and all 40 retained historical files remain unchanged.

The fixed [release declaration](../../manifests/patient-corpus-releases/existing-nine-plus-ixi265-v1.json)
binds all ten candidates, roles, source files and exact bundle hashes. Full and
resume runs qualify/reuse ten receipts. The catalog file SHA256 is
`2a7ba1c26f101261e3a5fecf84ae998b637a70c5a7f66d43d224153c738ee673`;
its semantic seal is
`sha256:554c172ee7061c1ec803a4ea56c8025b5e6969f777bfecfe44987ec8aa2479a3`.
See the [qualification record](../patient-corpus-ixi265-v1/qualification-audit.json)
and [reproduction guide](../../docs/patient-corpus-ixi265-release.json).

Cases lists the ten source candidates alongside native studies and saved versions.
Search matches each row's name, dataset and present anatomy. Opening a catalog
entry rechecks the fixed metadata, permitted source location and streamed bundle
checksum; recent/prepared aliases inherit that check. Renderer IDs confer no
arbitrary path authority. Listing requires only bounded metadata, with no image
decoding or model execution.

Actual Electron interaction found all six BTC entries, opened BTC05's acquired T1
with its 11.4 mL threshold-derived annotation, and verified that `vessels` returns
only the IXI and Lausanne source candidates. IXI opened through its catalog ID;
the vesselFM estimate was inspected on native MRA, separately labeled from source
annotations and with fusion still disabled. No source bundle was saved over.
Screenshots and source bindings are recorded in [GUI evidence](ui-demonstration.json).
The two source opens are interaction checks, not an anatomical accuracy review.

The [independent audit](independent-qa.json) passes candidate/receipt/code seals,
protected and generated-lineage refusal, scoped search, metadata-only listing,
opaque IDs, renderer fields and tiny generated hash/path/symlink controls. It
opens zero real payloads. Root verification also passes 169 focused Python tests,
129 app-host tests, 497 renderer/viewer tests and TypeScript/production build.
Historical full-repository failures remain described in the preceding result;
this is not a claim that the complete legacy test suite passes.

Presence counts are seven brain+tumor, one brain+skull and one brain+vessels across
saved grids. All accepted-fusion combinations remain zero, as does complete
brain/tumor/skull/vessel coverage. All anatomical locations remain UNKNOWN.
Four people have 56 verified embedded-original records in total; the six legacy
BTC bundles still reference original files externally. Their SynthStrip brain
proposals remain unselected evidence, not automatically accepted working anatomy.
Cross-dataset participant independence, tissue-conditioned growth, physiological
validity, broad accuracy and publication readiness remain open. The separate
uniform IXI lesion is excluded from this patient catalog.
