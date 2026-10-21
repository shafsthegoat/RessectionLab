# Two more acquired CT/MRI cases in Electron

The next two TRAIN people in the frozen SynthRAD center order, `1BB033` and
`1BC077`, now have actual TotalSegmentator brain/skull estimates and portable
native cases. Both cases were opened, inspected, saved separately and reopened
in a fresh Electron process. The ten-person qualification catalog and previous
audits remain frozen; these are two additional source/model candidates outside
that catalog, with no anatomical or training acceptance.

## Source and execution

The [source admission](../../manifests/poc-synthrad-bc-source-admission-v1.json)
fixes the identities before pixel inspection. A complete checksum of the existing
14,471,900,926-byte Task1 archive precedes decoding only four named acquired CT/MR
members. ZIP/gzip length, CRC, coded geometry and finite-value checks pass.
These are acquired radiotherapy-source scans under CC-BY-NC-4.0; the dataset name
does not make these CT images synthetic or establish healthy hosts.

The [preparation receipt](../poc-synthrad-bc-preparation-v1/receipt.json) binds the
exact sources, declarations, software, checkpoint, outputs and resource records.
The unchanged fast3mm TotalSegmentator model runs sequentially on MPS with two
threads. The campaign completes in29.50 seconds. Sampled family RSS peaks are
3.20/3.77GiB, below5GiB; minimum sampled system availability is2.76GiB. No retry,
registration, download or parameter search runs. Process RSS and Metal allocations
overlap and must not be added; transient peaks may be missed.

| Person | Native shape, 1mm | Estimated brain cells | Estimated skull cells |
| --- | --- | ---: | ---: |
| 1BB033 | 212×246×181 | 1,281,637 | 677,099 |
| 1BC077 | 229×277×182 | 1,246,316 | 737,268 |

Each case embeds five originals: CT, MRI, the two model masks and inference
record. Primary CT arrays and separate acquired MRI arrays remain exact. Equal
coded grids do not establish anatomical alignment; fusion stays disabled.
Tumor, vessels, ventricles and functional evidence remain unavailable.

## App and independent checks

The [source/model/bundle audit](independent-qa.json), SHA256
`90eadfb9fd95d4c64778a7356791c49da8735bee8b21562ebe70f3f84d617346`,
passes exact arrays, affines, masks, originals, fixed source roles, budgets and
checkpoint checks. Full multiclass model predictions were not retained; the
audit proves saved binary-mask transfer and code identity, without repeating
inference or independently testing model accuracy.

The [separate native-save audit](gui-save-independent-qa.json), SHA256
`1d8b36e84916d8dc88c20212eb927b7a63c40ea62d9089c447d06f0170b53847`,
verifies exact source/model arrays, all ten originals and unreviewed provenance
after native Save. Source and model output directories remain149,452,038 bytes;
GUI copies are stored outside them. Prior audits remain unchanged.

The [actual interaction inventory](ui-demonstration.json) records seven captures
and their accessibility observations. A fresh process restores1BB033 with native
T1 selected; switching to CT restores brain/skull opacity.18/.32 and structures
only. Opening1BC077 in that process restores CT, opacity.20/.44 and structures
only. Both separate MRI views show no attached CT segmentation layers.

Source cases are under `outputs/poc-synthrad-bc-source-v1/{subject}/` and the two
reviewed saves under `outputs/poc-first-qa/synthrad-bc-ui/saved/`. No patient arrays
or model weights enter Git. Thirteen adapter controls,132 Electron host tests
and the TypeScript/production build pass. This is scoped verification; the
previously recorded legacy bridge failures are not declared fixed.

This change also lets a clean checkout browse native cases when the optional
generated catalog is absent. Existing corrupt or redirected catalog metadata
still fails closed; present catalog aliases still verify source checksums.
Missing-anatomy labels use readable names without changing stored metadata.

Anatomical completeness, HU calibration, model/source overlap and external
accuracy remain unresolved. Visual inspection and software preservation do not
close those gates or the larger publication goal.
