# Critical annotation source audit

Checked October 8, 2026 through bounded primary-source research and one original
Lausanne annotation intake. No new agreement, outreach or patient-role change
occurred. These findings support acquisition decisions; none is positive vessel
planning acceptance.

## Current candidates

| Candidate | What was established | Remaining admission dependency |
|---|---|---|
| TopCoW external Lausanne, Zenodo 15692630 | Existing metadata inventory maps 20 duplicate people to the original Lausanne family; 17 are TRAIN | Annotation-level rights, per-case manual/model-assisted origin, initializer ancestry and source-grid linkage. No new unique people. See [ingestion notes](lausanne-original-ingestion.md). |
| Original TopCoW 2023 | [Official release](https://topcow23.grand-challenge.org/data/) describes direct manual annotations and noncommercial terms | Bind the presently obtainable file tranche to those methods and resolve preprocessing lineage before payload admission. |
| IXI 100-subject vessel annotations | [Primary methods](https://www.nature.com/articles/s41597-025-06354-1), “Skull Stripping and Brain Mask Creation,” explicitly use `mri_synthstrip` to exclude skull vessels despite Frangi initialization and human review | Known synthetic-trained upstream dependency makes the derived labels ineligible under the current rule. Human correction does not erase ancestry. CoW/major-vessel emphasis also leaves smaller vessels incomplete. |
| COSTA dataset v1.0 | [Official repository](https://github.com/iMED-Lab/COSTA) links [Zenodo 11025761](https://zenodo.org/records/11025761), published 2024-04-22; [API](https://zenodo.org/api/records/11025761) reports restricted access and an empty public file list | Actual versioned file inventory/fixity, patient crosswalk, separate image/annotation permissions and release-specific annotation/preprocessing history. API MIT metadata and the [creator page's](https://imed.nimte.ac.cn/costa.html) academic-research-only condition do not settle those rights. |
| UNC/Kitware TubeTK `Normal-002` | The [maintainer's clinical example](https://github.com/KitwareMedical/TubeTK-pypbm#example-clinical-data) associates `Normal002-MRA.mha` with `VascularNetwork.tre` and structural MRI | Resolve the linked MIDAS release, dataset rights, exact fixity, native frame/units, extraction and actual review provenance. Tubes contain centerlines/radii, not an independently verified voxel mask. |

COSTA's [primary abstract](https://pubmed.ncbi.nlm.nih.gov/39012728/) reports
manual annotation, but full-method retrieval failed in this audit. The repository's
BET2/HD-BET/iCVMapp3r options for users' own images do not establish which method
produced the released labels. Shipped histogram-landmark fitting ancestry also
remains unresolved. Classify COSTA as access/rights/lineage unresolved, not as
proven synthetic or permanently excluded.

[Zenodo 10957925](https://zenodo.org/records/10957925) is COSTA/CESAR **model
weights**, not the dataset: `Task099_COSTA.zip`, 3,559,838,158 bytes, MD5
`70a0d868cd41f95a1ac054dbdf6955ec`, version 1.0, published 2024-04-11. Do not
count that archive as acquired patients or labels. Software and article licenses
cannot substitute for dataset permissions.

TubeTK's clinical example must remain separate from its expressly artificial
Circle-of-Willis toy drawings and simulated deformations. The latter are
ineligible. The historical handle `1926/1687` did not resolve during this pass.
MRA, structural MRI, tubes and derivatives must be grouped as one person and
deduplicated across UNC/Bullitt/MIDAS/TubeTK aliases before roles are assigned.

## Next acquisition requirement

Acquire a legitimately accessible, version-pinned same-person image/annotation
pair with original hashes, source linkage, image and label rights, nonprohibited
derivation ancestry, actual review and documented annotation domain. Verify the
native frame or an accepted transform before constructing the typed record.
A healthy vascular component case could test integration; it would not validate
glioma surgery, small-vessel completeness, injury mechanics or clinical safety.

## Exact Lausanne source-manual candidate

The author's pinned [voxelwise crosswalk](https://github.com/connectomicslab/Aneurysm_Detection/blob/5ecdf6e5b9a811e4ec7472c210dada42e60cc3dc/extra_files/patients_with_voxelwise_labels.pkl)
includes `sub-476_ses-20140519` exactly. Its published release sidecar declares
the original angiogram as `RawSources` and `Space: orig`. The full [primary
methods](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9931814/fullTextXML)
describe axial slice-by-slice manual drawing in ITK-SNAP3.6.0 and senior
neuroradiologist double-check. Weak spherical labels elsewhere are not contours.
Dates in the author's sub022/sub450 entries differ from the dataset release;
no undocumented date crosswalk was inferred.

The [sub476 result](../artifacts/lausanne-sub476-annotation-v1/RESULT.md) records
the successful82,420-byte binary-mask acquisition and failed direct-grid gate.
Shape matches but units are unknown and affine coefficients differ slightly.
An explicit source-reference normalization now preserves both original headers
and proves unchanged nearest voxel indices within a float32 serialization bound.
The derived retrospective component passes positive-exclusion, roundtrip and
desktop-inspection checks while retaining aneurysm-only support, unknown
background and unknown review/annotation availability. The planner's `vessels`
channel cannot turn it into complete vascular or negative coverage. Source-frame,
brain-support and surgical planning admission remain separate unresolved gates.
