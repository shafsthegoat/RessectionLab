# Functional evidence and patient-context contract

Verified October 4, 2026. Implements the meanings required by `MASTER_PLAN.md`
Sections 4, 6, and 8; this is an acquisition/QC record, not a clinical validation.

## Verified small public releases

Both pinned records declare **CC BY 4.0** in the Zenodo API. Neither archive
contains a separate license file. Retain the record metadata, attribution, archive
hash and exact member name. Bundled upstream templates/code require their own
provenance review before inclusion in an application distribution.

| Record | Archive | Bytes | Source MD5 | Locally computed SHA256 |
|---|---|---:|---|---|
| [10439149](https://zenodo.org/records/10439149) | `data.zip` | 8,413,614 | `9a4427005a38e500057696db83893909` | `b1d3f39719a36344c620e09248a10c8b32b317c4c24d5a9ff8556275caeba0fd` |
| [16418628](https://zenodo.org/records/16418628), v2 | `pub_release.zip` | 19,149,132 | `aaa97d40e791dcd2a80a9de0b9251dc8` | `ea7967a34b8feba8983443afdfbdcbb9103cdf4b09708c110b0d2f271ab48d41` |

Download endpoints:

- https://zenodo.org/api/records/10439149/files/data.zip/content
- https://zenodo.org/api/records/16418628/files/pub_release.zip/content

The bytes were downloaded and both published MD5 values matched. Inspection
copies are under `/tmp/ressectionlab-functional-evidence/`, named
`10439149-data.zip` and `16418628-pub_release.zip`; this temporary directory is
not a durable application cache. No archive or medical volume belongs in Git.

## Actual map contents and frames

| Archive member | Shape / spacing | Meaning |
|---|---|---|
| `data/netw_no_thr/MOTOR_POSITIVE_no_thr_N_599_seeds.nii.gz` | 91 × 109 × 91 / 2 mm | Population motor functional-connectivity concordance |
| `data/bipartite_netw/MOTOR_positive.nii.gz` | 91 × 109 × 91 / 2 mm | Released motor positive-network derivative |
| `pub_release/maps_paper_release/brain_maps/functional/{PHONOLOGICAL,SEMANTIC,SPEECH_ARREST}_union_randomise.nii.gz` | 91 × 109 × 91 / 2 mm | Three separate population language functional maps |
| `pub_release/maps_paper_release/brain_maps/structural/{PHONOLOGICAL,SEMANTIC,SPEECH_ARREST}_union_norm_thr_bin.nii.gz` | 182 × 218 × 182 / 1 mm | Three binary normative structural-network masks |
| `pub_release/maps_paper_release/brain_maps/templates/MNI152_T1_{1,2}mm_brain.nii.gz` | Corresponding 1 / 2 mm grids | FSL MNI152 reference images |

The maps use the FSL MNI152 template frame. For the inspected 2 mm maps the
voxel-to-RAS-mm affine is `diag(-2, 2, 2, 1)` with translation `(90, -126, -72)`;
the 1 mm maps use `diag(-1, 1, 1, 1)` with that translation. Array axis zero
therefore decreases the RAS x-coordinate. Do not flip by array appearance.

**A real loader edge case:** the motor maps have valid `qform_code=1` but
`sform_code=0` and zero srow entries. Their qform quaternion is `(0, 1, 0)`,
qfac is `-1`, offsets are `(90, -126, -72)`, and scalar slope/intercept are
`(1, 0)`. Respect the valid qform; an all-zero, disabled sform is not their
physical transform. Language maps have both forms enabled. Their form codes
alone do not prove template provenance: retain the release and template identity.

The motor concordance maximum in the inspected map is approximately 0.99833;
language functional maxima are approximately 0.94340, 0.88889 and 0.94 for
phonology, semantics and speech arrest. These are data-integrity observations,
not patient risks. A file called `negative` denotes an anticorrelated functional
network, not negative intraoperative stimulation or a safe-tissue mask.

The first archive contains 4,137 point rows from 612 distinct `SUB_ID` values
(2,906 cortical, 1,231 subcortical), verified from the eight CSV files. Columns
are `SUB_ID,CATEGORY,X_MNI_LIN,Y_MNI_LIN,Z_MNI_LIN`. Preserve patient grouping
if constructing any held-out localization experiment. Different points from one
person are not independent patient samples.

## Supported use and registration gate

The original motor/DES paper describes population connectome maps and sampling
bias from operative locations and incomplete negative stimulation acquisition.
Its public atlas supports prior overlays and localization research; absence of
atlas signal does not establish absence of function.
([Coletta et al., Brain](https://academic.oup.com/brain/article/147/3/1100/7458468))

The language paper describes affine alignment to FSL MNI152 of postoperative
source T1 imaging, visual registration checks, and repositioning of subcortical
points toward white matter/deep nuclei, excluding moves of at least 5 mm. The
released maps remain normative after registration to a new patient. The paper
explicitly withholds raw glioma MRI, derivatives and clinical scores; figure
data do not supply the coupled patient/action/outcome set needed to calibrate
counterfactual surgical deficits.
([Coletta et al., Communications Medicine](https://www.nature.com/articles/s43856-025-01121-0))

Before a patient overlay is enabled, retain template and patient image hashes,
patient frame, transform direction, interpolation, method/version, lesion mask
use, alignment review, and an explicit pass/fail state. Use nearest-neighbor
interpolation for categorical masks and an appropriate scalar interpolation for
concordance maps. Keep the acquired patient geometry authoritative. An affine
or nonlinear registration can fail near mass effect and must not silently become
an individualized functional boundary. Unreviewed alignment remains unassessed.

Retain motor, phonology, semantics and speech-articulation evidence separately.
Any combined language field is a declared research aggregation. Do not infer
left-only function from handedness, mirror a left map into the other hemisphere
as observed evidence, or replace failed patient tracts with the atlas while
retaining a patient-specific label.

## Minimal inspectable evidence fields

Each displayed layer needs `source_id`, source/member hashes, `evidence_type`
(`observed`, `estimated`, `prior`, or `simulated`), target system/component,
frame, transform hash, method/version, units/meaning, parameters, QC state,
and a separate coverage/unknown mask. Zero signal and unknown coverage are
different states. Public source masks and fitted patient tracts stay separate.

- **H0:** distance in millimeters plus a chosen proximity penalty. Its scale is
  a sensitivity parameter, not a clinically established safe margin.
- **H1:** support fraction across declared coherent anatomy reconstructions.
  Show ensemble size and assumptions; shared reconstruction bias remains.
- **H2:** event count over whole-plan executions, such as full-tool contact
  with a modeled motor structure. Store event definition, numerator,
  denominator, world-generator hash/partition and sampling interval separately.
- **Clinical deficit probability:** `null`, with
  `no_validated_clinical_outcome_model`. Never convert a map intensity, contact
  frequency or lack of data to a percentage of paralysis or aphasia.

Shaft radius, pose tolerance, registration error and anatomical uncertainty
are distinct terms. Do not inflate an anatomy mask for the shaft and then
apply the same shaft radius again. Tract dropout must introduce unknown coverage
or failed eligibility, not a newly low-cost route.

## Time-aware patient context

Each context entry retains value, unit, source, measurement time, availability
time and evidence type. `planning_as_of` defines the decision cutoff. A value is
eligible for primary preoperative inputs only if its availability is established
at or before that cutoff. Missing timing is unavailable, even if the dataset
column is populated. Preserve the original value in an audit record while
excluding it from actor observations and default objectives.

Store assayed IDH1/IDH2, 1p/19q, MGMT promoter methylation and integrated diagnosis
when supplied. Preserve assay provenance and distinguish confirmed measurements
from imaging predictions. MGMT methylation is an epigenetic measurement.
The EANO guideline addresses assays and diagnosis; it supplies no
genotype-to-tool-tolerance or voxel-injury equation.
([EANO molecular diagnostics guideline, full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10547522/))

Prior-biopsy results may be used when availability is verified. Future resection
pathology, future intraoperative findings and later function scores are excluded
from the preoperative primary track. A separate oracle-context or biological
scenario must be explicitly labeled and versioned. Geometry and protection
constraints must remain unchanged when molecular data are missing or aggressive
tumor biology is recorded. The planner must run without molecular metadata.

Acceptance cases include unknown availability, timezone-equivalent cutoffs,
available-exactly-at-cutoff, known prior biopsy, postoperative pathology,
estimated genotype and absent baseline function. Source timestamps must not be
replaced with file download time. Any context edit invalidates dependent results.
