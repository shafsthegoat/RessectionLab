# Independent PAT16/PAT20 extraction audit

All eight saved inference results (sixteen mask/distance artifacts) from the
frozen declaration committed as `2e54b98` passed the independent engineering
audit. Each case has two repeated runs of
the no-CSF and main SynthStrip models. No network inference or training was
performed by the auditor; source cases and extraction outputs were unchanged.

`scripts/audit_pat16_pat20_extraction.py` reuses the established independent
NiBabel/SciPy helpers, rather than the production extraction QC functions. The
machine-readable results are `brain-extraction-pat16-pat20-independent-qc.json`;
actual inspection of both rendered six-panel sheets is recorded separately in
`brain-extraction-pat16-pat20-independent-visual-review.json`. The numerical
report deliberately leaves visual acceptance pending until that second record.

The audit verified source manifests and images, unchanged prepared-case bytes,
model assets, frozen wrapper snapshots, generated MPS runners, retained child
records, output hashes and the declared order of runs. All mask and predicted
distance artifacts retain the native 160 × 256 × 256 voxel grid, with
**zero maximum corner displacement** from their source T1. Every saved mask
reconstructs exactly from predicted distance <1 mm, largest connected component
and hole filling. Each has one connected component and zero input-face contacts.

Repeated masks and float32 predicted distance arrays are identical within each
patient/model pair, with zero differing voxels and zero maximum distance
difference. Their compressed artifact hashes are also identical. These are
within-patient repeatability checks, not independent patients or accuracy tests.

| Source annotation at threshold 0.5 | No-CSF omitted voxels | Main omitted voxels |
| --- | ---: | ---: |
| PAT16: 45,400 voxels | 2,045 | 19 |
| PAT20: 12,451 voxels | 413 | 125 |

All omissions reproduce the original reports and remain explicitly flagged.
No target voxel was added to an extraction to improve agreement. The supplied
annotation is a fractional source-intensity threshold scenario, so complete
inclusion would not establish anatomical accuracy either. The distance arrays
include an upstream 100-mm exterior fill; they are not measured surgical
clearance fields.

Both overlay sheets were opened and inspected at three native planes through
the envelope centre and three through the source-annotation centre. PAT16 shows
local no-CSF boundary incursions into the annotated region. PAT20's annotation
approaches the superficial envelope boundary, where the two contours differ
locally. Neither sheet shows gross global overlay displacement in the selected
views. Both estimated envelopes include inferior structures and remain
unsuitable as reviewed cortical access surfaces. Whole-volume anatomical
accuracy, patient function and clinical safety remain unverified.

Recorded runtime versions agree between declaration, preflight and child
records. The current interpreter hash is also recorded. **Historical interpreter
and installed-package binary hashes were not predeclared**, so agreement of
version strings does not prove execution-time binary identity. Retained logs
and timestamps likewise cannot prove that no unrecorded runs occurred.

The numerical audit took 10.35 seconds with a peak process RSS of 824,393,728
bytes on macOS (about 0.768 GiB), using one requested numerical thread and no
GPU. Twenty-eight focused metadata-contract and independent-array-helper tests
passed in 0.35 seconds. No audit failure occurred. Masks remain estimated,
`review_required`, with no working brain mask attached, no anatomical approval,
no cortical-access permission and no clinical probability claim.
