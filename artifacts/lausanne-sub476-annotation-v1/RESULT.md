# Original manual aneurysm annotation: retrospective component verified

The frozen TRAIN person `sub-476/ses-20140519` is an exact member of the author's
38-person voxelwise crosswalk. Membership was inspected with `pickletools`
opcodes; pickle code was never executed. Original source metadata and dataset
CC0 rights were authenticated before the single mask request. Existing scans,
patient roles and held-out records were unchanged.

The 82,420-byte source mask matches published annex MD5
`4b8717630500d48edb362542a283a07b` and acquired SHA256
`1687ac04ca817691ce91de4ccb4457b31dbbbfca414db019d4e29bd88b0a03ee`.
The bounded acquisition/QC completed in0.790 seconds; binary values0/1 contain
193 positive voxels. No learned initializer or generated label was used.

The original TOF and mask both have shape306×384×160. Direct exact-grid admission
**did not pass**: the mask declares unknown spatial units and sform code2,
while the original uses millimeters and sform code1. The maximum affine
coefficient difference is1.52587890625e-5. These observations require a documented
source-coordinate interpretation; no affine, unit field or source payload was
changed to satisfy the gate. Scanner-world and interscan registration remain
unaccepted. Zero annotation-to-planner admissions or optimizer updates occurred.

The label supports only an aneurysm-positive region. It supplies no whole-vessel
coverage, negative vascular domain, force response, glioma boundary or surgical
outcome. The paper reports source radiologist drawing and senior review; neither
is a clinician review of this importer or a timestamped surgical approval.
Annotation/review availability remains unknown. Source-date mismatches prevent
the earlier sub022/sub450 crosswalk candidates from being treated as resolved.

Reproduce preflight with `.venv/bin/python scripts/acquire_lausanne_sub476_annotation.py`.
The single scientific attempt used `--execute`; its existing marker prevents an
automatic repeat. The initial acquisition-free preflight exposed a Path indexing
typo, repaired before scientific access. Original and source-snapshot bytes are
retained under the ignored data directory; Git contains only provenance and
compact results. The current source and frozen declaration reproduce the checks.

## Explicit derived component

Subsequent independent coordinate checks found at most three float32 steps of
coefficient difference and 2.80333e-5 mm displacement over the full voxel-support
boundary. A separately named `source_reference_grid_normalization` now inherits
units from the exact `RawSources` image and retains the original array ordering.
It preserves both untouched headers, file and sidecar hashes, inference rationale
and a precision-envelope proof that every voxel centre retains its nearest
reference index. No registration, interpolation or resampling occurs. The
untouched headers are still not described as identical.

All 193 positives reach canonical exclusions and survive save/reopen and desktop
inspection, with positive-only coverage and unknown background. Motor/language
remain missing. Unknown annotation/review timestamps exclude timestamped use.
The desktop receipt explicitly exposes normalization, original unknown units
and false scanner/spatial-planning admission flags. Source-review text binds
the paper's reported review; it is not a clinician signature on our conversion.

The focused actual-source/refusal suite passes 77 checks in 14.73 seconds. After
adding authenticated crosswalk-license binding, all 13 affected component checks
passed again in 10.29 seconds. Independent review found no remaining code blocker.
Both the methods XML's CC BY 4.0 notice and pinned repository Apache 2.0 license
were authenticated. See [component receipt](component.json),
[verification](component-verification.json) and [license source](membership-license.json).

Run `.venv/bin/python -m resectionlab.lausanne_sub476_component` after acquiring
the hash-pinned originals and source records. This builds an inspectable real
component; it creates no target, brain support, route, simulator episode or
training example. Surgical route acceptance and source-to-scanner accuracy
remain unproved.
