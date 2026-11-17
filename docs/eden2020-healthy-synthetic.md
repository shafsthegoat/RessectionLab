# EDEN2020 healthy-source synthetic projects

This separate healthy-volunteer collection uses the public Zenodo release
[3994749](https://zenodo.org/records/3994749), not the restricted glioma dataset
5714812 handled by the EDEN intake adapter. It prepares fifteen acquired baseline
projects and eight explicit synthetic-tumor variants per person: 120 generated
scenarios, fifteen source people, all DEVELOPMENT. No training is performed.

Original ZIPs, complete extracted MRI/diffusion/tractography, metadata, published
MD5 verification, SHA-256 receipts and a prospective development declaration
live under `sources/eden2020-healthy-3994749/` in the configured external data root.
Native projects use `cases/controls/EDEN2020-*` and
`cases/synthetic/EDEN2020-*`. Recipes, labels, comparisons, failure evidence,
app checks and the browsable `index.html` are under
`prepared/eden2020-anatomy-v2/`. The earlier `prepared/eden2020-synthetic-v1/`
and all v1 projects remain preserved on the data drive. Local ignored `testcases/` links keep these
HDD projects visible in the checkout app's **Cases** list; search **EDEN2020**.
They are local generated projects, not independently accepted release entries.

From the primary checkout with `RESSECTIONLAB_DATA_ROOT` set to the mounted
RessectionLabData folder:

```sh
.venv/bin/python scripts/acquire_eden2020_healthy.py
.venv/bin/python scripts/build_eden2020_anatomy.py
.venv/bin/python scripts/register_eden2020_projects.py --collection eden2020-anatomy-v2 --prefer-segmented-v2
```

Acquisition and generation may run concurrently. Keep model inference and case
builds serial. Existing source bytes and completed projects are checked and
reused; conflicting bytes are refused. Sources are never modified in place.
The small-range transfer helper authenticates assembled ZIPs against the
publisher checksum before extracting or generating anything from them.

The current v2 preparation uses pinned FreeSurfer8.2 SuperSynth source and model
assets in the external model cache. It evaluates the unchanged segmentation
backbone/head using overlapping128-cubed blocks,80-voxel stride and Gaussian
logit blending. This memory-bounded adaptation covers the entire input FOV;
whole-volume equivalence and clinical accuracy are not asserted. The unused
synthetic-image, registration and QC heads are not evaluated. Exact1mm
categorical results, receipts, failed attempts and model hashes stay on the HDD.

Every present model class is retained as an independently toggleable layer:
cerebral gray/white matter, deep nuclei, brainstem/cerebellum, ventricles/CSF,
skull/scalp, optic nerves, arteries/veins and other supported head structures.
Absent classes and tiny labels lost at the declared rehearsal resolution are
recorded. The full-head grid adapts its spacing to remain below the existing
1GiB expanded workspace cap; it does not silently replace the native scan.
Every project includes the exact native healthy T1 reference and original bytes.

VesselFM runs on a declared0.6mm MRA derivative. Its estimates, baseline FLAIR
and SWI are rigidly registered to the rehearsal grid. Each registration retains
the source forms, physical forward/pull maps, coverage and an unreviewed status.
Source qform/sform conflicts are explicitly recorded; registration uses the
coded sform as a working basis without changing or declaring the original
clinical scanner frame resolved. Labels use nearest-neighbor interpolation.
Published tract overlays are decoded only after a clean high-intensity gap and
agreement with the matching tract-only TRK coordinates are verified. Their
unknown units are inherited only from the exact corresponding FLAIR grid.
Original overlays and TRK bytes remain archived. These are baseline tract
reconstructions, not complete or clinically reviewed functional mapping.

Eight seeded placements per person span physical left/right, anterior/posterior
and superior/inferior cerebral-tissue strata with a midline exclusion and at
least18mm center separation. Targets use rotated asymmetric radial fields with
three to six lobes, smoothly closed against estimated cerebral GM/WM. Separate
core, rim and edema masks are preserved. An independent mesh check rejects
cuboids, disconnected targets and excessively regular/solid shapes. Positions,
physical sizes, seeds, containment, shape measures and3D previews are recorded.
These strata are not validated lobe assignments, clinical incidence
distributions or operability judgments.

The T1 appearance is a generated observation model. Baseline values outside the
lesion stay exact. There is no mass effect, growth, modeled vessel displacement
or calibrated tissue response. Every variant includes the original healthy T1
reference plus registered baseline FLAIR/MRA/SWI references. Brain tissue under
the generated target is replaced in active material masks while healthy source
memberships remain preserved. Baseline vessels/tracts remain undisplaced, with
independent protected memberships. Material assignments for bone, CSF, vessels,
tracts and tumor are explicit and survive bundle key sorting/save/reopen.

Each project passes the production save/reopen, source-original, array/geometry,
synthetic-label and scene-initialization checks. Source-comparison figures and
representative hidden actual desktop open, inspect/cancel, save and reopen
checks provide additional engineering evidence. Clinical validation, reviewed
vascular/skull/function evidence, dura segmentation, surgical access and automatic planning/training
admission remain unavailable.

The legacy v1 preparation remains reproducible with
`scripts/build_eden2020_synthetic.py`. It had only a SynthStrip brain envelope
and generated ellipsoidal tumor compartments. The preferred v2 registration
replaces only verified v1 prepared-case symlinks after all135 upgraded projects
exist; it never deletes the historical HDD packages or source data.

The publisher API specifies **CC BY-NC-ND 4.0**. This workflow creates private
local noncommercial adaptations and retains attribution; modified imaging and
derived projects must not be redistributed without separate permission. Source:
Castellano, Pieri, Galvan, Rodriguez y Baena and Falini, *EDEN2020 Human Brain MRI
Datasets for Healthy Volunteers*, version 1.0, DOI 10.5281/zenodo.3994749.
