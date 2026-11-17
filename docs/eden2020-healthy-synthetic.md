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
`prepared/eden2020-synthetic-v1/`. Local ignored `testcases/` links keep these
HDD projects visible in the checkout app's **Cases** list; search **EDEN2020**.
They are local generated projects, not independently accepted release entries.

From the primary checkout with `RESSECTIONLAB_DATA_ROOT` set to the mounted
RessectionLabData folder:

```sh
.venv/bin/python scripts/acquire_eden2020_healthy.py
.venv/bin/python scripts/build_eden2020_synthetic.py --watch
.venv/bin/python scripts/register_eden2020_projects.py --watch
```

Acquisition and generation may run concurrently. Keep model inference and case
builds serial. Existing source bytes and completed projects are checked and
reused; conflicting bytes are refused. Sources are never modified in place.
The small-range transfer helper authenticates assembled ZIPs against the
publisher checksum before extracting or generating anything from them.

Brain extraction uses the installed SynthStrip model with its exact source and
model inference receipt. The generator samples seeded random admitted interior
voxels, alternating physical RAS hemispheric proxies. Placement support combines
the estimated brain, smoothed high-T1 signal, an inferior exclusion and a midline
exclusion. The requested core/rim/edema ellipsoid must fit wholly inside this
support without clipping. Size adaptation and minimum center separation are
recorded when required to obtain distinct positions. These constraints are not
validated GM/WM/CSF classes, lobe assignments, clinical incidence distributions,
or operability judgments.

The T1 appearance is a generated observation model. Baseline values outside the
lesion stay exact. There is no mass effect, growth, modeled vessel displacement
or calibrated tissue response. Every variant includes the original healthy T1
reference. Baseline projects additionally include eligible FLAIR/MRA/SWI references on
separate grids. A conflicting original qform/sform is recorded and its channel
is withheld from display before any resampling; its original bytes remain preserved. Oversized reference channels have an explicitly labeled 1 mm
display derivative; full original bytes remain preserved. Supplied FLAIR-based
tracts and raw diffusion are retained without inventing a T1 registration.

Each project passes the production save/reopen, source-original, array/geometry,
synthetic-label and scene-initialization checks. Source-comparison figures and
representative hidden actual desktop open, inspect/cancel, save and reopen
checks provide additional engineering evidence. Clinical validation, reviewed
vascular/skull/function evidence, surgical access and automatic planning/training
admission remain unavailable.

The publisher API specifies **CC BY-NC-ND 4.0**. This workflow creates private
local noncommercial adaptations and retains attribution; modified imaging and
derived projects must not be redistributed without separate permission. Source:
Castellano, Pieri, Galvan, Rodriguez y Baena and Falini, *EDEN2020 Human Brain MRI
Datasets for Healthy Volunteers*, version 1.0, DOI 10.5281/zenodo.3994749.
