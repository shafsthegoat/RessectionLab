# Diffusion reconstruction slice — October 4, 2026

`resectionlab.diffusion` produces actual diffusion-derived scalar maps and a
bounded probabilistic tracking audit. Its current real-case output is a raw-data
diagnostic. It is not eligible for tract-aware planning, and it identifies no
motor or language bundle.

## Source and coordinate contract

The inspected case is BTC preoperative `sub-PAT28`, OpenNeuro `ds001226` version
5.0.1, pinned commit `359d372c5e972a161966312128adb365870df949`. The acquisition
manifest under `data/diffusion_source/ds001226-v5.0.1/` retains source URLs,
version IDs, hashes, and CC0 provenance. This is the documented BTC feasibility
departure from the UCSF priority, not a replacement of the planned benchmark.

The AP DWI contains 102 volumes on a 96×96×60, 2.5 mm grid: six b0, sixteen
b700, thirty b1200, and fifty b2800 volumes. It has an oblique LAS image grid.
The two reverse-phase PA b0 volumes have a slightly different affine; they are
not concatenated or declared corrected by this module.

The [BIDS gradient specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#required-gradient-orientation-information)
defines its FSL b-vector convention, including a first-component sign inversion
when image axes are right-handed. `gradients_in_image_axes` implements that rule
and explicit world-RAS/world-LPS conversions. Image spacing never scales unit
direction vectors. Sheared grids, unknown conventions, invalid norms, count
mismatches, and rank-deficient gradients fail with named errors.

Output affines remain the acquired DWI voxel-to-RAS transform. Principal-vector
components are unit RAS directions with antipodal sign ambiguity. Native-grid
retention is not a registration to T1. No transform or tumor-overlay alignment
is inferred from a shared patient identifier.

## Implemented models and gates

The tensor fit uses [DIPY weighted least squares](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dti.html),
restricted to b0/b700/b1200 for this case. It excludes b2800 rather than treating
all shells as a single low-b tensor acquisition. Processing uses chunks of
2,048 masked voxels. Derivatives are FA, mean diffusivity in mm²/s, principal
RAS direction, normalized signal-fit residual, and a separate fitted-voxel mask.
Values outside the fitted mask are NaN; missing reconstruction is not zero risk.
The automatic mean-b0 median-Otsu mask remains unreviewed.

Default reconstruction rejects unresolved motion correction, susceptibility
distortion correction, gradient rotation, brain-mask review, and DWI-to-T1
registration. Correction references must match the loaded DWI and b-vector
hashes. A declared DWI-RAS-to-T1-RAS rigid transform, structural source hash,
and registration review are required for the preprocessing contract. Passing
these checks alone still does not validate tract endpoints or functional
coverage. Every tensor result therefore retains
`usable_for_tract_aware_planning: false`.

`--diagnostic-only` permits an explicitly labeled reconstruction while retaining
every failed gate. This does not change the planner's eligibility. The current
module does not perform motion/eddy/susceptibility correction or T1 registration.
The source authors describe a fuller preprocessing and multi-tissue CSD workflow
in their [dataset paper](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9637199/fullTextXML).
Reusing source acquisition files does not mean those processing steps have run.

The optional crop audit separately fits [constant-solid-angle Q-ball](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_csa.html)
on b0 plus the fifty b2800 directions, using order-six spherical harmonics.
The design must identify all 28 coefficients. It clips negative ODF samples and
uses DIPY's [probabilistic tracking mechanism](https://docs.dipy.org/stable/examples_built/fiber_tracking/tracking_probabilistic.html)
within a fixed 24³-voxel crop. The initial protocol uses 64 seeds, three sampling
realizations, 0.75 mm steps, a 30° turn limit, and an unreviewed FA≥0.2 stopping
mask. Paths shorter than 5 mm are counted and discarded.

The crop audit is a computational diagnostic, not a whole-brain tractography
protocol. Repeating tracks from one fixed ODF measures sampling variation under
that fit. The saved visitation fraction is explicitly conditioned on the crop,
seeds, fit, and stopping rule. It is not a reconstruction ensemble, anatomical
absence map, or clinical event probability. There are no CST/language labels,
endpoint claims, or automatic planner integration. DIPY 1.12.1's CSA model uses
its legacy Descoteaux SH basis internally and emits a pending-deprecation
warning in tests; the basis is recorded, not silently changed.

## Executed validation and actual results

Fifteen focused tests passed. They cover BIDS left/right handedness, oblique
anisotropic image axes, RAS/LPS conversion, invalid gradients, missing raw
directions, stale preprocessing evidence, an improper registration reflection,
analytically known FA/MD/principal directions, high-shell exclusion, unknown
coverage, rejection of a different patient's tensor, and reproducible bounded
probabilistic paths in a reflected physical frame.

The current real-case artifacts are under
`artifacts/diffusion/PAT28-diagnostic-v2/`. All saved numerical derivative hashes
were verified, fitted principal-vector norms were checked, and outside-mask
NaNs were checked. `diagnostic_qc.png` was inspected at acquired DWI slices 25
and 35 against mean b0. The scalar and direction maps occupy the corresponding
native image regions. This technical inspection is not an expert tract or
registration review.

| Recorded quantity | Result |
|---|---:|
| Fitted native DWI voxels | 101,311 of 552,960 |
| Tensor fit, excluding loading and masking | 2.243 s |
| FA median / 95th percentile within fitted mask | 0.1939 / 0.6408 |
| Median mean diffusivity | 0.0009356 mm²/s |
| Median signal RMSE divided by mean b0 | 0.06359 |
| Diagnostic paths retained across three sampling runs | 176 |
| Paths discarded because shorter than 5 mm | 16 |
| Crop reconstruction and tracking time | 0.173 s |

These are diagnostic observations from one patient. The residual is goodness of
signal fit, not anatomical accuracy. The times are one local run, not a latency
guarantee. The strict real-input planning call was also exercised and rejected
with `DWI_MOTION_UNCORRECTED`; its complete gate list is saved in
`planning_gate_check.json`. The earlier v1 diagnostic remains retained; v2 adds
source-code hashing, stale-derivative checks, and full CSA design-rank validation.

## Reproduction

Install the project's diffusion extra into its Python 3.12 environment. The
tested optional library is DIPY 1.12.1. From the repository root:

```sh
.venv/bin/python -m pytest tests/test_diffusion.py -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m resectionlab.diffusion \
  --dwi data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/dwi/sub-PAT28_ses-preop_acq-AP_dwi.nii.gz \
  --bvals data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/dwi/sub-PAT28_ses-preop_acq-AP_dwi.bval \
  --bvecs data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/dwi/sub-PAT28_ses-preop_acq-AP_dwi.bvec \
  --gradient-convention BIDS_FSL --diagnostic-only --probabilistic-crop \
  --output outputs/diffusion/PAT28-diagnostic
```

The next required experiment is corrected AP/PA preprocessing with rotated
gradients and independently inspected DWI/T1 alignment. Then test a suitable
multi-shell reconstruction and explicit motor/language endpoint protocols,
retaining tumor-domain failures and missing coverage. None of the current
diagnostic tracks should enter route scoring before those gates are resolved.
