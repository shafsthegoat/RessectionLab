# Preprocessing and rigid alignment — October 4, 2026

An actual BTC `sub-PAT28` mean-b0/T1 rigid registration now runs locally, with
explicit physical transforms, reproducible numerical QC, and inspectable image
overlays. The current result uses raw diffusion data and remains a diagnostic.
It does not resolve EPI distortion, volume motion, eddy currents, gradient
rotation, functional coverage, or anatomical approval.

## Available tools and scope

The initial local inspection found no `topup`, `eddy`, `eddy_cpu`, `bet`,
`flirt`, `epi_reg`, `antsRegistration`, or `mrconvert` executable on the shell
path. DIPY 1.12.1 was already installed. SimpleITK 2.5.6 was installed into the
project `.venv` for this slice. No paid infrastructure was used.

[FSL supports Apple Silicon macOS](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/macos.html),
and its [official package repository](https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/osx-arm64/)
lists native CPU correction tools. Their separate installation and input
preflight are being handled independently. A missing local executable does not
establish that the platform is unsupported. No FSL binary is bundled with this
registration implementation.

The raw mean-b0 is the arithmetic mean of AP volume indices
`[0, 1, 26, 51, 76, 101]`. Its source hashes and averaging operation are saved in
`artifacts/preprocessing/PAT28-raw-rigid-v1/mean_b0_lineage.json`. This averaging
does not itself correct motion. Registration uses the previously generated
mean-b0 median-Otsu mask only to restrict its similarity metric. That mask is
unreviewed and is not a supplied brain/cortex segmentation for route planning.
T1 remains a full-head structural image.

## Physical transform contract

NumPy image arrays use XYZ order and RAS millimetres. `to_sitk` explicitly
converts array order to ZYX and image geometry to SimpleITK's LPS convention,
preserving spacing, direction, origin, obliquity, and handedness. No left/right
flip is inferred from anatomy appearance.

The optimizer fixes mean-b0 and samples moving T1. As explained in the
[SimpleITK registration overview](https://simpleitk.readthedocs.io/en/v2.3.0/registrationOverview.html),
the returned transform maps fixed physical points into moving physical points.
Thus the exported forward matrix is **DWI-RAS-mm → T1-RAS-mm**. Its inverse is
also saved. The `.tfm` file separately retains the native ITK **DWI-LPS →
T1-LPS** transform. Matrix extraction includes the Euler transform's center of
rotation; its translation cannot be copied from the parameter vector alone.

`T1_on_b0.nii.gz` samples the T1 through the forward mapping onto the acquired
DWI grid. Its affine remains the DWI grid affine. This resampled visualization
does not move the authoritative T1 geometry. A later transform estimated from
corrected b0 data must supersede this raw-data diagnostic for downstream use.

## Registered experiment

The protocol uses an Euler rigid transform, identity initialization in physical
space, Mattes mutual information with 32 bins, a fixed random sampling seed of
41, 25% sampling, three resolution levels `[4, 2, 1]`, and Gaussian smoothing
`[2, 1, 0]` mm. Regular-step gradient descent has at most 120 iterations per
level. Registration uses two threads, and the executed CLI also capped ITK's
global default at two threads. The configuration follows the components in the
[official SimpleITK example](https://simpleitk.readthedocs.io/en/release/link_ImageRegistrationMethod4_docs.html).

Numerical QC uses a separate NumPy implementation of histogram normalized
mutual information, `(H(fixed)+H(moving))/H(joint)`. It compares exactly the
same paired voxels before and after registration and uses fixed intensity
normalization windows. This prevents apparent improvement caused by changing
the comparison support. It is an independent implementation of an image
similarity measure, not independent anatomical ground truth.

The latest inspected artifacts are in
`artifacts/preprocessing/PAT28-raw-rigid-v2/`:

| Quantity | Recorded result |
|---|---:|
| Optimizer time | 3.330 s |
| Complete CLI, including I/O and QC image | 4.87 s |
| Complete process maximum resident memory | 941.75 MiB |
| Optimizer iterations across levels | 64 |
| Identical paired voxels for before/after QC | 99,922 |
| Common support fraction of registration mask | 0.98629 |
| Histogram NMI before → after | 1.04024 → 1.06627 |
| Displacement at the masked image center | 2.409 mm |
| Rigid rotation angle | 1.378° |

The runtime record, report hashes, and transform serialization checks are saved
in `runtime_and_validation.json`. These are measurements from this local
development run, not timing guarantees. A second run retained the same NMI
values and essentially identical transform. No search over patient-specific
registration settings was used to select a favorable result.

The overlay figure was inspected at acquired DWI slices 21, 30, and 39. It shows
the source images share the expected gross image region and permits comparison
of T1 contours before and after the modest rigid adjustment. Residual modality
and geometric differences remain visible. No lesion-local landmark error or
expert anatomical pass is claimed. `registration_reviewed`, `mask_reviewed`,
and `preprocessing_ready` remain false; motor/language coverage remains unknown.

## Executed tests and retained failure evidence

Eight focused tests passed, covering XYZ/RAS to ZYX/LPS image coordinates,
rotation-center-aware transform export, inverse point mapping, rejection of
reflections/scaling, histogram-QC range failures, recovery of a known injected
physical translation, cancellation, invalid masks, and excessive compute
budgets. The analytic registration test checks recovered translation to 0.6 mm
on a synthetic asymmetric phantom; it does not establish real-patient accuracy.

All declared v2 artifact hashes were verified. Loading the serialized ITK
transform reproduced the RAS matrix, and the stored forward/inverse product
was identity within `1e-10`. The resampled T1 had finite data and the expected
DWI affine. The original v1 output is retained. Its broad output-directory hash
scan incorrectly included an actively written redirected log; v2 hashes only
the known completed artifacts. No registration result was changed to repair
that bookkeeping issue.

## Correction work still required

The source AP/PA sidecars declare opposite `j-`/`j` directions and the same
total readout time, but preflight identified two items needing explicit
reconciliation: approximately 2.37 mm corner disagreement between the AP/PA
affines, and reported total readout `0.0266003` s versus `0.03610038` s from
the reported effective echo spacing times `(96−1)`. The
[BIDS MRI metadata specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#in--and-out-of-plane-spatial-encoding)
defines the relevant fields. This registration module neither overwrites those
values nor treats its rigid estimate as correction of that discrepancy.

After reviewed correction inputs and local tool setup, use the supported
[TOPUP workflow](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/users_guide/index.html)
and [EDDY workflow](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/users_guide/index.html)
with retained outputs and rotated-gradient provenance. Rerun this registration
on the corrected mean-b0 and inspect lesion-adjacent alignment. The whole-image
rigid transform must not be substituted for per-volume motion-related b-vector
rotation. A reviewed brain/cortex support and functional endpoint QC remain
additional prerequisites for tract-aware route planning.

## Reproduction

The tested environment needs the project's scientific dependencies and
SimpleITK 2.5.6; generating the optional QC plot also uses Matplotlib.

```sh
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 .venv/bin/python -m pytest tests/test_preprocessing.py -q
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 .venv/bin/python -m resectionlab.preprocessing \
  --mean-b0 artifacts/preprocessing/PAT28-raw-rigid-v1/mean_b0.nii.gz \
  --t1 data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/anat/sub-PAT28_ses-preop_T1w.nii.gz \
  --b0-mask artifacts/diffusion/PAT28-diagnostic-v2/fit_coverage.nii.gz \
  --source-state raw_uncorrected --save-qc \
  --output outputs/preprocessing/PAT28-rigid-reproduction
```

The API accepts an existing mean-b0 image; it does not force reuse of the raw
diagnostic. Corrected inputs must be labeled `correction_outputs_pending_review`
until their processing and anatomical review are established.
