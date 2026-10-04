# Population prior registration: UCSF-PDGM-0004

Local development experiment, October 4, 2026. The implementation creates
inspectable candidate transforms and review previews. It does not establish
patient functional localization or clinical route safety.

## Sources and exact frames

The source T1 and supplied lesion annotation are the pinned UCSF-PDGM-0004
structural files described in `manifests/data_acquisition_ucsf.json`. Geometry
remains in that processed release's coded physical frame. This experiment does
not reconstruct an unavailable original acquisition frame or native skull.

The two reference images were already bundled in
[Zenodo 16418628](https://zenodo.org/records/16418628), under
`pub_release/maps_paper_release/brain_maps/templates/`. No additional template
download or FSL executable was needed. The 1 mm template matches the three
structural language masks; the 2 mm template matches the motor map and three
functional language maps, with exact shape and voxel-to-RAS-mm affine agreement.
The saved grid report records both template hashes.

The 2 mm registration reference, `MNI152_T1_2mm_brain.nii.gz`, has SHA256
`09c925ad165dfbe6b41b572876d67e9bbf8c5e26517e7817f9b43a8256227f53`.
Its shape is 91 × 109 × 91 and its affine has diagonal `(-2, 2, 2, 1)` and
translation `(90, -126, -72)`. Matching a voxel grid is an integrity check,
not validation of its fit to a new patient.

**Template redistribution is unresolved.** The Zenodo record declares CC BY
4.0, but the embedded template has distinct upstream provenance. FSL documents
its template as the asymmetric counterpart of the MNI sixth-generation
symmetric template. McGill explicitly permits use and redistribution of its
symmetric release with attribution; that does not by itself establish rights
for these exact asymmetric bytes. These templates remain local research inputs
and are excluded from a distributable product pending verification. Replacing
them with another MNI generation would require an explicit inter-template
registration; a shared “MNI” name is not sufficient.
([FSL template description](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html),
[FSL licensing](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html),
[McGill sixth-generation release and license](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin6))

## Procedure

SimpleITK runs locally with the patient T1 fixed and the reference template
moving. The bridge preserves voxel locations when converting NumPy RAS arrays
to SimpleITK LPS images. SimpleITK returns a patient-to-template sampling
transform; the exported 4 × 4 matrix is inverted and converted to
**template-RAS-mm → patient-RAS-mm** before use by `anatomy.register_prior`.
Nonzero rotation centers are included in the exported translation.
([SimpleITK registration conventions](https://simpleitk.readthedocs.io/en/master/registrationOverview.html),
[physical image geometry](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html))

Brain support uses positive intensities in these skull-stripped inputs. The
supplied lesion mask plus a declared 5 mm distance buffer is excluded from
fitting. This is a registration sensitivity choice, not a surgical margin.
Images are scaled by their positive-intensity 99.5th percentile. Fitting uses
Mattes mutual information, 32 histogram bins, 18% seeded sampling, and a
multiresolution rigid comparison at 0°, −6°, and +6° initial axial rotations.
An affine candidate starts from the rigid candidate with the lowest final
optimizer metric. The affine stage is centered on brain support, uses a smaller
step, and refines only the finer two levels.

Each candidate has a finite time budget, a bounded iteration count, cancellation
support, source hashes, settings, software version, optimizer trace and stop
reason. Failed candidates remain in the report. The fitted transforms are
frozen artifacts; refitting can vary with numerical threading and optimizer
basins even when the sampling seed is fixed. Do not replace a saved transform
silently with a newly fitted one.

## Measured development results

All values below are image-fit diagnostics from saved reports, measured on the
same development anatomy. They are not independent validation endpoints.

| Experiment / candidate | Brain-support Dice | Lesion-excluded intensity correlation | Observation |
|---|---:|---:|---|
| Initial regular-step rigid, 0° | 0.868 | 0.302 | Broad alignment; residual contour mismatch |
| Initial affine from rigid | 0.818 | 0.264 | Worse fit; retained negative result |
| Initial unconstrained line-search variant | — | — | +6° and affine candidates failed with overlap errors |
| Centered finer-level affine, regular-step pilot | 0.929 | 0.456 | Improved global fit |
| Bounded line-search affine pilot | 0.911 | 0.487 | Different fit tradeoff; not automatically preferred |
| Final v3 multistart rigid, 0° | 0.868 | 0.302 | Stable coarse candidate |
| Final v3 multistart rigid, +6° | 0.675 | 0.086 | Visible inferior displacement; poor candidate |
| Final v3 affine from best rigid | 0.917 | 0.480 | Candidate exported for review, not accepted |

The final four-candidate fitting run took 2.102 seconds on the local machine,
excluding rendering and seven-map preview export. It excluded 79,193 source
brain voxels from fitting because of the lesion-plus-buffer mask. Both the best
rigid and affine stages reached their iteration bound; no convergence guarantee
is claimed. The fitted affine's principal scale factors were approximately
0.976, 0.896 and 0.804 and require anatomical review.

Visual inspection confirmed that the poor +6° solution displaces the inferior
brain and that the refined affine better follows the global boundary. Ventricular,
sulcal and lesion-adjacent mismatches remain. No independent anatomical landmark
annotations were available: `independent_landmark_error_mm` stays null. Image
similarity cannot resolve tumor-associated distortion or patient functional
reorganization. A global affine is especially limited near the lesion.

## Artifacts and application contract

Ignored local experiment directory: `outputs/prior_registration/UCSF-PDGM-0004/`.
It retains the initial comparison, failed line-search variants, corrected pilots
and original comparison montage. `final_v3/` contains:

- `registration_comparison.json`: four candidates, matrices, source/configuration
  hashes, traces, stop reasons and limitations.
- `implementation_snapshot.py`: exact numerical-run source snapshot.
- `alignment_qc.png`: patient T1 with reference contours and source lesion outline.
- `functional_overlay_inventory.json`: seven population-prior overlays, each
  `alignment_review_required`, clinical deficit probability null, and complete
  transform provenance.
- Seven `*_review_preview.npz` files with values, sampling coverage and the
  patient-grid RAS affine. Sampling coverage describes the atlas field of view,
  not patient functional coverage.

The final case hash is
`sha256:cd4756876845943c54d94f98a851c0f07d092631998b4ae327714295cc236b13`.
Importing a preview must not call `review_alignment(accepted=True)` automatically.
Recreate an immutable `RegisteredPrior` from the chosen saved matrix, confirm
the current case hash, and show the original MRI during review. Acceptance only
records alignment review. The layer remains a population prior, and all source,
transform or case changes invalidate previous review.

The reusable entry point is `register_template_to_patient(...)`; rendering is
`render_registration_qc(...)`. A command-line entry is available through
`python -m resectionlab.prior_registration --help`. It requires explicit patient,
template, template identity, case hash and output directory. SimpleITK is an
optional registration dependency; Matplotlib is needed only for the QC montage.

Validation executed: **17 tests passed** across `test_prior_registration.py` and
`test_anatomy.py`. These include an actual synthetic registration with a known
physical translation, oblique/anisotropic RAS-to-LPS landmarks, transform-center
and inverse-direction checks, reflection/shear rejection, finite budget and
cancellation gates, qform-only ingestion, categorical interpolation, unknown
sampling coverage and stale-review rejection.
