# Case4 DEVELOPMENT scan-preparation QA, attempt 01

**Decision:** saved scan-only preparation is numerically self-consistent and
ready for further *review*, but its anatomy is **not accepted** for whole-brain
coverage, cortical access, model inference, or planning. This is an image QC
reading by the implementation team, not a neuroradiologist assessment or
clinical evidence. No tumor label, postoperative image, model, new registration,
or planning output was used. The independent software/receipt audit is separate
at `../resect-case4-scan-preparation-v1/INDEPENDENT_REVIEW.md`.

The source is exact `preparation-result.json` SHA-256
`92fdc93f1240c81ac0ecbd7910a8256a18693480bd4bed48b19e109fe308d72c`
from the one-shot RELEASE SHA-256
`6c004e45cca84c7488f66d905c281ded82a85eb3cb8bdf211f655d09aa2d5fbc`.
The script checked all listed saved-output SHA-256s and the three source scan/mask
SHA-256s before computing these measurements. `metrics.json` SHA-256 is
`354148c981c7e477e6abf941e5d61ab2b3b3a5671bdf493ddc955504df2d406f`.

| Check | Saved result | Interpretation |
| --- | ---: | --- |
| Native limited mask / atlas limited mask | 1,186.021 / 1,186.055 mL | Volume ratio 1.000029 after rigid/nearest-neighbor resampling; this is self-consistency of the **same estimated mask**, not accuracy versus anatomy. |
| Native mask within explicit registered FLAIR support | 100% | Both source scans cover this limited mask. It does not establish coverage of excluded brain. |
| Common T1c/FLAIR field of view in atlas | 72.61% of entire atlas | A field-of-view statistic, not brain or tumor coverage. |
| Prepared channel nonzero voxels outside limited mask | 0 for each | Channel masking was applied as intended. |
| Saved atlas mask returned to native T1 using the saved inverse and nearest-neighbor interpolation | Dice 0.99744 with its *own source mask* | Checks transform direction and discretization; no ground-truth anatomical validation. |
| Saved rigid point transform inverse closure | 1.3121×10⁻¹³ mm max for atlas↔T1; 4.0818×10⁻¹⁴ mm for atlas↔FLAIR chain | Algebraic numerical closure on 33 sampled mask points, not registration accuracy. |
| Same-patient T1/FLAIR mutual information | 0.32573 nats with physical-frame identity resampling; 0.37082 after saved rigid registration | +0.04509 nats on 18,517 common-support sampled voxels. ANTs optimized a Mattes MI objective on these images, so this is an in-sample QC signal, not independent anatomical evidence. |

The identity comparator used SciPy linear resampling; the saved registered image
used ANTs interpolation. That implementation difference also limits attribution
of the MI change to the rigid parameters alone.

The [native montage](../../build/scan-target-estimator-research/case4-postprep-qa-v1/native-registration-montage.png) shows plausible central
T1/FLAIR structural correspondence at one axial, coronal and sagittal level.
The green contour is the *estimated* SynthStrip mask. In the sagittal view it
plainly excludes the cerebellum and lower structures; the coronal lower contour
is also cut short. These are consistent with the previously recorded
inferior/cerebellar omissions. The [atlas montage](../../build/scan-target-estimator-research/case4-postprep-qa-v1/atlas-preparation-montage.png)
shows plausible coarse rigid positioning against SRI24, but the public template
and this patient's anatomy visibly differ. These center slices do not constitute
an exhaustive 3-D or expert anatomical review.

The source manifest calls the first scan a T1c candidate based on the RESECT
cohort protocol, without a per-file contrast-administration record. The source
FLAIR role is likewise based on the supplied file/protocol metadata; visual
appearance alone was not used to certify modality identity. Unknown GlioMODA
training lineage and Case4 overlap remain unknown. The limited mask does not
establish whether a future tumor estimate/ROI is fully inside reliable support.
Any bounded diagnostic inference would need a separate reviewed release and
local ROI/coverage checks; these QA results do not admit it automatically.

## Next bounded diagnostic

The Case4 source extraction already used the pinned SynthStrip **main** weights
with a 1 mm border. In prior separate cases, the pinned no-CSF variant produced
*smaller* parenchymal masks than main (`docs/brain_extraction.md`); no Case4
evidence shows that variant would recover the missing inferior tissue. Changing
the border or dilating this mask would expand an estimate, not validate anatomy.
HD-BET was only considered and lacks the exact rights/checkpoint/run evidence
needed here. Thus no currently qualified configuration can be claimed to repair
the observed omission from the saved evidence.

The next useful **source-only preparation** is a fixed, scan-derived support
map and explicit `unknown_outside_support` output contract for a *cerebral-region
research diagnostic*. Retain the current mask and both original scans; record
which native/atlas voxels and any prospective ROI are inside both modalities and
the limited mask, without using tumor labels or model output to choose that ROI.
Review its 3-D cerebral-region bounds and inferior exclusions before calling it
qualified. A later model forward, if independently resource-feasible at its real
128³ patch and separately released, could report only a candidate *within that
documented support*. It cannot infer absence of tumor outside the mask or become
a whole-brain/planning input merely because an internal model score looks good.

![Native T1/FLAIR registration montage](../../build/scan-target-estimator-research/case4-postprep-qa-v1/native-registration-montage.png)

![Atlas preparation montage](../../build/scan-target-estimator-research/case4-postprep-qa-v1/atlas-preparation-montage.png)
