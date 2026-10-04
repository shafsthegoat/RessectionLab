# Estimated native brain extraction

This research slice runs a pinned public model on the full-head BTC PAT28 T1 and
preserves its native voxel grid. Every output remains `brain_reviewed=false` and
`cortical_access_permitted=false`. A brain-extraction envelope does not identify
the pial surface, individual cerebral structures, safe cortical entry, or a
clinical injury probability. Source annotation overlap measures inclusion of that
annotation only; there is no reviewed brain ground truth for this experiment.

## Model and rights

We tested official SynthStrip version-1 no-CSF weights as a parenchymal-envelope
candidate, with the main weights as a separate boundary-sensitivity comparison.
Neither is automatically selected as accepted anatomy. Each checkpoint is
30,851,709 bytes. The original upstream script is
pinned to FreeSurfer commit `cf4bccf24875a47245b4df9fc9372d0f1c3d784f`.
The [official model page](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
offers its weights under MIT or CC BY 4.0; this experiment chooses MIT. That
statement concerns weights, not the entire FreeSurfer codebase.

The [script's original license](https://github.com/freesurfer/freesurfer/blob/cf4bccf24875a47245b4df9fc9372d0f1c3d784f/LICENSE.txt)
is the FreeSurfer Software License Agreement 1.0, February 2011. Part B permits
proprietary incorporation and sublicensing subject to its retained terms and
notices, modification marking, and research-use/clinical-use disclaimers. The
unchanged license is retained locally in `docs/licenses/FreeSurfer-SynthStrip.txt`.
All or portions of this licensed product (such portions are the "Software") have
been obtained under license from The General Hospital Corporation "MGH" and are
subject to the terms and conditions in that file. No FSL component is used by this
extraction path. Commercial permission does not establish clinical suitability.

The downloader verifies SHA-256 before execution:

| Artifact | SHA-256 |
| --- | --- |
| No-CSF weights | `62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28` |
| Main weights | `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33` |
| Upstream script | `291c253ab2f7c0cbcb84afa769ae10909549537e727db5b92d268754b7cc7871` |
| Original code license | `632d404ea17b9101d9ac87cf8f09e651d18eb20d2cc0ba8dbee77e85b8516307` |

HD-BET was considered: its [original official weight release](https://zenodo.org/records/2540695)
has five folds totaling 327.2 MB and a less specific repository license label.
We did not establish a comparable exact-checkpoint rights record or execute it.
No performance comparison against HD-BET is claimed.

## Training overlap

The [original paper, section 3.4.1](https://arxiv.org/pdf/2203.09974), describes
synthetic-image training from anatomical labels of 40 Buckner40 adults, 30 HCP-A
adults, and 10 Boston Children's Hospital infants. Its evaluation includes QIN
glioblastoma scans. The [official evaluation-data README](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/README)
lists IXI, FSM, ASL, QIN, Infant, and CIM cohorts; it does not identify BTC as a
cohort. These descriptions do not certify patient-level disjointness for the
downloaded version-1 main or no-CSF checkpoint. Exact released-checkpoint subject
membership and subsequent training changes remain unverified. PAT28 is therefore
a development/QC case, not a certified held-out test of this pretrained model.

## Reproduction and measured limitations

`resectionlab.brain_extraction` bounds each child process by elapsed time and
sampled resident memory, records failure, and refuses to reuse pre-existing
outputs. The original script supports CPU/CUDA. After the CPU negative result
below, a clearly marked device adapter substitutes Apple Metal for CUDA and adds
observational allocator sampling, with two CPU helper threads. It preserves the
architecture, weights, preprocessing, and resampling operations. GPU allocations
are limited using [PyTorch's process allocator setting](https://docs.pytorch.org/docs/2.14/generated/torch.mps.set_per_process_memory_fraction.html)
to the lower of 6 GiB or 70% of recommended device memory; automatic CPU fallback
is disabled. The acquired T1 conforms
internally to a 192 × 256 × 256 model input. Estimated masks are returned by the
upstream implementation to the acquired grid and then independently checked for
binary values and physical affine agreement. Signed-distance outputs must also
have the native shape/physical affine and entirely finite values before a run can
record successful artifacts. Distance values remain model predictions, not
reviewed geometry or probabilities.

Early compatibility testing failed with the project's NumPy and Surfa 0.6.3:
Surfa attempted to assign a one-element array as an integer during orientation
conversion. The same native T1 passed with NumPy 2.2.6 in a separate interpreter;
the shared project NumPy was not downgraded. The local inference interpreter
inherits the project's Torch 2.14.1, Surfa 0.6.3, SciPy 1.18.1, and NiBabel 5.4.2.
The extra NumPy wheel is about 5.1 MB. Runtime versions are recorded alongside
every inference. This compatibility profile is explicit, not an upstream patch.

Create an isolated interpreter, install `numpy==2.2.6`, and add the project's
site-packages directory through a `.pth` file after the isolated site-packages.
Pass that interpreter using `--inference-python`. The project environment needs
`surfa==0.6.3` (and its `xxhash` dependency), in addition to its existing Torch,
NiBabel, NumPy, SciPy, and scikit-image dependencies.

```sh
.venv/bin/python -m resectionlab.brain_extraction \
  --t1 data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/anat/sub-PAT28_ses-preop_T1w.nii.gz \
  --tumor-fractional data/diffusion_source/ds001226-v5.0.1/derivatives/tumor_masks/sub-PAT28/anat/sub-PAT28_space_T1_label-tumor.nii \
  --cache data/models/synthstrip-v1 --output artifacts/brain-extraction/PAT28-mps-v4 \
  --inference-python .tools/synthstrip-runtime/bin/python --compare-main-model --device mps
```

Use `--allow-model-download` once to acquire the pinned public assets. No patient
image is uploaded. Source inputs are never rewritten.

The comparator uses positive-intensity Otsu thresholding, a physical 2.5-mm
opening, the largest component, hole filling, and a 1-mm erosion. It is an erosive
T1 intensity core, not a validated whole-brain segmentation. This intentionally
simple comparator exposes gross dependence on intensity/morphology assumptions;
its Dice agreement with the model must not be called accuracy. The source tumor
annotation is independently reindexed to native T1 coordinates and thresholded
at 0.5 under the importer’s explicit fractional-annotation contract.

Automated tests cover asset mismatch, native grid preservation, nonbinary/empty
outputs, opposite-axis fractional annotations, physical volume, disconnected and
edge-touching masks, invalid affine/spacing, a synthetic scalp-shell comparator,
actual process timeout, and stale-output rejection. Child-process fixtures with
valid masks but wrong-frame, wrong-shape, or nonfinite distance outputs are rejected
even when the child exits successfully. Any omitted annotation voxel is counted
and flagged, including omissions smaller than 2%. Expert anatomy review,
brain/cortex ground truth, and clinical validation remain outstanding.

## PAT28 observations, 2026-10-04

The native T1 SHA-256 is
`70c5c1e4e28143318d2820788343e239efdc205344204f8f50497b81281b32c6`.
Its native grid is 160 × 256 × 256 with approximately 1-mm spacing. The source
fractional tumor mask is opposite on the first voxel axis; explicit physical
reindexing and threshold 0.5 yield 10,269 annotation voxels.

The unchanged CPU model exited with signal 9 at 242.90 seconds, after a sampled
resident peak of 5.60 GiB, before the 300-second timeout and 6-GiB sampled-RSS
watchdog. No successful model output was accepted. The exit reason is not proven;
the local CPU backend lacks MKLDNN and memory pressure is a plausible contributor.
This negative experiment is retained in `artifacts/brain-extraction/PAT28-v1`.

Before full MPS inference, the actual loaded no-CSF network was compared on a
seeded 64³ random input: maximum absolute signed-distance output difference was
0.00000644 mm, with identical threshold masks. This establishes limited numerical
parity, not anatomy accuracy. `cpu_mps_parity.json` records that test.

The complete MPS experiment and measured repeat are retained in
`artifacts/brain-extraction/PAT28-mps-v1` and `PAT28-mps-v2`. A final
`PAT28-mps-v3` run also checks the license-copy and existing-directory protections:

| Measurement | No-CSF | Main |
| --- | ---: | ---: |
| First complete inference, seconds | 8.082 | 6.529 |
| Repeat inference, seconds | 6.487 | 6.609 |
| Final wrapper run, seconds | 7.129 | 6.319 |
| Repeat sampled process RSS, GiB | 0.626 | 0.630 |
| Sampled Metal tensor allocations, GiB | 3.835 | 3.835 |
| Sampled Metal driver allocations, GiB | 5.540 | 5.540 |
| Estimated mask volume, mL | 1337.639 | 1529.550 |
| Source tumor-annotation inclusion | 97.916% | 100% |
| Dice agreement with intensity baseline | 0.74794 | 0.70646 |

The allocator measurements sample module boundaries and may miss transient
workspace peaks; driver allocation includes cached memory. Process RSS alone
would substantially understate MPS memory usage. These are three sequential runs,
not a latency distribution or clinical-performance benchmark. Masks and distance
maps were numerically identical across all three runs. Source and output hashes, original
and executed-runner hashes, runtime versions, budgets, and QC are in the reports.

Both estimates have one connected component and no input-grid face contacts.
No-CSF excludes 214 source-annotation voxels and raises
`SOURCE_ANNOTATION_EXTENDS_OUTSIDE_EXTRACTION`; those voxels were not silently
unioned back into the mask. Main and no-CSF Dice agreement is 0.932964. That
agreement and complete source-annotation inclusion do not establish anatomical
correctness. The models include inferior brain structures and do not constitute
a reviewed cerebrum/cortex segmentation.

The intensity baseline has volume 1540.930 mL. Visual inspection of the six native
planes showed extra-cranial/neck support and excluded brain regions in that
baseline, demonstrating that it is unsuitable as a complete brain mask. The model
contours broadly follow the intracranial envelope in those planes, with expected
differences near CSF; this limited engineering inspection is not expert review or
proof of whole-volume coverage. The native tumor overlap discrepancy remains an
explicit review item. `extraction_qc.png` preserves the inspected views.

Independent review identified that the earlier wrapper validated the mask before
hashing the distance map but did not validate distance-map geometry or values.
The corrected wrapper adds these gates and rejects malformed child outputs;
`PAT28-mps-v4` records a fresh complete run. Historical v1–v3 reports and existing
frozen implementation snapshots are preserved rather than silently assigning them the
new validation contract. The v4 QC report also records explicit annotation
exclusion counts and flags every nonzero exclusion. The updated 17-test suite
passes. The v4 no-CSF/main runs completed in 7.317/6.301 seconds, with native masks
and distance arrays identical to v3. Both distance maps pass geometry and finite
value checks; their ranges are −4.112 to 100 mm and −5.228 to 100 mm respectively.
The 100-mm value includes upstream out-of-support fill and must not be interpreted
as a calibrated boundary distance throughout the image.
