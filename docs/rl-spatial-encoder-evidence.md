# Scan-conditioned policy: compact 3D evidence and implementation direction

Evidence checked October 4, 2026. This is a proposed architecture pathway, not a
trained model or a patient benchmark. No pretrained weights were downloaded or
imported, and no public case was executed for this review.

The current `MaskedPatientPolicy` scores each candidate with a configurable
small MLP (16 hidden units in the recent axis pilot) over 15 native action
features and six state features. Its inputs describe simulated
removal/contact, nominal anatomical fields and episode state; the policy does
not consume MRI voxels. Those summaries are legitimate in the declared
annotation-assisted track, but attaching an image encoder while retaining
reference-derived target/risk summaries would not establish scan-only inference.
The spatial observation contract must identify which fields can actually be
obtained from the permitted preoperative images and reviewed inputs.

The agreed first prototype uses six spatial channels and a randomly initialized
3D Conv8→16 feature grid with per-action ray sampling. A small residual CNN is
a subsequent comparison. MONAI provides configurable 3D residual blocks and
SegResNet, including small initial widths; its SegResNetDS additionally supports
resolution-aware anisotropic kernels. These are implementation precedents, not
evidence of planning accuracy. A transformer is unnecessary for the first
controlled comparison. [MONAI architectures](https://monai.readthedocs.io/en/stable/networks.html#segresnet)

Use bounded scan context plus physical sampling along each candidate's
entry/path/endpoint; add local patches only after the first prototype. Feed global and local embeddings, declared tool
geometry, physically located candidate coordinates, permitted scalar state and
modality availability into a shared candidate scorer; retain a separate STOP
logit and value head. Apply the existing legal-action mask afterward. The image
branch must receive gradients and affect decisions. Separate static scan encoding
from dynamic cavity/observed-state encoding: cached image features cannot replace
the changing cavity. Candidate generation must obey the same permitted-input
contract; a reference-mask-driven proposer remains annotation-assisted even if
the actor sees only images.

For an initial 16-GiB Mac experiment, propose batch one, a 64³ global tensor and
32³ local patches processed in small candidate chunks. Sixteen float32 channels
occupy 16 MiB at 64³, versus 54 MiB at 96³; these are input storage calculations,
not training-memory measurements. Activations, optimizer state, source volumes,
simulator arrays and Metal allocations also matter. Profile CPU and MPS
forward/backward separately with explicit process/driver-memory and wall-time
stops before scaling. Start with float32; enable reduced precision only after
finite-gradient and numerical-agreement checks. Architecture fit and throughput
remain unmeasured.

Specify crops in RAS millimetres, retain native and observation-grid affines,
and sample through their explicit transform. Record coverage, padding,
interpolation and downsampling; use nearest-neighbour sampling for categorical
masks. A voxel offset is not a millimetre offset, especially on oblique or
anisotropic images. Coarse encoder views must never replace native tool-clearance
geometry. Keep an explicit array-axis convention when moving repository `X,Y,Z`
arrays into tensor `D,H,W` order.

Represent T1, T1c, T2 and FLAIR using fixed slots with separate modality-presence
and spatial-coverage indicators. Zero intensity alone cannot distinguish missing
data, padding and tissue. Train declared modality-dropout patterns; never fabricate
missing acquisitions. HeMIS demonstrated modality-specific embeddings fused over
available inputs, including mean/variance features and modality dropout; its
reported implementation was 2D, so a 3D policy adaptation needs new validation.
[HeMIS, MICCAI 2016](https://arxiv.org/html/1607.05194v1)

Normalize each available channel using only permitted preoperative image support;
freeze its statistics and crop definition before patient optimization. MONAI's
normalizer supports image-derived mean/standard deviation, nonzero support and
channel-wise operation. Nonzero full-head support is an intensity convention,
not a brain mask. Never choose statistics or crops from hidden reference lesions,
postoperative outcomes or evaluation worlds. A fixed per-case transform applied
to an uploaded scan is permitted preprocessing; fitting population transforms
or tuning clipping thresholds on held-out patients is leakage.
[MONAI normalization](https://monai.readthedocs.io/en/1.5.2/transforms.html#normalizeintensity)

Rich training annotations can supervise auxiliary segmentation or representation
losses and score optimization-world rollouts. Keep them in a training-only
target/evaluator object, outside the policy observation and inference checkpoint
inputs. Search-generated teaching targets remain simulated, not surgeon
demonstrations. Test that changing hidden labels changes supervision/reward but
leaves permitted observations, candidate proposals and pre-action logits unchanged.
Evaluate image-only and explicitly annotation-assisted tracks separately, with
missing-modality, image-shuffle and scalar-only ablations.

Self-supervised pretraining is a later comparison, not a prerequisite. Swin UNETR
used 5,050 CT scans, which does not establish brain-MRI transfer. The 2025
Residual-Encoder MAE study used 39,168 MRI scans from a proprietary cohort;
public weights would not themselves establish subject disjointness. Audit every
checkpoint's pretraining and downstream fine-tuning cohorts, aliases, licenses
and overlap before import; retain unknown overlap explicitly. Compare scratch,
frozen and adapted encoders under equal permitted inputs and measured budgets.
[Swin UNETR](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.html),
[Residual-Encoder MAE](https://arxiv.org/html/2410.23132v3)
