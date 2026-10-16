# One image-only ultrasound fit is frozen for landmark scoring

The single predeclared RESECT Case3 fit completed from preparation commit
`fa44c9e`. It uses only the two existing TRAIN ultrasound images and
image-derived signal/coverage support. The full-tag landmark coordinates remain
unopened at this stage. No new person, acquisition, neural training or accepted
anatomy is added.

The [fit record](fit-record.json), SHA256
`35519def02baac35797076b33eabf2eb77dcf1b2314977c8c7eec642fb9e599d`,
binds the candidate, frozen transform, source checks and supervision. Full
matrices and the candidate remain under
`outputs/poc-resect-case3-image-registration-v1/`, outside Git. The frozen
transform SHA256 is
`e6648339d69841985f8fa644ae5d69168422ce12ca5dadc4659b275a8b9cd8af`.

The [supervision record](fit-supervision.json) reports exit zero, 1.301515 s
worker wall time, 1,590,722,560 sampled worker/family RSS bytes and at least
6,302,629,888 available bytes. Seven samples use the fixed 0.2-second interval;
the 4 GiB sampled stop and 240-second worker limit do not fire. The worker is
reaped and cleanup attempted. Sampling is not an exact memory peak or OS hard
limit. The optimizer phase alone takes 0.021308 s; these are single executions,
not comparative latency measurements.

Matched image NMI rises from 1.069655 to 1.078429 over 9,052 paired coarse-grid
voxels, with full declared overlap retained. No predefined image-QC flag fires.
This measures image similarity, not anatomical alignment. The next fixed stage
must compare identity and the frozen rigid transform against every published
full-tag landmark, without refitting, case replacement or outcome-based
exclusion. No landmark accuracy result is claimed here.

The [independent saved-fit audit](../resect-case3-registration-result-independent-v1/independent-fit-qa.json)
passes 24 generated/tamper controls and checks the actual saved hash chain,
runtime/source bytes, rigid inverse, direction, optimizer events and supervision.
It does not decode images or independently recompute the image objective. Its
report SHA256 is
`3277460d1529126ea04828e8785ffd0ebfebdc1ae6209b77e7d4cd92eed1a955`.
The separate [scoring preparation](../resect-case3-registration-evaluation-preparation-v1/preparation.json)
fixes all-point Euclidean TRE, native-image inclusion reporting, duplicate
retention and the rigid-minus-identity RMS endpoint. Its 52 generated controls
pass before actual landmark access. The scorer and fit records are committed
before the separate evaluation release; preparation is not an accuracy result.

Both acquired source files are rechecked after decoding and after fitting;
code/runtime continuity and transform/inverse/direction are checked before the
freeze. The [separate root release](root-fit-release.json) grants exactly one
fit on the fixed outputs path. An unused initial release is retained after its
ambiguous model-fitting category was replaced with explicit neural-training
and registration-fitting fields, before any worker started. Scientific inputs
and settings are unchanged; there is one actual fit attempt and no retry.

The earlier [independent preflight](../resect-case3-registration-preflight-independent-v1/independent-qa.json)
and its exact generated-auditor snapshot are preserved. That audit is software,
source/header and runtime evidence. Sparse landmark scoring, effective imaging
resolution, scanner/tracker calibration, dense deformation, tool causality and
clinical validity remain separate questions.
