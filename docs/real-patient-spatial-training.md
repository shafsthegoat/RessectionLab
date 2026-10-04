# Real-patient spatial policy: next executable slice

October 4, 2026 steering: population learning uses real patient datasets only.
The supplied `CODEX_STEERING_PROMPT.md` prioritizes one probability-aware patient
planner while this learning pipeline progresses; useful learning is evaluated
against strong search, whose product role does not depend on RL winning.
The procedural spatial screen remains unrun; its development checks and negative
teacher diagnostic are retained. Simulated interventions on actual patient
anatomy remain simulated experience, not surgeon demonstrations or clinical
outcomes. The primary specification remains `MASTER_PLAN.md` §§9 and 15.8.

## Patients and inputs

The frozen BTC cohort manifest assigns PAT05/16/20/22/25/28 to training and
PAT26/27 to checkpoint selection. PAT29/31 remain unopened for a later frozen
development-transfer pilot. These ten patients do not establish external
generalization; the reserved UPenn evaluation remains untouched. Group every
visit, crop, derivative and augmentation by `BTC:participant_id`, reconciling
older bundle spellings without creating new patients.

Load verified bundles with `resectionlab.imaging.load_case` from
`outputs/cases/BTC-sub-PAT##.ressectionlab`; bind each bundle and source manifest
hash. The four new bundles have been prepared; independent review remains pending. The actual
structural input is T1 MRI with its native affine. The primary common-input
experiment cannot assume T1ce, T2, FLAIR, vessels or patient functional maps.
Unknown fields retain false availability/coverage and null clinical risk.
Unknown-timed pathology stays excluded.

Use a declared intensity normalization based only on that patient's permitted
image and support estimate. A fixed whole-image or access-centered policy grid
must retain its physical affine; never choose a crop using a withheld tumor
mask. Native-resolution geometry remains authoritative. The current CNN accepts
at most 64 voxels per axis, so passing the full-resolution MRI directly is not
an executable option.

## Two explicit observation tracks

The fastest planning comparison can be **annotation-assisted**: provide the
source-derived tumor annotation openly to both spatial policy and search, along
with MRI and permitted anatomical estimates. This tests learned planning given
segmentation, not scan-only tumor recognition. New BTC preparations currently
mark tumor labels private for scan-only use. Bind their source annotation as an
observed input in the separate annotation-assisted adapter; retain the original
inference-only metadata. The user's steering explicitly permits supplied
reviewable segmentations.

The **scan-only** track keeps source tumor labels exclusively for training and
evaluation outcomes. It requires either a tumor estimator fitted only on TRAIN
patients, or label-independent candidate generation plus a declared observed
planning objective for search. Native axis proposals currently read target
labels to choose endpoints; simply hiding that channel does not remove leakage.
The synthetic 0.35/0.5 intensity thresholds are invalid MRI estimators and must
never be reused here.

## Blockers and next measurement

Full-head T1 is not a brain mask. None of the newly acquired four cases has
approved working support or cortical access; nonzero-intensity support is
forbidden. An explicitly unreviewed scan-derived envelope may support a separately
bounded engineering experiment only after its geometry/access contract is
declared; it cannot silently become reviewed anatomy. The native adapter also
needs real-image provenance, bounded patient proposals and correct memory for any
history-dependent contact cost.

Once these contracts are ready, run one TRAIN patient preflight: prepare spatial
observations, execute independently checked legal actions, perform one actual
gradient update and measure preprocessing, proposals, forward/backward, memory
and replay. This sets patient-scale budgets before population training. Reuse
the existing runner's checkpoint/cost logging; replace procedural generation
with role-locked patient loaders and the native adapter.

Then train on all six permitted patients, select checkpoints using only PAT26/27,
and freeze the entire procedure before opening PAT29/31. Compare SEARCH,
PATIENT_SCRATCH_RL, POPULATION_FROZEN and independently initialized
POPULATION_ADAPTED. Freeze each patient's objective and world generator; isolate
optimization, selection and evaluation worlds. A fixed deterministic anatomy
alone cannot support a claim of independent uncertainty-world evaluation.
