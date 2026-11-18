# Bounded supratentorial procedure contract

`surgicalProcedurePolicyV1` declares one research tumor-resection workflow before
search. It supports generated engineering controls starting after scalp access
or at an explicitly supplied post-access checkpoint. Earlier costs are unknown;
neither start represents completed skull-to-exit surgery. Acquired sources are
currently unsupported (`PROCEDURE_ACQUIRED_SOURCE_QUALIFICATION_UNAVAILABLE`):
existing receipts do not yet bind complete intracranial and critical coverage to
this task. A review string or research acknowledgement cannot supply that join.

On the November 18 four-channel UCSF-PDGM-0004 study, the selected left-superior
path ends at the intact admission boundary. The geometric comparison evaluated
five windows and returned three. The selected 16.32 mm corridor contains
151 mm³ of source-supported non-target brain by voxel centers. Physical costs
for traversing and withdrawing through that corridor are unknown. The inspection
derivative samples the T1c reference, selected brain mask and supplied enhancing
target. All four acquisitions and the separate BraTS estimates remain bound
through the native reconstruction revision. The exporter forecasts STOP before access. The engine measures zero
actions, zero target and mapped non-target removal and 15,672 mm³ of original
target remaining. Both tools start and finish outside. No physical treatment
candidate was examined on this revision. The historical preparation on the
earlier revision incurred 66 mm³ of mapped non-target removal over 53 actions.

The current local release review gives UCSF-0004 structural MRI and supplied
tumor inspection capability. Its brain estimate and structural-MRI vessel
candidates lack independent operative review. The reviewed TopCoW-001 CTA
release covers a different person with no tumor target and partial vascular
coverage. Neither release supplies reviewed operative permissions. The current
integrated catalog contains 368 rows and reports zero subjects with matched
acquired tumor MRI, skull CT and dedicated angiography. A particular modality
combination would still require task-specific review.

A future acquired admission needs a versioned, independently issued decision
that binds the exact patient, timepoint, input checksums, geometry, registration,
sampled original memberships and coverage domains to this procedure. The
importer must verify that decision against an independently maintained release,
rebuild its masks and protection on the engine grid, and match the declared
stage, permitted entry, working compartments, swept instrument domain, tissue
permissions and harm limits. Acceptance needs a reviewed positive case and
refusals for changed pixels, patient, revision, partial critical coverage and
foreign or withdrawn review decisions. A study-supplied assertion of review
cannot supply release authority. [Private release and demonstration guide](local-storage-ledger.md)

The versioned declaration independently binds entry masks, permitted compartment
masks, the whole-tool work box, observed source coverage, tissue effect permissions,
and budgets. Brainstem and cerebellum are required neural memberships with
explicit negative-annotation coverage over the intracranial domain. Traversing
either is forbidden in this workflow. Names do not classify anatomy or material.
Overlapping source memberships remain independent of categorical material owners.
All declared anatomical cells outside permitted compartments, unknown cells,
undeclared intracranial entry boundaries, and space outside the work box are
excluded. An empty categorical voxel alone never establishes observed free space.

`make_procedure_policy` serializes supplied source-cell IDs losslessly as runs.
Every evidence row declares provenance, an evidence SHA256, and independent source
origin. Generated controls require explicit generated source qualification;
contradictory acquired-source metadata takes precedence. `procedure_source_binding`
retains the adapter's full pre-policy case manifest hash to avoid a hash cycle
when the policy is persisted inside case metadata.

`SurgicalEngine` binds the policy automatically. Its existing whole-instrument
motion and source protection checks cover both hands, shafts, intermediate poses,
effects, and withdrawal. A detached candidate is checked against cumulative bone,
nonneural preparation, neural removal, existing cut-edge source incidence, structure
exclusions, and modeled-time budgets before publication. Refusal preserves the
accepted state, history, time, and treatment credit. Actual search computation is
accounted separately. The drill remains bone-only.

`sequence-outcome.v2` and `surgical-consequence-vector.v2` report direct source
removal separately for bone, nonbone preparation, neural tissue, protected tissue,
and every declared structure. Structure overlaps are deliberately not additive.
Nonbone preparation means all non-target, nonbone, nonneural source removal over
the complete accepted history. Cut-source quantities count unique source cells
incident to existing native cut/transection edges; they are not physical incision
volume or injury. Neural cutting defaults to a zero engineering allowance.
Motor/language outcomes remain unavailable. Mechanical, vascular, and thermal
quantities retain their existing model and coverage qualifications.

The existing consequence comparator first applies hard bounds and retains the
Pareto frontier. The explicit selection rule maximizes additional target treatment
within those bounds. Equal treatment compares neural removal, neural cut incidence,
structure effects, preparation, bone, then time; no combined bone/neural sum is
used. All alternatives and refusals remain visible. No useful admissible
continuation yields executed STOP with residual when cleanup is executable.
Source-supported mechanical, thermal and blood-generation evaluators also enter
the required comparison objectives; a continuation cannot drop one to gain an
apparent zero-cost advantage. Unavailable source evaluators remain unavailable.
Only the nominal deterministic scenario is supported by this milestone; it does
not claim uncertainty calibration or clinical optimality.

Policy snapshots carry the exact schema, ID, and content hash. Historical partial
protection snapshots cannot masquerade as this workflow. Unchanged historical
research scenes retain their existing admission semantics. The resident C++ engine
does not consume this policy and is explicitly declined; compiled numerical and
trajectory kernels still run under the Python policy authority. Verified replay
caches include the policy and outcome implementation identities. Native spatial
learners remain incompatible with this task; the compatible RL path is evaluation
only and preserves its existing physics qualification holds.

The design uses the neurological-function priority in the
[EANO guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC7904519/) and the
tumor-specific scope of the [EANS–EANO guideline](https://pubmed.ncbi.nlm.nih.gov/40973061/)
as reasons to separate eligibility from treatment preference. It imports no
clinical thresholds. Reward-independent enforcement follows the engineering
principle in [Safe RL via Shielding](https://arxiv.org/abs/1708.08611).
[Tractography limitations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8280000/)
also motivate retaining unavailable functional outcomes rather than treating
an anatomical membership as a postoperative deficit evaluator.

Focused checks:

```sh
.venv/bin/python -m pytest -q tests/test_surgical_procedure_policy.py tests/test_surgical_procedure_comparison.py tests/test_surgical_rl_procedure.py
RESSECTIONLAB_NUMERICAL_BACKEND=reference .venv/bin/python -m pytest -q tests/test_surgical_procedure_policy.py tests/test_surgical_procedure_comparison.py
RESSECTIONLAB_NUMERICAL_BACKEND=cpp .venv/bin/python -m pytest -q tests/test_surgical_procedure_policy.py tests/test_surgical_procedure_comparison.py
```
