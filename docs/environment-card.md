# RessectionLab environment and model card

Recorded October 4, 2026. **Local research prototype.** This card describes
implemented behavior and retained evidence, not a surgical recommendation or a
claim of clinical readiness. Requirements remain in the three specification
documents; ongoing work is tracked in [PROJECT_STATUS](../PROJECT_STATUS.md).

## Identified application and purpose

The inspected arm64 Electron application uses renderer `c194a8b` (source digest
`762973e2…`) and the separately frozen Python `0b4e334` engine (executable SHA256
`85a92eff…`). Its [native validation receipt](../artifacts/electron-priors-v1/native-workflow-final.json)
contains the full identities and exact scope: final focused MRI/prior checks,
487 independent GPU sampling checks, and earlier seven-layer, save/reopen and
historical-replay checks on the same engine. No new training was performed in
that final UI inspection. Later learner changes and the feature-unit study are
outside this application snapshot; this card claims no outcome from that study.

Supported uses are local inspection of MRI and source annotations, comparison
of declared complete-tool routes, inspection of unreviewed evidence proposals,
bounded patient-simulation refinement, and source-bound replay/export. The
desktop's selected-route learner chooses between STOP and its declared native
stroke; it does not learn a free-form surgical trajectory. Numerical development
also explores small finite multi-stroke inventories. Neither supplies a complete
resection plan, robotic control, intraoperative navigation or postoperative
functional prediction.

## Meaning of the reported quantities

| Quantity | Implemented meaning and limit |
|---|---|
| Route accessibility | Target voxel centers inside the active-tip swept tube, conditional on the static route. The tissue array is unchanged and simulated removal remains null. |
| Route normal-tissue exposure | Conservative unique-voxel overlap of the complete swept tool with supplied non-target support. It is neither removal nor injury. |
| Simulated removal | Native cells fully contained by the active distal sweep and connected to exterior/cavity. Target and modeled non-target volumes are reported separately. “Normal” means outside the supplied target annotation, not histologically verified normal tissue. |
| Partial contact | Contact without full-cell removal. A contacted cell remains occupied and earns no removal credit. Cumulative contact, currently retained contacted tissue, and previously contacted tissue later removed are separate quantities. Do not add overlapping categories into “total injury.” |
| Return | A declared research preference combining target credit, normal/functional surrogate costs, partial-contact costs and action/tool costs. It is not a clinical utility or probability. |
| Clinical deficit probability | Always null, with `no_validated_clinical_outcome_model`. Missing motor/language evidence remains unassessed, not zero risk. |

Definitions are enforced in [route planning](../src/resectionlab/planning.py),
[native simulation](../src/resectionlab/native_simulation.py) and
[native replay accounting](../src/resectionlab/native_refinement.py).

## Geometry, observations and uncertainty

The source voxel grid and physical affine remain authoritative. “Native” refers
to the imported release grid; original acquisition geometry may be unavailable.
Native removal supports orthogonal grids, including oblique rotations; unsupported shear is
rejected. Tools are rigid research profiles with declared tip, shaft, length and
angle dimensions, not verified commercial devices. Access is a hypothetical
disk, not an inferred skull opening or reviewed cortical surface. Each stroke
inserts along one straight line and fully retracts along it before reorientation.
No corridor is cleared initially. The swept shaft must clear the prior cavity
before new endpoint removal is credited. All eight physical cell corners must
fit inside the active sweep; partial boundary cells stay occupied. This can
reject a physically plausible maneuver at the available image resolution.
[Geometric contract](native-simulation.md),
[native engine](../src/resectionlab/native_resection.py).

The actor uses nominal evidence and a masked finite action list, with STOP always
available. One coherent latent registration world is sampled per episode;
hidden anatomy does not enter actor features or action masks. World assumptions,
reward weights, tools and action inventory are frozen during optimization.
Separate optimization, selection, final-evaluation and stress manifests exist.
Completed patient development comparisons used **zero perturbations**; distinct
seed IDs there do not demonstrate anatomical robustness. Final/stress worlds
remain unopened in those records.

| Uncertainty level | Current capability |
|---|---|
| H0: proximity | Backend physical-distance/Gaussian halo with an explicit research scale and coverage. It does not define a clinical safe margin. Unreviewed desktop priors do not enter route scoring. |
| H1: anatomical support | Backend counts membership across supplied reconstructions, retaining assessed counts and unknown coverage. No accepted patient functional reconstruction ensemble is demonstrated. |
| H2: plan events | Backend summaries retain event definition, counts, world/partition identity and finite Monte Carlo interval. Incomplete coverage yields unknown frequency; deterministic replay has no uncertainty interval. No calibrated patient injury probability is available. |

These are [world-model](../src/resectionlab/worlds.py) and
[evaluation](../src/resectionlab/evaluation.py) capabilities, not validation of
the underlying anatomy or uncertainty distribution. Brain shift, tissue forces,
vascular injury, microscopic infiltration and patient functional reorganization
are not modeled adequately for clinical decisions.

## Evidence and source boundaries

UCSF-PDGM-0004 uses pinned public structural-mirror bytes whose equivalence to
official TCIA remains unverified. Its research simulation uses an explicitly
estimated, unreviewed intracranial envelope. BTC full-head cases cannot obtain
cortical access from nonzero MRI or an unreviewed extraction mask. SynthStrip
main/no-CSF proposals remain separate from working anatomy; annotation overlap
checks are not segmentation accuracy scores. [Source registry](../manifests/cohort_registry.json),
[structural proposal inspection](../artifacts/electron-structural-proposals-v1/native-validation-final.json).

The seven motor/language layers remain **population priors requiring alignment
review**, with no planning eligibility. Their T1 registration, T1c display,
lesion labels, transform, values and coverage are bound separately. Coverage
means atlas field of view, not patient function. Covered zero, positive atlas
signal, outside coverage and incomplete interpolation support remain distinct.
Binary structural networks are not this patient's tracts; concordance values
are not BOLD intensity, p-values or deficit probabilities. Language dominance
remains unknown without patient evidence. Adding unused proposals changes audit
identity but not planning inputs. [Import receipt](../artifacts/desktop-priors/ucsf0004-prior-import-v1.json),
[functional evidence meanings](functional_evidence.md).

Context records retain source and availability timestamps. Unknown or
post-cutoff information is excluded from primary planning inputs and seed
identity. No genotype-to-injury or molecular-to-tool-tolerance rule is supplied.
[Context contract](../src/resectionlab/core.py).

## Validation, failure modes and reproducibility

- Static feasibility does not imply a cuttable corridor: all 54 original UCSF
  routes failed native action checks. Two separately declared native access/tool
  alternatives were introduced; blocked configurations now stop before learning.
  One exact-stroke workflow made 32 updates and changed selection return from
  0 to 171.58, with 174 target and 11 non-target mm³ independently checked.
  This was a workflow smoke test, not a comparative efficacy benchmark.
  [Exact-route receipt](../artifacts/desktop-native-bridge/ucsf0004-exact-selected-route-v1.json).
- The earlier coarse-removal pilot failed native-footprint checks; its returns
  are not valid resection performance. The corrected two-stroke source-grid
  example removed 249 target and 17 non-target mm³. Its 32 mm³ cumulative
  partial normal contact comprises 26 retained and 6 later removed mm³.
  [Checked replay/accounting](../artifacts/desktop-native-bridge/ucsf0004-30s-v1.json).
- The completed procedural-to-patient comparison found no learned arm better
  than SEARCH/GREEDY (245.24). It used one reused development patient, two
  procedural families, zero human pretraining patients and three optimizer
  seeds. All 13 frozen candidates passed geometric checks; the best removed
  only 249 of 41,919 target mm³. Actor weight changes alone do not establish
  improvement; one scratch run retained its initial selected policy.
  [Completed result and negative findings](../artifacts/learning/procedural-native-to-ucsf-v2/RESULT.md).

Cancellation is cooperative. Invalid stroke previews do not alter cavity state.
The local bridge preserves resumable checkpoints under the original budget and
source/runtime/configuration contract; unsigned crash recovery is refused.
Replay requires a completed selection result, source/model consistency and an
independent history check. Case or model changes invalidate dependent results;
imported certificates alone cannot authorize replay. Geometric certification
checks the declared simulator, not surgical safety. [Replay implementation](../src/resectionlab/native_refinement.py),
[bridge integrity and cancellation](../src/resectionlab/desktop_bridge.py).

The application runs locally through a narrow JSONL sidecar, without a listening
service or paid infrastructure. Local ad-hoc signing is verified; distribution
signing, notarization and a second physical Mac remain unvalidated. Data, model
and template redistribution terms and required notices remain separate release
work; the current package is not a completed distribution release.
[Release evidence audit](release-evidence-audit.md).
