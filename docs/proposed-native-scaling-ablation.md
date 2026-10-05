# Native actor feature-unit study

Status: reviewed design and [machine-readable declaration](../manifests/experiments/procedural-native-feature-units-v1.json); implementation, tests, independent audit and root execution release are still required. No registered training has run. This replaces the unexecuted joint feature/return-scaling draft.

The comparison is **RAW versus FEATURE_UNITS only**. It asks whether fixed actor input units change bounded scratch learning or procedural transfer in the same physical planning problem. Returns, critic inputs and outputs, critic initialization, value weight 0.5, entropy weight 0.01, global gradient clip 5, Adam learning rate 0.003 and network width 16 stay unchanged. There is no 100-unit return transformation.

The earlier proposed scalar-loss rescaling did not preserve parameter-gradient balance: rescaling the critic output layer changes its coordinate derivatives differently from actor and critic hidden-layer derivatives. Adjusting one global clip threshold cannot undo that difference. A feature-only intervention avoids that confound, although it still changes both initial actor behavior and subsequent gradient conditioning. It does not prove that tanh saturation alone caused the earlier results.

## Fixed inputs and preserved planning problem

The declaration inherits the exact [v2 physical design](../manifests/experiments/procedural-native-to-ucsf-v1.json): two procedural families, one previously studied UCSF-PDGM-0004 structural mirror, four candidate rays, two generic tools, three-removal horizon, source grid, rewards and optimization/selection panels. Geometry and world source hashes remain pinned. No final/stress worlds are opened and no additional proposals are introduced.

| Action feature | FEATURE_UNITS divisor | Source of reference unit |
|---|---:|---|
| Target/normal volumes, motor/language spatial integrals and their three partial-contact counterparts | 648 mm³ | The full 9 × 9 × 8 procedural tissue support at 1 mm³ per cell |
| Insertion distance | 120 mm | Maximum declared generic-tool working length |
| Tip radius | 2.25 mm | Maximum declared generic-tool active-tip radius |
| Shaft radius | 1.10 mm | Maximum declared generic-tool shaft radius |
| Binary and fractional features | 1 | Unchanged dimensionless quantities |

The ordered vector is `[1,648,648,648,648,120,2.25,1.10,1,1,1,1,648,648,648]`; all six state divisors are one. The profile version is `native-fixed-action-units-v1`. These are fixed reference units, not bounds on patient measurements or functional surrogate integrals. No patient extrema, running statistics, clipping or centering are used. The historical `depth` feature remains the fraction of the action budget consumed. Critic state aliasing and the fixed proposal-coverage ceiling remain unchanged.

RAW preserves the legacy identity forward path and state keys. Paired profiles use identical seeded **trainable tensors**; their action probabilities and initial returns may differ intentionally. Record both initial and selected performance. FEATURE_UNITS stores its divisor vector in a nontrainable persistent buffer. Checkpoint, resume, frozen and adapted validation bind the profile identifier/version, exact named feature order, divisors, state identity and profile hash. Dimensions alone do not establish compatibility. Reject profile mismatch or altered buffers. Legacy checkpoints lacking profile metadata are accepted only as RAW without a divisor buffer. Preserve existing analytic population guards.

## Fixed run order and costs

Run two fresh offline pretrainings, RAW then FEATURE_UNITS, both seed 101. Run shared STOP/GREEDY/SEARCH once under the unchanged physical model and inspect one frozen checkpoint per profile, RAW then FEATURE_UNITS. For each seed 11, 23 and 47, execute **RAW scratch → FEATURE_UNITS scratch → RAW adapted → FEATURE_UNITS adapted**. This order is fixed before execution. Use private cold simulator state for every arm; learned checkpoints and optimizer state never cross profiles or seeds.

Each offline run retains the 60 s / 32 actual Adam update / 256 optimization-transition caps. Each online scratch/adapted run retains 30 s / 32 update / 256 optimization-transition caps; two episodes per update, selection every two updates, and four episode slots including STOP. The 12 online learning runs, two offline runs and two frozen runs remain in the denominator if interrupted or failed. Retain six scratch initial candidates, 12 selected candidates, two frozen candidates and three shared baselines: **23 candidate identities** before any deduplicated geometry audit.

The comparison matches wall allowances, not executed updates or total transitions. Initial and later selection consume the online learning clock; initialization and cold setup are separately measured. Report full offline and per-arm costs, actual optimization/selection transition counts, deadline overshoot, checkpoint loading, export, extraction and independent validation. Shared baseline sequences have one explicit identity and original measured cost; their cost is neither hidden nor charged twice. Run RAW again under the same new frozen implementation rather than using the historical v2 timing as its only comparator. Heavy parallel work pauses during timing, while normal desktop/OS background load remains uncontrolled and is recorded.

## Gates and interpretation

Before gradients, validate exact patient identity, geometry/world/configuration and both profiles for all three seeds; verify paired trainable tensors and identical initial critic outputs. Test named scaling, unchanged masks/rewards/removal, legacy RAW equivalence, malformed or mismatched profile checkpoints, resume, fresh adapted optimizers, zero-update frozen arms and incomplete-selection refusal. An independent audit precedes execution. Commit the declaration, freeze the tested implementation and exact source bytes, and obtain root's execution release. Failed attempts are preserved before any implementation repair or new attempt declaration.

Report per-seed FEATURE_UNITS-minus-RAW physical selection return separately for scratch and adaptation, improvement from each initialization, and cost/performance relative to shared SEARCH. Entropy, ranking, tanh activation ranges, critic predictions, gradient statistics and completed updates are secondary diagnostics. Scaling the entire pipeline cannot attribute an effect separately to offline versus online learning. Preserve negative results. One reused structural case and three optimizer seeds support no patient-level significance, clinical safety, uncertainty-calibration or population-generalization claim; motor/language evidence and reviewed anatomy remain missing.
