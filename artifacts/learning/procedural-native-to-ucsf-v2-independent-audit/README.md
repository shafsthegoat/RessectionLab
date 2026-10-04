# Independent audit of the completed v2 artifacts

The retained artifacts pass the consistency and accounting checks in `audit.py`.
This audit reads saved records and checkpoint tensors and recounts source-cell
labels. It does not train a model, execute simulator transitions, or rerun the
independent collision checker. Its results are in `report.json`; the original
experiment directory is unchanged.

Verified provenance and model contract:

- All 138 frozen files match their recorded hashes. Numerical source bytes also
  match commit `68e4fde13b1fa13411e59af663bd17ae63885947` directly. The launch
  snapshot has no Git revision, but the worker revision and actual committed
  numerical bytes bind the implementation.
- The scientific declaration is byte-identical to commit `f26a72c`. The source
  bundle, target model, action inventory, objective, budgets, and permitted world
  records agree with that declaration across all arms.
- All six saved initial/latest/selected tensor hashes match the reported hashes.
  Scratch initialization reproduces its specified random seed. Every adapted
  run starts from the identical exported shared weights. Saved Adam step counts
  match the actual update counters, and all six actors changed.
- Each selected checkpoint is the earliest maximum among completed selection
  panels. Scratch seed 23 legitimately retained its initial checkpoint.
- All 13 candidates remain in the audit denominator, including STOP and the
  three initial policies. Eight distinct sequence certificates cover them; all
  report complete-tool and frontier checks with zero unsupported removal.

Source-cell removal, cumulative partial normal contact, normal removal, action
costs, and tool changes reproduce every reported candidate score. Partial cells
later removed remain distinguishable from retained partial contacts.

| Method | Seed 11 | Seed 23 | Seed 47 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Scratch | 245.24 | 171.42 | 245.16 | 220.607 |
| Procedural adapted | 171.42 | 171.42 | 245.17 | 196.003 |

The frozen procedural policy scores 139.62; SEARCH and GREEDY score 245.24.
Adaptation's descriptive mean is 24.603 points below scratch. No learned arm
exceeds search. Three optimization seeds within one previously studied patient
do not establish statistical or patient-population superiority.

The offline call costs 31.218 seconds and performs 32 updates, 169 optimization
transitions, and 188 selection transitions. Both procedural families contribute
32 gradient episodes, with no patient source among those episodes. Online
initialization is recorded outside the learning allowance; initial selection is
inside it. Cooperative overshoot ranges from 0.0028 to 0.2255 seconds.

Explicit, nonoverlapping measured phases total 285.281 seconds, leaving 1.909
seconds of worker orchestration within the 287.189-second worker duration.
The complete launch takes 288.873 seconds. Independent geometry validation costs
33.124 seconds; the largest recorded process peak is 3.317 GiB. These are v2
costs; the separately retained failed v1 attempt also consumed development work.

Saved contracts and replay seeds use optimization and selection roles only. The
frozen runner contains no final-evaluation path, and no final/stress result or
evaluation ledger was produced. Final/stress manifests are declarations, not
evidence of their execution.

Functional evidence remains missing, clinical probabilities remain null, the
working tissue envelope is unreviewed, and access is hypothetical. The geometric
certificates do not establish tissue mechanics or clinical safety. Sequential
arm order and uncontrolled system load limit runtime comparisons.
