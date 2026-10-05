# Synthetic scan-conditioned opening task

`spatial_task.py` is a small representation and sequence-learning experiment.
It composes the existing `SequentialSimulator`; no native or desktop model was
changed. A complete rigid instrument enters through one declared window, acts at
an exposed terminal footprint, and withdraws along the same line. The analytic
primitive removes **tip-intersected whole cells**, unlike the native engine's
contained-cell rule. Its independent check verifies this discrete geometry; it
does not validate native removal, deformation, incision, retraction or clinical
clearance. Policy transfer to the native engine remains separate work.

## Observations and private outcomes

The primary actor receives the shared six-channel spatial observation contract.
Available channels are generated structural intensity, support estimated as
intensity greater than 0.05, and previously committed removal. Target, motor and
language channels are unavailable, with false coverage and zero storage. The
remaining actor inputs are access/budget state and complete tool/entry/tip
geometry. There are no per-action target volumes, reward estimates, private
removed-target fractions or hidden functional summaries in actor tensors.

The numerical geometry engine receives the estimated support, declared access
and tools, and an **all-zero target-label volume**. Every observed frontier cell
is considered with every tool; no target-informed ordering, scan limit omission
or hidden hazard filter selects the candidates. Complete-tool access and current
cavity checks reject inaccessible primitives. The inventory records frontier ×
tool denominator, accepted IDs, geometry/cavity rejection count and zero
omissions. This is complete coverage of the declared local primitives, not all
possible surgical paths. Search and the policy receive the same inventory.

Private reference fields score committed cells in the evaluator. Source IDs
bind only permitted scan pixels and affine. Changing private labels, functional
fields or diagnostic world seed while retaining the same scan and action history
must preserve the full actor interface, including IDs, order, masks and metadata.

`planning_clone()` constructs a new case from the observed scan and replays only
the chosen action IDs. It copies no private outcomes, reward history or world
seed. Search uses a fixed synthetic class-midpoint estimator:

`target = intensity >= (.2 + (.2 + .30)) / 2 = .35`.

Normal signal 0.2 and minimum target contrast 0.30 are declared properties of the
generator. This supplies search with a strong, explicit observational prior; it
is not a universal MRI threshold. The first draft used 0.5, which was recognized
as weak near the generator's minimum contrast before any comparative experiment.
That draft was replaced before study declaration. A deliberately below-model
contrast fixture demonstrates that the current estimator can still be wrong.

## Shared objective and limitations

The first screen's objective is target removal minus normal removal, action,
tool-change and physical insertion/withdrawal effort. Motor, language and graph
weights must be zero for **every** method. A draft with private functional
penalties but no matched search estimate was rejected during review, before any
training or comparative result. The constructor now rejects that mismatch.

All removed volume is labeled function-unassessed. Private motor/language
surrogate exposure is a separate evaluator diagnostic; planning clones report it
as unavailable rather than zero observed risk. Clinical deficit probability is
null. This screen does not test functional preservation or least clinical harm.
World seeds vary only these excluded functional diagnostics, so they are not
independent geometric reward worlds. Generalization must be measured across
distinct anatomy shapes, with policy randomness labeled separately.

## Small API and checks

- `make_spatial_case(seed, morphology=..., tool_regime=...)` generates a 7×7×6
  scan from declared contrast, Gaussian noise, bias and support/lesion shapes.
  Morphologies are `paired_lobes` and `extended_lobes`; tool regimes are
  `reference` and `thick_shaft`. They permit a prospectively declared shape/tool
  shift without generating evaluation cases during development.
- `make_spatial_task(...)` provides `observation`, `reset(world_seed)`, `step`,
  `fresh`, evaluator `clone`, and observed-only `planning_clone`. Each STOP counts
  as a transition. The default horizon is six transitions.
- `metrics()` is evaluator-only. It records geometric reward, actual committed
  modeled volumes, separate unknown/functional diagnostics, insertion and
  withdrawal distance, candidate denominator and exact history. Distances are
  millimeters, not operative time.
- `anatomy_hash` binds support/target shapes and affine, excluding tools,
  intensity noise and all functional fields. It groups the eight XY rotations
  and reflections with Z and the affine fixed. All generated cases use the same
  identity affine; arbitrary affine reindexings are outside this grouping rule.
  A separate `reference_hash` retains the original unrotated support, target,
  affine and private diagnostic fields. Different seed labels alone do not prove
  held-out anatomy; experiment manifests must compare these canonical hashes.

The three-step `make_opening_task()` fixture is deliberately small enough to
enumerate every reachable action/STOP sequence. It has 7 nonterminal states,
32 terminal sequences and 38 transitions. The unique best path first removes two
normal cells and then one target cell. Its return is
`1 - 2*.2 - 3*.03 - .002*(.5 + sqrt(1.25) + sqrt(3.25))`, approximately
0.50315838. One-step greedy search chooses STOP with return zero. The complete
optimal sequence passes the separate whole-tool/frontier checker.

Owner tests also cover private-label swaps, scan-only planning isolation,
missing-channel flags, complete local candidate accounting, illegal travel,
clone/reset isolation, frozen reward, unknown exposure and physical effort.
These are synthetic development checks. They do not constitute a population
training result, held-out study, public-patient experiment or clinical validation.

Independent pre-freeze attacks found that replacing a task's access/tools or
changing its horizon could alter actor parameters while its geometry engine kept
the original configuration. The three failed checks and pre-repair source are
retained under `artifacts/spatial-task-development-v1`. The task now checks one
observed source/access/tool/horizon/reward contract before observations, actions,
reset and cloning, in addition to the underlying geometry fingerprint.
The private reference contract is frozen separately, so replacing evaluator
labels cannot silently change recorded anatomy identity or future reset outcomes.
That private hash never enters actor inputs or the observed decision-model hash.
