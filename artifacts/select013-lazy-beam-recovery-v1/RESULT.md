# Deferred geometry completes the exposed SELECT013 beam

One search-only execution completes all 24 planned layers in **79.720 seconds**,
evaluating 725 transitions and 47 expanded nodes. The previous eager execution
completed four layers before its 300.087-second cap. The recovered search still
selects **STOP**, with zero target or outside-target removal. This resolves the
computational cap for this search configuration; it establishes no useful target
progress, learned improvement, global infeasibility or clinical benefit.

The only search change is the existing `lazy_planning` transition: prepare a
successor's expensive geometry inventory when the retained node is expanded.
Current-action full-tool checks, native transition, rewards, public anatomy,
ordered candidates, cache condition, horizon 24, beam width 2, depth/volume
retention, 5,640-call and 300-second search limits remain fixed. The complete
initial world and the first four completed layer projections exactly match the
earlier execution. Later successor-availability fields are separately accounted
for, since more retained nodes are now expanded.

All 725 evaluated prefixes have negative cumulative return. The final retained
prefixes also have zero target progress. They are explored alternatives, not the
accepted patient's plan: their simulated outside-target removal must not be
attributed to the selected STOP. A finite beam does not enumerate every path.
The original incomplete four-arm experiment remains incomplete in its original
record; this separately completed beam does not revise the old gate or rerun the
three earlier greedy/IL/RL arms.

## Costs and checks

| Phase | Seconds | Native previews |
|---|---:|---:|
| Reference checkpoint reload | 0.013634 | 0 |
| Public task construction | 1.520596 | 117 |
| Search, including wrapper | 79.727863 | 5,382 |
| Selected-plan collection | 0.095119 | 0 |
| Fresh independent replay | 1.351520 | 117 |

The owned process takes 86.929393 seconds, including startup and supervision;
the worker reports 82.769673 seconds. Sampled peak is 561,709,056 bytes, output is
690,036 bytes, exit is zero and no owned process or cleanup error remains.
Total previews are 5,616. Planning constructs 46 successor inventories rather
than eagerly constructing one after every transition. Inclusive inventory,
transition and clone timers overlap and must not be added together.

The existing admission/collector needs one original IL64 checkpoint reference;
it performs **zero policy forwards, gradients or optimizer updates**. Search
authors the plan, which is sealed and independently replayed. The independent
saved audit passes **2,206 checks** across 151 source/JSON files, without loading
arrays/weights or rerunning the simulation (0.056609 seconds, 27,475,968 bytes).

Seven generated driver/parity controls pass in 1.89 seconds. Two earlier test
fixture failures are retained: a planning clone was incorrectly supplied as
the authoritative independent evaluator. The corrected control constructs fresh
authoritative tasks; the actual scientific runner already did so. Existing
canonical lazy/search controls pass 55 tests, with 17 historical-stage-dependent
skips. Those skips are not counted as validation of this run.

## Scope and reproduction

013 retains its SELECT role but was already development-exposed. Held SELECT037
remains in denominator two; EVAL067 and private references are unopened. Inputs
include supplied tumor and automatic cerebrum annotations, partial source
coverage, a declared post-exposure aperture and an unvalidated S-union-T material
assumption. No vascular/functional completeness or tissue-force fidelity follows.

`preservation-index.json` binds original bytes to the compact archived projection.
`observed_beam/selection.json` keeps the full layer ledger; duplicate copies in
the result and parent receipt are omitted only from their archival projections.
`runtime-source-map.json` identifies canonical sources at `fe857ee` and preserves
the six untracked experiment/helper Python files under `runtime-source/`.
The original result/receipt SHA256 values are respectively
`96e0b98069689a7ad6f1b986423f527ce931e04034b33605d3ebaf48e9e9d761` and
`83031cb5bdeb9c9ee6bd47dcf53f28e3c7ea7f83207bc94b174d8501b994f232`.

To reconstruct the source, restore the six experimental files to the original
repository-relative paths and use the pinned canonical commit and dependencies.
`HANDOFF.txt` records the exact freeze/launch interface. Original local data,
prior experiment records and authenticated checkpoint prerequisites named by
the source index are required; they are not redistributed here. The historical
release is an execution receipt, not permission to overwrite its output.

Next: distinguish candidate/tool reach from retention failure using the saved
plateau and public definitions before choosing one further planning contrast.
