# Executed experiment log

All entries below are development evidence. No final held-out patient benchmark
has been run. The patient remains the independent unit for population claims.

## October 4, 2026 — local runtime probe

`scripts/benchmark_runtime.py` ran on the Apple M5 with two PyTorch intra-op
threads and one inter-op thread. Exact workload definitions, warmups, all timed
samples and environment versions are in `docs/runtime-benchmark.json`.

| Synthetic workload | Median | p95 |
|---|---:|---:|
| Distance transform, 96³ grid | 40.34 ms | 41.65 ms |
| Distance transform, 160³ grid | 202.34 ms | 208.51 ms |
| Capsule-distance proxy, 400,000 point/segment pairs | 12.76 ms | 13.14 ms |
| Compact policy-style Adam update, CPU, batch 128 | 0.495 ms | 0.618 ms |
| Same update, MPS, batch 128 | 2.041 ms | 2.284 ms |
| Compact policy-style Adam update, CPU, batch 512 | 0.823 ms | 0.841 ms |
| Same update, MPS, batch 512 | 1.985 ms | 2.343 ms |

MPS execution succeeded, but CPU was faster for this workload. The initial
compact policy will use CPU. These are runtime microbenchmarks, not planning
quality, completed patient RL, or validated interaction-latency results. Distance
transforms must run outside the UI thread and be cached by input version.

## October 4, 2026 — initial independent contract attacks

The orchestrator reran `pytest tests/test_core.py tests/test_adversarial.py -q`
against the in-progress implementation: **26 passed, 10 failed**. The failures
showed that optional string metadata accepted mutable containers and malformed
halo masks silently promoted NaN, infinity, negative or fractional values to
anatomical evidence. Module owners are fixing the implementation; passing unit
tests alone had not exposed these issues. No scientific result relies on this
failed state.

A further review identified that hashing future context values into simulation
seeds could alter primary optimization. Separate permitted-input planning hashes
are being introduced while retaining full provenance hashes for audit/staleness.
The initial sequential fixture exposed only STOP; its insertion-boundary rule
is under correction before any patient learning result can be accepted.

## October 4, 2026 — acquisition findings

TCIA's current UCSF-PDGM v5 fixes diffusion metadata compared with v4 cited by the
specification. Its public transfer service reported unavailable files. A pinned,
hashed public structural mirror unblocks viewer development; equivalence to the
official source bytes remains unverified and is disclosed in the case manifest.

A separately acquired OpenNeuro BTC glioma case includes directional diffusion
and gradients, verified against release git-annex hashes. Its source tumor map
is fractional and has different voxel-axis ordering from T1. Strict categorical
import correctly requires an explicit, recorded derivation/alignment step.
Neither source is treated as clinical functional-outcome ground truth.

## October 4, 2026 — executed search and actual learning

The connected synthetic paid-access fixture requires paying normal-tissue cost
before reaching two target cells. SEARCH reaches the known return 1.74 in seven
transitions; greedy STOP returns zero. Three scratch REINFORCE seeds reach 1.74
after real updates, but require roughly 963–986 optimization transitions and
1.72–2.63 seconds. SEARCH took about 0.0025 seconds. This establishes no RL
advantage. Raw records and exact implementation snapshots are retained in
`artifacts/learning/connected-synthetic-v1/`.

The later branching fixture introduces competing tool paths and harmful
continuations. The development screen saved its configuration before execution,
used seeds 11/23/47 and never opened final/stress worlds. Three bundled
width/learning-rate/entropy configurations ran on this fixture and one UCSF
coarse structural case: **18 runs, 32 actual gradient updates each**. Bundling
settings prevents causal attribution to width alone.

| Development scenario | SEARCH score | Scratch selected scores, configuration A / B / C |
|---|---:|---|
| Branching analytic fixture | 1.30 | [0, 0, 1.26] / [0.08, 1.24, 0.08] / [0, 0, 0] |
| UCSF coarse model, subsequently invalidated | 1473.04 | [1429.82, 1429.82, 1429.78] / [1429.78, 1429.78, 1429.78] / [1473.04, 1429.78, 1429.80] |

Every run changed actor weights. No learned candidate beat SEARCH; one tied it.
SEARCH took about 0.049–0.057 seconds on branching and 0.456–0.480 seconds on
UCSF, compared with 0.339–1.717 and 4.800–7.381 seconds of training respectively.
These are saved development-run timings, not complete end-to-end planning
latency or an externally validated patient benchmark. Raw records:
`artifacts/learning/development-config-screen-v2/`.

Six branching follow-ons increased training to 128 updates and compared entropy
weights 0.01/0.05. Both retained [0, 0, 1.26]. Diagnostics found nonzero actor
gradients and increasing STOP probability in seeds 11/23. Of 256 sampled
optimization-world episodes, the later checkpoints stopped immediately in
235/228 episodes and produced only one positive episode each. Harmful sampled
continuations made local abstention attractive. Extra width, updates and this
entropy increase did not resolve the exploration problem. Records:
`artifacts/learning/branching-budget-entropy-followon-v2/`.

## October 4, 2026 — independent native-grid falsification

The separate checker accepted the UCSF candidates under their declared coarse
grid, but rejected **every non-STOP sequence** when removal was checked against
original 1 mm image cells and the actual distal tool capsule. First action:
92 mm³ of claimed tissue, zero fully contained source-cell volume, first
offending index (180, 150, 102), world RAS (-180, 89, 102) mm. Across retained
sequences, 1604–1820 mm³ of claimed volume lacked footprint containment.

This invalidates the coarse UCSF scores as physical resection performance; it
does not invalidate their use as a negative software experiment. STOP alone
passed this check. The footprint audit does not independently certify the
complete tool trajectory. Exact checker source and raw per-sequence failures are
saved beside the development screen. Native-grid connected removal, causal
microstep checks and a separate native-history evaluator are the corrective
vertical slice. The app must not present these rejected histories as approved
removed/residual-volume replay.

## October 4, 2026 — source diagnostics and packaged UI

BTC `sub-PAT28` raw-data tensor fitting ran on 101,311 voxels, excluding b2800;
FA median/p95 were 0.1939/0.6408 and median normalized signal RMSE was 0.06359.
CSA crop tracking retained 176 unlabeled diagnostic paths. Signal fit and
sampling repeatability are not anatomical accuracy. Strict planning rejected
uncorrected DWI. Reports and inspected overlay:
`artifacts/diffusion/PAT28-diagnostic-v2/`; reproduction and caveats:
`docs/diffusion-pipeline.md`.

The initial standalone arm64 application passed dependency, real MRI/3-D window,
internal-link and native-library checks. Local ad hoc signing is verified;
notarization and another physical Mac are untested. First package size was
1.28 GB; a prior failed Cocoa/VTK render-loop attempt is retained in build logs.
Interactive real-case inspection loaded the UCSF bundle in 0.30 seconds and
generated 54 routes in 1.32 seconds. Twelve geometric Pareto alternatives were
shown with source MRI and full-tool overlay. Screenshot review identified
clipped slice scale labels, opaque route IDs and poor accessibility of route
selection; fixes are underway. These are research usability observations.
