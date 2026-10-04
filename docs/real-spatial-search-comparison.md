# First real-only comparison runner

`scripts/compare_real_spatial_search.py` currently performs only a bounded PAT05
SEARCH/untrained-policy comparison. Optimizer execution is absent. A declaration
requesting nonzero updates, another method, another patient, a changed cohort or
unbound source is rejected. No population training, SELECT loading, policy
selection or external evaluation is enabled by this first slice.

The default CLI action validates a frozen declaration and numerical source
closure without opening a patient bundle. Actual execution additionally requires
`--execute`, the exact `--expected-declaration-sha256`, and a new output directory.
The prospective `manifests/experiments/pat05-real-spatial-search-comparison-v1.json`
binds the measured nominal64 configuration, lazy beam width 2, at most 512 search
transitions, 180 seconds of search and a 300-second/6-GiB process limit. It remains
unexecuted pending review, commit and explicit release. The supporting JSON
coverage receipt and source-profile declaration are both verified by path/hash.

The declaration binds the original BTC cohort, exact PAT05 bundle and semantic
hash, source-bound provisional support acknowledgment, access, expected source
and native frame record, shared tools/objective/adapter settings, explicit policy
configuration and architecture hash. It must name exactly `SEARCH` followed by
`untrained_policy`, and set `settings.optimizer_updates` to integer zero. Explicit
seed, horizon, RSS, whole-run wall time, online episode time, search transition
cap, beam width, search time and eager/lazy transition mode are required.

The same source, native task and objective are used by both methods. The initial
candidate inventory, nominal target/crop coverage and proposal coverage are
recorded before either method. This is not identical representation: search can
read full permitted nominal fields while the actor sees its declared crop. The
comparison records that limitation. Inputs remain annotation-assisted, functional
and vascular evidence remain unassessed, and rigid-cell removal does not establish
validated incision, deformation, retraction or instrument-force mechanics.

SEARCH returns a bounded nominal-model sequence and replays it on a fresh native
task. The policy follows its untrained deterministic actions on another fresh
task. Both actual histories receive the independent native audit. Shared setup,
online work, audit time and native previews are reported separately. Shared
initial diagnostics precede online timing; neither method adds per-step coverage
diagnostics. Call-cap truncation and weak STOP incumbents remain visible. Search
timeouts preserve partial accounting as failures and never become fabricated
teacher routes. Method starts are saved before long work, and every completed
replay transition is retained in order, including explicit committed interruptions.
Method failure also preserves cumulative native preview work.

The existing process supervisor enforces whole-run wall/RSS bounds and preserves
partial output. Completed runs verify unchanged model state and numerical source.
Source integrity, anatomy eligibility and audit failures remain failures; this
runner does not alter masks, substitute patients or manufacture successful data.

The committed six-TRAIN/two-SELECT role and batch helpers remain unchanged for a
later explicitly scoped extension. PAT16/PAT20 support conflicts and the original
six attempted-patient denominator must remain visible if that extension proceeds.
The current user direction prioritizes validated tissue mechanics before further
optimizer execution or population-model sweeps.
