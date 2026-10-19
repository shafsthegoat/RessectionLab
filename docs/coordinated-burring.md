# Bounded two-instrument learning demonstration

The task is **shallow bone burring followed by debris aspiration** on the existing
sub-06 development anatomy. It uses the unchanged `SurgicalEngine`, V2 work and
thermal mechanics, full-tool collision checks, source arrays, strict protection
contract and native replay. It does not complete a craniotomy or remove tumor.

Success requires 30–40 mm³ of bone removed, at most 1 mm³ residual debris, and
zero soft-tissue/tumor removal, vessel damage, generated bleeding or protected
removal. Limits are checked after each public action and can overshoot within
the final action; an overshoot fails the task. Existing collision/protection
guards still act at microsteps. Bone heating is recorded, not clinically
calibrated or asserted safe. Missing anatomy remains unknown.

The fixed BURR and CLEAR macros use the existing checked approach and explicitly
park one instrument before inserting the other through the same corridor. The
policy chooses STOP, BURR or CLEAR, with at most four decisions. STOP deactivates
both hands through public actions. The policy sees only bone/debris simulator
counters and remaining decisions. Those counters are declared simulator
observations, not assumed surgical sensors. Motion and spatial targeting remain
scripted; this demonstrates **learned sequential instrument scheduling**.

The causal necessity check runs before fitting: BURR creates 34 mm³ of debris;
CLEAR aspirates it. BURR alone leaves debris and repeated burring exceeds the
bone cap. CLEAR alone removes no bone. CLEAR → BURR → STOP leaves newly created
debris. Thus either instrument alone and the reversed one-pass sequence fail;
BURR → CLEAR succeeds. There is no added rule that refuses burring when suction
is absent, and no new physical effect is introduced to force coordination.

Three tabular Q learners start from zero values, with STOP resolving ties.
Each gets 48 epsilon-greedy episodes, epsilon 0.35, learning rate 0.25 and
discount 0.95. These settings and rewards are frozen before baseline execution.
Actual engine-generated rewards drive temporal-difference updates; no expert
action labels or initialization from the successful script are used. Exact
deterministic prefixes are cached by complete action history. Cache reuse is
reported separately from new engine transitions. The cache never merges states
based on the policy's aggregate observation.

All three frozen policies learn BURR → CLEAR: 3/3 successful assessments versus
0/3 untrained and 8/30 seeded random assessments. Learned return is 1.2916 in
each seed, matching the script. Random mean return is 0.2516733. Q updates are
92, 93 and 93; the read-only verifier independently reconstructs every update
from stored trajectories and exactly matches all parameter tables. These are
repeats of one deterministic development task, not independent patients,
held-out anatomy, transfer evidence or learned continuous bimanual control.

Reproduce locally using the already prepared pinned source:

```sh
.venv/bin/python scripts/train_coordinated_burring.py --output testcases/coordinated-burring-new-attempt
.venv/bin/python scripts/verify_coordinated_burring.py testcases/coordinated-burring-run03
.venv/bin/python -m pytest tests/test_coordinated_burring.py -q
./scripts/launch_poc.sh --case "$PWD/testcases/coordinated-burring-run03/learned-seed-17.ressectionlab"
```

In **Rehearse**, the native timeline has ten public actions. State 5 is after
burring; state 10 is after aspiration. Both instruments, acquired source slices,
anatomy, strict exclusions and play/scrub remain in the existing interface. Each
seed has an exact fresh-engine replay, fresh-process native reopen and independent
source-cell/material and debris-mass accounting. All source arrays are unchanged.
The local telemetry export preserves microstep motion and effects for all seeds.

The initial shorter drill-intent variant failed the baseline gate (3 mm³ rather
than 30) and ran zero updates. Two later export failures exposed plain-case
bundles without prior toolbox/workspace records. The identical repeat fit
produced identical checkpoint hashes for all seeds; export recovery then used
those frozen policies with zero extra updates. Earlier attempts remain locally
preserved. The successful fit/assessment took 27.998 s and 69 measured engine
calls; export recovery took 54.429 s and 100 calls including replay/telemetry.
Interrupted-export cost was not captured, so these are not total campaign costs.
The prior surgical pilot's negative result is unchanged.

The [saved numerical summary](../artifacts/coordinated-burring-v1/summary.json)
retains all assessment rows, seeds, checkpoint hashes, ablations and measured
cost scopes. Generated cases, complete learning trajectories, frozen protocol,
native saves, telemetry and local testing records are under ignored `testcases/`.
The implementation stops at this measured result; no broader task, anatomy
change, population fit or further parameter sweep is included.
