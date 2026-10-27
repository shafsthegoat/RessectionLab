# Source-bound sector strategies and learned execution

The desktop can compare access corridors, target-sector order and the active
CUSA hand, then execute a frozen hierarchical policy on a selected corridor.
All tissue effects use the shared native surgical engine. Missing skull,
vessels and functional anatomy remain explicitly unsupported; these are
bounded development simulations, not complete operations or clinical guidance.

## Compare executable strategies

Start from an intact prepared environment. In **Research Planning**, acknowledge
the source assumptions and generate the three access approaches. Modeled brain
corridors require the existing explicit opt-in and removal cap. In **Resection
order and tool arrangement**, generate the sector/hand comparison. It retains
every attempted route × order × hand combination, including failed preconditions,
refused actions and incomplete withdrawal. It keeps the access comparison and
prior native branches intact.

Two disjoint target sectors are declared from source geometry before execution.
The fixed task requires at least 30 mm³ additional target removal after access,
at least 10 mm³ in each sector, two admitted sector strokes, no more than
20 mm³ additional and 200 mm³ total non-target removal, and checked cleanup.
The common limits are 48 native actions per strategy and 120 seconds aggregate
comparison computation. Requested order must agree with actual first progress
in the source sectors; a swept stroke that removes the other sector first does
not pass the ordering gate.

Each table row opens its recorded sector boundary. The full timeline includes
the intact source, access and both persistent tool slots. Use **Play replay**,
pause, the action timeline and **Inspect strategy endpoint** to inspect actual
native states. **Recorded model consequences** reports residual target location,
per-label removal, working-tool cell-seconds, available mechanics/heat and local
fluid ledgers. Working-tool exposure is not complete swept-shaft or bimanual
contact measurement. Missing measures remain unavailable; neurological risk and
global perfusion are unknown.

The CLI equivalent uses a saved source access comparison:

```sh
.venv/bin/python scripts/compare_anatomy_sectors.py \
  --case "$SOURCE_ACCESS_CASE" --output "$NEW_TESTCASES_OUTPUT"
```

## Fit and audit the hierarchy

The actor chooses an access hand at the intact boundary, then near/far sector
options and STOP from current published observations. Each option appends native
actions; it cannot replace an executed prefix or silently switch hands after
access. Checked duration retries and reverse source-waypoint withdrawal retain
their refusals and actual costs.

Before admitting a sector option, the controller executes the proposed macro
and its checked STOP on detached native branches. It admits the macro only if
the complete option and withdrawal pass the same tissue and action limits.
A refused check preserves the current accepted prefix. The comparison shows
these exit checks separately from the recorded trajectory. Discarded probe
previews and transitions count toward computation costs; they receive no
removal credit. The actor receives no prospective outcome or feasibility mask.
Its saved decision evidence retains observation identities, controller refusals
and forced cleanup separately from policy-selected STOP.

The development runner first measures scripted, bounded search and untrained
controls. It fits BC from a successful native demonstration, checks frozen BC,
then performs actual PPO updates and cold/native audits for three fixed seeds.
It retains failed seeds and transfer outcomes in the denominator:

```sh
.venv/bin/python scripts/run_patient_sector_learning.py \
  --train-source "$DECLARED_DEVELOPMENT_SOURCE" \
  --transfer-source "$SECOND_DECLARED_DEVELOPMENT_SOURCE" \
  --source-role DEVELOPMENT --fit --output "$NEW_TESTCASES_OUTPUT"
```

Simulator experience is generated training data, not patient observations.
The actor receives local fields, declared sector geometry and task ledgers,
plus published tool/goal state including normalized absolute coordinates.
Source/frame shortcuts remain possible. Matched same-case success and actor
weight changes do not establish PPO benefit, transfer or clinical validity.

## Execute a checkpoint anew

Return to an intact source, choose the access approach in **Run a learned
continuation**, click **Run frozen sector checkpoint**, and select a compatible
hierarchical checkpoint. The desktop snapshots its input, runs new frozen
observed-state decisions with zero optimizer updates, independently replays the
history, and opens the saved native outcome. Task and cleanup status remain
separate. **Cancel frozen policy execution** preserves the source. Failed display
admission restores the input snapshot.

The single-checkpoint CLI path has no fitting option:

```sh
.venv/bin/python scripts/execute_patient_sector_policy.py \
  --source "$SOURCE_ACCESS_CASE" --access-method superior-access \
  --checkpoint "$COMPATIBLE_CHECKPOINT" --source-role DEVELOPMENT \
  --output "$NEW_TESTCASES_OUTPUT"
```

Checkpoints bind producer code, runtime, action/observation schema and fitting
receipts. Use matching source snapshots. This action currently requires the
local checkout and matching runtime. The separate **Run frozen CUSA checkpoint**
retains the earlier stroke-policy interface and uses its own checkpoint family.

The integrated Electron verification harness runs with `--automation-hidden`.
It captures rendered pages with `stayHidden: true` and fails if its window was
ever shown. This keeps automated inference, playback and save checks off the
user's monitors. For an intentionally visible session, `--secondary-display`
selects a nonprimary display before creating the window and refuses to open
when none is connected.

Local dataset inventories, versioned experiment results, failed attempts and
desktop evidence are under ignored `testcases/`; consult its README. Portable
bundles carry exact source/weights/checkpoint/trajectory bytes and a `VERIFY.md`
with matching-runtime reproduction commands. Existing historical bundles are
kept separately.
