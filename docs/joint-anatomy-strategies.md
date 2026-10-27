# Joint corridor and sector strategies

The joint planner selects source corridors with the intended resection strokes
in view, then executes each candidate in the existing native engine. Route
ranking is deterministic source geometry. It is separate from the learned
hand/sector policy and does not estimate clinical risk.

## Generate and inspect

Use an intact prepared workspace with saved access and sector comparisons:

```sh
.venv/bin/python scripts/compare_joint_anatomy_strategies.py \
  --case "$INTACT_SOURCE_WITH_COMPARISONS" \
  --output "$NEW_TESTCASES_DIRECTORY"
```

The output directory must be new and under the checkout's ignored `testcases/`.
The original archive is read-only. The derived `joint-comparison.ressectionlab`
retains every original branch and parent, archives the original comparison pair,
and adds new access, sector and stopped-prefix branches. Save/reopen verification
replays every new branch through the same native-call meter.

Open the derived file in the desktop and choose **Rehearse**. **Deterministic
joint strategies** shows the selected geometry count, native access and sector
results, checked cleanup and unmet gates. Recorded-start and endpoint buttons
open actual native states; **Play replay**, pause and the timeline inspect the
authoritative trajectory. **Original classic comparison** keeps the stronger
original alternatives directly inspectable. Stopped branches are explicitly
unpassed and are not admitted strategy alternatives.

The automated integrated Electron verifier runs hidden and records whether its
window was ever shown. It must keep the user's primary monitor free.

## Source selection and physical checks

The source-only scan has fixed bounds: up to 4,096 exterior mapped-brain samples,
eight nearby target-boundary candidates and 32,768 pairs. It retains at most one
route per native entry axis and requires both 12 mm source-surface entry
separation and 30 degrees between inward directions. Inadequate or missing
routes remain inadequate; the planner does not synthesize a third entry.

The score includes the intended far transit, far working region and return to
the near sector. Source-contact differences beyond the estimated corridor are
ranked against the declared target-sector and non-target thresholds, followed
by whole-corridor/shaft burden, length and action headroom. These undeformed
source measurements are heuristics, not upper bounds or predictions of native
removal. Selection is sealed before native execution and makes zero physics
calls. None of these scores becomes a policy observation or a future-success
mask.

Every selected route then uses the existing access controller, full-tool and
other-hand checks, duration-aware retries, and a fixed far-first left-CUSA
sector strategy. The original task remains 30 mm³ additional target, 10 mm³
in each source-defined sector, two admitted strokes, at most 20 mm³ additional
and 200 mm³ total non-target removal, ordered progress, 48 native actions and
checked cleanup. Sector macro/exit preflight can refuse the complete option
while retaining its accepted prefix. Refused probes receive no removal credit.

One ledger charges public previews, direct steps, discarded probe transitions,
failed-path recovery, independent native replay and export replay. A preview's
internal step is not charged twice. The global limits are 1,500 physics calls
and 120 seconds end to end, with 288 calls reserved for up to six exported
branches and separate failure-recovery headroom. Trace and failure receipts
are saved in the sealed joint report. A failed withdrawal stays failed.

## Scope and limitations

The present joint mode requires an intact diagonal-grid scene with a mapped
brain and target, and no supplied skull. It claims no craniotomy or complete
operation. Missing vessels, functional boundaries and unreviewed reconstruction
remain unknown. Geometric source coverage does not establish executable success;
native modeled success is still not physical or clinical validation.

New source/controller code changes the authoritative runtime fingerprint.
Historical checkpoints require their exact matching producer runtime; use the
versioned reproduction package for historical execution. A newly selected joint
route needs an explicitly selected, source-bound and verified checkpoint before
its continuation can be described as learned execution. The joint comparison
itself performs zero optimizer updates.

Versioned patient results, failed experiments, native-call audits and desktop
evidence belong under ignored `testcases/`; consult that directory's README.
