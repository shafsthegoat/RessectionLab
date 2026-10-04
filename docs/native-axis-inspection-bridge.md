# Sidecar contract for neighboring-path inspection

`inspectAxisPlanning` exposes the [read-only axis facade](native-axis-inspection.md)
through the existing Python JSONL worker. This operation does not execute a path,
train, remove tissue, save a run, resume a checkpoint or authorize a population
policy. It does not add preview IDs to route, replay or training caches.

## Request

The operation accepts exactly these fields:

| Field | Contract |
|---|---|
| `caseHash` | Current cached immutable case identity, `sha256:<64 lowercase hex>`. |
| `planningHash` | Current planning identity with the same prefixed format. |
| `routeId` | Current-session feasible route identifier, at most 128 characters. |
| `routePlanningModelHash` | Exact cached route-model identity; existing routes use **bare** 64-character hex. |
| `toolIds` | Explicit selection of one or both `native-fine-aspiration` and `native-wide-aspiration`, without duplicates. |
| `acknowledgeNeighboringColumns` | Must be the Boolean `true`. |
| `acknowledgeEstimatedSupport` | Required Boolean; `true` is additionally necessary when the facade requires estimated-support acknowledgment. |
| `expectedBindingHash` | Optional prior inspection binding, using the prefixed SHA-256 format. A mismatch withholds the new result. |

The route supplies its **window only**. Its selected entry/target ray and selected
instrument are not retained as the only allowable stroke. The requested native
instruments are resolved from the server's unchanged catalog in catalog order.
Their dimensions and provenance are returned in the binding. Generic instrument
IDs, caller-provided geometry, model settings, evaluator results and execution
instructions are refused. Existing selected-route operations remain separate.

The server-owned v1 preset uses the thirteen declared source-voxel column offsets,
maximum 26 primary rays, RAW inputs, `max_steps=3`, 0.25-mm native microsteps,
partial-contact weight 0.05, default `RewardSpec`, and default
`WorldGeneratorConfig` with zero perturbation scales. Every effective setting is
retained in the raw binding. The configured horizon does not imply that any cut
was executed. No world partition or experimental role is created. Source-voxel
offsets are not millimeter displacements; spacing and affine determine those.

## Result

The camelCase envelope is:

```
{
  schemaVersion: 1,
  caseHash, planningHash, routeId, routePlanningModelHash,
  accessSource: "selected_route_window_only",
  anchorWindowRas: {center_mm, normal_inward, radius_mm, window_id},
  anchorWindowNormalization: {normalAbsoluteTolerance, normalMaximumDifference},
  requestedToolIds: [/* canonical server catalog order */],
  inspection: /* unchanged AxisPlanningInspection.to_dict() */
}
```

`anchorWindowRas` retains the exact cached window converted to RAS+. Constructing
an `AccessWindow` from its JSON can renormalize an already unit-length normal by
roundoff. Only an absolute component difference of four float64 epsilons is
allowed, with zero relative tolerance; centers, radius and window ID stay exact.
The observed difference and bound are explicit. The raw
`binding.requested_access_ras` retains the resulting typed boundary; the facade's
separate LPS conversion record describes requested-to-native normalization.
Neither step snaps source coordinates or changes unsupported access directions.

Before publication, the bridge compares the returned source identity, frame,
shape, canonical affine, requested window and actual native window directly with
its cached inputs. The declared normalization tolerance and measured difference
are checked against those inputs. It also compares the returned tool catalog,
proposal rule, reward, world generator, input profile and horizon with the exact
server-owned preset. A report's self-consistent hashes do not replace these
comparisons. An explicit `expectedBindingHash` must match the returned binding as
well as passing the facade's check.

The raw snake_case report and its hashes are preserved. Its `inventory` retains
all column/tool slots, omissions and primary/fallback attempts. `actions` retains
STOP plus every native-certified non-STOP preview in order, with exact geometry,
source/cavity/model provenance and unknowns. Proposed rays, preview attempts and
accepted actions are separate counts. A complete `no_actionable_moves` result
means this bounded initial inventory has no certified action; it does not mean
surgical inaccessibility. Rejection may reflect a numerical limit or unsupported
model condition rather than an anatomical obstacle.

Preview cell counts are unexecuted. Contact counts overlap contained cells and
must not be summed with them or relabeled as exclusive partial contact. Actual
transition, native-commit, gradient and removed-volume counts are zero.
`candidate_eligible` and `removal_authorized` remain false; clinical probability
remains null. No independent history certificate or clinical clearance is implied.

## Limits and lifecycle

Before numerical construction, admission checks enforce at most 16,000,000 source
voxels, the existing 1-GiB case-array limit, and 256 KiB for exported support-related
metadata, unknowns and structural evidence records. The complete result envelope
must fit 2 MiB before publication, below Electron's 8-MiB protocol limit. These
are input and message limits, not a measured peak-RSS bound or an OS allocation
limit. No transfer arrays, result cache or files are created by this operation.
An oversized result is withheld in full. A stale expected binding is checked by
the facade after preparing the complete model, so this refusal can still be costly.

The operation uses existing `started`, `progress`, `result`, `error` and
`cancelled` events with request IDs. Default request timeout is 120 seconds within
the existing allowed timeout range. Progress contains no partial inventory.
Cancellation or timeout sends one terminal event and suppresses late results;
ongoing native work drains at a cooperative boundary. No transition can commit
in this operation, so there is no canceled-but-committed removal to reconcile.
Source/window/tool/acknowledgment changes must invalidate a displayed inspection;
new inspection is always a deliberate user action.

Named refusals include `AXIS_ACCESS_UNSUPPORTED`,
`AXIS_SUPPORT_ACKNOWLEDGEMENT_REQUIRED`, `AXIS_SUPPORT_UNAVAILABLE`,
`AXIS_BINDING_CHANGED`, `AXIS_INPUT_SIZE_LIMIT`, `AXIS_METADATA_SIZE_LIMIT`,
`AXIS_RESULT_SIZE_LIMIT`, `AXIS_SOURCE_BINDING_MISMATCH` and
`AXIS_INCOMPLETE_INSPECTION`, alongside the existing
case/route/input/cancellation errors. Full-head anatomy and unreviewed extraction
restrictions remain independent of both acknowledgments. Generic search windows
may legitimately be nonaxial or have fractional transverse origins; the server
preserves that refusal rather than manufacturing a usable window.

Focused tests use tiny synthetic source grids. They cover complete wire geometry,
explicit tools and acknowledgments, stale sources/models, unsupported access,
RAS/LPS conversion, admission/result limits, no route/file/run mutation, refusal
of promoted/partial reports, and cancellation without a late inventory. Public
patient execution and desktop packaging require separate orchestration release.

Initial backend validation passed 31 focused inspection checks in 3.51 seconds
and all 48 existing bridge/adversarial regression checks in 27.68 seconds. The
latter include the existing synthetic training/replay workflow; they are
compatibility checks, not expanded-model training or patient execution. Renderer,
transport and independent contract review are recorded separately.

Independent fault injection then exposed six returned-binding gaps: coherently
rehashed reports with a changed requested center, native radius, source frame,
source shape or native affine could pass; a different returned binding could
also pass after the facade had accepted an explicit prior binding. These were
internal facade-output adversaries, not renderer-supplied geometry. Their failed
source-bound receipts remain in
`native-axis-inspection-bridge-independent-review.json`. The direct comparisons
above repair this boundary; subsequent validation is recorded separately below.

The repaired bridge passed all 32 owner inspection checks plus all 48 existing
bridge/adversarial checks in one run (80 passed in 32.04 seconds). The independent
review passed 20 checks in 3.74 seconds. Both runs used bridge source SHA-256
`2d32d9d0f8b19701ff9ec9e008a8b0ae20a111dc9536b2e34a6474e38b939df8`.
The additional owner test checks a changed returned binding even when its
geometry and preset are unchanged, so the explicit prior-binding comparison is
exercised separately from the geometry refusals. No public patient run, expanded
training, removal or desktop packaging occurred in this validation slice.
