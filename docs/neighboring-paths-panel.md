# Neighboring-path inspection panel contract

`NeighboringPathsPanel` is an isolated, controlled React component. It is not
mounted in the app in this slice and does not call the engine, train, save a run,
export a candidate, mutate source data, or select an A/B route. Its tests use a
clearly synthetic JSON contract fixture; they are not patient or native-app QA.

The panel follows the existing compact route sidebar: explicit controls first,
then counts and expandable per-path records. It says **Inspect neighboring
paths**, **Research instruments**, **Preview passed**, **Preview rejected**, and
**Omitted**. Passing means only the initial native-engine preview. The selected
route supplies its window; entries, endpoints and explicitly selected native
research instruments may differ from that route. The report does not grant
candidate or removal authority.

## Wire and display binding

Use the shared `InspectAxisPlanningRequest` / `InspectAxisPlanningResult` types
and the named `inspectAxisPlanning` operation. Requests contain only case,
planning, route and route-model identities, explicit `toolIds`, two independent
acknowledgments, and an optional expected binding hash. No renderer coordinates,
instrument dimensions, rewards or world configuration enter the request. The
server resolves the selected route's current window and canonical tool catalog.

The result preserves the full raw `AxisPlanningInspection.to_dict()` beneath
`inspection`; the envelope includes exact source/route/model IDs, requested tool
IDs, `accessSource: selected_route_window_only`, the cached canonical-RAS window
and its explicitly bounded constructor-normalization receipt. The panel's local
context separately receives that selected window converted once to RAS, the
trusted engine tool catalog, and the source-bound support gate. Neither geometry
nor support is inferred from a friendly label or an acknowledgment checkbox.

`validateNeighboringPaths` checks the display joins: case/model/window/tools,
source/cavity hashes, complete initial ledger, proposal/attempt/action counts,
primary-before-fallback ordering, missing-evidence flags and zero execution. The
v1 display contract pins the 13 declared source-voxel offsets, cap 26, RAW profile,
three-step configuration and existing fallback ordering. Thirteen columns times
the selected tools is the denominator; omissions remain rows. Geometry checks
include primary and fallback attempts and are a different count. No count is
labeled accessible or removed tissue. Requested failed endpoints are never
presented as first-collision locations. This validator is not independent
geometric certification or cryptographic validation of a report.

The cached-window normal may differ from its typed constructor by at most four
float64 epsilons, as disclosed by the envelope. Requested-to-actual native access
has the separate existing LPS-only normalization convention. Centers, radii and
window identities stay exact; no origin snapping or direction correction occurs.

## Required host lifecycle before app integration

- Start work only from the button. Loading a case, selecting instruments, or
  checking either acknowledgment must not start inspection. Instrument selection
  starts empty; acknowledgments start false. Estimated-support consent is separate
  from neighboring-path consent and cannot unlock a blocked full-head case.
- Hold a generation-bound request snapshot. Invalidate output and pending requests
  when case/planning identity, route/window/model, tool selection, acknowledgments,
  engine availability or component lifetime changes. Recheck the full snapshot
  before publishing a terminal result. The controlled component checks displayed
  records again; the host must also discard late responses and guard callbacks.
- On cancel, immediately clear the visible result and mark the operation cancelled,
  then send cancellation for its request ID. A current native geometry check may
  still finish. Do not publish its late result or announce that CPU work has
  necessarily stopped. Errors and unsupported windows remain errors, not empty
  successful inventories. No timeout may fabricate a partial denominator.
- Keep this namespace outside normal route, run and replay caches. Population
  overlays do not become model inputs. Any future 3D tool inspection needs its own
  bound, passive display contract and clear/restore action; no row currently
  supplies route-selection, training or removal authority.

These host lifecycle requirements are documented for the next integration slice;
they are not claimed as implemented by the standalone component.

The isolated slice passes 14 owner contract/rendering checks, two independent
display regressions, and a strict scoped TypeScript check. The full renderer suite
passes 100 checks. Independent review first reproduced hidden extra uncertainties
and stale results under engine unavailability; both are now withheld or exposed
appropriately. Native-window rendering, actual patient inspection and host
cancellation remain outside this component-only validation.
