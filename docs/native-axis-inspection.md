# Read-only native axis inspection

`native_axis_refinement.inspect_axis_planning` is a separate backend facade for
inspecting the [expanded axis adapter](native-axis-adapter-design.md). It does not
change selected-route refinement, the Electron bridge, saved runs or the renderer.
It creates one fresh model and returns the complete initial native preview
inventory. It performs no transitions, tissue removal, gradients, search,
checkpointing, final evaluation or independent history certification.

Callers provide `CaseData`, expected case and planning hashes, canonical RAS+
`AccessWindow`, complete `ToolGeometry` configurations, `RewardSpec`, and
`WorldGeneratorConfig`. They must explicitly acknowledge that neighboring columns
broaden the action model. The optional proposal rule, native step length, cut
horizon and partial-contact weight are recorded in the binding. A supplied
source-grid hard-exclusion mask requires separate provenance; it never implies
that all vascular or functional anatomy has been assessed. This first facade
accepts no functional maps, population initialization, feature transformation,
resume option or geometry cache.

The returned `AxisPlanningInspection` holds an immutable JSON string. `to_dict()`
returns a detached copy; modifying an export cannot change the retained result.
`binding_hash` identifies source, configuration and assumptions independently of
local timings. `expected_binding_hash` can reject reuse after any bound setting
changes. Neither hash grants candidate eligibility or authorizes committing a
serialized preview.

## Coordinates and support

The public access is always canonical RAS+, including for an LPS+ source. The
facade adapts it to the direct native builder's documented source-frame input;
the builder returns RAS+ native geometry. Source samples and voxel indices are
never resampled. Centers, radii and window IDs must agree exactly. For LPS sources only, the
existing window constructors can renormalize a unit direction by roundoff; the
facade allows an absolute difference of four float64 epsilons per component
(about 8.88e-16), with zero relative tolerance. Requested and actual directions,
the observed difference and the bound are retained. RAS input remains exact;
provider and engine geometry tolerances are unchanged. Obliquely
rotated source grids are supported when access follows a source axis; a direction
oblique to the source axes or a fractional transverse origin is refused with the
provider's explicit unsupported reason. No replacement direction is generated.

Neighboring-column acknowledgment cannot grant anatomy approval. The existing
`planning_brain_support` review and source-bound assumption checks remain in
force. With no supplied brain mask, fallback requires both a declared
skull-stripped source and the separate `acknowledge_estimated_support=True` flag.
Full-head or explicit nonzero-support prohibitions still refuse it. An unreviewed
supplied research support estimate also requires that separate acknowledgment;
an accepted source-bound review or a synthetic fixture does not. Even a reviewed
brain envelope leaves cortical access hypothetical and unverified. The returned
binding retains the actual support method, review/assumption details and mask
hash; it never installs a derived mask into `CaseData`.

## Inventory, cancellation and limits

The complete provider ledger retains every column/tool slot, including omissions.
The preview ledger retains rejected primaries and subsequent fallback attempts;
a feasible primary never triggers fallback because of its score. Certified
non-STOP actions retain exact entry, endpoint, axis, tool ID, proposal ID, endpoint
phase, source/native/cavity identity, geometry unknowns and unexecuted preview
cell counts. Whole-tool portions outside the source image remain unassessed. Their
certificate scope is explicitly `native_engine_preview_only`. Full tool geometry
and access are in the binding. Failed-attempt endpoints are requested geometry,
not claimed first-failure poses; the current adapter does not expose those poses.
No preview cell count is reported as actual removed tissue.

The result distinguishes complete `ready` and complete `no_actionable_moves`
inventories. An all-omitted or all-rejected supported inventory retains its full
denominator and reasons. Unsupported access, stale identities, exceptions or
cancellation raise without returning a partial result or publishing STOP-only
success. Cancellation is cooperative at existing adapter boundaries and after
snapshot serialization; it is not a hard latency guarantee. Model validation also
runs after the final external callback. Full source manifests are hashed at entry
and exit as well: cached case identities alone cannot detect illicit frozen
dataclass field mutation. Those two source reads are real inspection overhead.
The reported preparation timer excludes final JSON serialization and exit checks.
Constructor seed zero has no experimental
role; no optimization, selection, final or stress manifest is created.

Motor and language evidence remain unavailable, vascular anatomy remains
unassessed, clinical probability is null and `candidate_eligible` is false. Local
preparation, initialization, integrity and preview timings overlap; do not add
them as disjoint phases or infer patient throughput from the tiny tests.

The focused synthetic checks cover complete inventory accounting, no commits or
source changes, detached immutable exports, RAS/LPS physical equivalence on an
oblique grid, explicit unsupported directions, separate acknowledgments and
anatomy prohibitions, real primary-reject/fallback acceptance, complete rejected
inventories, stale source/model bindings and interruption during a native preview.
The first run retained one error-message mismatch: a redundant RAS window
construction renormalized a nonaxial normal before the provider could report its
unsupported direction. Reusing the already canonical RAS window fixed that
boundary; the next run passed all 15 checks. Independent review then retained
two actual failures: a late source-field mutation could evade cached case hashes,
and a valid LPS unit direction changed by 1.11e-16 during constructor
normalization. The uncached source seals and the explicit LPS-only roundoff
convention address those findings. Negative receipts and the independent final
result remain in `artifacts/native-axis-inspection-independent-v1/`.
No public patient inspection or training was performed in this slice.

Final focused validation passes 16 owner checks, including unchanged exact
selected-route readiness before and after expanded inspection, and 33 independent
checks. The independent set includes eight composed oblique frames, the retained
one-ulp LPS failure, larger direction and center/radius/ID drift rejection, and
late source/tool/reward/world mutations. These are synthetic backend checks, not
an end-to-end desktop workflow or a public-case latency measurement.
