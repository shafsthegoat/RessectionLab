# Conservative algebraic corridor preparation

The [conditional source pilot](continuum-closed-loop-pilot.md) passes its numerical
and replay gates, but six fresh preparations cost 97.95 of 132.24 worker seconds.
Its 96-coordinate spring core leaves the CUSA removal stencil in the exterior,
forcing a fresh exterior factor for each new owner. The next measured hypothesis
puts every potentially removable in-domain source element in the algebraic core.
All physical elements, supports, constitutive parameters and history remain.

For the frozen repeated absolute-pose intervention, the resolved spherical CUSA
tip stays on the initial-to-requested segment. The unchanged center-displacement
guard bounds each current cell center by `U=3 mm` from rest. Legacy occupancy
rounds that center to the nearest voxel, adding at most one source half-diagonal
`H`; the voxel cube supercover adds another `H`. Thus a contacted in-domain
source owner must have its resting center within `radius + U + 2H + 1e-9 mm`
of that segment. The bound includes raster membership, rather than assuming
that CUSA contact is a distance test against the continuous source center.

The source geometry has `radius=1.5 mm`, `H=0.8660254 mm`, and a 6.23205081 mm
bound. A read-only geometry preflight selects 501 source cells and 2,574 free
core coordinates, including every actual spring stencil. It leaves 30,786 free
exterior coordinates; the physical root still contains 13,211 cells, 49,260 total
and 33,360 free DOFs. Selection takes 0.03 seconds in the tracked implementation,
with identical source-owner/core hashes to the initial independent prototype.
No source response solve or MRI decode occurs in these geometry preflights.

`corridor_partition` freezes that root selection before future events are known.
It selects whole source-element corner DOFs, respects inherited fixed supports,
and refuses nonorthogonal grids, inconsistent source-corner ownership, invalid
bounds and oversized/empty partitions. It does not select source owners outside
the supplied physical domain or admit their removal. Their existing refusal
remains. Every queried current field must satisfy the full displacement guard
first; the declaration is not a guarantee for an arbitrary future public plan.

Removing a selected whole element changes its element contribution only in I–I.
This mathematical observation is not permission to reuse a factor: source-sector
splitting or floating assembly can alter the induced coupling. Every update still
requires an injective free-coordinate map, an unchanged E bijection, and bit-exact
E/E and both coupling matrices. The profile disables fresh fallback, so a failed
proof is an explicit failed cost hypothesis rather than an uncharged rebuild.

Eight generated controls pass, including a native CUSA case that would escape
a bound with only one half-diagonal. Native raster owners remain included along
20 sampled poses in each of two bounded-field controls, including rotated,
anisotropic source geometry. Remote spring stencils and malformed geometry/bounds
are covered. Geometric containment explicitly grants no factor-retention or
physical-validation claim.

`continuum-corridor-profile-v1.json` freezes a 90-second combined, 3-GiB, 8-MiB,
single-thread/no-retry profile: at most 80 worker and 10 checker seconds. It
measures the initial preparation and one actual 12-BE-step microtick, comparing
its full before/after checkpoint against the accepted spring-core pilot under
the unchanged 1e-9 mm budget. A separate hypothetical deletion of the current
leading contact owner measures transfer and exact retained-preparation cost,
then restores the original numeric binding and checks the complete state hash.
That deletion is not a CUSA event or a new episode outcome.

The independent export checker reconstructs the root corridor without application
imports, validates the unchanged physical source domain, integrates the actual
Q1 checkpoint, and checks exact retained history, removal energy, free lineage,
whole-element core ownership and exterior bijection. It reuses the hash-bound
independent quadrature/checker code already qualified by generated controls.
The backend's exact floating coefficient proof remains authoritative for reuse.

This is a preparation profile, not a complete two-pass pilot. Any full run needs
its own prospective protocol and complete fresh replay/independent checks.
Public scheduler qualification must still start from a prefix evolved by the
coupled model. Failed boundary-stability and measured mechanics evidence,
source-supported vascular/neurological consequences and both RL holds remain.
The fidelity-first scientific goal remains active.
