# Conditional continuum contact feasibility

The complete prescribed-input continuum histories pass numerical qualification
at `a6b32ddb`: 4,218 fixed steps, exact fresh replay, all 16 removal events,
release and relaxation, with 26 independently checked Q1 sample worlds. That
gate takes 449.72 seconds including independent verification. It does not
establish that the larger-domain geometry would accept the native actions.

## Frozen capture and conditional query

`continuum-contact-capture-v1.json` prescribes a native-only observer replay of
the same three programs. It captures actual post-solve corner displacements at
the 26 existing selected BE phases, resolved tool poses, effective parameters,
holds, removal/source identities and protected configuration. The observer
publishes only after the whole native microtick commits. Intermediate BE phases
are conditional geometries within that microtick, not separately accepted
public states. All canonical native action receipts, full snapshots and the
already qualified mechanics input projection must remain exact. No 20 mm solve
is performed. The capture has a 300-second, 2-GiB, 32-MiB ceiling, one numerical
thread and no automatic retry; all outputs remain under ignored `testcases/`.

The subsequent geometry-only query joins each captured native phase exactly to
its frozen 20 mm field by branch, step, operation, slot, action, timestep,
tissue state, field hash and source-operator identity. It compares native and
coupled geometry at the same resolved endpoint poses. Each declared retained
owner is represented either by its displaced Q1 cell or its resting source
cube outside the field. Translated non-target holds and detached fragments
refuse, since their motion is not represented by this query.

For every shaft and working capsule on both hands, the query records legacy
center-raster membership, positive Q1 material-point witnesses, and unresolved
overlaps. Nonnegative Q1 basis functions enclose each element in its displaced
corner AABB; a separated AABB excludes contact numerically at the fixed 1e-9 mm
tolerance. An explicit interpolated material point strictly inside the capsule
establishes a positive numerical witness. Every remaining overlap, including
tangency, stays indeterminate. This is not an interval-arithmetic certificate.

Only a shaft's own hand-held source cells receive the engine's shaft exemption;
the other hand's held cells remain obstacles. Protected contacts remain visible
even with that exemption. Working contacts are descriptive: compatible tissue
may be intended contact, and no working-contact legality is inferred. Tool-pair
endpoint envelopes are also reported. Post-field instantaneous CUSA candidate
ordering follows the original source-center rule, but no capacity or removal is
applied and no pre-effect contact limiter is validated.

The tissue/obstacle survey reports positive sampled points strictly inside
declared bone, vessel or unknown source cells. Absence of such a point does not
establish separation; unannotated anatomy remains unknown. The query performs
no pre-effect sweep, public action, physiology update or solver admission. Its
prospective ceiling is 60 seconds, 1 GiB and 16 MiB, serial and without retry.
Small controls cover clipped CUSA poses, exact phase refusal, outside-field
source coverage, own-versus-other held shafts, unresolved working contact,
fragment refusal, protected visibility and center-raster contact blind spots.

## Completed source capture and endpoint survey

Native capture at `bd1f2598` completes in **169.62 seconds supervised**
(168.95 seconds worker). All three complete native programs preserve every
public receipt, final snapshot and mechanics input exactly. It records all
26 selected phases in 24 distinct field payloads: five committed tick endpoints
and 21 intermediate conditional phases. Sampled peak RSS is 481,067,008 bytes;
the OS worker peak is 483,819,520 bytes. Retained output is 1,054,662 bytes.

The frozen query at `930c7993` first passes its three-world preflight in
3.28 seconds supervised, at 374,865,920 sampled bytes. The maximum paired-world
cost of 0.747 seconds projects 19.90 seconds for the complete set. The actual
26-world survey completes in **22.25 seconds supervised** (21.45 seconds worker),
at 379,142,144 sampled bytes and 379,224,064 OS peak bytes. Retained output is
545,352 bytes. Every bound source/input remains unchanged. These costs include
full scene coverage and full native/coupled fields, with no mechanics solve.

All 130 capsules per model are reported, including both hands' shafts and all
working pieces. Neither native nor coupled geometry has a positive forbidden
shaft witness, an indeterminate forbidden shaft overlap, or a raster-forbidden
shaft contact in these samples. Endpoint tool envelopes do not conflict. Raster
membership, positive material-point membership and the instantaneous post-field
CUSA candidate order are identical across the two models. Six right-hand working
capsule comparisons have fewer indeterminate owners in the coupled geometry;
eight owner incidences disappear (ten native versus two coupled). Each model
retains 161 positive working-contact incidences. None is promoted to working
contact legality. The point survey reports no
declared blocking-label or outside-grid point witnesses. That negative survey
does not validate obstacles, missing anatomy or physical bounds. No protected
source IDs were configured, so critical-anatomy coverage remains unknown.

The selected regression suite passes **197 tests in 8.45 seconds**. Independent
small controls verify Q1 interpolation and conservative contact classifications;
a scalar obstacle-point oracle agrees in 60 generated cases, including oblique
affines, nonaffine fields, boundary points, removal and different batch sizes.
An independent export audit reconstructs all 322 positive source Q1 points and
capsule distances exactly at stored precision. The three preflight worlds match
their complete-query counterparts exactly after excluding measured timing.
Initial fixture/import failures remain in versioned ignored logs. The source
capture, preflight and full query are terminal, with no retry or process left
running. Exact local paths, hashes, commands, receipts and independent reviews
are indexed under `testcases/outputs/simulator-qualification-20261029/continuum-exterior-events-v1/`.

## Remaining admission gates

The 20 mm model still fails both original 16-to-20 mm stress-stability checks
against the unchanged 5% threshold. Its full 33,360 free DOFs and 13,211 source
cells exceed live admission budgets. Complete larger-domain public action
checking, calibrated tissue/tool response, source-supported vascular physiology
and neurological consequences remain open. Conditional geometry results cannot
clear any of those gates. Physical validation remains false and physiological
and large-scale RL remain on hold.

The subsequent [pre-effect study](continuum-pre-effect-qualification.md) now
completes all 16 original removal-transition comparisons in 13.26 seconds.
Two conditional comparisons change contact ordering and fail whole-cell removal
equivalence despite identical raster candidates and resolved poses. Endpoint
agreement cannot replace those temporal/effect checks. The next action gate is
a state-driven closed-loop mechanics/CUSA pilot; outer-support adequacy and full
live resource admission remain separate required gates.
