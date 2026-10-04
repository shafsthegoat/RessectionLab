# Experimental native axis-column proposals

`native_proposals.py` packages the rule explored in
`artifacts/native-frontier-expansion-v1` as a separate source-bound proposal
provider. It is not connected to learning, selected-route refinement or the UI.
The fixed-ray production workflow remains unchanged.

Create `PreparedAxisColumnProposer(native_config, AxisColumnProposalConfig())`
once, then call `propose(engine)` for the current engine or one of its clones
sharing that exact configuration. The immutable result contains exact tool,
entry, primary endpoint and optional fallback endpoint for each proposed ray.
The fallback is eligible only after the primary native preview is rejected.
After committing a cut, request a new batch for the changed cavity.
`validate_batch(batch, engine)` rejects a stale or altered batch by comparing it
with a freshly regenerated current batch; successful validation grants no tool
clearance.

The default rule considers thirteen declared transverse source-grid offsets,
two endpoint depths per tool at most, and a cap of 26 primary rays. It selects
the distal remaining annotated cell in each column, with the proximal remaining
cell as fallback. Every declared column × tool slot receives one disposition:
uncertified proposal, out of image, no remaining target, full-tool aperture
prefilter failure, candidate cap, or an explicit unsupported-input reason.
The cap counts primary rays; actual geometry-check budgets must count fallback
previews separately. Counts include omitted slots and never imply that a
proposal passed geometry checks.

Source hash, complete native engine fingerprint, frozen rule fingerprint and
cavity ancestry hash bind the exported model and proposal IDs. Preparation
reconstitutes the native configuration and rejects a cached fingerprint that no
longer describes its actual contents. Later configuration replacement or
mutation is also rejected. Before emission, the provider verifies the engine's
committed-history digest chain and derives expected remaining, removed, contact
and exterior-connected masks from that history. Direct changes to any live mask
are rejected, even when its cached ancestry string is unchanged. This performs
whole-grid integrity checks on every call; it is deliberately not a cached
clearance or a new geometry audit. The descriptor reconstruction is explicitly
bound to the native V2 history schema; a changed native-engine version requires
review before this provider will operate. Static affine transforms,
entry positions and aperture filters are cached; remaining target cells are
rescanned on every call. No cavity, feasibility certificate or removal result is
cached, and no source tissue is altered.

Finite invertible homogeneous transforms are required. Orthogonal oblique and
mirrored source grids are supported, but the access normal must align with one
source axis and its transverse center must lie at integer source coordinates.
Nonaxial access and fractional origins produce an explicit abstention, with
every slot recorded. Invalid native configurations, including shear, reject at
preparation. Checks use zero relative tolerance, including explicit
off-axis component limits; small tilts do not pass merely because their cosine
is close to one. Invalid or nonfinite transforms raise before proposals exist.

Every emitted ray must still pass a fresh `NativeResectionEngine.preview_stroke`
and the existing state-bound `commit_preview`. The provider does not certify
whole-tool clearance, create a cavity, exempt tissue, claim removal or authorize
clinical access. A future adapter integration needs a separately hashed action
model, a declared policy action-limit rule, replay tests, and the same proposal
and checking budgets across comparators. It must not silently broaden an exact
user-selected entry/target pair.
