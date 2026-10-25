# Recorded field transfers

Observed replay can show the fluid and debris writes recorded during its previous
observation interval. In **Rehearse**, open an observed microstep and use
**Show transfer glyphs**. The panel identifies the interval and keeps anatomy,
instruments and quantitative field markers at the exact recorded endpoint.
Static historical arrows and local diamonds remain available during review.
During playback, an additional GPU droplet overlay illustrates the preceding
recorded interval once. Its emission time, source cell, lifetime and volume
come from each recorded operation. A labeled 3× radius display gain improves
visibility. Deposits grow locally, uptake shrinks locally, and transfer motion
uses only a recorded neighboring-cell edge. It supplies no route to a tool tip,
fluid surface or measured velocity. Pausing and scrubbing stop animation.

Arrows connect actual neighboring native cells that were free when the engine
transferred fluid or sediment. Diamonds mark local fluid release, suction or
escape. Irrigation and wound releases identify their real free destination;
the display invents no path through an unknown outlet or to a suction tip.
An off-grid irrigation escape retains its numeric record without a glyph.
Transported fluid remains mixed. Release provenance does not imply that blood
and irrigation remain separately tracked after mixing. Debris is separate.

Glyph size is fixed display resolution, independent of transferred volume.
The expandable operation records show the sampled quantities in mm³. Summing
transport across several edges would count the same material repeatedly; these
records do not replace the complete fluid, vascular and debris ledgers.
Creation of debris by tissue removal remains in the endpoint state and events.

Only observed replay owners install the bounded recorder. Ordinary headless
engine execution has no recorder, trace state, sampling arrays or new physical
events. The hooks retain the existing numerical expressions and write order.
Samples enter an interval only after the complete microstep, including its
post-effect protection checks, succeeds. Rejected work and canceled unpublished
actions cannot publish transfer indicators. If a positive duration is too small
to advance the floating-point clock, its indicators are omitted and truncation
is reported; the physical action keeps its original outcome.

Each interval keeps the first 256 operations, with a 64 KiB encoded metadata
ceiling. Reaching a limit marks the sample truncated. Existing replay, action,
manifest and retained-memory limits remain unchanged. Optional samples can be
dropped to a truncated empty block; if even that envelope cannot fit, the trace
is unavailable and static endpoint review remains possible. This is distinct
from an untruncated interval containing no recorded operations. Metadata budget
retries repeat serialization only, never physics.

The optional `playback.flowDisplay` block uses
`resectionlab.recorded-field-transfers.v1`. It binds scene identity, action and
frame ordinals, previous/current observed state hashes and interval endpoints.
Each operation declares kind, field, provenance, native source/destination cell,
positive volume and its successful substep times. Canonical observed and
prepared-prefix manifests authenticate the block. Readers check its context,
types, bounds and native adjacency. Historical free occupancy is certified at
the original write; a later endpoint cannot reconstruct intermediate occupancy.
Older absent traces remain usable as static observations; malformed supplied
metadata is rejected. Implementation seals include the recorder and invalidate
older prepared packages.

Only the demanded response supplies the overlay. Prefetched successors remain
undisplayed wire data. Transfer glyphs and their endpoint enter the existing
atomic model-publication path together. Preview, boundary navigation, source
changes and display toggles remove stale overlays. Glyphs follow the scene's
actual affine and active clipping planes, cannot be picked, and do not affect
camera fitting. Three static instanced meshes allocate 256 slots each, with at
most 512 glyphs. One additional droplet mesh uses at most 256 instances; a shader
uniform advances it without rewriting instance buffers. Replacement releases instance buffers, geometry and
materials. Allocation telemetry includes instance matrices and colors.

Pooling surfaces and compositor delivery remain open. Working tool parts can
also animate during accepted activation: recorded-RPM drill phase and explicitly
illustrative jaw close/release, while root poses stay exact. The optional follow
camera and cutaway consume newly accepted endpoints only. This display performs
no learning and establishes no patient physiology or clinical calibration.

Generated validation passed 1,173 Python tests with two intentional native
alias skips and two existing Gymnasium advisories. The pilot module was
excluded entirely. The focused recording/cache selection passed 166 tests;
722 renderer/viewer tests, 231 Electron tests and the production build passed.
Checks include complete ordinary/observed state and receipt equality, a real
tiny-duration suction/overflow action at an unchanged floating-point clock,
failed-write rollback, authenticated tamper rejection, cached seek without
physics, optional metadata budget fallback and renderer resource cleanup.
Independent static review found and checked repairs for the rounded clock,
release-color wording and asynchronous screenshot context. Patient application
validation is recorded separately in the performance results.

The first fresh GUI attempt stopped after one observation because compact
prefetch omitted scene context. Wire validation now uses the same replay
owner's immutable scene and request-time acknowledged timeline generation in
a temporary view. It neither hydrates future field values nor advances stream
acknowledgments. The demanded receipt still undergoes normal hydration and full
admission before display. The failed v12 run and partial cache remain preserved
and unqualified; subsequent qualification uses a fresh cache.

The optional **Close tool view** now offers a pinned local anatomy cutaway.
Static field points, cut lines/faces and transfer glyphs follow that 3D window;
complete instrument geometry remains outside clipping, and source MRI panes stay
available. Refocus explicitly chooses a new accepted tip. Turning the window
off restores previous clipping; full Restore recovers the prior view/cursor.
One prepared-only actual-app probe made the local instrument and nearby colored
markers distinguishable with the window on. Faceted anatomy, poor tool visibility
with it off and the below-fold legend remain limitations. No moving-fluid or
all-tool visual acceptance follows.

Current v15 fresh replay after the bounded native field-token optimization
passed 73 actions / 1,149 observations and a separate no-step numerical audit.
Every v14 complete descriptor, field token, transfer block, public receipt and
compressed physical-state hash/length matches, along with the full canonical
records and earlier audited trajectories. Its full/warm CPU rates are
0.705×/1.722×; dense anatomy still obscures small transfer indicators in actual
pixels. Older caches are historical because package Python source changes
invalidate normal seals. The current prepared launch selects fresh v15 data.
See [performance results](rehearsal-performance.md) for current measurements,
source bindings and the remaining visual/performance gates. No RL ran here.
