# Recorded field transfers

Observed replay can show the fluid and debris writes recorded during its previous
observation interval. In **Rehearse**, open an observed microstep and use
**Show transfer glyphs**. The panel identifies the interval and keeps anatomy,
instruments and quantitative field markers at the exact recorded endpoint.
This first display uses static historical arrows and local diamonds. It does
not reconstruct a moving fluid surface, intra-cell velocity or tool trajectory.

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
camera fitting. At most three instanced meshes allocate 256 slots each, with at
most 512 visible instances. Replacement releases instance buffers, geometry and
materials. Allocation telemetry includes instance matrices and colors.

Animated droplets, synchronized flow playback, pooling surfaces and compositor
delivery remain open. This display performs no learning and establishes no
patient physiology or clinical calibration.

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
