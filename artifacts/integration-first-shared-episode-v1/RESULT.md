# Shared execution, replay and source-image workspace

The existing native simulator now connects to the Electron app through one typed
episode record. Scripted actions, bounded search and compatible policy actors use
the same persistent state transition. The desktop executes the generated scripted
or SEARCH episode and displays the recorded complete tool and exact removal history.
This is a working integration milestone, not physical or clinical validation.

## What is integrated

- Aspiration removes legally accessible tissue; a separate geometric tangential
  probe records exposed-surface contact without removing tissue. Earlier removal
  changes later access. Probe contact is not a force or sensor observation.
- Public observations, nominal planning and sealed actions remain separate from
  private reference scoring. Mixed-mode observations carry action modes and probe
  contact history; the existing aspiration-only policy rejects them explicitly.
- Replay includes insertion and reversed withdrawal poses, tool geometry and
  authoritative native-cell deltas. Source view restores acquired/generated source
  tissue instead of displaying reachability as removal.
- Additional scalar NIfTI images and aligned labels can be imported into a
  session-only imaging workspace. Each keeps its source identity and native frame.
  Unregistered attachments suppress primary planning/replay overlays and never
  become policy inputs merely because they were opened.

The generated scripted episode commits six actions and 96 frames, removing two
target and three other-tissue cells, with 14 historical probe-contact cells. SEARCH
commits two actions and 72 frames, removing four target and three other-tissue
cells. It omits probing because this task assigns no information benefit to it.
These are distinct modeled outcomes under the same simulator, not patient results.

## Verification

Root's canonical runs passed 13 shared-engine/policy tests, nine bridge tests and
six imaging tests. The desktop worktree suite passed 282 tests; it also contains
pre-existing unfinished neighboring-path tests. Both the worktree production build
and a separate projection of the exact staged desktop sources passed TypeScript
and Vite. The production bundle still triggers a large-chunk advisory.

The selected backend regression run reported **137 passed, five failed**. All five
failures were separately reproduced on committed baseline `23570f7`: one stale
case fixture lacks context, and four legacy learner tests omit the now-required
generated-development context. Production admission guards remain unchanged.
This record does not describe the entire repository test suite as passing.

Independent review passed 14 backend controls and audited all 168 saved replay
frames, including full-tool geometry, state ancestry, removal prefixes, private
reference invariance and tampering rejection. Root then exercised the actual Mac
app: generated execution, probe microsteps, STOP, source restoration and SEARCH.
The independent review predates final cosmetic display fixes; root build and live
inspection cover the final version. Exact source hashes and raw test outputs are
included beside this report.

## Real-image display check

Root opened the public RESECT Case4 T1 image and attached its FLAIR image through
the native file dialog. They displayed in their distinct source-reported frames;
no registration, target annotation, planning world or training admission was
invented. The first live check exposed stale metadata/footer coordinates and an
incorrect annotation-assisted label. All three were corrected and inspected live.

A suspicious visual orientation prompted an independent CPU resampling check of
the exact source affine. It visually reproduced the Electron appearance; this
supports implementation consistency with source coordinates, not independent
anatomical orientation acceptance. No affine correction was justified. The UI
explicitly keeps source orientation and planning suitability unreviewed.

The first real-display session closed normally after 114.54 seconds at a sampled
964,935,680-byte owned-process peak. The final verification session reached its
180-second wall cap and was reaped at a sampled 1,005,387,776-byte peak. The latter
exit is the supervisor's bounded stop, not a normal-close success. These samples
exclude unobserved transients and do not establish GPU memory consumption.

Source attribution: Xiao, Fortin, Unsgård, Rivaz and Reinertsen (2017),
[RESECT, DOI 10.11582/2017.00004](https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2),
CC BY 4.0. Patient image arrays, screenshots and local display bundles are not
included in this Git artifact. No restricted SynthRAD pixels were used.

## Remaining work

Multimodal session persistence, explicit registration/evidence qualification,
useful learned mixed-mode behavior and physical interaction validation remain
open. The legacy trained-policy execution is recorded separately. This milestone
provides no learned mixed-mode efficacy, unseen-patient generalization, neurological
injury probability or RL advantage over matched search.
