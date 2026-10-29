# Surgeon-perspective rehearsal increment

Use the existing primary checkout on `shaf/surgeon-physics-rehearsal`. The source
anatomy, RAS-mm affine and persistent source-cell identities remain authoritative.
No policy fitting, RL retraining or dataset acquisition is part of this increment.

The latest visibly excavating, active-suction example and precise local evidence
are described in [Operative demonstration](operative-demonstration.md).

In Rehearse, **Surgeon / Microscope** installs a fixed optical camera with angle,
working distance, field size, focus-plane offset and operative light controls.
Tools never drive it after initial placement. Opaque retained anatomy occludes
the field, including cavity walls; ordinary hidden-layer/transparency/clipping
choices cannot create an unlabelled see-through surgeon view. Use the explicitly
labelled inspection controls for those views. Center on the accepted tip is a
one-time command. The external camera and microscope settings survive native
save and cold restart. Focus changes the optical plane; depth-of-field blur and
stereo optics are not modeled.

**Enable revised instrument mechanics** is an explicit action after withdrawal
of both hands. New UI actions select revision 2; revision-1 episodes offer
**Upgrade operative tool geometry**. It selects finite CUSA V3, actual capsule working contacts, bounded
blade cuts, closed bipolar contact, checked fragment evacuation and physics-state
jaw articulation. Older recorded prefixes keep their prior semantics. See the
[15-tool audit](surgeon-tool-audit.md) for the baseline, fixes and approximation
limits. Generic metal/capsule proxies are shared with collision geometry;
manufacturer geometry, hand dynamics and calibrated forces remain unresolved.

**Patient position and gravity** supplies supine, prone, left/right lateral,
park-bench and semisitting presets, table tilt, head orientation and estimated
operative-site height. A recorded setup action settles one second of existing
local lattice mechanics and conservative field flow. The comparison describes
gravity, local modeled shift and static-column height/pressure changes. The
[source-backed posture model](patient-position-model.md) documents coordinates,
units, numerical bounds and explicit assumptions. Venous outflow, CSF buoyancy,
ICP, perfusion and neurological outcomes are unknown.

## Run and verify

Launch a prepared saved rehearsal on the secondary display without taking focus:

```sh
./scripts/launch_poc.sh --secondary-display --case /absolute/path/to/example.ressectionlab
```

`scripts/prepare_surgeon_rehearsal.py SOURCE NEW_OUTPUT_DIRECTORY` prepares a
source-bound example under ignored `testcases/`. It records an explicit research
exposure using historical CUSA, then finite-capacity revised CUSA, irrigation and
same-prefix posture branches. Source arrays remain intact. The source must carry
an existing standard V2 mapping/profile and mapped tumor; no missing anatomy is
invented. The script refuses overwrites and checks canonical replay.

Local patient paths, exact launch/reproduction commands, capture inventory,
recording, test logs and known failed attempts belong in the ignored
`testcases/outputs/surgeon-physics-20261028/README.md`, following local working
instructions. They are not required source assets in a fresh checkout.

General checks:

```sh
pnpm -C desktop build
pnpm -C desktop test
.venv/bin/python -m pytest tests/test_surgical_revised_tools.py \
  tests/test_surgical_position.py tests/test_surgical_position_integration.py \
  tests/test_surgical_tools.py tests/test_surgical_cusa_v3.py \
  tests/test_surgical_physics_v2_integration.py tests/test_surgical_frame_replay.py \
  tests/test_shared_surgical_guards.py tests/test_workspace_persistence.py -q
```

`desktop/electron/verify-surgeon-rehearsal.cjs` uses `SURGEON_CASE` and a new
`SURGEON_OUTPUT` under `testcases/`. Set `SURGEON_PREPARED=1` for the prepared
example, and `SURGEON_RECORD=1` to collect observed native frames. The verifier
starts the real app with `--automation-hidden`, captures with `stayHidden`,
checks that the window was never shown and preserves its input. It verifies
changing native anatomy against fixed optics, both tool poses, blocked exchange,
playback/pause/scrub, posture differences, optical/physiology separation,
Undo/Redo, branch, actual position application and native save.
`SURGEON_REOPEN=1` and `SURGEON_EXPECT=<previous report.json>` perform a separate
cold-restart check. CPU timings report frame submission including test overhead;
they are not guaranteed FPS. Replay may buffer while exact native observations
and local surfaces are admitted.
