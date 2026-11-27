# Surgeon-perspective rehearsal

The source anatomy, RAS-mm affine and persistent source-cell identities remain
authoritative. No policy fitting, RL retraining or dataset acquisition is part
of this work.

## What the surgeon needs the night before

The rehearsal exists to help one surgeon decide seven things about one patient.
Each change to the rehearsal names the need it serves, and work is ranked by how
much it helps these decisions. Literature on surgical rehearsal reports gains in
anatomy, strategy and confidence more than in measured outcome, which is why
corridor anatomy on the patient's own scan ranks above finer tissue physics.

| Need | Decision | What the rehearsal must show |
| --- | --- | --- |
| N1 | Where to put the craniotomy and corridor | The head as positioned, the incision and flap on the real scalp and skull, and the corridor from cortex to tumour |
| N2 | Where eloquent cortex, tracts and vessels sit relative to that corridor | Named arteries, veins and tract bundles with their distance from the working tip, each marked with its evidence |
| N3 | What each depth will look like | Scalp, bone, dura, arachnoid and pia, gyri and sulci, white matter, tumour rim and core, drawn from the scan at the depth reached |
| N4 | How tumour and brain will behave | Tissue that yields, tears or bruises under the instrument, tumour that debulks at a believable rate, and a cavity that forms |
| N5 | Where bleeding will come from | Bleeding bound to the vessel that was cut, arterial or venous in character, pooling and hiding the field until controlled |
| N6 | What navigation gets wrong after brain shift | The planned target against the shifted one once the dura is open, with the pointer's registration error |
| N7 | When to stop | Residual tumour, the estimated distance to a tract, the mapping response and the harm already done, with unknowns left unknown |

Ranked work, highest first:

1. A complete procedure, positioning to closure, on real head anatomy (N1, N2, N3).
2. The depth picture: the scan carved by the removal, scan-textured cut walls and a smooth cavity (N3).
3. Tract and vessel distance at the working tip, the stimulation estimate and the stop cues (N2, N7).
4. Bleeding from the named vessel, pooling, and haemostasis that depends on vessel size (N5).
5. Brain shift at dural opening with a navigation pointer (N6).
6. Injury in place of refusal when an unsafe move is made (N4, N7).
7. Instruments that work at operative rates with two hands (N4).
8. Interaction speed, which supports every item above.

The gap ledger in [current status](current-status.md#operative-realism-gap-ledger)
records which of these are closed.

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
