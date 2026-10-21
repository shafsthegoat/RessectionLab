# Surgical toolbox engineering walkthrough

This workbench exercises explicit instrument mechanisms against declared material cells and imported geometry. Its generated coupons and numeric properties are engineering test inputs. They are not patient anatomy, measured surgical mechanics, or clinical response predictions.

## Open the desktop workbench

```sh
./scripts/launch_poc.sh --case "$PWD/outputs/poc-cases/UCSF-PDGM-0004-rehearsal-ready.ressectionlab"
```

Choose **Rehearse → Prepare instrument scene**. Review Material mapping before using a tool. The supplied patient bundle has no bone, so use **Open engineering test scene** to exercise the drill and craniotome. Select a hand, exchange only while withdrawn, choose **place** with a free-space numeric pose, and then choose **apply**, **hold**, or **grasp** for the selected mechanism. The viewport selector separates navigation, drawn strokes, translation, and rotation. Preview shows an uncommitted outcome; Apply records it; Cancel restores the recorded pose and geometry. A red marker and result text identify the first blocked step.

The two slot cards persist across actions. Inspect **Last action result** and **Recorded effect details** for cut connectivity, exposure, attachments, fluid, and fragment accounting. Probes display supplied evidence or explicit unknown coverage. Undo/Redo and the timeline restore recorded states; acting from an earlier state retains the original continuation as another branch. The native Save/Open workflow preserves the rehearsal alongside source data.

For the captured desktop sequence, open `outputs/surgical-toolbox-verification/surgical-desktop-verified.ressectionlab`. See [verification notes](surgical-toolbox-verification.md) for exact coordinates, screenshots, timings, and test boundaries.

## Reproduce all fifteen instruments

From the repository root:

```sh
.venv/bin/python scripts/surgical_toolbox_demo.py > /tmp/surgical-toolbox-demo.json
.venv/bin/python -m pytest tests/test_surgical_tools.py -q
```

The demonstration runs fifteen isolated coupons through the public engine action API. Each run exchanges the left instrument, places it in free space, and performs a checked application stroke. It verifies saved-action replay and fluid conservation, then reports actual events, metrics, and state identities. A mechanism with no effect raises an error. Isolation keeps one instrument's removal or deformation from masking another's behavior. The separate `mixedSequence` result runs 18 actions in one shared scene: drill removal, retractor hold and displacement, CUSA removal while the hold remains, release, then irrigation and suction. It records the temporary displacement, verifies two removed cells remain removed after release, verifies fluid balance, and restores the saved checkpoint by replay. Both demonstrations are engineering exercises, not simulated surgical procedures.

The fixture declares attachment kinds, strand cells, fluid source rates, a source exposure threshold, and synthetic mapping evidence where needed. These numbers demonstrate deterministic transitions only. Imported anatomy needs its own explicit material mapping and properties.

## Mechanisms to inspect

| Instrument | State change |
| --- | --- |
| Cranial drill | Bone cell removal and local debris |
| CUSA | Compatible soft/tumor cell removal |
| Brain retractor | Persistent source-cell hold and lattice displacement |
| Scalpel | Oriented cut edges; cells remain |
| Periosteal elevator | Declared periosteal attachment release |
| Craniotome | Thin oriented bone kerf; retained connected components |
| Dural scissors | Closed-jaw sheet cuts and retained seam records |
| Microsurgical dissector | Declared soft attachment release |
| Frazier suction | Local fluid/debris removal; no tissue removal |
| Bipolar forceps | Local exposure and bleeding control; vessel patency unchanged |
| Micro scissors | Closed-jaw eligible strand edge transection; pieces remain |
| Tumor forceps | Closed-jaw connected-fragment grasp; attachment-constrained movement/extraction |
| Irrigation | Directed fluid addition and bounded redistribution/spill |
| Cortical probe | Supplied cortical evidence at contacted cells, otherwise unknown |
| Subcortical probe | Supplied subcortical evidence at contacted cells, otherwise unknown |

Scissors and forceps require jaw opening at or below 0.5 mm for their cutting/grasp effects. This is an explicit coarse interaction rule, not a calibrated tissue threshold. Roll changes the blade plane. Radius and depth bound cutting eligibility; source cells are the spatial resolution limit.

## Inspect and replay

`engine.view()` exposes cuts, removed cells, source-cell displacements, holds through slot state, thermal exposure, fluid/debris, source state, and topology records. `engine.snapshot()` stores the scene identity, action sequence, and a full state checkpoint with its action position and identity. `SurgicalEngine.restore(scene, snapshot)` rejects changed scenes, differing replay results, or a checkpoint that differs from verified replay. Preview evaluates a clone without mutating the live engine.

For retained meshes, `meshCutEdges` identifies original vertex pairs and `meshSeams` supplies RAS millimeter coordinates. Sheet seams identify their original face; strand breaks identify their original edge. Meshes imported through case adapters retain geometry in `source.adapter`, which the effects layer recognizes. The graph records changed connectivity while preserving original geometry: faces are not remeshed and physical opening displacement is not simulated.

Fluid addition equals present fluid plus aspirated fluid plus escaped fluid, up to numerical tolerance. Present fluid includes enclosed source reservoirs. Suction and irrigation do not close bleeding sources. Bipolar exposure can stop a declared source's bleeding while preserving its separate `patent` value. Fluid displaced by newly occupied cells moves to an adjacent free cell or the explicitly accounted spill reservoir. Local overflow carries a proportional share of debris; suctioned and escaped debris are accounted separately.

## Limits that affect interpretation

The model uses whole affine cells, conservative capsule collision envelopes, reversible lattice patch displacement, and local scalar energy exposure. It does not provide force feedback, calibrated deformation, thermal diffusion, vessel physiology, inferred functional mapping, or clinical risk estimates. Mesh seams are connectivity records, not a reconstructed surgical opening. Cutting retained native meshes after lattice displacement is explicitly rejected because mesh vertices are not remapped by the cell deformation model. Cell-volume cutting follows displaced source cells. Missing required attachment/strand/source properties cause explicit errors; missing mapping evidence returns unknown. Generated demonstrations must not be used to infer patient-specific safety or outcomes.
