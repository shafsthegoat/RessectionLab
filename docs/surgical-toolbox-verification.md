# Surgical toolbox verification — October 18, 2026

## Automated checks

Commands run in the primary checkout:

```sh
.venv/bin/python -m pytest tests/test_workspace_persistence.py tests/test_workspace_imaging.py tests/test_patient_workspace.py tests/test_imaging.py tests/test_rehearsal_episode.py tests/test_surgical_tools.py tests/test_surgical_adapters.py tests/test_surgical_integration.py -q
pnpm --dir desktop test
pnpm --dir desktop build
.venv/bin/python scripts/surgical_toolbox_demo.py
```

The source/imaging/save/rehearsal/backend selection passed 186 tests. The complete desktop test command passed 163 main-process, 382 renderer, and 133 viewer tests. The production desktop build passed. A final 59-test tool/integration run also passed after adding conserved debris transport. The all-instrument script exercises 15 distinct tool mechanisms and an 18-action mixed sequence, checking deterministic replay and fluid conservation.

An additional run of older `test_desktop_bridge.py` and `test_desktop_bridge_adversarial.py` did not pass: their legacy synthetic/training fixtures request `createSyntheticCase`, which the existing runtime policy disables (`SYNTHETIC_CASE_DISABLED`). No training was enabled. Those suites are not counted as passing. A transient Vite optimizer teardown failure in the renderer tests was resolved by disabling background dependency discovery in the test harness, consistent with the existing anatomy renderer harness.

## Actual desktop interaction

Used the rebuilt Electron application and native controls:

1. Prepared the existing UCSF-PDGM-0004 source bundle. The drill reported no compatible bone, as expected.
2. Selected the right-hand CUSA, placed it at RAS `[-124,100,14]`, radius `0.6 mm`, and previewed a stroke to `[-124,100,19]`. Preview showed 4 mm³ non-target removal without changing the committed state. Apply recorded the removal. Undo restored the complete action; Redo reproduced it.
3. In the explicitly generated engineering scene, earlier drill/CUSA/retractor interaction created a bone opening, tissue cavity, and held displacement. Backend checks verify release and removal preservation.
4. In the final build, exchanged the right slot for dural scissors, closed jaws, placed at `[15,42,18]`, and applied to `[15,42,21]`. Three cut edges persisted without tissue removal.
5. Kept the scissors parked and used left-hand irrigation at `[28,40,20]`, 20 mm³/s for one second. Fluid increased by 21 mm³: 20 irrigated plus 1 from the fixture's explicitly declared source. Removal stayed zero.
6. Saved through the native Save dialog to a separate verification file. Quit and launched a fresh Electron process with that file. Both instruments, three cut edges, and 26 mm³ fluid reopened correctly.
7. Withdrew irrigation and exchanged it for the separate subcortical probe. Applied at `[32,12,19]`. The UI reported unknown result, unknown coverage, no matching evidence, six contact cells, and the supplied stimulus parameters.
8. Dragged the translation gizmo, verified the numeric draft pose changed, and canceled. The recorded pose was restored.

Local screenshots and the saved fixture are in `outputs/surgical-toolbox-verification/` (generated artifacts, not source-controlled): `patient-cusa.png`, `scissors-irrigation.png`, `linked-cut-and-fluid.png`, and `surgical-desktop-verified.ressectionlab`. The original patient bundle was not overwritten.

## Measurements

On this 16 GB Mac, the supplied patient grid is 240×240×155 with 1,475,621 occupied cells and four mapped regions. One read-only benchmark measured load 168 ms, adaptation 65 ms, inspect previews 1.1–2.0 ms, exchange/placement about 3/2.5 ms, and a 5 mm CUSA stroke removing four cells 19.9 ms. Peak Python RSS was about 330 MB. These are backend timings, not frame latency guarantees.

A process RSS snapshot of the reopened engineering fixture totaled about 400 MiB across Electron main, GPU, network utility, renderer, and Python. This is a point-in-time OS measurement, not a peak or a patient-case memory bound. GPU memory and OS compression are not separately measured.

## Modeling and delivery boundaries

Displacement is a deterministic lattice-patch constraint model. Solid cuts sever cell adjacency and display retained affine cut faces with matching slice intersections; native mesh cuts retain seam/topology records. Native mesh faces are not remeshed into finite-width physical apertures, and cutting a native mesh after lattice displacement is explicitly rejected. Fluid transport is bounded nearest-neighbor redistribution; thermal exposure is a scalar proxy. No calibrated mechanics, physiology, or clinical response is inferred.

All 15 mechanisms have API tests. Manual desktop verification covered the workflows listed above; it did not independently operate every instrument on every input format. See the [walkthrough](surgical-toolbox-walkthrough.md) and [API contract](surgical-toolbox-api.md) for reproducible operations and explicit limitations.
