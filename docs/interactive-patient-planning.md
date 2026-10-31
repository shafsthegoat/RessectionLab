# Interactive patient approach planning

**Rehearse → Plan an approach · post-exposure** generates new straight CUSA
approaches to a selected source target region within an existing exposure.
Choose the target and allowable entry patch with the linked MRI cursor or RAS
coordinates, edit the tip radius and mapped non-target clearance, and generate.
Select a candidate to inspect its 3D trajectory and MRI slice intersections.
Preview checks native insertion **and reverse withdrawal** without committing;
Install creates a normal editable/replayable branch at the inactive approach
endpoint. It stops before retained tissue. Aspiration requires a separate action.

## One documented development case

The admitted source is the existing local
`testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab`, branch
`original-exposure`, boundary 4. It combines acquired T1c MRI, an unreviewed
source tumor annotation and an unreviewed SynthStrip brain estimate. Source
voxel ownership, original files and affine remain unchanged. No images, skull,
vessels, tracts or functional labels are synthesized or transferred from another
patient. Native source-layer provenance remains inspectable in the report.

The historical authored exposure removed **560 mm³ target and 4,940 mm³
non-target**, leaving 18,808 mm³ target. These costs remain visible. This
milestone does not plan or validate that exposure. It supports withdrawn,
released instruments at the baseline and the bundle's recorded generic CUSA
geometry (40 mm shaft, 0.5 mm shaft radius; the recorded working-part revision).
It does not plan coordinated retractor motion or a complete resection.

The default target sphere is centered at RAS `[-84,118,66]`, radius 12 mm;
the entry disk is centered at `[-84,118,42]`, radius 7 mm in the recorded
exposure plane. The UI's entry sphere bounds that disk. Target membership means
retained owned tumor voxel centers inside the selected sphere. A candidate's
first prospective contact must belong to that selection. A deep or obstructed
selection can produce no route; the system reports the limiting condition.

## Shared generation and verification

The new adapter calls the existing joint-geometry planner's bounded
post-exposure sampler and the existing native-axis candidate generators. It
uses unchanged engine occupancy, physical capsule geometry, protection guards,
tool-pair checks and public actions. It adds no physics engine or scalar risk
model. Fixed-axis continuous capsule covers screen the complete shaft and
working volume, current secondary-tool geometry and reverse withdrawal before
any public execution. Four small swept translation probes describe local
geometric maneuverability; they do not characterize every possible maneuver.

Only retained candidates are materialized with the original physics profile.
Generation runs no continuum solve and removes no tissue. The separately
authored `verify_interactive_geometry.py` reconstructs the source prefix and
audits complete insertion/withdrawal with `geometry.check_route`, including
working geometry, secondary tools, source protection and the requested
non-target margin. It also has shaft-only, working-tip, secondary-tool and
withdrawal negative controls. This is independent checking code using existing
geometry, not independent anatomical or physical validation.

The comparison separates path length/action cost, entry angle, local geometric
probes, target-boundary contact assumptions and zero additional traversal/removal.
Uncomputed minimum clearance and neurological/deformation injury remain null.
Requiring absent vascular, functional, shift or skull evidence refuses all
routes. A mapped-tissue clearance is an engineering constraint, not a safe
clinical margin. Outside-image anatomy is unassessed.

Reports bind the exact case, source labels, mapping/scene, profile, protection,
native action history and checkpoint. Any draft edit clears selection, guides
and preview authority. Changed checkpoints make results historical; returning
to the exact checkpoint revalidates them. Frozen source/report revisions and
installed branches survive native save/reopen. Editing their programs forks a
new branch. Saved reports are checked against canonical replay and recomputed
geometry; unfamiliar installed receipts also replay their complete native
approach and withdrawal. Bounded caches key complete source and content.

## Reproduce

The commands require the existing local source archive; they download no data.
Use fresh output directories to preserve earlier results.

```sh
.venv/bin/python scripts/check_interactive_planning.py \
  --output testcases/outputs/interactive-planning-next/bridge
.venv/bin/python scripts/verify_interactive_geometry.py \
  testcases/outputs/interactive-planning-next/bridge/01-base-approaches.json \
  --source testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab \
  --output testcases/outputs/interactive-planning-next/independent.json
pnpm --dir desktop build
INTERACTIVE_PLANNING_CASE="$PWD/testcases/outputs/interactive-planning-next/bridge/interactive-planning.ressectionlab" \
INTERACTIVE_PLANNING_OUTPUT="$PWD/testcases/outputs/interactive-planning-next/desktop" \
  desktop/node_modules/.bin/electron desktop/electron/verify-interactive-planning.cjs
INTERACTIVE_PLANNING_CASE="$PWD/testcases/outputs/interactive-planning-next/desktop/desktop-planning.ressectionlab" \
INTERACTIVE_PLANNING_OUTPUT="$PWD/testcases/outputs/interactive-planning-next/cold-desktop" \
INTERACTIVE_PLANNING_REOPEN=1 \
  desktop/node_modules/.bin/electron desktop/electron/verify-interactive-planning.cjs
```

For an interactive demonstration, open the generated
`interactive-planning.ressectionlab` with `./scripts/launch_poc.sh --case PATH`.
The saved comparison starts at its verified planning checkpoint. Narrow entry
radius from 7 to 0.5 mm: only one route can satisfy the 2 mm separation
requirement. Restore 7 mm, require functional evidence, and generate: no route
is recommended because that evidence is absent. Restore the requirement,
generate, select, preview and install a candidate. A 0.5 mm mapped non-target
margin also excludes the current default candidates with a geometric reason.

Relevant regression checks:

```sh
.venv/bin/python -m pytest tests/test_surgical_interactive_planning.py \
  tests/test_surgical_joint_geometry.py tests/test_interactive_geometry_verifier.py \
  tests/test_interactive_planning_review.py tests/test_surgical_consequences.py \
  tests/test_surgical_consequence_analysis.py tests/test_surgical_prefix_bridge.py -q
pnpm --dir desktop test
```

Local timings, native captures, cold reopen receipts and preserved failed
verification attempts are recorded under ignored
`testcases/outputs/interactive-planning-20261031/`. The complete bridge run took
31.64 s including the 3.80 s independent audit, all three exact previews,
installation, edits/refusals, save and canonical reopen. Bridge generation took
0.99–2.98 s; exact preview took 0.98–1.10 s on this machine. These are measured
operations on one development case, not throughput or clinical performance
claims. Native click-to-accepted-display generation took 1.04–2.92 s, including
geometry, persistence checks and rendering; exact preview took 1.29 s. No
discarded candidate ran the continuum. A separate run loaded the committed
engine, motion and continuum modules in memory to check independence from
unrelated uncommitted retractor work, without changing the checkout.

The final 162 targeted Python tests, 1,209 desktop tests and desktop build pass.
Hidden native runs cover both window sizes, target cursor selection, entry
restriction, mapped-tissue protection, missing functional evidence, exact preview,
installation, save and a fresh-process reopen. The window remained hidden.

All existing lattice, continuum/domain, physical-response and physiological RL
qualification failures remain. No finite-blade expansion, model fitting or RL
training is part of this milestone.
