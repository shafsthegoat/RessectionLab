# Canonical finite-blade engineering maneuver

One opt-in approach–indent–hold–withdraw maneuver now runs through the actual
desktop and the same native headless actions. The generated engineering fixture
passes kernel/native agreement, transaction, save/reopen and replay checks.
**No compatible acquired-source maneuver was admitted.** The subsequent
[temporal correction](planar-blade-temporal-correction.md) passes the unchanged
frozen work gates and three-resolution canonical work checks. New declarations
bind the refined integration method; existing saved declarations preserve original
BE and its failed work gates. Physiological, patient-prediction, planning and RL
eligibility remain unchanged.

## Supported configuration

The [versioned supported-domain declaration](../manifests/experiments/planar-blade-maneuver-v1.json)
describes the boundary. A complete `planarBladeManeuverV1` scene declaration binds
actual labels/affine/region/provenance, the continuum material profile, source
cells, exposed faces, independent support faces, blade frame/footprint, rigid pad
and shaft, and the fixed program. Native scene/state identities also bind the
complete source case. Only the derived outer case hash and environment tag are
excluded from the separate geometry digest to avoid self-referential metadata.

The left retractor performs four ordered `bladeManeuver` actions. Each has an
explicit `maneuverPhase` and fixed duration; the right instrument stays parked.
At 0.025 s increments, offsets move −0.1→0→0.02→0.02→−0.1 mm over
0.1/0.2/0.2/0.2 s: **28 kernel increments in 0.7 modeled seconds**. No cutting,
removal, native rigid hold, body loading or topology change is admitted.
Unrelated native actions and planning/full-case commands refuse before mutation.

The pad is a 0.2 mm thick rigid rectangle with its contact face at local Y=0;
the shaft has radius 0.3 mm and stays on the exterior side. Backend-owned rigid
primitives and the exact pose drive rendering and collision bounds. Collision
uses source-cell/Q1 corner bounds, including the coupled linear space-time sweep
between increments, and refuses unresolved positive overlap. Q1 boundary faces
must remain planar to 1e-9 mm so the rendered triangles represent the same
surface. Loaded footprint migration must be ≤1e-9 mm. This is a restricted
reference-footprint model, not finite-strain/sliding edge contact.

The runtime calls `surgical_planar_contact.PlanarContact.step` directly. It retains
the existing material/history equations, kernel residual checks, and stricter
source-profile limits. Independent supports are required while separated;
contact cells never supply three-axis rigid support.

## Canonical state and desktop

Native JSON owns phase, acquisition/release events, offsets, Q1 displacement,
Maxwell memory, source ownership, and cumulative work/dissipation. Kernel wall
timing is excluded from state and history. Each public phase is atomic: a refused
increment or cancellation discards its candidate, including earlier increments
in that phase. Preview uses the same transaction without publication. Existing
branching, timeline, save, canonical restore and observed replay use these same
actions. Loaded-code cache fingerprints include both planar modules. Untagged
historical episodes retain their previous behavior.

The default panel shows contact/no-contact, modeled reaction, maximum displacement,
the next preview/commit actions and any active refusal. Numerical/material details
remain in a mounted disclosure. The 3D field uses source-owned Q1 boundary corners
and the matching pad/shaft; original MRI/source arrays remain unchanged. First
microscope entry frames the declared surface once. Subsequent actions never move
the camera. The inactive parked hand is omitted from this one-tool optical field.

## Original measured engineering result (before temporal correction)

The clearly generated 4 mm cube has eight source Q1 cells, a 4×4 mm blade,
independent bottom-face clamps, μ=1,000 Pa, ν=0, and one 500 Pa / 0.5 s Maxwell
branch. These are numerical assumptions, not patient measurements.

| Endpoint | Modeled reaction | Maximum displacement | Reference contact area |
| --- | ---: | ---: | ---: |
| Approach: touching without load | 0 N | 0 mm | 16 mm² |
| Indent | 224.632 µN | 0.020 mm | 16 mm² |
| Hold | 203.746 µN | 0.020 mm | 16 mm² |
| Withdraw | 0 N | 0.00235031 mm | 0 mm² |

The residual after release is free Maxwell recovery, with no attractive force.
All 28 public increments match the corresponding preceding-kernel trajectory
exactly for displacement, memory and reaction. A separate scalar uniaxial BE
oracle agrees with the public BridgeSession endpoints. Canonical preview/commit,
cancellation during execution/finalization, late-refusal rollback, branches,
verified caches, intermediate frames, rotated geometry, save/cold reopen and
complete restored/replayed states pass. Source arrays and the retained initial
branch are unchanged.

The current-checkout regression selection passes **128 Python checks**, including
19 maneuver checks. A separate **79-check staged-delivery control** verifies the
new path without the checkout's unrelated unfinished contact/runtime work.
Desktop checks pass: **626 renderer, 254 viewer, 318 main-process tests**, plus
TypeScript/production build. Hidden native checks visually pass at 1440×900 and
1180×760 with details closed, explicit preview/cancel, fixed optics and cold native
reopen. Windows never showed or took focus. Earlier adapter-binding and optical
framing failures are retained with their corrected proofs.

## Source result and limits

Three existing archives of the **same UCSF-PDGM-0007 source** were audited without
editing evidence. The operative archive lacks the continuum material declaration.
Recorded exposure/contact/continuation checkpoints retain removed topology,
native holds and/or nonzero displacement/Maxwell history, which this milestone
cannot reset or transfer. Clean main checkpoints lack an explicit footprint/support.
A hypothetical untouched-source geometry probe using the original ν=0.45 material
has 0.002680735 mm loaded footprint migration, exceeding 1e-9 mm. No material,
support or anatomy was changed to make a source case qualify. This is an unavailable
source demonstration, not a successful operative exposure.

An original 29-action `short-hold` episode still restores its complete snapshot
exactly with `use_cache=False` in 21.839 s. The original archives, source masks,
materials and branch histories remain unchanged.

The original BE 5% net-cycle-work sensitivity gate fails at **10.08% / 11.41%**;
the [correction report](planar-blade-temporal-correction.md) records current results.
Small-deformation contact/replay agreement does not establish time-converged cycle
work, measured tissue behavior, patient prediction, clinical safety or RL readiness.

## Timing and reproduction

Five fresh BridgeSessions, with no previews before commits, were measured in one
fresh Python process on an Apple M2 Pro / 16 GiB, macOS 26.5.2, Python 3.12.12,
NumPy 2.5.3. Imports, fixture construction and source hashing are outside intervals;
process-level pure caches may remain warm. Medians (min–max), n=5:

| Backend operation | ms |
| --- | ---: |
| Source load | 1.672 (1.639–5.467) |
| Cold model inspection/assembly | 9.248 (9.083–16.713) |
| First approach, including lazy timestep factor preparation | 14.492 (14.387–15.087) |
| Indent | 26.204 (25.887–27.439) |
| Hold | 27.115 (26.354–27.416) |
| Withdraw | 25.017 (24.675–25.459) |

Inspection does not prepare the first timestep factor. The final native sequence
observed approximately 187–190 ms per action, including 100 ms polling and PNG
capture/encoding/write overhead. Those are instrumented desktop observation
intervals, not presentation latency or a display-speed improvement claim.

```sh
pnpm --dir desktop build
.venv/bin/python scripts/check_planar_blade_maneuver.py \
  --prepare-engineering testcases/outputs/planar-blade-maneuver-NEW
.venv/bin/python scripts/check_planar_blade_maneuver.py \
  --case testcases/outputs/planar-blade-maneuver-NEW/generated-planar-blade-maneuver.ressectionlab \
  --actions testcases/outputs/planar-blade-maneuver-NEW/actions.json \
  --uniform-oracle --output testcases/outputs/planar-blade-maneuver-check-NEW
```

Open the generated native case in the actual desktop, select Rehearse →
Surgeon / Microscope, and use the four fixed blade buttons. For raw samples,
source refusals, actual native saves, screenshots, hashes and hidden/cold-reopen
commands, see the ignored local evidence index:
`testcases/outputs/planar-blade-maneuver-20261030/README.md`.
The source audit can be reproduced with the checker's `--audit-source` option;
it requires the existing local source archives and never downloads missing data.
