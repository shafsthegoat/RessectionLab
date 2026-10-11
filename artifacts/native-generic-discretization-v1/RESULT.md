# Actual generic-tool resolution control

The fixed continuous scene and fixed ten-cut sequences produce different native decisions when only the sampling grid changes. Complete sequence acceptance is **2/6 at 1 mm, 3/6 at 0.5 mm, and 6/6 at 0.25 mm**. Acceptance is not monotonic for every sequence, and the finest grid has not demonstrated volume convergence. This is a generated representation control, with no patients, learned policy, fitted parameters, or clinical outcome.

All 18 declared rows completed. The saved-only audit independently joined the declaration, release, recorded source identities, action sequences, state chains, scalar costs and geometry certificates. It did not rerun geometry, load arrays, or execute native tests. `saved-audit.json` contains the exact unrounded scalar projection; `audit_saved.py` reproduces these joins using only the standard library.

## Fixed comparison and verification

The anatomy is the analytic half-space z >= 0, with target z >= 2.8 mm. The actual generic suction has tip/shaft radii 1.0/1.4 mm and active length 3 mm; aspirator has radii 1.7/2.7 mm and active length 4 mm. Each sequence has nine axial preparations ending at z=1 mm on its predeclared physical lattice, then one central target stroke ending at z=2.2 mm. Anatomy, aperture, action coordinates/order, tools, reward and maximum microstep 0.0625 mm are unchanged across grids. Sampling is aligned, shifted by half a voxel in all axes, or tilted 20 degrees about y relative to the physical scene. This is not a joint rigid transformation of the anatomy and instrument.

The same physical action digest is present in all nine rows for each tool; both tools share the identical sampled anatomy within each spacing/sampling pair. Recorded prospective and terminal source maps agree. Release and control hashes match. Canonical source revision: `958e587f73ccfb8db3df843b99695475894d52ab`.

The predeclared continuous enclosure gives a positive clearance reference. Every preparation shaft stays below physical tissue (highest possible z is -0.6 mm for suction, -0.3 mm for aspirator). The nine ideal cuts cover the entire central target shaft's positive-z extent, with positive lateral and depth margins. Preparations end below the target plane; the final distal cap intersects it. This establishes feasibility in this idealized geometric construction, not native certification or physical tissue mechanics.

There were 137 native calls: 130 accepted and 7 rejected. All 43 remaining prescribed suffix actions are explicitly unexecuted following rejection, within the original 180-slot denominator. Rejections preserve the prior state and credit zero removal. Four rows reject with empty histories. All 14 nonempty histories passed the separate full-tool/frontier checker, including all three accepted prefixes that later reject a prescribed cut. Thus 11/18 complete sequences have full accepted certificates; accepted partial prefixes do not turn the other seven sequences into successes.

The parent finished with exit 0 and no stop reason in 28.055755 s; worker 27.710535 s; native calls 18.099388 s; independent audits 8.483284 s. Peak worker RSS was 351,191,040 bytes. One thread, no missing rows, incomplete audits, or exceeded caps. The saved launcher reports completion and unchanged source; process cleanup was owned by the root launcher.

## Exact first refusals

Failure positions below are endpoints of the first refused microstep, not exact first-contact locations. The other 11 rows accept all ten cuts.

| Grid/tool/spacing | Accepted cuts before refusal | Refused cut | Reason | Failure tip (mm) |
|---|---:|---:|---|---|
| Aligned aspirator, 1 mm | 0 | 1, preparation | Remaining-tissue shaft blockage | (0, 0, 0.9375) |
| Half-phase suction, 1 mm | 0 | 1, preparation | No newly fully contained surface cells | (0, 0, 1) |
| Tilted suction, 1 mm | 0 | 1, preparation | Remaining-tissue shaft blockage | (0, 0, 0.8125) |
| Tilted aspirator, 1 mm | 0 | 1, preparation | Remaining-tissue shaft blockage | (0, 0, 0.25) |
| Aligned suction, 0.5 mm | 9 | 10, target | Remaining-tissue shaft blockage | (0, 0, 1.3970588235294121) |
| Tilted suction, 0.5 mm | 9 | 10, target | Remaining-tissue shaft blockage | (0, 0, 1.1500000000000004) |
| Tilted aspirator, 0.5 mm | 2 | 3, preparation | Remaining-tissue shaft blockage | (0, 1.8, 0.9375) |

The four preparation shaft failures occur despite a strictly negative physical shaft height throughout those preparations. Their obstruction is therefore introduced by the represented occupied cells and shaft contact rule in this analytic scene, rather than by an actual intersection with z >= 0. The half-phase suction refusal is the complementary removal-quantization effect: its static union contains zero whole tissue cells, although the physical cutter has positive tissue intersection.

The two 0.5 mm suction target failures occur after all nine preparations succeed. The ideal continuous preparation union covers the target shaft, but those native cuts leave cells occupied where whole-cell containment was not established. The saved report does not retain blocker indices or per-cell cumulative coverage, so it cannot isolate a particular leftover cell or prove that accumulating unions alone would repair these two failures. Whole-cell geometry, prior-cavity ordering, and frontier rules remain part of the native decision.

Aligned suction is explicitly **nonmonotonic**: complete at 1 mm; nine preparations then target rejection at 0.5 mm; complete at 0.25 mm. Even the static contained physical union decreases from 14.5 to 13.1875 mm³ between the first two grids. The aligned center lattices at these resolutions do not define nested voxel cells: a 1 mm cell centered at zero has faces at +/-0.5, while a 0.5 mm cell centered at zero has faces at +/-0.25. Refinement therefore need not preserve which boxes are fully contained or which residual boxes contact the shaft. Aggregate acceptance improvement does not establish monotonicity.

## Removal sensitivity

Volumes below are **physical tissue volume**, obtained by clipping removed cells against the analytic half-space, not native full-cell credit. A partial/refused sequence's row contains only its committed prefix. Preparations remove zero physical target in every row; all listed target removal is from the final stroke.

| Spacing (mm) | Sampling | Suction accepted / 10 | Suction tissue / target (mm³) | Aspirator accepted / 10 | Aspirator tissue / target (mm³) |
|---:|---|---:|---:|---:|---:|
| 1 | Aligned | 10 | 14.5000 / 0 | 0 | 0 / 0 |
| 1 | Half-phase | 0 | 0 / 0 | 10 | 56.0000 / 0.8000 |
| 1 | Tilted | 0 | 0 / 0 | 0 | 0 / 0 |
| 0.5 | Aligned | 9 | 11.4375 / 0 | 10 | 78.6875 / 1.5875 |
| 0.5 | Half-phase | 10 | 14.5000 / 0 | 10 | 76.0000 / 1.3000 |
| 0.5 | Tilted | 9 | 12.5625 / 0 | 2 | 18.9182 / 0 |
| 0.25 | Aligned | 10 | 22.7891 / 0.114063 | 10 | 95.4922 / 3.204688 |
| 0.25 | Half-phase | 10 | 21.5625 / 0.150000 | 10 | 93.6875 / 3.037500 |
| 0.25 | Tilted | 10 | 21.8203 / 0.115517 | 10 | 94.7734 / 3.161778 |

Native sequence acceptance does not require removal of analytic target: aligned suction at 1 mm and half-phase suction at 0.5 mm accept the final stroke but remove zero target. At 0.25 mm all sequences accept and remove positive target, but the target volumes are still far below the continuous reference.

A post-run analytic calculation from the already fixed geometry supplies an exact target-volume reference. Only the final distal spherical cap reaches z >= 2.8. For radius r and cap height h = 2.2 + r - 2.8, its volume is pi*h²*(r-h/3): **0.4356341813 mm³ for suction**, **5.0684361478 mm³ for aspirator**. The fine-grid target shortfall is **65.6–73.8% for suction** and **36.8–40.1% for aspirator**, depending on sampling. This calculation selects no new action or parameter and is not an additional simulator run.

The static contained/contacted unions for the full prescribed geometry also remain widely separated at 0.25 mm. For aligned/half-phase/tilted suction they are [22.7891,37.8047], [21.5625,36.8125], [21.8203,37.4922] mm³; for aspirator [95.4922,131.8828], [93.6875,129.7500], [94.7734,132.2109] mm³. These bound the planned geometric union and do not certify an executable sequence. In each complete row, achieved physical removal equals the static whole-cell-contained lower bound. Passing all six fine rows therefore does not remove the geometric under-credit.

Even among sequences complete at both 0.5 and 0.25 mm, half-phase suction tissue removal increases 48.7%; aligned and half-phase aspirator increase 21.4% and 23.3%. Fine-grid phase/orientation differences remain, especially for small target caps. Three resolutions with these changes do not demonstrate convergence.

Native credit also includes outer-voxelized volume outside the true z >= 0 solid. At 0.25 mm the excess is 0–1.9453 mm³ for suction and 0–6.6328 mm³ for aspirator. Half-phase is zero because the horizontal tissue boundary aligns with cell faces; aligned and tilted grids retain excess. These quantities remain separate in the raw report and audit. All nonempty sequences have negative total frozen reward; this control does not establish a useful policy or positive complete-route return.

## Supported next representation change

The next representation candidate should retain **residual solid geometry inside boundary and cut cells**, for example with adaptive subcells and conservative inner/outer residual bounds. Subtract already executed active sweeps chronologically; evaluate shaft clearance against that residual before crediting the current cut; measure removal and target volume from the same residual intersections. Keep source labels, physical actions and reward weights fixed. In the analytic control, initialize boundary subcells from the known half-space rather than turning every intersecting source box into an entirely solid box. This does not justify inventing subvoxel anatomy for an acquired image.

Accumulating cut coverage may help residual cells, but cannot by itself remove the first-preparation false shaft collisions caused by initial outer voxelization. Conversely, changing reward fractions alone leaves the collision failure untouched. Neither contact alone nor an unlocalized volume fraction is sufficient evidence that a shaft path is clear.

Evaluate that candidate against the same fixed 18 rows and exact target-cap reference before more corridor-ranking optimization. The required improvement is stable physical acceptance plus smaller bounded volume error across sampling, with prior-cavity ordering and independent full-tool checks retained. This run proves that the current representation can create failures without learning or deformation in a simple scene. It does **not** establish that discretization is the principal cause of SELECT013's 48-action failure, validate a patient resampling rule, prove an optimal preparation count, or validate mechanics.

## Archive recommendation and identity

Retain the compact stage files listed in `archive-manifest.json`: adapter, declaration, tests, historical source-ready/README, generated-control receipt/log, root release/dispatch, prospective source pins, terminal report/launcher, this result and its saved-only audit. Keep the complete 252,072-byte report; it already contains every action row, certificate and scalar cost. Preserve the empty worker log for completeness. No arrays or bulk histories were saved. Do not duplicate canonical modules already recoverable from the pinned Git revision. The total curated archive is comfortably below 0.5 MiB; no new run or archive copy is performed by this review.

Key SHA-256 identities:

- Terminal report: `8e40ef8f6c4504c8eedb020687250c59cb4e58f869fe02bf530a9a001761bc9f`
- Launcher: `9df34b8d3ae5ddc4dda95d63bff83373427eb9e69c194c7ba7b7e235ee072628`
- Prospective source/declaration: `cb379d2bdeca324e27b04eafbd64453314a2985e4e17b3bb18e31ef3eadfc8c3`
- Adapter: `b4f07e29324a3dcc8fce2c861fc45b34fc1d2051176832f87e76ef3b3c235797`
- Declaration bytes: `a815e89510b153778a82307c3ae142e939c422187a51dcb618ba643ed1164081`
- Seven-control receipt: `e7a08a4670f1bd75f0e9d9c68c5286cd2f172df50aecd93f046dad2b08786f13`
- Saved-only audit: `b2cda9f3cb42bf15066be5bb638036054c8a45898beef9b3f4f2abd9df2a20a3`

The source-ready and README files preserve their historical pre-execution wording. The terminal report and launcher establish the later completed status.
