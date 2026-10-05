# Frozen TRAIN transfer: access and proposal diagnosis

This diagnosis uses only the saved corrected float64 transfer records and existing geometry code. No patient arrays, new native previews, training, SELECT cases or unopened cases were accessed. The [receipt and JSON-only reproduction script](../artifacts/frozen-transfer-access-diagnosis-v1/receipt.json) bind the preparation records and inspected code hashes.

| Initial inventory | PAT22 | PAT25 | PAT28 |
|---|---:|---:|---:|
| Supplied target cells | 13,915 | 16,526 | 10,269 |
| Emitted / accepted actions | 78 / 76 | 78 / 0 | 78 / 70 |
| Rejections | 2 working reach | 78 shaft blocked | 8 working reach |
| Target centers outside accepted sweep AABB | 10,186 (73.20%) | No accepted envelope | 7,509 (73.12%) |
| Target outside support / actor crop | 0 / 0 | 0 / 0 | 0 / 0 |

All three inventories contain every declared slot: 13 columns × two tools × three endpoint families, with no cap omissions, duplicates or unavailable slots. This is completeness of the declared catalog, not completeness of possible tool paths. Both generic research tools have 120 mm working length; tip/shaft radii are 1.25/0.45 mm and 2.25/1.1 mm.

## PAT22 and PAT28: narrow lateral sampling

The [provider](../src/resectionlab/native_proposals.py) uses parallel source-axis columns at offsets 0, ±2 and ±3 source voxels around one access. It proposes the first exposed remaining cell and the proximal/distal nominal-target cells in each column. These approximately 1 mm grids therefore sample a narrow lateral neighborhood. Although the tool specification permits 35° access angles and the aperture radius is 6 mm, this provider generates no oblique directions or other access windows. Untested directions are not established as feasible.

The accepted axial bounds are 37.75 mm for PAT22 and 31.75 mm for PAT28. Their most distal target centers are 36.50 and 30.50 mm respectively: **zero target centers exceed those axial bounds**. The large initial omission is consequently lateral relative to this narrow catalog, rather than insufficient target depth coverage. The rejected rays are exposed-opening proposals exceeding working reach, not target endpoints. Every accepted ray is inside the actor crop; both methods retain the full supplied nominal target.

The [diagnostic AABB](../src/resectionlab/spatial_policy_diagnostics.py) encloses the active sweeps optimistically. Centers outside cannot lie inside that initial envelope; centers inside are not thereby removable. It ignores gaps between rays, cell containment and sequence interactions. These percentages are neither removed fractions nor a proof about later dynamic inventories or complete resection.

## PAT25: valid rejection within a poorly established access choice

Every family fails for both tools: 39 shaft rejections per tool, 26 per family. The central exposed-opening endpoint lies only **0.500003 mm** inside the access plane, yet both tools fail. This is not solely a distal-target reach problem. Frozen policy, greedy and random therefore all return STOP with zero removal.

The [access rule](../scripts/prepare_real_training_cases.py) selects the target cell nearest its centroid, walks to the first non-support cell along each of six source axes, then chooses the shortest physical exit. PAT25 chooses boundary voxel `[91.5,139,118]`, 17.50 mm from representative `[109,139,118]`, with inward +x. This rule checks neither the full entering shaft nor whether that local exit connects to the external free region.

The [native checker](../src/resectionlab/native_resection.py) tests the entire shaft against remaining tissue before crediting that microstep’s active-tip removal. The tested paths genuinely fail this declared geometric model. However, saved rejection rows omit the first failing pose and blocked cells. They cannot distinguish support behind the access, neighboring-cell collision, or a later insertion obstruction; neither an internal gap nor an alternative viable path is proved. The acknowledged estimated support remains unreviewed, and no cortical or clinical clearance follows.

## One prospective diagnostic

Predeclare PAT25’s six already-recorded axis exits as six separate hypothetical accesses, keeping source labels, support, tools and all checks unchanged. Evaluate only their initial 78-slot catalogs: at most 468 previews, one CPU, 180 seconds, 6 GiB; retain timeouts and all failures. Record first failure pose, insertion phase, blocked-cell bounds and external-free connectivity, alongside accepted counts and initial target envelopes. No optimization, cuts or outcome-selected window replacement. This would isolate access selection from rejection mechanics while preserving the original STOP-only result; it would not establish clinical paths or solve the separate lateral-coverage problem.
