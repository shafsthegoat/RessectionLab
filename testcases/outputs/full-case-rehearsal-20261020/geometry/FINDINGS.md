# Exact sub-06 case geometry and API findings

Inspected the primary checkout and the exact intact `testcases/cases/meningioma-sub-06-standard-v2.ressectionlab`, without source mutations, model runs, new data, RL, or UI automation. All counts are freshly computed from source arrays. `source_geometry.json` retains hashes, full provenance, regions, affine, matching strict guard report and metadata. Original source file SHA256: `ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838`.

## Denominators

| Region | Source membership cells/mm³ | Owned material cells/mm³ |
|---|---:|---:|
| Brain | 1,667,660 | 1,389,147 |
| Tumor | 45,115 | 43,573 |
| Bone | 753,872 | 732,802 |
| Vessels | 141,681 | 141,681 |

Ownership is brain → tumor → bone → vessels; later layers own overlaps. 1,542 source tumor cells are owned by later bone/vessel layers. Source tumor overlaps bone by 1,184, vessel by 364 (six overlap both). The engine removal percentage denominator must be 43,573 mm³; source annotation coverage can be reported separately against 45,115 mm³. All voxels are 1 mm³.

Shape `[220,220,181]`; affine diagonal `[-1,1,1]`, translation `[109.53500366210938,-123.558837890625,-554.25]`. Frame RAS+, units mm. Tumor bounds are `[-42.464996,-20.558838,-450.25]` to `[7.535004,41.441162,-410.25]`. Owned centroid `[-15.768074,7.887471,-428.875892]`.

The intact original main branch has zero actions, zero removals and no embedded protection contract. Matching protection JSON/NPZ load successfully into a separate strict research engine: 11 source-bound unreviewed estimated structures, clearance 2 mm, critical displacement allowance 0 mm, no missing required mask. This does not establish complete functional anatomy. Guard contract `sha256:bd086e3832d76c44e09ccec7e5cc64a77a27f2b2affcd76d5c51ea3fc06cebfa`.

## Access geometry

`access_routes.json` checks straight axis columns from image exterior toward first owned target, allowing bone and excluding all owned normal brain/vessel above it. This is an access hypothesis from supplied geometry, not reviewed anatomical identification.

Superior approach is most favorable: 2,204 candidate columns. Maximum lateral window clearance is 11.18 mm at index `[120,154,133]`, target entry RAS `[-10.464996,30.441162,-421.25]`. Center ray contains bone at -420.25 through -406.25 mm Z. Axis inward -Z, xyzw quaternion `[1,0,0,0]`. Image-top exterior staging plane chosen by helper is Z=-366.25.

Left lateral (-X outward) has 907 columns, maximum window clearance 7.62 mm at target entry `[-17.464996,26.441162,-423.25]`; center overlying bone is 27 mm thick. Anterior (+Y outward) has a main 535-column window, clearance 5.66 mm at `[-28.464996,18.441162,-424.25]`; overlying bone is 32 mm. Other axial directions have no meaningful brain/vessel-free window.

Superior target cells above the highest normal brain/vessel in their column total 40,360 (92.63% of owned target). This is an upper bound before exact shaft, guard and bone constraints. Window inset >4 mm leaves 21,099 candidate target cells, inset >5 mm leaves 14,284. A single ~9.68 mm radius opening around the maximum has only ~3,072 target cells under its conservative disk; multiple overlapping openings/frontiers are needed for >10,000 mm³. `engine_access_candidates.json` lists eight deterministic broad opening centers.

## Public API route mechanism

`SurgicalEngine.preview(action)` clones and calculates a transition without committing; `step(action)` commits accepted/partial transitions. `clone`, `snapshot`, `restore`, `replay`, `metrics` are available. Every physical action needs exact preview/step; geometry is not an admission override. CUSA effect removes only soft_tissue/tumor, but it can remove normal brain because normal brain is compatible: a healthy-free director must reject candidates whose predicted or actual removed source labels include non-target normal tissue. Bone must first be opened with source-preserving drill actions. Shaft collisions and guards are independent constraints.

`src/resectionlab/full_case_geometry.py` contains authorized read-only proposal functions:

- `access_candidates(engine, limit=8)` proposes bone-only superior windows and poses, drill radius ≤ the declared profile maximum, and geometric target capacity.
- `next_removal_candidates(engine, access, slot='left', limit=8, radii=None)` returns ranked raw public `apply` actions for a CUSA already oriented superior-down. Every path first retreats along its existing vertical shaft, moves laterally above the source image, then descends. The proposed radius is checked over the whole path. Working cover must be target/free; shaft material must already be in the preceding aspirator removal cover. Deep columns clip above neighboring brain/bone/vessel boundaries to produce shallower target frontier proposals. Root exact preview remains mandatory.

`place` requires free space for tip plus 40 mm shaft, and cannot be used while inserted/active. `exchange` requires deactivation, release and withdrawal. `withdraw` checks existing material contacts. The helper therefore does not generate placement or exchange while inserted. Whole-cell removal lets a drill remove an intersected last bone voxel before its tip enters adjacent target; broad drill radius will still stop at the highest target/bone boundary in the footprint, so tapered or overlapping access may be needed.

## Evidence limits and performance

Tumor is an unreviewed Raidionics meningioma proposal without supplied ground truth; brain is a SynthStrip estimate; bone is a CT 300 HU boundary proposal with unreviewed CT/MRI registration and source brain/bone conflict warning (~8.9%); vessels are experimental T1CE vesselFM with unknown vascular completeness. 44,496 source tumor cells overlap the brain extraction mask. Extra-axial character and surgical cleavage plane cannot be established from this overlap alone.

Attachment, dural tail, artery/vein/sinus identity, tumor feeders, adherent vessels, cranial nerves, optic pathways, motor/language cortex and complete functional tracts: **UNKNOWN** where absent. The source has no explicit attachments, strands, bleedingSources or functionalEvidence property records. Do not synthesize attachment evidence from adjacency or missing records.

Source/workspace/environment/strict-mask load and geometry summary took ~1.32 s. No full action engine benchmark was run by this delegate, avoiding simultaneous physics runners with root. Candidate enumeration is geometry-only but can involve many exact capsule queries; it returns a small ranked list. Module passed `py_compile`; root retains transition/replay verification ownership. Existing source arrays were only read. Source preservation hashes are in JSON for root's final after-check.

## Exact tapered frontier probe outcome

Root supplied `heartbeat/broad-probe-snapshot.json`: strict source-bound engine, 2,632 mm³ bone removed by three broad burr strokes. This delegate restored it with the public replay API (85.27 s), legally withdrew/exchanged/placed a CUSA, and exact-previewed/committed one exposed target voxel. Remaining broad-hole candidates had no actual tip effect; that failure identified and fixed a proposal-ranking bug: collision working cover includes a proximal segment, while the remove effect uses only the tip centerline sweep. Ranking now uses the latter and rejects zero-effect proposals.

A second and final public restore of that saved prefix took 91.83 s. At the same superior center, an authorized 3 mm tapered burr stroke removed five more bone cells before legitimate target contact. Direct center CUSA still stopped at residual bone, but the helper found three off-center continuous vertical frontier paths at radius 1.3 mm. Exact preview and commit removed 138, 137, and 33 further target cells (309 target total). The next geometry enumeration returned no further legal candidates at this opening, so the worker stopped searching it. No brain/vessel tissue was removed; no vessel wound or displacement was produced. Candidate enumeration with the retained-roof filter took ~3.5–4.3 s, with individual exact CUSA preview/commits about 7–8 s on that process's earlier engine implementation.

The actual native artifact is `tapered-frontier-prefix.ressectionlab`, branch `tapered-frontier-prefix`: 19 recorded public actions, strict matching protection, 309 owned target + 2,637 bone removed. State `sha256:310c464e860a85ac68721995698e86e655b650d4e626e86e00426501e1083085`. Native Save/Open preserves original source semantic hash and exact snapshot equality. The source case file SHA256 remains unchanged. Details: `taper_results.json`, `taper_snapshot.json`, `tapered-frontier-prefix-verification.json`. Root owns final optimized replay verification and continuation.

The revised `access_candidates(engine, limit=160, max_radius_mm=3., separation_mm=3.5)` returns 99 dense narrow windows, beginning with the same maximum-clearance center so it can follow a broad exposure. Later centers score superior target column capacity and stay within declared brain/vessel-free windows. Their conservative disk union contains 21,629 owned target cells in 1,092 columns (upper bound only). Files `dense_access_candidates.json` and `dense_access_capacity.json` retain the concrete proposal. This supports a real continuation strategy; the 21,629 cells are not claimed removed.

## Proven subvoxel passage and preserved 516 mm³ prefix

A concrete false negative in the original planner clipped above a lateral obstacle by its full radius plus a fixed 0.7 mm, eliminating legal target contact. Replaced it with the exact vertical capsule/cell cap computed from horizontal distance to the voxel box. Small radii 0.3/0.45/0.6 and a bounded fallback of ±0.25/0.4 voxel offsets on six exposed columns avoid conflating best poses with voxel centers. Candidate ray search extends to drill radius + 2 mm (override `searchRadiusMm`).

The unchanged engine inflates swept envelopes according to the actual action microsteps. Explicit 0.05 mm waypoints through the bony entry **and** at the final target frontier provide a faithful narrow contact path; fine waypoints only at a deep endpoint leave the bony entry coarse and may fail. No global collision or physics law changed. Candidate prefilters use the corresponding 0.0251 mm frontier inflation and public exact preview remains the admission authority.

The retained live public-engine worker opened one additional real R3 access at index `[119,125]`, adding 529 bone cells. A direct 0.6 mm CUSA stroke removed one target cell, while same-center 1.3 mm and same-center fine strokes remained blocked. An R1 burr added no bone and was not repeated. A subvoxel CUSA ray at RAS XY `[-8.464996337890625,1.191162109375]` with radius 0.6 mm and endpoint Z `-444.3747333570265` then passed the actual exact engine. Its 217 public path poses refined the entry and final frontier. Preview and identical commit both accepted, adding **206 target cells (310 → 516)**, with zero added bone, zero brain/vessel removal, no wounds or displacement. This proves passage and actual whole-cell target effect rather than a geometric capacity estimate.

Actual native recovery artifact: `subvoxel-frontier-516.ressectionlab`, 36 actions, 516 target + 3,166 bone removed, strict matching source contract. State `sha256:c63c247f5a7bf622f69297c23dad11d033b7a2046281898277e5ad04a6b73005`. Native reopen preserves the exact snapshot and original case semantic hash; source file SHA256 remains unchanged. `first-passage-preview-commit-proof.json` preserves the exact successful action, matching preview/commit state IDs and timings. `subvoxel-frontier-516-verification.json` records native checks. The earlier 19-action native remains untouched. The live worker was closed cleanly after this checkpoint; no further physics runs were performed by this delegate.

## Optional local pocket proposals (not executed)

`src/resectionlab/full_case_debulking.py` separately provides `local_debulking_candidates(engine, slot='left', limit=6, radius_mm=1.5)` with optional maximum 2.5 mm. It proposes bounded axial core positioning at the currently recorded aperture before any widening near a healthy boundary, or static/short axial core removal at the generic 1.5 mm catalog aperture. The physical shaft stays 0.5 mm. It checks complete working/shaft covers, ranks actual tip target effects, retains target/free-only constraints and requires public preview. Positioning may have zero removal and must be admitted as a separate bounded phase. Widened aperture persists and withdrawal must be checked or explicitly narrowed in recorded history.

Read-only reconstruction of the exact 516 mm³ checkpoint (matching state digest, explicitly not a replay) produced four positioning proposals in 0.40 s: 2–5 mm axial retreats at radius 0.6, each exposing a predicted 20-cell static 1.5 mm pocket. These are **unexecuted geometric proposals**, preserved in `local_debulking_proposals.json`. The proposer is not integrated into the director. Assumptions are generic engineering apertures and whole-cell effects, without calibrated patient energy or aspiration mechanics.

## Stationary second-hand field assistance proposal

`src/resectionlab/full_case_assist.py` supplies the read-only `assistant_field_candidate(engine,current_access)` contract. It groups actual removed target source cells by column, chooses a point 2 mm above the highest removed target in that column, and requires at least 15 mm XY separation from both current left tip and planned access. The full vertical physical tip/40 mm shaft insertion capsule is clear in current shared occupancy at radius 0.67 mm (unchanged 0.5 mm physical radius plus conservative 0.17 mm motion margin). The separate stationary suction field radius is 1.5–3 mm, flow 20 mm³/s, force zero. Candidates must contain actual current debris/fluid; empty fields are declined.

Read-only position-58 source-bound inspection found RAS `[-9.464996337890625,30.441162109375,-418.25]`, outside Z=-366.25, superior-down quaternion `[1,0,0,0]`, field radius 3 mm. The current left and access are 27.82 and 27.17 mm away. Its field contains **176 mm³ actual existing debris**, fluid zero. Proposal generation took 0.024 s and left the reconstructed checkpoint state unchanged. `assistant_field_proposal.json` explicitly marks this as unexecuted geometry inspection. Root owns legal exchange/place/apply/activation and actual shared-clock suction verification. Right-hand suction must deactivate and withdraw before a planned left access approaches within 10 mm XY. No retraction, manufactured fluid or anatomy modification is implied.

## Narrow-channel withdrawal proposal

`src/resectionlab/full_case_withdrawal.py` adds `withdrawal_candidate(engine,slot='left')`, a read-only public-action planner. It retains the existing physical shaft and collision law, selecting outward 0.333/0.05/0.025/0.01 mm waypoints only when the actual conservative swept shaft/working cover and other-hand capsules permit them. It caps paths at 512 poses. A complete checked exterior route uses `withdraw`; an incomplete inactive prefix uses `move` and remains inserted. Any necessary target-only CUSA active pullback is a separate `apply` proposal requiring normal target-only public-preview admission, and never claims the hand withdrawn. Root setup must deactivate/release first and explicitly record any aperture adjustment before planning.

The exact stopped position-58 checkpoint has inactive inserted CUSA radius 0.45 at `[-11.464996337890625,2.691162109375,-416.25]`, quaternion superior-down. A default coarse withdrawal can inflate its 0.5 mm shaft by ~0.167 mm and hit the narrow bony entry that was entered with fine waypoints. Read-only inspection found a complete **212-pose withdraw** to `[-11.464996337890625,2.691162109375,-366.25]`, using adaptive local segments and unchanged radius/shaft. Proposal generation took 0.618 s and left the checkpoint state unchanged. `withdrawal_proposal.json` contains the raw action and explicit nonexecution/nonreplay labels. Root owns actual public preview/step verification; no physics run was performed by this delegate.

The same lifecycle issue can apply to a wide stationary suction field: its generic field radius should be explicitly reduced in recorded parameters before withdrawal through the narrow entry if the existing withdrawal tip-contact rule otherwise refuses it. This does not alter the physical 0.5 mm shaft or the collision law.

## Large source-pure core and pivot fan extension

`full_case_debulking.py` now treats `radius_mm` as a bounded maximum, default 1.5 mm. An explicit 6 mm request falls back through 4/2.5/1.5 when a larger whole working pocket is unavailable. It ranks core positioning by the largest independently checked source-pure pocket capacity. Small target effects near an interface no longer tie with a much better interior core. A verified static tip effect of at least 500 cells returns immediately rather than spending time on many fan proposals.

The preserved 516 mm³ passage gives concrete source-geometry capacity along its existing cleared axis: 2 mm retreat permits a 1.5 mm pocket (20 new tip cells); 4 mm permits 2.5 mm (101); 8 mm permits 4 mm (395); 12 or 16 mm permits 6 mm (**1,228**). Whole working capsules contain only target/free material and the unchanged physical shaft is free. `core_aperture_scan.json` records the source-bound read-only scan.

A single focused **public** verification then freshly replayed all 36 actual actions from `subvoxel-frontier-516.ressectionlab` (110.78 s), with exact snapshot identity. The helper selected a 12 mm axial retreat at the existing 0.6 mm aperture. Its public preview and actual step both accepted with matching state IDs and zero removal. At the actual resulting pose, the helper returned one static R6 action. **Public preview accepted in 0.498 s, adding 1,228 owned target cells (516 → 1,744), with zero added bone/brain/vessel removal, no wounds/displacement, and unchanged committed state.** The R6 removal was previewed, not committed in this delegate's test. Root owns actual primary-branch continuation. Full actual proof: `r6_core_public_preview.json`; compact proof: `r6_core_proof_summary.json`. No source or collision law changed.

The optional lateral extension uses continuous ≤512 fine poses with orientation rotating about a source-derived inner-aperture pivot on the current cleared shaft ray, rather than assuming lateral shaft translation through intact bone. Each swept working and shaft envelope is checked in engine order. Prediction credits only preceding tip removals before checking each subsequent shaft, never future removal or normal tissue. It ranks actual new source target tip cells, checks other-hand capsules, and preserves public preview admission. The pivot is an engineering fulcrum inferred from the existing source bone/ray, not a measured surgical attachment or calibrated mechanical boundary.

At an unexecuted 2 mm retreated core pose, default 1.5 mm fan proposals were limited by full geometry to ~0.3–1.1 mm lateral reach and 30–35 new cells. This is weaker than the proven large interior pocket; the implementation prioritizes the latter. Fan tests are source-bound geometry predictions only, preserved in `pivot_fan_geometry.json`; no lateral patient physics run was executed.

## Bounded six-axis residual access comparison (no physics)

`axis_access_comparison.json` compares the exact original source and the saved position-100 checkpoint, with 1,268 owned target cells removed and **42,305 mm³** residual. There are no holds or active deformation nodes. Bone and free cells may precede the target along a ray; any retained normal brain, vessel or other nontarget label blocks that ray. Counts are optimistic direct-point-ray capacities before physical tip/shaft, guards, cutting geometry and registration review. They are not globally attainable removal volumes.

| Outward direction | Initial capacity | Residual capacity | Residual candidate columns | Best window clearance | Extra residual outside superior set |
|---|---:|---:|---:|---:|---:|
| Superior +Z | 40,360 | 39,097 | 2,177 | 10.82 mm | — |
| Left lateral −X | 20,692 | 20,108 | 907 | 7.62 mm | 1,555 |
| Anterior +Y | 16,887 | 16,465 | 536 | 5.66 mm | 1,147 |
| Posterior −Y | 3 | 3 | 1 | 1.00 mm | 0 |
| Right lateral +X | 0 | 0 | 0 | 0 | 0 |
| Inferior −Z | 0 | 0 | 0 | 0 | 0 |

Superior remains the strongest source-geometry route. Its residual maximum window center is index `[132,150,129]`, target entry `[-22.464996,26.441162,-425.25]`, with 17 center-ray bone cells outward before the target. Even a >5 mm inset of the residual superior candidate window retains 10,574 mm³ direct-ray potential (before exact tool geometry).

The two meaningful alternatives are left lateral and anterior. Left lateral has the larger and wider window: entry `[-17.464996,26.441162,-423.25]`, 27 overlying bone cells, inward axis +X, quaternion `[0,0.70710678,0,0.70710678]`; exterior staging X=-117.464996 at the same Y/Z. Anterior entry is `[-28.464996,18.441162,-424.25]`, 32 overlying bone cells, inward axis −Y, quaternion `[0.70710678,0,0,0.70710678]`; staging Y=103.441162 at the same X/Z. These poses only declare frame/orientation, not exact physical admission. The unusually thick CT-proposal bone and unreviewed registration remain material limits.

Left and anterior together add only **1,862 mm³** beyond the superior direct-ray set because their capacities overlap it heavily; anterior contributes only 307 additional cells after left is also considered. The six-axis residual union is 40,959 mm³. The other 1,346 residual cells are outside these straight fixed-axis point-ray sets, not proven globally unreachable: five already removed cells were outside the initial six-ray union, illustrating how valid curved/multi-segment paths can exceed this limited representation.

Right and inferior have no direct bone/free-only target window under the supplied labels. Posterior's isolated one-column, three-cell ray with 60 bone cells outward has no meaningful robust physical window and is not a useful next engineering candidate. No new anatomy, attachment/functional assumptions, production changes or physics executions were introduced by this comparison.
