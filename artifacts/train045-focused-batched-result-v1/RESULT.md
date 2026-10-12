# Complete focused-catalog comparison

Search remains the baseline. Both methods now finish their matched 24-action routes and pass fresh independent full-tool geometry evaluation. Frozen TAIL removes 89 target cells versus GREEDY's 90, with zero outside-target removal in either route. The learned useful-gain criterion fails. This is annotation-assisted simulation on one exposed TRAIN patient, not measured surgical performance or held-out validation.

| Complete endpoint | GREEDY | Frozen TAIL |
|---|---:|---:|
| Target cells removed | 90 | 89 |
| Target volume removed, mm³ | 85.8307949024 | 84.8771194035 |
| Outside-target volume removed, mm³ | 0 | 0 |
| Nominal return | 84.7363573379 | 83.7840818390 |
| Tool motions / changes | 24 / 5 | 24 / 5 |
| Tool travel, mm | 224.4375644794 | 223.0375644794 |
| Online planning, seconds | 87.419608500 | 98.937831417 |
| Online previews | 2,484 | 2,478 |

Both routes share the genuine twelve-action prefix. The full target is 24,128.943798 mm³; even GREEDY removes only 0.3557%. The final TAIL return deficit is one target cell's value (−0.953675499) plus 0.0014 shorter-travel savings. At the common state for action 14, the tail prediction changes the choice between equal-immediate-reward rays. At its later states 22 and 23 it chooses one-cell suction over available two-cell aspirator actions. These are local ranking observations, not proof that changing one choice repairs the entire route.

## Runtime improvement

The change authenticates each page once within a validation pass while retaining every packet, certificate, score and canonical commit check. No result is cached across state mutations. Models, rewards, action order, positive-immediate selection guard, geometry and planning caps remain unchanged. The original 23-action trace and both later partial/complete traces reproduce exactly after separately authenticating each release's own admission identity.

On one 26-action, five-page catalog, four alternating reference/batched timing pairs reduce median validation from 0.934992 to 0.202199 seconds (78.37%). Both predefined timing gates pass; snapshots, certificate/score counts and outcomes remain identical, with zero new previews/transitions during the timing calls. Catalog construction and the profile phase cost 208 previews and 9.971959 seconds. This narrow repeated-state measurement is not a general end-to-end speed claim.

## Evidence and complete costs

The [inventory](INVENTORY.json) pins source, configuration, control receipts, sealed plans, final outcomes and independently checked reports. Bulk traces remain local with hashes. All 59 generated controls and the actual isolated import check pass. The [independent saved-result audit](snapshot/independent/REPORT.txt) passes 7,447 checks; it verifies saved geometry certificates rather than implementing a second geometry engine.

The scientific worker completes in 325.794646 parent seconds, with 1,263,321,088 bytes sampled peak memory, 10,932 previews, 84 transitions, 12 frozen forwards, two checkpoint loads and **zero optimizer updates**. Including the three preserved failed attempts: 27,635 previews, 201 transitions, 23 forwards, eight checkpoint loads and 890.950853 parent seconds. Controls and saved audits are additional costs. The first saved reader exceeded its 128 MiB allowance; the identical reader passed under an explicitly revised 512 MiB allowance. Both receipts are retained. Every owned process is terminal and reaped.

The shared stage closes within its 8 MiB reservation. The [dated storage record](STORAGE.json), before this compact archive and later Git writes, records 1,822,773,248 bytes additional allocated growth and 352,410,738,688 bytes free. No dataset or weight acquisition occurred.

## Next campaign under October 12 steering

The next deliverable is a mature RL-trained checkpoint, learning curves and a matched search comparison, using development partitions only. A candidate local task uses the blocked TRAIN045 suction goal that becomes executable after four clearing actions. Fresh reference replay must establish solvability before freezing a prospective goal-completion reward, environment and evaluation rules. The new task must permit costly preparation; the old positive-immediate guard remains only in its historical baseline.

Use progressively larger experience budgets and compare frozen versus trainable spatial encoders under the same PPO settings. Count unique patients separately from episodes, repeated states, decisions, model queries and optimizer updates. Keep simulator rewards and outcomes explicitly labeled. Physical validation continues separately and is not a prerequisite for this bounded experiment. Retain search unless complete-task performance improves or planning cost falls without worse outcomes. The current negative comparison does not authorize an accuracy or generalization claim.
