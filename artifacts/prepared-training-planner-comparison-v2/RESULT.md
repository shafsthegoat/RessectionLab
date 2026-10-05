# Prepared TRAIN planner comparison V2

The declared V2 attempt completed with **four complete comparisons and 12 accepted simulation histories across six prescribed TRAIN cases**. Greedy search obtained higher geometric reward than the frozen imitation policy on all four completed cases, while consuming more guarded online time and native previews. This is a development comparison of an existing policy with search, with **zero optimizer updates or adaptation**; it does not demonstrate an RL improvement, held-out generalization or clinical safety.

PAT05 is the case used to train the frozen imitation checkpoint. PAT22, PAT25 and PAT28 are development-transfer cases within TRAIN. PAT16 and PAT20 retain their historical support conflicts: respectively 19 and 125 annotated target cells lie outside fixed support. Their three arms were not executed and have null outcomes, including STOP. They remain in the six-case denominator; no target clipping or support expansion resolves them here.

| TRAIN case | Frozen IL reward | Greedy reward | Target removed, IL / greedy (mm³) | Normal removed, IL / greedy (mm³) |
|---|---:|---:|---:|---:|
| PAT05 — trained case | 290.506010 | 410.312426 | 345.001 / 493.002 | 271.001 / 412.001 |
| PAT16 — support block | null | null | null | null |
| PAT20 — support block | null | null | null | null |
| PAT22 — development transfer | 672.297005 | 789.096658 | 697.998 / 819.998 | 127.000 / 153.000 |
| PAT25 — development transfer | 584.914177 | 753.905095 | 615.003 / 787.004 | 149.001 / 164.001 |
| PAT28 — development transfer | 493.534013 | 612.537775 | 530.999 / 643.999 | 186.000 / 156.000 |

Each completed STOP arm has zero reward and zero removed volume. Every completed IL and greedy arm makes three non-STOP simulated actions. The frozen reward is target removal − 0.2 × normal removal − 0.03 per action − 0.001 × complete-tool path length − 0.03 per tool change. Partial contact receives zero removal credit. Greedy's larger target removal accompanies greater normal removal on PAT05/22/25 and lower normal removal on PAT28; its reward advantage is 116.799653–168.990918 units. These are simulated contained-cell removal volumes, distinct from contacted-tissue upper bounds. **Normal removal is not neurological harm: clinical deficit probability, motor surrogate and language surrogate remain null.** Target removal is only 3.02–5.17% for IL and 4.31–6.27% for greedy under this short horizon, not a complete operation.

All arms start from separate clones of the same prepared state, with matching source, selected access, initial inventory, tools, objective and horizon. Access selection uses any statically admissible entry, then distance/axis/sign, without reward ranking or fallback. The representations differ: the CNN sees its declared 64³ crop, whereas greedy search uses the full permitted nominal fields. The selected crops retain all nominal target mass; this does not establish complete tool visibility or functional/vascular coverage. The comparison cannot isolate architecture from representation or privileged simulation access.

| Case | Shared preparation (s) | IL online (s / previews) | Greedy online (s / previews) | IL / greedy audit (s) |
|---|---:|---:|---:|---:|
| PAT05 | 26.807027 | 18.833501 / 140 | 35.875431 / 264 | 3.788512 / 3.752817 |
| PAT22 | 24.706589 | 17.287635 / 140 | 32.546892 / 280 | 3.926142 / 4.046549 |
| PAT25 | 22.945717 | 22.352285 / 140 | 42.618243 / 264 | 3.662700 / 3.906170 |
| PAT28 | 24.849664 | 16.547935 / 140 | 30.771239 / 264 | 3.409498 / 3.352527 |

The equal online limits are **90 s and 468 actual native previews per arm**. Guarded cost includes clone work, decisions, search planning, replay, successor inventories and durable terminal export. Shared loading/access screening/selected inventory construction and independent audits are outside that clock; shared preparation includes another 156 native previews per completed case. STOP takes 0.316–0.326 online seconds and zero previews. IL's summed actor-decision time is 0.775–0.971 s, substantially less than its full online cost. Greedy planning separately takes 15.154–21.029 s and scores 188/214/210/192 non-STOP candidates on PAT05/22/25/28. Its microsecond `actor_decision_seconds` records only replay callbacks, **not search latency**. Candidate scoring counts are distinct from native preview counts; global optimality is not established.

Patients and arms ran in fixed order, STOP → IL → greedy. Single-run timings are descriptive and confounded by ordering/cache effects; no randomized speed benchmark is claimed. Root recorded 383.645944 s for the whole call. Per-patient supervisor intervals were 86.798628–103.369040 s against 600 s. The maximum supervisor-sampled worker RSS was 2,751,889,408 bytes (2,624.40625 MiB); the separate worker high-water measurement reached 2,786,803,712 bytes (2,657.703125 MiB). Both are below 6 GiB, but sampled RSS may miss transient peaks. Online checks are cooperative, not a kernel-enforced 90 s interruption guarantee. No retry occurred.

The [independent saved-record audit](independent-check.json), SHA `b98dd4141082c9a88b2caf0ecc8a224b318f0cca58568a4a48657eb60401710f`, passed its first execution in 5.519 s. It verified the 70-file source archive, 214 unchanged inputs, histories, cell-union/reward arithmetic, accounting and closure. This checks software/simulation records, not physical or clinical validity. V1's tuple/list durable-history failures remain failed and preserved. All 12 complete metrics/history records match V2 exactly; decision records match after excluding `decision_seconds`, `transition_seconds` and `status`.

The [compact results](compact-results.json) retain unrounded values and input hashes. The [comparison figure](comparison.png) displays all six cases with null blocked rows and separate score/online-cost panels; its first render passed visual inspection without changes. Reporting read saved JSON only and performed no model, native, array or training execution. The parent's lossless archive preserves 136 raw files (242,388,356 bytes) in 6,236,986 compressed bytes, SHA `9de5fba4c15c7fb75f08cb7060ac4a96c85ccb2bf4ef0859f8bcfc572ad4916a`.
