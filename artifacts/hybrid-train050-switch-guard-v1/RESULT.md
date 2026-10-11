# Model-guided checking matches search with fewer checks

On development case TRAIN050, the frozen proposal scorer plus two conditional
checking rules reproduces every physical action and outcome of fresh lean
exhaustive search. The independent saved audit confirms the unchanged quality
gate passes. The original transfer failure and intermediate fallback result
remain unchanged.

| Measure | Model-guided checking | Fresh exhaustive search |
| --- | ---: | ---: |
| Supplied target cells removed | 37 | 37 |
| Outside-target removal, mm³ | 0 | 0 |
| Frozen nominal return | 34.830407 | 34.830407 |
| Movements / tool changes | 11 / 1 | 11 / 1 |
| Complete tool travel, mm | 95.570343 | 95.570343 |
| Online geometry previews | 870 | 1,092 |
| Source preparation through plan, seconds | 8.781208 | 10.248682 |

This pair uses **20.3% fewer previews and 14.3% less online wall time**, with the
same 37 removed cells and contact upper bounds. Timing is one fixed-order pair,
not a stable latency estimate. TRAIN050 has been used for development after its
original frozen transfer failure; this is not an untouched held-out result.

The original top-eight shortlist removed only six cells. Checking the rest of
the emitted catalogue when no positive action remained recovered all 37, but
incurred an extra suction action and two switches. The new rule additionally
compares unqueried current-tool choices before accepting a positive tool-changing
choice. It recovers the rank-46 search winner at the measured shared state.
It does not prohibit useful switches, hardcode patient geometry or change model
weights, reward, tools or world rules. This is supervised ranking plus conditional
search, not an RL update.

Current-tool comparison triggers at four states and adds 193 previews; the full
fallback triggers at seven. All 774 added checks, feature construction, gate
scoring, source construction and logging are charged. Both plans are sealed
before fresh uncached authoritative replay. Independent saved validation compares
all eleven physical actions, removed indices, source states, affine bindings and
each exact once-only expansion rule. Total planning/replay work is 3,924 previews,
48 transitions and 24 ridge scores. The parent finishes in 51.683 s with
521,355,264 B sampled peak, 4,700,747 B output and clean termination. One existing
ridge is loaded; zero fitting or optimizer updates occur. Original fitting and
data-generation costs remain in the balanced-v2 result.

Seven generated controls pass, including an improved current-tool choice, a
worthwhile switch still accepted, private-reference permutation, no repeated
queries and complete two-arm replay. The first fixture had no positive small-tool
action, causing three precondition failures; its evidence is preserved. Correcting
only invented tool dimensions made the intended controls executable, with
unchanged production bytes. No acquired comparison was rerun.

This qualifies the development-case comparison, not surgical use. Search itself
removes only 0.1892% of the supplied whole-tumor annotation, which is not necessarily
a prescribed resection target. These are rigid-tool geometric simulation results,
not measured tissue interaction, neurological harm probabilities or clinical
accuracy. No new physical-measurement agreement or population transfer is shown.

Next, check regression on the original TRAIN cohort and evaluate complete routes
that can clear sequential blockers. Keep EVAL reserved. Reuse current data; no
additional dataset is justified by these decision failures. The Mac workflow
remains intact; this prototype is not newly exposed in it.

The compact preservation index retains changed sources and exact results. Unchanged
helpers remain in preceding archives; canonical source is at `19f273b`. Full
repeated inventories remain local with audit hashes. Patient arrays and weights
stay outside Git.
