# Public retained-surface contact on the shared native engine

The backend can now execute an explicit generated goal: contact one declared cell
with the probe while retaining it. This adds a public goal and scoring contract
to the existing persistent tissue/tool transitions, observation, strategy sealing
and exact full-tool frame exporter. It does not add a separate simulator.

Near-goal scripted and SEARCH sequences aspirate then probe for modeled return
+0.708. For the fixed costly goal, scripted contact costs more than its declared
value, while SEARCH chooses STOP for zero. All removal is charged; tumor-removal
reward and private anatomical labels are unused. The geometry-only probe measures
no force and reveals no additional anatomy. No learned policy was run here.

The distinct v2 envelope binds the public goal to the source, native cell, crop,
physical frame, objective and committed contact state. The bridge exposes
`executePublicSurfaceContactEpisode` with exactly the fixed fixture, near/costly
goal and scripted/SEARCH selector. Same-case installation clears the old actor
and matched pair. This new replay is transient: Save explicitly refuses it until
v2 persistence is implemented. Legacy aspiration execution and persistence remain
separate and unchanged.

Root canonical regression passes 34 tests in 18.88 seconds. Independent core
review passes 33 controls, including four full saved replays and private-reference
invariance; independent bridge review passes 21 cancellation/publication/scope
controls. Exact sources and the canonical command are in [source-index.json](source-index.json).
Desktop v2 admission, goal rendering and live inspection are the next integration
step. These generated software results establish no tissue-force accuracy,
patient transfer, surgical outcome or RL advantage.
