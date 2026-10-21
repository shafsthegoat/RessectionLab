# Bounded live-profile diagnosis

Evidence: primary-v2/profile-live-slice-from-1436.{json,pstats}, captured actual public actions1436→1438, original source unchanged. Pstats SHA256 `c41c738c5c98b9dd177e8b135a68ed39b517ffb5e51082bec771a50210fd3656`. Profile commit `3e0bd07c7a5ad7b84a9cf921c20f1c2564cd2c69`; checkout at inspection `3e0bd07c7a5ad7b84a9cf921c20f1c2564cd2c69`. No engine construction, replay, live-console/process/flag interaction, GUI, production/test edit or new benchmark.

**Rollback deepcopy is directly implicated and dominant in this slice.** Four public step/executions (two previews plus two committed actions) perform1280 microstep checkpoints. Direct `_execute`→deepcopy attribution is **125.249393s**, 48.28% of the instrumented total. Overall top-level deepcopy is 162.936224s, including 35.249173s from21 clone copies. Deepcopy has 174,197,578 recursive calls (1,328 outer calls); its atomic dispatch has167,235,866 calls. These cumulative categories overlap and must not be added together.

| Function | Calls | Self seconds | Cumulative seconds |
|---|---:|---:|---:|
| step | 4 | 0.012320 | 231.216694 |
| _execute | 4 | 1.028931 | 206.762511 |
| clone | 7 | 0.087812 | 35.337244 |
| occupancy | 6,402 | 0.640992 | 39.160143 |
| _thermal | 1,280 | 8.978216 | 16.710222 |
| persist | 1 | 0.032013 | 8.861436 |

Occupancy performs6402 exact key checks, including287,021,329 integer-type generator visits (23.997050s generator self time); this is a separate substantial repeated scan. Thermal is16.710222s. Persist is8.861436s, so this sample does not support saving/compression as its main cost. It does not establish the same proportions for every access phase or for the earlier1400→1412 interval.

**One narrow later recommendation:** apply the existing deferred rollback memo proposal only: freshly validate exact builtin `removed:list[int]` and `debris:dict[str,int|float]`, seed independently owned `.copy()` containers into the existing deepcopy memo, retain all other deepcopy and current flap-summary sharing. Never share the original mutable list/dict, skip checkpointing, change keys/values/order or weaken rollback. Respect existing memo identity entries and use legacy deepcopy for unexpected nested values, NumPy/custom scalar or container subclasses. The prior writer audit and exact-copy kernel are `rollback-copy-writer-audit.md` / `rollback-copy-kernel-audit.json`: saved760-action state,34,573 removed IDs and6,717 debris entries, current13.362914ms vs proposed6.823871ms per copy. This directly measures the operation now implicated; it is older/smaller state and is **not** a prediction of current action/episode speed. Do not simultaneously expand scope to clone/history ownership or occupancy invalidation.

Require legacy-reference accepted/partial/rejected public-action parity, mutation/alias ownership and float/order checks, first/late microstep rollback, observers/protection subclasses, source guards and cancellation tests before any adoption. Arbitrary custom objects inspecting deepcopy memo are outside the proven flat-JSON equivalence; preserve fallback rather than claiming universal Python-object semantics. Root owns whether/when any reviewed source change is loaded.

**Timing limitation:** 259.415800s wall includes cProfile overhead over 1,387,165,504 calls (1,195,504,915 primitive). Python scalar dispatch and generator costs are disproportionately inflated. This is attribution/count evidence only, not a normal throughput benchmark or a clinical/physics finding. No extrapolated whole-action speedup.
