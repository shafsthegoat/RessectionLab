# HBE axial mesh resolution: terminal bounded failure

The separately released four-case study stopped after its second attempted solve. **N16 compression passed individual numerical checks; N24 compression reached the fixed 420-second call limit. Neither tension case ran.** There is no aggregate comparison artifact: both required N12→N16→N24 and N8→N16→N24 convergence judgments remain unevaluated. The previous experiment's mesh-convergence failure remains unchanged.

| Declared case | Actual status | Recorded solver interval |
| --- | --- | ---: |
| Compression N16, S60 | Completed; individual numerical checks passed on 61 states | 88.58572033396922 s |
| Compression N24, S60 | Timed out; killed and reaped, exit −9; no completed numerical readout | 420.039369917009 s |
| Tension N16, S60 | Not executed | — |
| Tension N24, S60 | Not executed | — |

N24's `TimeoutExpired` message reports a **419.9953113750089 s wait timeout**. Its final recorded interval is **420.039369917009 s**, including termination and cleanup; it is retained as a failed above-cap interval, not a successful solve. The saved solver log confirms convergence through **step 36 of 60, printed pseudotime 0.6**, then the start of step 37 at 0.616667. It has no normal-termination banner. These are log-confirmed observations; buffered output may omit later work, and partial primitive files have not been promoted to a complete numerical pass.

N16's native summary reports 60 completed steps, 308 equilibrium iterations, 308 linear-solver calls, 368 right-hand evaluations and 60 stiffness reformations. It prints linear-solver time `0:01:01` versus elapsed summary `0:01:28`; its detailed timing table gives 60.7794 s in the linear solver and 88.4268 s solve time. This identifies linear solving as a major cost in the source-reported timing. It is not an independent precise profiler decomposition or a forecast for another mesh.

Supervised elapsed time was **517.519936500001 s**, worker-body time **516.011924624909 s**, and peak sampled process-group RSS **1,563,459,584 bytes** (1,491.03125 MiB). The 1,200 s aggregate and 3 GiB RSS limits did not trigger an aggregate kill. The terminal supervisor exit was 1 with no cleanup error. A read-only process snapshot confirmed the supervised process family was absent before the collector froze files. No retry, budget increase or resumed solve occurred.

The closed experiment contains **22 files / 358,464,168 bytes**, including all partial N24 outputs. They are indexed, read-only and rehashed after the permission change. Only the experiment directory was frozen; its parent study directory retains its original writable mode. All **194 launch-bound inputs**, the **28-file accepted mesh preparation**, and the **169-file previous failed experiment** still match their saved hashes and sizes. The unchanged executed source commit is `5b80e3fee2349bc78d3f2ceb9b211efb305a1fc9`; exact source archive and release bindings are in `outcome.json`.

No Gmsh generation, calibration, fitted material parameter, measured CSV member access, or held-out response access occurred in this solve phase. The collector used only saved metadata, scalar reports, small solver logs, hashes, sizes and permissions; it did not replay field calculations or run FEBio. This result leaves the mesh-convergence question unresolved and establishes no material fidelity, patient-specific mechanics or clinical adequacy.

Evidence: [outcome](outcome.json), [per-case records and logged progress](run-outcomes.json), [complete raw index](raw-output-index.json), [preservation](preservation.json), [process closure](process-closure.json), [read-only freeze](readonly-freeze.json), [access chronology](access-chronology.json).
