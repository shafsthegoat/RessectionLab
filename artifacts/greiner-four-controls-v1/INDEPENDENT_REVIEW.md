# Independent saved-result audit

**PASS: numerical software controls only.** The exact reviewed remaining-three release executed at historical commit `1ede7500a1771323b3cf817cdb14906e8fee9bf5`. This audit launched no solver and changed no saved evidence. No blocking finding.

The saved checker results reproduce exactly for all three new cases: 105 fine-shear states, five zero-arm shear states, and five zero-arm volume states. Together with the separately authenticated 53-state original coarse replay, this is 168 recorded rows and 164 positive-time material-check rows. Every frozen state check, residual criterion, and continuum envelope passes. The zero-arm shear also passes the archived plain-elastic implementation crosscheck. Initial rows remain initialization/format evidence; deviatoric SED remains outside acceptance.

| New case | Recorded states | Maximum discrete stress error (Pa) | Native seconds | Finalized case bytes |
| --- | ---: | ---: | ---: | ---: |
| Fine one-arm shear | 105 | 5.15e-10 | 0.073704 | 1,047,499 |
| Zero-arm shear | 5 | 2.68e-11 | 0.073581 | 185,126 |
| Zero-arm volume | 5 | 1.60e-9 | 0.080616 | 186,142 |

Halving the shear timestep gives coarse/fine continuum-error ratios **3.9960218827–3.9998751344** across the four compared stress components at the three frozen observation times. At the ramp end, the maximum error falls from approximately 0.0752247 Pa to 0.0188087 Pa. This is consistent with the declared second-order discrete trend and remains diagnostic, not physical validation.

The audit verified the root release, preparation and independent-review hashes; seven tracked helper blobs at historical `1ede750`; the complete accepted runtime/profile and source closure; all **198 saved audit entries per pre/post/final phase**; exact case inventories including `.xplt`; every receipt/checker/output hash; command, backend and one-thread environment; and unchanged inputs and outputs after replay. Each new call exited zero without a kill or cleanup error. The original coarse failure remains unchanged and charged once; coarse was not rerun. There are three new calls and four combined calls, with no retries.

Finalized remaining output is **26 files / 1,463,352 bytes**. Including the original nine files / 503,011 bytes gives **35 files / 1,966,363 bytes**. This exact recount includes the last summary write; the controller's earlier byte snapshot was 42,022 bytes smaller. The new sequence records 1.431000 seconds; combined sequence time is 1.803089 seconds and combined native time is 0.309289 seconds. Peak sampled native-group RSS is 1,622,016 bytes. All 45/10/10-second, 8-MiB-per-case, 90-second/24-MiB aggregate, and 256-MiB RSS limits pass. Sampling can miss brief between-sample peaks.

Every solver warning is the retained `No force acting on the system.` message, once per positive-time state in each console and solver log. The source emits this in its small-residual convergence branch. Signed reactions, force and moment balance, free-center residual, and all frozen residual checks still pass; warnings were not removed or hidden.

`audit.json` contains the complete hash inventory, per-case metrics, historical source bindings, continuum errors, ratios, and exact accounting. `audit.py` records the saved-only verification procedure and blocks subprocess creation after the historical Git reads. Preparation/runtime replay and numerical replay use saved bytes only.

These arbitrary-constant source-matched PK2 controls establish numerical software behavior. They do not establish physical validation, measured-specimen accuracy, patient parameters, clinical suitability, or substantial-RL readiness. Literature CSV schema and measurement chronology remain unresolved.
