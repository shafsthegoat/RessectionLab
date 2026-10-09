# Independent tension N8 S60 saved-result audit

**Decision: GO for the completed ordinal-7 numerical specimen-fixture row only.** The one bounded independent replay reproduced the entire saved **61-frame** readout byte for byte, including all **60 solver states and 75 displacement probes**. No native rerun occurred. This first tension row does not establish tension mesh convergence, measured-force accuracy, material calibration, full twelve-row qualification, or patient mechanics.

The full-native fixture contains **1,045 nodes and 768 Hex8 elements**. Direct stream counts confirm 61 ordered frames in both native logs, every expected ID in order, and header times within `3.333334e-10` of `i/60`. Native and full loading coordinates are identical, ending at **+0.0007360099999999 m**. The representation is `full_native_fixture` with `reconstruction_provenance=full_native`; no half-domain reconstruction or native-half energy multiplier is used.

The final **simulated signed tension force is +0.0271175517171575 N**. All 60 non-rest force values are positive, and none decreases between successive states; the smallest loaded force is `0.00052327678577962 N`. All 61×75 probe vectors are finite. `signed-path.json` retains every signed force, coordinate, stored energy, integrated work, and per-state maximum probe norm. Those paths are diagnostics of this numerical fixture, not comparison with measured tissue response.

All frozen numerical criteria pass. The largest criterion ratio is **0.000861247**, free-DOF reaction; solver residual ratio is `4.033485305978043e-11`. Minimum sampled `J` is **0.8821918661896644**, and direct scanning confirms minimum logged `J=1.0`. Sampled positive Jacobians do not prove positivity everywhere. Both native logs have exactly one Accelerate selection, no fallback marker, and normal termination, alongside **60** “No force acting on the system” warnings under prescribed displacement.

Independent trapezoidal integration reproduces the complete saved work vector. Final stored energy is `1.047266774295893e-5 J`; final work is `1.0472530374363516e-5 J`. Maximum work–energy error is **`1.373685954178213e-10 J`**, below **`2.0953162029917868e-7 J`**. Logged strain-energy density is not used for the energy gate. Native-half work is correctly absent for this full-native row.

| Stage | Calls | Wall time | Peak sampled process-group RSS | Fixed wall / RSS ceiling |
| --- | ---: | ---: | ---: | --- |
| Original native FEBio | 1 | 2.180279 s | 40,599,552 B | 90 s / 3 GiB |
| Original saved-output readout | 1 | 1.561998 s | 199,049,216 B | 600 s / 3 GiB |
| Independent saved-stream replay | 1 | 1.746642 s | 196,935,680 B | 600 s / 3 GiB |

All stages exited 0 without a kill or retry. Original preparation used **7.310108 s** against 150 s. The exact nine-file closed directory is **15,857,006 B**, below 64 MiB. Audit replay peak sampled output was **1,194,877 B**, below its separate 64 MiB cap. Numerical-thread settings were one. Sampling can miss brief peaks. This audit made **zero native calls and one full saved-stream replay**; the replay is separate audit overhead.

The validated primary ledger now has **8 native calls**, **1,867.924604543252 s** native time, **172.83288833172992 s** original readout time, **42.376802461221814 s** preparation and **2,822,230,478 B** closed output. With the separate N12 supplement: **2,089.417941628257 s** known combined time and **2,822,787,870 B** combined output. All remain below frozen aggregate limits. **The original compression N12 remains `failed_or_incomplete`; its admitted saved-output supplement does not erase the missing original post-run source/runtime guard record.** Historical N8 preparation/readout times remain unknown.

Exact bindings:

- Release: `c91e41806e70093644714072e7a22d24642aae40b8f10310f8cf8bc8bd69a843`.
- Terminal receipt: `3499f1787dacca9d24e463cc09ec539437c1046252b616041a00b89f91ce08b5`.
- Independently reproduced readout: `7e572166c5734bbdab14c468c18d8bb6253e6525c70505935bc7ff55d5127bd6`.
- Released/audited commit: `4e89ef83805ea9282f59fc4ff1c4543b60110908`.

All **20** executing sources, seven release inputs, eight non-receipt outputs, seven-receipt ancestry, 13 installed runtime files, private OpenMP, and retained tiny analytical controls verified. The complete source and output guards passed again at audit completion; root was explicitly released from the HEAD/compute freeze afterward. No measured response or patient data was accessed, no material was fitted, and this audit made no tracked edit or commit. The independent replay uses the existing reader, not a separately implemented constitutive solver. Compact receipts, bindings, accounting, signed paths and audit source are retained; raw logs and full response/probe arrays remain in their original/ignored locations.
