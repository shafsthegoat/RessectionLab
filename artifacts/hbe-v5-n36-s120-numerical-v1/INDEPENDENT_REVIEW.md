# Independent N36 S120 saved-result audit

**Decision: GO for the completed numerical specimen-fixture row and a passing S60/S120 load-step diagnostic.** One independently supervised replay reproduced all **121 frames and 120 solver states** and the entire saved JSON byte for byte. The frozen comparison passed at **all 61 common states and all 75 probes**. This does not qualify the full twelve-row study, spatial convergence, measured force, material calibration, or patient mechanics.

| Frozen common-state comparison | Observed maximum | Declared limit |
| --- | ---: | ---: |
| Absolute signed-force change | `1.8068706253426825e-12 N` | `7.271855906202017e-5 N` |
| Probe displacement-vector change | `2.313705266673336e-14 m` | `8e-7 m` |

The largest force difference occurs at S60 state **38** / S120 state **76**; the largest probe difference occurs at states **26/52**, probe **43** (zero-based). Every common state was compared, so the decision includes non-monotone differences and internal peaks. The maximum force difference exceeds the endpoint difference; the decision is not an endpoint-only check. There are **no force or probe exceedance states**. Both existing helpers, `v4.compare_pairwise_arrays(..., "compression_temporal", ...)` and `v5_comparison._signed_pair(...)`, agree on the signed arrays, fixed limits, and passing decision. The force allowance uses only the **61 even S120 states**. No generated-row admission function or full-study comparator was called.

The S60 input is the previously independently replayed, receipt-bound N36 row. Its complete replay remains byte-identical to the current saved S60 readout. S120 uses the same material, mesh, boundary topology, and half-domain reconstruction. All 61 shared full loading coordinates match exactly. Final full displacement is **−0.00073726 m**. Final **simulated** force is **−0.03635127953101008 N**, compared with S60's **−0.03635127953163304 N**; the signed endpoint change is **6.229600169049831e-13 N**.

The independent stream count confirms 121 ordered node frames of **39,610 nodes** and 121 element frames of **34,992 Hex8 elements**, with every ID ordered. Header times differ from `i/120` by at most `3.333334e-10`. All 120 solver states and numerical criteria pass; the largest criterion ratio is **0.000479166** for free-DOF reaction. The solution is `reconstructed_full` by reflection of the native lower half, not a direct full-domain solve. Force scale-one and energy scale-two checks pass. Minimum sampled reconstructed `J` is **0.5525190653076201**; the directly scanned logged minimum is **0.979324390909**. These sampled minima do not prove positivity everywhere.

Independent trapezoidal integration reproduces the complete saved work vector. Final full work is `1.2727781117296621e-5 J`; full stored energy is `1.272773476505392e-5 J`. Maximum full work–energy error is `4.635224269400318e-11 J` against `2.546338877859324e-7 J`; native-half error is `2.317635240911927e-11 J` against `1.2731694389758748e-7 J`. Logged strain-energy density is not used for this gate. Both native logs retain **95** “No force acting on the system” warnings under prescribed displacement, alongside normal termination and passing numerical checks.

| Stage | Calls | Wall time | Peak sampled process-group RSS | Fixed wall / RSS ceiling |
| --- | ---: | ---: | ---: | --- |
| Original native FEBio | 1 | 919.116218 s | 1,935,507,456 B | 2,400 s / 3 GiB |
| Original saved-output readout | 1 | 78.893169 s | 802,177,024 B | 600 s / 3 GiB |
| Independent saved-stream replay | 1 | 78.576757 s | 887,422,976 B | 600 s / 3 GiB |

All stages exited 0 without a kill or retry. Original preparation used **8.360866 s** against 150 s. The original exact nine-file closed directory is **1,303,460,128 B**, below 2 GiB; the audit replay's sampled output peak is **1,741,883 B**, below its separate 1 GiB cap. Numerical thread controls were one. Sampling can miss brief peaks. This audit made **zero native calls and exactly one full saved-stream replay**; its replay is separate audit overhead.

The primary ledger now contains **7 native calls**, **1,865.7443257092964 s** native, **171.2708908317145 s** original readout, **35.066694086184725 s** preparation, and **2,806,373,472 B** closed output. Including the separately charged N12 supplement gives **2,078.365556919249 s** known combined time and **2,806,930,864 B** combined output, within frozen aggregate limits. Historical N8 preparation/readout times remain unknown. **The original N12 remains `failed_or_incomplete`; its separate admitted supplement does not erase the missing original post-run source/runtime guards.**

Exact release: `1ea9f55f9d1db13cb214e8eac649713fb39b2492d0721c8bed9921d4cd886a58`; terminal receipt: `a1357ecb8cd129e10923d2d3a0837b62d892bf8b4bdb7d83ccb12cb7b6d041f4`; independently reproduced readout: `14ca49981b0bf68df1a2a1d75489b9e8966c017635d3c4ba20a794e64a8eb18b`. All **20** executing sources, seven release inputs, eight non-receipt outputs, six-receipt ancestry, 13 installed runtime files, private OpenMP, and retained analytical controls verified. Source and output guards passed again at audit completion with HEAD **`0f7eee3a17b6743fe15549cf8aba396fa4e98a2c`**. The comparator's additional static source closure is bound to that commit. Root was explicitly released from the compute/HEAD freeze only afterward.

No measured HBE response or patient data was accessed, no material was fitted, and no tracked file was edited or committed by this audit. The independent replay uses the existing numerical reader; it is not a separately implemented constitutive solver. The compact package retains exact receipts, accounting, comparison summaries, source bindings, and audit source; bulk logs and full response/probe arrays remain in their existing output/ignored audit locations.
