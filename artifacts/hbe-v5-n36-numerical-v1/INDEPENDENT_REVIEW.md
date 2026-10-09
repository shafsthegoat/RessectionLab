# Independent HBE v5 N36 saved-run audit

**Decision: GO for a specimen-fixture numerical software result only.** The completed `compression:N36:S60:reference` native attempt and its saved readout pass the declared checks. This audit made **zero native calls and exactly one saved-stream replay**, with no retry. No measured HBE forces or patient data were accessed. The result does not establish physical tissue accuracy, patient-brain validity, clinical relevance, or spatial convergence.

## Exact provenance and closed outputs

- Frozen release SHA-256: `ffe757b78dfcf82ff81ceb3016ab724615a8338d84283335831f984ac7449034`. Receipt SHA-256: `b7f59dada629cd1fb8e61eac48a6eda86724b755055bdb7aa5dbf812d83fb013`.
- Released source commit, receipt preflight/postrun HEAD, and independently observed checkout HEAD all equal `05d4fddf897de20b49f1651b96f57d29e8e29f32`. All 20 source files match their release hashes, receipt before/after maps, working bytes, and Git blobs at that commit. The runner's preflight committed-source check, executing-import closure, postrun source/input/runtime checks, and final output rehash were inspected. Current source and output hashes were checked again after replay and at audit completion.
- All seven release input bindings match: preparation, v5 declaration, source map, old source deck, native mesh, backend profile, and runtime identity. All 13 installed executable/library hashes and the separately bound private OpenMP library match; the profile verifier also accepts its bound control ancestry. Both native logs contain exactly one Accelerate selection, no fallback marker, and normal termination.
- The attempt is the exact expected nine-file closed directory, **657,160,204 B**. All eight non-receipt output hashes and lengths match, including the identical five-file native subset. Only the expected silent-worker `readout-console.txt` is empty. The work order matches the release, one-time-token digest, old source/native mesh, adapter receipt, and four native-output bindings. Full evidence is in `source-audit.json`.

## Resources and preserved accounting

| Stage | Calls | Wall time | Peak sampled process-group RSS | Limits |
|---|---:|---:|---:|---|
| Original native FEBio | 1 | 481.205380 s | 1,830,125,568 B | 2,400 s / 3 GiB |
| Original saved-output readout | 1 | 41.294305 s | 882,704,384 B | 600 s / 3 GiB |
| This independent saved-stream replay | 1 | 40.720741 s | 811,630,592 B | 600 s / 3 GiB |

All three stages exited 0 without a kill. Original preparation took 7.519800 s against 150 s. Original native/readout peak sampled active output was 656,601,072 / 657,154,657 B against 1 GiB. The independent replay used the runner's reviewed process-group supervisor, private environment with numerical-thread variables set to one, 1 GiB audit-output cap, and no retry; peak sampled audit output was 1,196,252 B. Its evidence is `receipt.json` and `replay-comparison.json`. Sampling can miss brief RSS/output excursions; these are sampled peaks, not continuous maxima.

The exact predecessor ledger validates all five earlier receipts and the narrowly declared N12 supplement. The cumulative ledger through N36 contains six native calls, 946.628108 s native time, 92.377722 s original readout time, 26.705828 s preparation, one prior N12 supplemental replay of 2.847431 s and supplemental preparation of 3.436215 s: **1,071.995304 s recorded combined time**. This independent audit replay is additional audit overhead, not a new native run or an alteration of that ledger.

The original N12 receipt remains `failed_or_incomplete` with `ValueError: Native output absent or above bound`. Its exact supplemental replay supplies the narrowly reviewed numerical exception; it does not rewrite the failed attempt or repair its historical gap: **original post-run source/runtime guards were not recorded after parent packaging failure**. Historical N8 preparation and readout wall times were not recorded, so the cumulative known timing total is incomplete in that specific respect.

## Complete replay and numerical readout

The fresh call to `mechanics_hbe_v5_stream.read_bound_run` reproduced the **entire** saved JSON object and a byte-identical serialized file, SHA-256 `093777276e6270cea6ab76c8dc6d9e615acde78fedaa1517b2a8c04555398a48`. This is independent execution of the existing reader, not an independently implemented constitutive solver. Direct streaming counts additionally confirm 61 ordered node frames of **39,610 nodes** and 61 ordered element frames of **34,992 Hex8 elements**, with every ID in sequence and header times within `3.334e-10` of `i/60`.

The native domain is **lower half-height**. The reported response has `representation: reconstructed_full` and `reconstruction_provenance: reflected_native_half_not_native_full`; it is a reflection of the native half, not a direct full-domain solve. Force scale-one and energy scale-two reconstruction checks pass. Their maximum criterion ratios are `1.5086891e-12` (bottom force), `2.4943624e-06` (midplane force), and `1.1025104e-06` (energy).

All 61 frames and 60 solver states pass the declared numerical gates. The largest criterion ratio is **0.0007281404309**, native work–energy; the free-DOF reaction ratio is `0.0007268447271`, and solver-residual ratio is `1.4854956095e-10`.

- Final full displacement: **−0.00073726 m**; reconstructed full **simulated** force: **−0.03635127953163304 N**.
- Final full stored energy: `1.272773476505392e-05 J`. Independent trapezoidal integration exactly reproduces every saved full-work value, ending at `1.2727920175878835e-05 J`.
- Maximum full work–energy error: `1.8541082490741733e-10 J` against `2.5463666895757667e-07 J`; native-half error: `9.270562693600344e-11 J` against `1.27318334483078e-07 J`. Logged strain-energy density is not used for the energy gate.
- Minimum sampled reconstructed `J = 0.5525190653628169`; minimum logged native element `J = 0.979324390904`, also confirmed by directly scanning the element log. These sampled positive values do not prove positivity everywhere.
- Both native logs retain 32 `No force acting on the system` warnings under prescribed displacement, despite normal termination and passing numerical gates.

N32's endpoint force is `−0.03640072215288983 N`. The N32→N36 absolute endpoint-force step is **0.000049442621256785835 N (0.136013% of N36 force)**, smaller than N24→N32's `0.0001393526336920456 N` and N16→N24's `0.0002448427926464228 N`. The earlier N12→N16 step was smaller than N16→N24, and the study changes from direct full-native rows to half-height reconstructions. The decreasing recent force steps are observations only; they do not establish asymptotic convergence or physically accurate force predictions.

`summary.json` records the direct count, work integral, criteria, comparison, resources, and preserved gaps. All new audit files are confined to ignored `build/hbe-v5-n36-result-independent/`; no tracked files or native outputs were modified and no commits were made.
