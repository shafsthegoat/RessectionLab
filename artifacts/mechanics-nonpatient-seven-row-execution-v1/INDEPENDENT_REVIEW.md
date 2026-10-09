# Independent final n13 half-step and frozen seven-row comparison audit

Decision: **GO for the final saved solve as numerical software evidence; NO-GO
for the frozen spatial convergence milestone.** All seven bound native rows
record internally passing saved-output checks, but the predeclared n9-to-n13
top-reaction absolute gate fails. Do not relabel the overall benchmark a pass,
alter the frozen 1e-5 N limit after seeing this result, or claim physical or
patient-specific validation. This audit executed no FEBio call and edited no
source.

## Final call provenance and resource bounds

- Receipt SHA-256 `d4949d98340b1a74adabdbb3eda9da9805d16e1ce7aceb630349e0793c5fc34e`;
  one-call release SHA-256
  `39b23070a4a76908102fba6c46bcc9cc9ec6950406df878729fa04ba07a4b462`.
  Release and receipt bind frozen source commit
  `5a1dff7ade2bf56e5da3e7703a36d5f65f33813a`, generated deck SHA-256
  `8114d7840f9617c9384ea831017e3534f7b8d03b8976745d311b9ab7808e1c13`,
  unchanged declaration, preparation, backend/runtime, and six prior receipts.
  All 13 released source SHA-256 values match frozen Git blobs. All release
  inputs and every saved deck, console, main log, node log and element log
  match recorded hashes and byte counts; the attempt directory has no extra
  file. Every prior receipt hash matches its listed released binding.
- Reverified 115 backend inputs under `accelerate_csc_v1`; installed FEBio
  executable hash matches its runtime identity
  (`d7d4855d1d541a6407f22efd4381e23d7057cc49ec726144f06599aa5184f7b6`).
  The console records exactly one Accelerate solver selection, no fallback,
  normal termination, and eight `No force acting on the system` warnings.
  These remain disclosed for this displacement-prescribed numerical cube.
- The last receipt records one native call, exit 0, no kill or retry,
  33.465915 s supervised wall, 1,086,308,352 B peak **sampled** process-group
  RSS, 43,263,548 B peak **sampled** active output, 43,263,553 B final active
  output. Caps were 600 s, 3 GiB sampled RSS, 512 MiB active output, one
  numerical thread, and one attempt per row. Six prior receipts each record
  one call and sum to 40.403032 s; the complete declared sequence records
  seven calls and 73.868947 s. Receipt accounting cannot exclude unrelated
  external calls or unsampled resource spikes.

## Independent saved-field checks

- Re-ran the complete strict primitive parser and nonuniform G8 constitutive,
  local force and seven-point readout on the saved ASCII logs. Both structured
  results exactly match the receipt. All nine states (initial plus eight
  half-steps) contain 19,683 nodes and 13,182 tet10 elements. Eight final
  residual N2 values lie between 8.38e-24 and 1.72e-23, below the recorded
  required residuals near 3e-15. This is solver convergence on one mesh.
- Final minimum actual G8 deformation J is 0.9689565185 and reconstructed
  quadrature energy is 1.0139672134e-6 J. Maximum free-node internal force
  across all nine states is 6.62913e-13 N, beneath the frozen 1e-8 N local
  allowance. Separate central finite differences of the final G8 energy
  (`h=1e-8 m`) at a top-center constrained node agree with the negative of
  its saved signed reaction within 2.75e-13 N.

## Frozen comparison from raw endpoints

Reparsed the final raw node state for each nonuniform case, independently
located every one of seven fixed physical points in the tet10 mesh, applied
quadratic interpolation, and summed signed top-face nodal reactions. All four
cases have expected point owner counts `(6,2,2,2,2,2,2)` and maximum
multiple-owner sample disagreement below 6.10e-20 m. There is no detected
source-binding, sample-location, force-sign, or final-time endpoint mismatch.
The resulting metrics match root's saved comparison artifact SHA-256
`0eb7be91eb43ce8cbc8bb2aa1042ed146c1921f79c079a8997add33122303464`
to below 1e-15 in their respective units.

| Frozen metric | Independent value | Result |
| --- | ---: | --- |
| n5→n9 max seven-point field difference | 6.44935308675e-7 m | Reference trend |
| n9→n13 max seven-point field difference | 2.20870705150e-7 m | Pass, limit 5e-6 m |
| n5→n9 top-reaction vector difference | 1.03687238446e-4 N | Reference trend |
| n9→n13 top-reaction vector difference | **3.49903441891e-5 N** | **FAIL, limit 1e-5 N** |
| n13 full-step→half-step max field difference | 4.13906226659e-14 m | Pass, limit 1e-7 m |
| n13 full-step→half-step top-reaction difference | 4.31564345692e-12 N | Pass, limit 1e-6 N |

Both field and reaction refinement differences decrease by more than the
frozen ratio requirement (n9→n13 versus n5→n9); only the absolute reaction
gate fails, by 3.499 times its limit. The half-step agreement isolates time
step size from the observed n9→n13 reaction gap on this idealized cube. It
does not repair the spatial failure. Further mesh design, postmortem and a
**new prospective** benchmark version are reasonable next steps, but this
frozen seven-row result must remain failed. All rows use declared numerical
gauge constants `mu=1000 Pa`, `K=9666.6667 Pa`; they are not patient-specific
material estimates, and no physical brain interaction is validated here.
