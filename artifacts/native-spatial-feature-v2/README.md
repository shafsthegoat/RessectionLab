# Synthetic explicit-tangent V2 evidence

This directory preserves the initial V2 experiment. Its later validation repair
is recorded separately in `repair-01`; the original numerical results and source
capture are not overwritten.

Eleven focused checks passed in 0.09 seconds at one numerical thread. The isolated
algebra-only probe took 0.00237 seconds, with zero simulators, native transitions,
policies, optimizers, gradients and patient loads. Its inputs are the preserved V1
synthetic-coordinate report and the independent conditioning counterexample.

The externally declared Y reference gives maximum error **2.22e-16 mm** under the
exact saved proper rotation. Declaring the near-parallel X reference rejects in
both frames instead of selecting another axis. Sines 0.099 and the three interior
roundoff-band cases reject; 0.101 accepts. Both accepted and rejected receipts
contain the actual reference/source/frame declaration and measured condition.

The explicit 0.1 condition and 64-epsilon band are engineering choices, not clinical
thresholds or a global floating-point bound. No universal reference convention
across patients is supplied. External provenance authenticity is not proven by
matching metadata. The existing summary collision remains; contact history, tool,
budget, topology and candidate interactions are still missing or compressed.

`probe/report.json` contains exact results and limitations. `execution-receipt.json`
retains the commands and outputs. `source-capture.json` binds all source, test and
input bytes. Source was copied into isolation before either command, and numerical
source/input hashes were unchanged through the probe. The original V1 code,
counterexamples, source archive and execution receipts remain unchanged.

The independent reviewer preserved 14 failures and 16 passes on the initial V2
source: direct constructor spoofing/mutation, overflow and complex-input handling
needed correction. The V2.1 repair passes the same 30 independent adversaries in
0.08s and all 11 owner checks in 0.06s. Its algebraic probe took 0.00343s, with
zero simulator, transition, policy, optimizer, gradient or patient activity.
`repair-01/accepted-case-comparison.json` verifies exact numeric equality for 14
saved comparisons, including the accepted bases/coordinates and conditioning,
alias rows, summary collision and threshold classifications. This does not imply
a general numerical bound.

The independent negative and repaired evidence is under
[`native-spatial-feature-v2-independent`](../native-spatial-feature-v2-independent/).
The original and repair source files are also preserved in separate verified
`source.tar.gz` archives; each archive receipt lists its hashes and file count.
The archived V1 returns were read for context, not recomputed; no new native
geometry validation or learned-policy benefit is claimed. No production profile,
pilot or cache declaration was modified, and no further training is authorized.
