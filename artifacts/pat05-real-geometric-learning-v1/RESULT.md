# Real PAT05 geometric learning: first result

The two RL updates executed, but the chosen route did not improve. This is a
completed negative development result, not evidence of transfer or clinical benefit.

The existing 30,827-parameter 3D CNN used annotation-assisted T1, nominal tissue,
supplied target annotation, observed cavity, coverage/availability, and certified
entry/tip/tool geometry. Motor/language/vessel risk was unavailable. Seed 11,
masked on-policy REINFORCE with a spatial value baseline, Adam 0.001, gamma 1,
entropy 0.01, value weight0.5 and global gradient clip 5 were fixed. Each update
used two complete PAT05 TRAIN episodes. PAT26/27 SELECT and unopened patients
were not accessed. The fixed latest checkpoint was reported; there was no
reward-based checkpoint selection.

All methods used the same 3-decision horizon, including STOP, permitted annotation,
aperture, tools and geometric reward. The model ranks pre-certified proposals;
it does not generate or certify surgical paths from scans. The model sees its
64³ physical crop while search reads the full permitted nominal field; the
supplied target lies inside the crop, but these are different representations.
No deformation, force, functional injury or clinical accuracy is established.

| Method | Return | Target removed mm³ | Normal removed mm³ | Episode seconds |
|---|---:|---:|---:|---:|
| Initial policy | 12.7030 | 17.0001 | 20.0001 | 17.05 |
| Random legal, draw0 | 169.3756 | 213.0007 | 217.0007 | 18.31 |
| Random legal, draw1 | −25.0131 | 28.0001 | 264.0009 | 15.76 |
| Random legal, draw2 | 246.5479 | 323.0011 | 381.0013 | 19.04 |
| Greedy current-action search | 410.3124 | 493.0017 | 412.0014 | 20.09 |
| Latest after two RL updates | 12.7030 | 17.0001 | 20.0001 | 16.37 |

Search planning cost 17.40 s is additional to its20.09 s replay/audit. Shared initial
source/candidate preparation cost 8.18 s is additional to each method's table
cost. Every episode included initial state cloning, proposal/collision work,
actual execution, JSON reporting and independent evaluation. Initial/latest
actor decisions took 0.894/0.613 s across all three decisions; their native steps
including successor inventory took 13.209/12.900 s and independent audits
2.046/2.014 s. Actor-forward timing alone is not end-to-end latency. Complete
supervised run205.84 s, measured peak RSS1,910,718,464 bytes, below 600 s/6GiB caps.

All 10 completed episodes passed independent full-tool history and source-cell
reward/volume accounting; invalid attempted actions were 0. Any target-cell
removal is only the reachability criterion. It is easy in this bounded catalog
and does not demonstrate useful learning. Greedy removed4.31% of the supplied
11,437mm³ target; the initial/latest policy removed0.149%. The catalog does not
cover full resection.

The four optimization returns were 10.2130,99.0073,12.1790,−3.1250. Batch means
were 54.6102 and4.5270, with no monotonic gain. Both updates changed parameters,
including nonzero encoder and actor gradients. Loss/forward/backward/Adam took
2.783 s and2.618 s. Before clipping, critic/actor gradient norms were 292.0/23.3
and16.4/3.62. This documents unequal gradient magnitudes, not a proven cause of
failure. Saved first-state logits show the greedy first action improving from
rank 27 to 23, with probability 0.014176→0.014255; argmax stayed unchanged. A small
probability shift is not demonstrated route improvement.

Initial tensor hash: `sha256:2fa97c0e730db09371786d5ba903ffdf467b3cc7149a111c182d6a7499cdd9b8`.
Latest: `sha256:a5d777dc7c9ca7715112e67106371f6b25541798a6fedcc2d8446ffc2ccf5cf6`.
Exact configuration/source bytes, initial/update/latest checkpoints, attempts,
current action IDs/logits/probabilities, native histories and independent checks
remain in this directory. `summary.json` contains the compact learning curve;
`output-sha256.json` is the unchanged execution-time file inventory.

The next single diagnostic is eight fixed behavior-cloning updates from the
same initial weights on the three independently checked greedy decisions,
followed by a fresh independently checked rollout. That tests whether this
representation can learn a useful ranking from a cheap valid demonstration;
it must be reported as imitation, not an RL gain. No additional patient,
configuration sweep or favorable checkpoint selection is justified by this run.
