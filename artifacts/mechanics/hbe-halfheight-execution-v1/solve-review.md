# Independent saved-result audit

October 8, 2026. Root records the core-contracts reviewer report. The reviewer
ran no native solver, accessed no measured response curves and wrote no source.

The frozen readout was replayed for all four cases and reproduced every saved
field, with 61 states per case. A separate NumPy/plain-text node reconstruction,
using neither the repository parser nor lift routine, reproduced all-node
displacement and summed reflected raw-reaction maxima. All 193 baseline input
hashes remained exact before/after replay; 28 unique state/result bindings,
36 retained files and 14 archive members match. The archive matches Git
437bbeddb and the extracted source.

Four native solves, zero Gmsh calls: 40.918404292 supervised seconds and
164,626,432 bytes sampled process-group RSS, below 420 seconds/3 GiB. The
longest individual solve took 7.817677291 seconds, below 90 seconds. Recorded
output-watch maxima were 27,760,791 bytes active and 79,320,260 total; the final
study directory was 79,322,080 bytes.

Maximum node discrepancy: 4.2426433e-14 m; maximum 75-probe discrepancy:
2.7439923e-14 m, against the fixed 1.1337385e-8 m limit. Worst raw-reaction
error/tolerance ratio: 8.8286754e-5; signed-force ratio: 6.7662132e-5; energy
ratio: 3.4892829e-9, all with limit one. Absolute raw-reaction maximum was
1.4125881e-13 N; signed-force maximum 8.0934304e-12 N; twice-half energy
difference 5.1457252e-20 J. The original numerical criteria also passed
(worst upper-bound ratio 0.0010051721; minimum sampled Jacobian 0.7530810624).
Sampled positive Jacobians do not prove positivity everywhere.

Historical native full-model solve times total 49.1561 seconds, versus 19.2196
seconds for these half-model solves (2.5576 ratio, per-case 1.90–2.80). These
are single observations on the same pinned runtime, without randomized timing
or an equivalent concurrent workload; no robust speedup is established. The
whole current supervised phase includes readout/checking and took 40.9184 seconds.

The result establishes the declared N8/N12 axial symmetric-branch equivalence.
It does not establish spatial convergence, asymmetric stability, torsion, a
measured modulus, tissue fidelity or patient-specific forces. Prior spatial
convergence failure and full N24 timeout remain unchanged. No automatic finer
run or calibration is released.
