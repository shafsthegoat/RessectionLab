# RESECT Case2 conditional observation prediction

The fixed three-method comparison completed on one existing TRAIN patient.
Six source-selected observed before/during ultrasound correspondences condition
the predictors. All ten other landmarks were evaluated after the fields were
frozen, without exclusion or retuning. This requires an intraoperative update;
it is not prediction from a preoperative scan alone.

| Fixed method | Validation RMS (mm) | Median (mm) | Maximum (mm) | Six-point fitting RMS (mm) |
|---|---:|---:|---:|---:|
| No update | 3.347873 | 2.806697 | 5.193873 | 3.078905 |
| Proper rigid | 1.859573 | 1.508631 | 3.238462 | 1.340021 |
| Inverse-distance squared, all six | 1.855245 | 1.600958 | 2.934572 | 0.000000 |

The predeclared primary difference, rigid minus static RMS, is **−1.488300 mm**.
IDW improves RMS over rigid by only **0.004328 mm**, while having worse median
error. Exact fitting interpolation does not establish better generalization.
These ten points belong to one person; no patient-level confidence interval,
clinical threshold, learned-model promotion or tissue law follows.

The creator's [dataset instructions](https://yimingxiao.weebly.com/data-repositories.html)
and [methods paper](https://users.encs.concordia.ca/~hrivaz/Xiao_Database_MedPhys.pdf)
support paired anatomical correspondence and the world-coordinate convention.
They do not supply this case's tool action, force or elapsed surgical time.
Source hashes and intended-use limits are retained in the executed protocol.

All sixteen source and destination points lie within their respective native
image voxel-cell boxes, using inverse active sforms. This is geometric inclusion,
not proof of ultrasound signal quality, tissue support or dense deformation.
No image arrays were loaded and no force response was fitted. Calibrated
measurement covariance and predictive intervals remain unavailable.

The six input rows are 1, 14, 13, 9, 2, 11. Validation rows are 3, 4, 5, 6, 7,
8, 10, 12, 15, 16. Partitioning uses source coordinates only; six destinations
are exposed during fitting and the other ten only after the complete field
freeze. All ten are now consumed for future method development. Case2 retains
its TRAIN role, Case4 remains exposed DEVELOPMENT, and SELECT/EVAL roles and
payloads are unchanged.

| Stage | Supervised wall time (s) | Observed worker peak RSS (bytes) |
|---|---:|---:|
| Qualification | 0.546485 | 61,718,528 |
| Fit and freeze | 0.274680 | 53,772,288 |
| Evaluate | 0.283164 | 53,854,208 |

All phases exited successfully within their separate 30-second limits; the
supervisor reaped each owned process group. Memory is observed, not a hard RAM
cap. These are single executions, not a comparative latency benchmark. Ten
generated controls passed before acquired-data execution. The independent audit
passes **375 checks**. It reads the exact original tag after evaluation and
reconstructs rigid alignment through a quaternion eigenproblem, independently
of the production SVD, as well as IDW, every point error and native-cell inclusion.
Maximum numerical disagreement is **1.50e-13 mm**. Its first execution failed
because a reviewer-local variable shadowed the regex module; that failure is
preserved. The corrected audit changes no model, frozen field or result.

The compact archive contains executed source, immutable authorization and
supervision evidence, and [derived scalar errors](scalar-results.json). Original
images, tag coordinates and coordinate-bearing outputs stay outside Git; their
local bindings are recorded for verification. The byte-preserved runner retains
its historical build paths and exclusive output rules. Do not overwrite those
outputs or re-open this validation as untouched evaluation.

This adds conditional real-observation evidence. It does not validate causal
retraction/cutting, a learned action-conditioned world model, RL superiority,
dense brain shift, neurological injury or clinical benefit.
