# Unwired independent geometry batch control

**76 focused checks passed; the single numerical probe completed.** The
production evaluator and original scalar oracle are unchanged. These are
numerical geometry controls, with no patient data, native history replay, or
learning updates.

Every ordered contact set and first witness matched the scalar reference in all
15 workload/tolerance combinations. Of 5,120 squared distances, 4,927 matched
bitwise. The other 193 were in the oblique-segment workload, with maximum absolute
difference **4.44e-15 mm²**, within the declared 2e-14 relative/absolute comparison
tolerance. This is numerical agreement, not universal bitwise equivalence.
Near-threshold decisions use the preserved scalar routine.

Median distance-kernel time per 1,024 boxes, two repetitions per phase:

| Numerical workload | Scalar before | Batch | Scalar after |
| --- | ---: | ---: | ---: |
| Axial, anisotropic cells | 49.081 ms | 2.281 ms | 49.397 ms |
| Oblique local segment | 104.894 ms | 4.174 ms | 104.937 ms |
| Point segment | 17.908 ms | 1.575 ms | 17.863 ms |
| Common translation of 1e9 mm | 55.411 ms | 3.992 ms | 55.492 ms |
| Exact tangency | 49.582 ms | 1.690 ms | 49.425 ms |

The contact helper is **not uniformly faster**. Large-coordinate contact calls
took 60.14–60.77 ms because the conservative numerical guard invoked scalar
fallback. Exact tangency with zero tolerance took 51.60 ms. With the original
1e-10/1e-9 contact tolerances, that tangency workload took 1.79/1.75 ms. These are
single contact-call observations, not paired estimates or claims about complete
audit performance. The larger distance-kernel reductions exclude enumeration,
coordinate transforms, frontier checks, full tools, and causal tissue histories.

At batch size 256, traced allocation above the required output was **89,147
bytes** for both 1,024 and 16,384 input boxes. Inputs already existed before
tracing; output sizes were 8,192 and 131,072 bytes. Peak process RSS was
65,110,016 bytes. These measurements support bounded kernel working storage;
they are not a process-wide hard memory guarantee. The implementation enforces
a maximum of 4,096 boxes per batch and does not retain empty contact fragments.

The combined test run took 1.09 seconds of pytest time. The probe took 1.62
seconds inside its measured body and 1.97 seconds including parent launch,
below both the declared 45-second process alarm and parent timeout. No retries
or cap changes occurred. Timings are one local process on one runtime, not a
general speedup estimate.

Evidence is retained in `test-baseline.json`, `combined-tests.txt`,
`test-receipt.json`, `probe-launch.json`, `probe-launcher.json`, and `run-01/`.
The probe writes prospective configuration and loaded-source hashes before
measurement, then rechecks sources afterward. The independent static review
identified an ambient-import provenance gap before execution; the probe now pins
and verifies the loaded scalar and prototype origins. The independent review
receipt is under `../independent-geometry-batch-review-v1/`.

No production integration is authorized by these results alone. The proposed
next gate is complete-history parity and a separately released, single analytic
history audit comparison, as described in
`../../docs/independent-geometry-batch-prototype.md`.
