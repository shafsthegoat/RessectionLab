# Seven-row numerical verification: convergence failed

Completed 2026-10-09 UTC. All seven individual native runs passed their own
saved-output checks, but the frozen aggregate convergence test **failed**.
No threshold was changed and no additional solve was run after this result.

| Frozen endpoint | Observed difference | Limit | Result |
| --- | ---: | ---: | --- |
| n9 to n13 seven-point displacement | 0.220871 µm | 5 µm | Pass |
| n9 to n13 top reaction vector | 34.990344 µN | 10 µN | **Fail** |
| n13 half-step displacement | 4.1391e-8 µm | 0.1 µm | Pass |
| n13 half-step reaction | 4.3156e-6 µN | 1 µN | Pass |

Both refinement-decrease gates also passed their frozen maximum ratio of 0.8.
Earlier n5-to-n9 differences were 0.644935 µm and 103.687238 µN. These are
finite-mesh comparisons, not a bound on error against the continuum solution.
The half-step check concerns a static elastic problem; it does not validate
viscoelastic dynamics or surgical loading rates.

The four continuation rows used source commit
`5a1dff7ade2bf56e5da3e7703a36d5f65f33813a`, the repaired FEBio 4.13 Accelerate
runtime, one numerical thread, individual reviewed one-call releases and
hashed predecessor receipts. The first six rows each saved five states; the
last saved nine, for 39 states in total. Independent replay checked the native
outputs and the endpoint comparison. The final independent report is
`build/nonpatient-n13-half-step-result-independent/REPORT.md`; its hash and
the seven receipt bindings are retained in [summary.json](summary.json).

Total supervised native wall time was **73.868947 s**, excluding preparation,
replay and review. Maximum sampled process-group RSS was **1,096,187,904 bytes**;
sampling is not a continuous upper bound. Solver `No force acting` warnings
remain in the raw logs and individual reviews. No retries, fallback solver,
patient data, measured tissue response, parameter fitting or RL were used.

The seven-call declaration is exhausted. The next step is diagnosis of spatial
force resolution and resource scaling before declaring another numerical
experiment. This result does not admit patient meshes or establish force
convergence, physical fidelity, surgical validity or clinical safety.
