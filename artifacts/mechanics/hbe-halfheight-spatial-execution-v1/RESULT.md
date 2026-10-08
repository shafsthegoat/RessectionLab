# Finer axial spatial experiment: retained failure

All three native runs completed and passed individual numerical checks, but
the declared spatial gate **failed**. Measured curves remain unopened and
calibration is not released. This is a numerical model result, not biological
validation or surgical performance.

| Quantity | Compression | Tension |
|---|---:|---:|
| N16→N24 displacement change | 4.799680 µm | 3.638568 µm |
| Original displacement limit | 8 µm | 8 µm |
| N16→N24 reaction change | 0.242781 mN | 0.120352 mN |
| Original reaction-change limit | 0.726931 mN | 0.531616 mN |
| N12→N16 reaction change | 0.203454 mN | 0.124369 mN |

The sole failed comparison is compression's required decrease in adjacent
reaction changes for N12→16→24. All other15 group criteria pass. The separate
N8→16→24 triplet passes, but cannot replace the failed co-primary comparison.
The earlier coarse-grid failures and full-domain N24 timeout are preserved.

Preparation took7.469 seconds with zero mesher/solver calls. The solve/readout
phase took365.799 seconds, three native calls,761,856,000 bytes sampled peak
process-family memory, and477,486,884 bytes final retained output. Individual
native durations were130.3253,20.7245 and126.8612 seconds. All declared resource
limits passed. Exit1 records the scientific comparison failure, not a timeout.

Independent archived-source replay reproduced all eight complete readouts
(five reused, three new;488 states) and the full comparison with zero field
mismatches. Direct force/probe arithmetic confirmed the same failure.209 input
hashes and27 new retained files were unchanged. The independent review took
73.85 seconds and made no solver calls or measured-data reads.

## Posthoc diagnosis, not revised acceptance

Saved compression endpoint forces at N8/12/16/24 are −37.106421, −36.784801,
−36.581347 and−36.338566 mN. They change monotonically. For h=1/N, the last
two intervals have equal spacing differences but different refinement ratios.
A positive sublinear error law can therefore increase raw adjacent differences.
The diagnostic three-level endpoint order is0.48258; the four-level fit is0.40093.
Changing fitted orders across triplets prevents claiming an established
asymptotic regime. Estimated remaining N24 endpoint bias is1.12–1.39 mN
(3.1–3.8%); these are model extrapolations, not measured errors.

The [NASA/NPARC convergence methodology](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
explains why refinement ratio and observed order matter; it supplies numerical
precedent, not validation of this FEM or tissue model. None of these posthoc
calculations changes the frozen failed result or its thresholds.

Full N16 saved probes already satisfy midheight reflection within1.00004e-14 m.
A representation-only N16 force shift large enough to reverse the trend would
exceed the original equivalence allowance. One small N16 half/full comparison
could isolate that residual concern, but cannot by itself establish spatial
accuracy. Before further meshes or calibration, revise the numerical study
prospectively around remaining-error estimation and the supported formulation;
do not spend additional runs merely to obtain a passing adjacent-difference test.

Reproduction uses the18-file archive at commit3de9f37, the two saved phase
releases, unchanged original protocol and the repaired FEBio runtime. Run the
archived `mechanics_hbe_halfheight_spatial_experiment.py` with the appropriate
phase/release/hash. Existing attempt markers forbid overwriting or retrying
these results. Complete raw predictions remain under ignored `outputs/`.
