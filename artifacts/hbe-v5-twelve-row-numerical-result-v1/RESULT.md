# Twelve-row native mechanics numerical result

The frozen v5 numerical comparison passes, with independent saved-result verification. Compression uses the finest N32→N36 mesh difference and its conditional all-state screen; tension uses N16→N24 and the required trend checks. Halving the timestep preserves both classifications. These are discretization checks, not measured tissue accuracy or a continuum error bound.

| Check | Largest difference | Frozen limit |
|---|---:|---:|
| Compression mesh force | 0.049443 mN | 0.727186 mN |
| Compression mesh probe displacement | 1.76350 µm | 8 µm |
| Tension mesh force | 0.120703 mN | 0.533026 mN |
| Tension mesh probe displacement | 3.65054 µm | 8 µm |
| Compression temporal force | 1.81×10⁻⁹ mN | 0.072719 mN |
| Tension temporal force | 6.59×10⁻⁹ mN | 0.053303 mN |

The compression conditional envelope is 0.649314 mN against a 0.727186 mN allowance. Coarse-mesh failures and the prior accounting failure remain in the record. The [independent report](REPORT.txt) explains the reused scalar-order solve, independently recomputed pair/group reductions, historical timing gaps and exact numerical limitations.

The corrected saved comparison took 11.155 s, at 254.6 MB sampled process-group peak, with no solver or stream-replay calls. Its independent check took 0.21 s externally at 76.2 MB maximum resident set. The full signed-array comparison remains outside Git; this package preserves its hash, compact metrics, source-bound input manifest, receipt and reproduction sources. Reproduction is through `build/hbe-v5-native-manifest-preparation-v1/run_comparison_accounting_v2.py --execute` with the frozen local inputs and a newly named output; the completed attempt directory must never be overwritten.

Measured response access and calibration remain closed pending a separate bounded successor. This does not validate live brain retraction, cutting, patient-specific material properties or surgical decisions.
