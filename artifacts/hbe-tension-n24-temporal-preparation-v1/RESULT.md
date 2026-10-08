# Tension increment verification: preparation

The retained N24 human-specimen model is tested at 120 loading increments against
its existing 60-increment result. Only the schedule changes; geometry, material,
boundary conditions and solver settings remain fixed. This is numerical verification,
not comparison with measured tissue responses.

The five implementation files were frozen in commit `2911d31` before release.
The executable uses its exact 32-file Git archive. All 22 owner controls and 18
independent controls passed; the independent review also authenticated 63 source
and evidence bindings. The accepted individual tension result remains distinct
from its earlier globally failed study. No prior failure is rewritten.

The separate execution release permits one native solve, zero remeshing and no
retry: 420 seconds native, 600 seconds total, 60-second preparation nested in that
total, sampled process-family RSS below 3 GiB, 512 MiB active output and 1 GiB total.
All 121 native states and 61 shared load coordinates are required. Both original
spatial triplets are checked with only the finest result replaced. Measured curves
and calibration remain outside this release. The subsequent actual run is recorded
separately in `artifacts/hbe-tension-n24-temporal-execution-v1`.
