# Independent saved failure review

The experiment correctly stopped at the fixed aggregate numerical gate after
18 successful individual solver/readout checks. The two fitted confirmations
were not executed. Independent arithmetic reproduced all four mesh and four
load-step comparison groups exactly; these three of thirty aggregate metrics fail:

| Metric | Observed | Fixed requirement |
| --- | ---: | ---: |
| Compression, maximum medium-to-fine probe displacement difference | 22.356144 µm | ≤8 µm |
| Tension, maximum medium-to-fine probe displacement difference | 23.257690 µm | ≤8 µm |
| Tension, medium-to-fine reaction difference | 0.227742502 mN | <0.199483624 mN preceding difference |

The saved load-step and stiffness-scaling metrics pass. Complete raw replay of
tension N8/S60 and N12/S60, 61 states each, exactly reproduced the saved readouts.
These two replays took 1.03 and 2.88 seconds using the frozen source. The scale
primitives and other sixteen raw readouts were not replayed; their saved results
and bytes were authenticated. No solver was executed by this audit.

The parent exited 1 without a resource kill after 396.303 seconds of its
900-second allowance. The longest solver call took 41.288 seconds against 90.
Sampled peak process-group RSS was 253,722,624 bytes against 3 GiB. Output-watch
peaks and final 914,390,988 bytes remained within their fixed caps. Solver time
is nested within worker and parent time; those durations must not be added.
Sampled peaks do not exclude short between-sample peaks.

All 215 baseline inputs and selected/full indexed outputs remained unchanged;
the first audit checked 384 distinct files. The final collector comparison also
rehashed all 169 raw output files and verified read-only permissions. Its counts,
three failed metrics and timing fields match the independent result. Original
Skyline failure result/state witnesses remain unchanged; the owner's broader
historical-preservation receipt is separately bound.

State flags, source control flow and absence of access/fit/freeze artifacts agree:
no calibration or held-out access was attempted. This audit opened no measurement
ZIP, CSV or patient file. No fitted material parameters, held-out prediction or
physical-validation result exists. The observed failure concerns the frozen
numerical refinement criteria; it is not an accuracy estimate against measured
tissue behavior. No retry, tolerance change or follow-on solve is authorized here.

`review.json` records the independent result and exact executed audit source.
`collection-binding.json` binds the final collector and read-only raw inventory.
