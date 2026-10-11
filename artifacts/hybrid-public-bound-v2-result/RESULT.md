# Second public-bound comparison: preserve quality, reject added overhead

The local-window successor removes the prior global workspace failure, but
does not improve planning. All five pairs retain exact actions, histories and
outcomes; all ten independent geometry certificates are accepted. Both methods
use 1,446 online previews. Every one of 1,118 conditional planning queries and
1,118 replay queries refuses because its conservative window includes unknown
target coverage. Zero previews are saved.

| TRAIN case | Both arms' online previews | Baseline planning, s | Local-bound planning, s |
|---|---:|---:|---:|
| 002 | 117 | 6.482007 | 6.508387 |
| 015 | 197 | 3.900251 | 3.911593 |
| 018 | 75 | 2.689344 | 2.925955 |
| 045 | 187 | 6.061858 | 6.608945 |
| 050 | 870 | 11.204790 | 14.842893 |
| Total | 1,446 | 30.338249 | 34.797772 |

This is 4.460 s / 14.7% extra source-through-plan time in one fixed-order paired
development run. Bound setup takes 1.129 s and queries 2.476 s, already included
in that total. Full replay plus audit costs 56.643 s versus 65.699 s. The owned
parent completes in 190.403 s with 743,030,784 B sampled peak memory and
15,479,548 B output. Across planning and replay: 5,784 previews, 164 transitions,
164 ridge calls, 20 source visits, one frozen ridge load and zero fitting or
optimizer updates. Cleanup leaves no owned processes.

The unknown-coverage refusal is honest but conservative: it does not establish
that the actual swept tool intersects unknown tissue. The target domain cannot
be silently converted to known-empty. No third bound variation is selected;
the faster retained switch-guard baseline remains preferred.

All ten plans were sealed before fresh uncached evaluation. Targets remain
0/58/0/51/37 cells and outside-target removal remains zero. This is annotation-
assisted simulated geometry on already exposed TRAIN cases, not independently
measured tissue response, new-patient transfer, neurological safety or an RL
gain. SELECT037 remains held and EVAL067 closed. Existing real-observation and
physical validation evidence is unchanged.

Next: correct the demonstrated residual-blocker candidate omission and test
whether a newly offered preparation opens a useful target ray, charging its
negative tissue-removal cost. The original negative search and both bound
results remain preserved.

Runtime HEAD: `7778053311a8173409c1a123499f73fb087936b9`.
Result SHA256: `f25b2d46aad3f969c9cbd4152988d4705d2472885f35124751356ff518dbb69f`.
Receipt SHA256: `7a16060cde1725186f3963175de375d9eb0d2d71577d66b61418f25dd53ec65e`.
Independent saved audit: `a79450d5acf7f288303507633e242da1ab3279ce2194ffb1cea793f8909a316d`.
`preservation-index.json` pins the compact evidence; full repeated inventories
remain local with hashes. No patient arrays or weights are copied into Git.

At 10:56:34 UTC, shared extension growth is 168.211 MiB with 332.760 GiB free,
before this compact archive. No new dataset, dependency or checkpoint was added.
