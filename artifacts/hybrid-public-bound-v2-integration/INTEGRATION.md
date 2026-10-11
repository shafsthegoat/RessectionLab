# Local public-target window integration

The first five-case comparison saved no geometry previews: every one of 1,118
conditional queries hit a global unknown-cell indexing limit. The original
method and complete negative result remain unchanged.

`public_reward_window.py` stores bounded public target/coverage snapshots and
queries a conservative native-grid window around the complete active sweep.
It includes full affine cells, backward active length and caps. Unknown coverage
inside that window retains exact preview; distant unknown cells no longer
require global coordinate boxes. Reward, original top-eight proposals, frozen
ridge weights and the conditional current-tool check remain unchanged.

Independent source review found no blocking defect. Canonical controls pass
22 tests in 0.31 s (0.415 s parent). Generated attempt 01 passed 24 and failed two
fixtures that supplied NumPy scalar volume to the existing strict scalar API.
Only those fixtures were corrected; their two cases passed separately in 0.58 s.
Production bytes did not change. This is combined evidence, not a claim that
all 26 generated controls passed in one run. The ten-plan-before-replay worker
control passed in attempt 01. Failed fixtures and receipts are retained.

The prospective comparison repeats the same five already exposed TRAIN cases
with fresh baseline and window-bound plans, followed by ten uncached independent
replays. No fit, optimizer update or acquisition is included. The worker is
limited to 600 s, 3 GiB sampled memory and 128 MiB total extension storage.
Every snapshot, query, planning step and replay is charged. Source review and
generated controls do not establish acquired-data speed, accuracy or physical
fidelity; the local windows may still intersect unknown coverage everywhere.

See `preservation-index.json` for immutable evidence paths and hashes. The
canonical module and test are committed alongside this record; reused runtime
sources remain pinned by the one-shot runtime freeze before execution.
