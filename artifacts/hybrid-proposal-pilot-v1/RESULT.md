# Learned proposal priority: useful speed signal, preset quality gate missed

One fixed ridge scorer was fitted on all 29 existing decisions from four TRAIN
patients (002/015/018/045), then compared with a sampled-target heuristic and
exhaustive native checking. All twelve complete routes passed independent
full-tool replay. SELECT/EVAL remained closed. **The declared TRAIN gate is
false**, and this result is retained without changing its criteria.

| Measure, all four TRAIN cases | Learned top8 | Heuristic top8 | Exhaustive control |
|---|---:|---:|---:|
| Removed target cells, 002/015/018/045 | 0 / 58 / 0 / 51 | 0 / 0 / 0 / 0 | 0 / 58 / 0 / 51 |
| Online native previews | 232 | 32 | 2,592 |
| Source preparation through complete plans | 9.859446 s | 4.480848 s | 44.740926 s |
| Separate source/replay/audit time | 25.009537 s | 6.022099 s | 71.624767 s |
| 045 geometric return | 48.151409795 | 0 | 48.181962921 |

The learned shortlist uses 91.05% fewer previews and 77.96% less online wall
than this **instrumented exhaustive control**. That control also computed
unused features (0.195009 s) and ranking setup (0.000064 s); all arms wrote full
inventory ledgers, whose logging time was not isolated. This is one fixed-order
measurement, not a measured speedup against minimal original greedy or a
sustained estimate. No component subtraction is presented as a new benchmark.

The learned route on 045 omitted the better final action after eleven identical
motions. It removed the same last target cell with another tool, adding one
switch and 0.553125842 mm travel. Their exact penalties explain the entire
0.030553126 return loss. Seven of its other shortlisted choices were blocked.
The better candidate's exact predicted score/rank was not saved; only exclusion
from eight is established. The heuristic's 32 queries were all shaft-blocked;
its STOP does not establish that useful actions are absent.

Outside-target removal was zero in all routes; learned and exhaustive contact
measures agree. Contact is nonzero, includes target tissue and is not injury.
The original heuristic comparison also required no extra contact versus its
zero-contact STOP, so it fails despite useful target progress. Both literal gate
failures remain recorded. These annotation-assisted post-exposure simulations
remove less than 1% of each positive case's full target. They establish neither
physical fidelity, scan-only planning, generalization, an RL gain nor safety.

Full owned work took 231.007419 s at 715,456,512 B sampled peak: 28 source visits,
2,592 training rows, one 0.014672 s solve, 182 transitions and 10,896 previews.
Training-source/label/teacher-replay costs are separately retained. No neural
forward, optimizer update or checkpoint load occurred. All processes exited
cleanly. Bulk ledgers, coefficients and patient arrays remain local.

The [independent saved audit](../hybrid-proposal-independent-result-v1/REPORT.txt)
passes 3,582 checks, including all current certificates and original exhaustive
histories. Its initial overstrict historical identity expectation is retained
with the corrected reader. The [saved diagnosis](../hybrid-proposal-pilot-diagnosis-v1/REPORT.txt)
supports one prospective positive/zero loss-weighting contrast. This hypothesis
has not been tested here; the original model/result/gates stay unchanged.

Reproduction: source-index.json pins canonical source at commit 5fe581e and all
runtime/metadata dependencies. runtime-source-map.json archives the nine local
noncanonical source files at their original relative paths. Restore those paths
only in a separate reproduction checkout; an acquired rerun requires the same
local source manifests and its own declared attempt. The original frozen local
attempt remains build/hybrid-proposal-pilot-v1/attempt-01. The saved audit and
diagnosis can inspect it without reading arrays or coefficients.
