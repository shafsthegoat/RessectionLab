# Search fallback recovers target progress but retains an avoidable tool change

One targeted policy change recovers the 31 target cells missed by the original
frozen shortlist on TRAIN050. Before accepting a shortlist with no positive
feasible action, the policy now checks every remaining emitted candidate. Model
weights, public inputs, rewards, tools and world assumptions remain unchanged.
This is supervised proposal ranking plus conditional search, not an RL update.

| Measure | Ranker with fallback | Fresh lean exhaustive |
| --- | ---: | ---: |
| Supplied target cells removed | 37 | 37 |
| Outside-target removal, mm³ | 0 | 0 |
| Frozen nominal return | 34.733524 | 34.830407 |
| Movements / tool changes | 12 / 3 | 11 / 1 |
| Full tool travel, mm | 102.453157 | 95.570343 |
| Online previews | 1,017 | 1,092 |
| Source preparation through plan, seconds | 9.126845 | 9.959361 |

The original ranker removed six cells; the fallback reaches the same 37-cell
set as search. Both contact upper bounds also match. But the quality gate
remains **false**: return is lower by 0.096882815. Exhaustive remains the stronger
baseline. This one fixed-order pair uses 6.87% fewer previews and 8.36% less online
wall time, without a claim of stable latency or population generalization.

Independent saved alignment explains the complete residual. The extra second
action uses suction; deleting it leaves exactly the same eleven physical tool
motions as exhaustive. Its one removed cell is later absent from the matching
aspirator cut. The return gap equals one action cost (0.03), two additional tool
switches (0.06), and 6.882814962 mm additional travel (0.006882815). At the shared
second decision, 48 current-aspirator candidates were unqueried, including 16
positive choices and search's winner at learned rank 46.

The next single change will compare unqueried current-tool alternatives before
a positive shortlist choice changes tools, while retaining unrestricted reward
comparison and the no-positive fallback. This targets measured under-comparison;
it is not a switch ban or a promise of subsequent route quality. No fitting,
new patient acquisition or reward adjustment is needed to test it.

Fallback triggers in 11 of 13 states and pays for 913 additional checks. Source
construction, model load, gating, checking and uncached replay are separately
reported. Both complete plans are sealed before either authoritative replay.
The independent saved audit passes; total planning/replay work is 4,218 previews,
50 transitions and 26 ridge scores. Parent runtime is 53.111954 s, with
543,162,368 B sampled peak and 4,926,758 B output, clean termination and no updates
or held-out/private input access. Original fitting cost remains in balanced-v2.

Eight generated controls pass after preserving and correcting an attempted
assignment to a frozen copied case; the repair follows the existing planning
clone pattern and never changes the original case. A saved-audit-only assertion
initially rejected the intentional runtime-release hash difference; that failure
and its narrow correction remain recorded. No patient rerun was needed.

TRAIN050 is now development-exposed, not an untouched transfer evaluation.
Search still removes only 0.1892% of the supplied annotation. These are simulated
geometric outcomes; no new physical-measurement agreement, clinical safety,
neurological accuracy or complete surgery claim follows. The Mac workflow is
preserved, but this experimental policy is not newly admitted to its interface.

`preservation-index.json` maps the compact sources/results to their original
paths. Unchanged helper sources are in the existing transfer-preparation archive;
canonical source is at `d70d814`. Full repeated inventories remain local with
audit hashes. Patient arrays and weights stay outside Git.
