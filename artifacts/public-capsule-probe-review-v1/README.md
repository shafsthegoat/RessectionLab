# Prospective public cache probe review

No public probe has executed. `pre-review-script.py` preserves the first
prospective runner after its seven tiny tests passed. Independent static review
identified three additional gaps to exercise: measurement-hook exit failures
could omit a per-phase failure receipt, the common template preview count was
not explicitly checked, and launcher success did not require the result payload
itself to say `completed` when its worker status/hash did.

The current separate runner has been amended for those paths, plus an explicit
check that independent audits never increment cache calls. Static follow-up
also found malformed worker/result JSON could leave the launcher marked
`running`; the runner now records a failed parse receipt and requires the
worker's runtime identity as well as the result's to match the launch snapshot.
The fixed manifest,
work counts, source model, cache capacity and resource budgets are unchanged.
The engine, adapter and frozen cache prototype are unchanged.

After the parent released the compute hold, independent replay reproduced all
five selected gaps against the preserved source in 0.33 seconds. The repaired
runner passed all 25 independent cases in 1.47 seconds, and all seven owner
cases passed in 1.10 seconds. The first repaired review run also retained one
harness failure: comparing in-memory tuples directly with JSON lists. That
assertion now uses canonical history equality; no runner change was needed.

The independent logs and source-bound receipt are in
`../public-capsule-probe-independent-v1/`. Owner results and exact file hashes
are in `owner-tests.txt` and `owner-review.json`. No public case was loaded by
these tests, no public benchmark has run, and these validation timings imply
no cache speedup. Public execution still requires the parent's separate release.
