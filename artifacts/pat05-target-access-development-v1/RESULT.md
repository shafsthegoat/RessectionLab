# PAT05 geometric target-access proof

One certified native motion followed by STOP removed **175.0006 mm³ of supplied target annotation (1.5301% of 11,437.0397 mm³)** and 121.0004 mm³ of non-target modeled tissue. Independent full-history geometry accepted the motion, including complete-tool and connected-frontier checks; unsupported claimed removal was zero. This establishes a reachable bounded geometric task for the next learning experiment. It does not establish complete resection or clinical usefulness.

This was one preoperative TRAIN case, `sub-PAT05`, with no gradients and no other patients opened. The actor/model uses the supplied annotation and an explicitly acknowledged provisional, unreviewed structural envelope. Every one of the 70 accepted initial non-STOP choices was scored from its existing certified preview using the permitted nominal target and frozen geometric costs. The selected ID was `nominal-cavity-417f590381bc98e52ed7bc77`. The task allowed three transitions; this proof executed one motion and STOP. The fuller learning comparison has its own receipt.

## Preserved evidence

`result.json.gz` decompresses byte-for-byte to the original 4,671,008-byte `result.json`. Compression uses an empty filename, level 9 and timestamp zero; the raw local file was not rewritten. `summary.json` retains source/input identities, scalar outcomes, counts and timing. `probe.py` is the executed probe, and `profile.txt` retains the measured cumulative profile. The local raw JSON and binary profile are excluded from Git to avoid redundant evidence copies.

- Raw SHA-256: `de793c35cf682d2d8fa239b17c30bb66e449d718ee39ddcd8e56934c3562580b`
- Gzip SHA-256: `b340639d422b0dfbdb7c1319c4a68c9c380304665b81350fd7375bc8244c64da`
- Gzip size: 103,006 bytes.

The probe predates the public `observed_greedy_search` helper. Its recorded source hashes bind the evidence to the code actually executed. This proof must not be relabeled as a run of that later helper.

## Measured cost and limits

The instrumented wall time was 26.2084 seconds: 8.9143 seconds preparing the source and initial inventory, 0.4589 seconds for the initial observation/nominal clone and cached scoring, 9.5863 seconds executing the motion and STOP, and 6.6120 seconds for independent replay. The process peak was 1,010,958,336 bytes; memory enforcement was a cooperative phase-boundary check, not a hard allocator cap.

The 148 native previews consumed 13.395 cumulative seconds, including 9.343 seconds in capsule-cell queries. Independent history checking consumed 6.158 seconds internally; model/state assertions consumed 2.246 seconds over 12 calls. These are nested profile measurements and must not be summed. They describe this probe, not the uninstrumented remainder of the separate failed 227-transition beam run.

The local catalog does not establish full-target path coverage. Contacted tissue is reported separately from removal; retained contacted-tissue upper bound was 562.0020 mm³. Motor/language and vascular safety remain unassessed, clinical deficit probability is null, and the rigid source-cell abstraction has known discretization dependence. This receipt supports engineering reachability and profiling only.
