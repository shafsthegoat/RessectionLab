# Public bound v1: exact decisions, no savings

All five fresh pairs preserve exact actions, full histories, outcomes and 0/58/0/51/37 target cells with zero outside-target removal. All ten plans seal before ten fresh uncached independent replays; the separate saved audit passes. **Zero actions are pruned**: each method uses **1,446 online previews**.

| Case | Baseline / bound previews | Baseline / bound inference seconds |
| --- | ---: | ---: |
| ReMIND-002 | 117 / 117 | 6.469 / 6.505 |
| ReMIND-015 | 197 / 197 | 3.539 / 3.832 |
| ReMIND-018 | 75 / 75 | 2.749 / 2.841 |
| ReMIND-045 | 187 / 187 | 6.397 / 6.444 |
| ReMIND-050 | 870 / 870 | 11.222 / 12.760 |

Source-through-plan totals are **30.376 versus 32.381 seconds**, including setup, in one fixed-order comparison. This is an efficiency regression. All 1,118 planning bound queries and 1,118 replay queries return `public_index_workspace_limit`. The adapter supplies the correct validated target domain. Its 2.59–2.96 million unknown cells per case alone exceed the one-million-cell global index cap, before local envelopes are tested. Zero indexed target cells means early refusal, not absent target.

The targeted successor uses bounded local tool-envelope queries while preserving unknown coverage, exact fallback, model weights and rewards. No cap increase, invented full coverage or new data is justified. V1 and its negative result remain unchanged.

Whole run: **183.217 s**, sampled peak **743,079,936 B**, output **15,467,273 B**, clean cleanup. Totals: **5,784 previews, 164 native transitions, 164 ridge scores, 20 source visits**. One existing scorer is loaded with zero fits/optimizer updates/neural forwards; original training cost remains documented in balanced-v2. All first-runtime-control failure evidence is retained: production sealed ten plans, but a fixture incorrectly expected JSON key insertion order. The corrected test passes in 11.04 s with unchanged production.

These are four fitted TRAIN cases plus development-exposed050. SELECT/EVAL remain closed. This measures annotation-assisted geometric decision efficiency, not physical tissue fidelity, held-out transfer or clinical safety. Executed canonical source is `f8c7e18120475e895453c9964bbd65fe40792292`. Complete result, receipts, runtime, histories, certificates and independent audit are in the preservation index; arrays, weights and repeated inventories stay outside Git.
