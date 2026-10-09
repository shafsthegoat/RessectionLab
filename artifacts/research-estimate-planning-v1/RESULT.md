# Research-estimate planning interface

The reviewed generated-only interface is integrated at `src/resectionlab/research_estimate_planning.py`. Search, imitation, RL and hybrid callbacks receive the same nominal estimate and action geometry. A complete action sequence is sealed before a separate target-only evaluator loads private evidence. Method labels alone do not establish that any trained policy ran.

The source matches the reviewed prototype byte for byte; the promoted tests change only the module import. Root ran the promoted tests with the existing limited-observation boundary suite: **85 passed in 5.52 seconds**. Independent review ran the 25 focused tests. Four adversarial direct-evaluator controls preserve the discovered forged-STOP bypass as regression tests; the repaired evaluator repeats input, QC and footprint checks before accessing private evidence.

Source-only ROI selection, real file/asset receipts, patient identity and estimate QC remain requirements for a future patient release. Outside-image tool geometry remains unassessed. This is software boundary evidence on generated arrays, not patient inference, policy training, transfer, physical fidelity or clinical benefit. Existing production model admission is unchanged.

## Frozen promoted files

- `src/resectionlab/research_estimate_planning.py`: `1c9d862018a7432915fe94942e7d6cde26167be9bb56d32de7dfe66909b1dd23`
- `tests/test_research_estimate_planning.py`: `c875887f2fe735a3323fc9543de1766ba18629e5d4d99b1f534fa09e59f0e0ed`
- `docs/research-estimate-planning.md`: `31b34dbf4478e3b0a604cd775676cb5e0ab2da18fbbcd493c098c120b0d0a036`
