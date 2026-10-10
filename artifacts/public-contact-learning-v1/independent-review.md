# Independent review of exact desktop publication candidate

**GO for root-owned promotion of the exact five-file candidate, followed by canonical tests.** No blocking identity or scope defect was found. This review does not execute or authorize an actor, training run or native task. It permits publication of the completed negative result for TRAIN/SELECT inspection, with held-out interactive requests still refused.

Reviewed candidate: `build/contact-learning-publication-v1/`. HEAD was `4384ea0ae681e46986de9505edf226dcaa2a6c8c` throughout the metadata audit. No tracked files were changed by this reviewer.

## Exact reviewed pins

| Item | SHA-256 |
|---|---|
| Publication manifest | `4bc4a93f453fb4064c3291b55c60481de554adfe706ab7db24a3a023f5e20dae` |
| Release verifier | `df739d8a0c855e37aa7d7bce036af402f5dfc054a1fa2d7fcd3e24852a1cd08b` |
| Supervisor | `933c8e8df128691a77a0b346c1d0037fbc39038e1b99c7313b470212956b606f` |
| Desktop helper | `a9f78cc38fd383583473362f0839367c5d3445b7a976e556ca2b9a34034843e1` |
| Corrected portable test | `adbd11049ca78a0aea39d6063741c235d17dec511701fda1166e0bb0532df5d3` |
| Five-file promotion map | `585f1d9e40eef0db09e734211809d97e06507a2ec1936aca42225c45ab76b377` |
| Source/test patch | `049e35c17a47fd940543bf6b89fa12651db84611e15b8ef8381941f463f19da1` |

Independently restoring the three old literals reproduces each exact pre-change source hash. The only production-code changes are: manifest SHA replaces `None`; the supervisor binds that verifier SHA; the helper binds that supervisor SHA. All 26 declared worker source files match, including the unchanged worker. The supervisor checks these sources before execution and again before recording completion. No checkpoint weights enter the five-file promotion map.

The 1,371-byte manifest fixes the pilot's exact four artifact paths: result JSON (7,952,064 B), final-freeze JSON (845 B), and the IL/RL final checkpoints (185,320 B each). Every raw hash/size matches the completed pilot already independently audited. The release verifier rejects absent, substituted, oversized and symlinked artifacts and accepts only the fixed file slots. Both final parameter identities match the pilot and freeze. Both saved training-lineage digests match the freeze's lineage hashes; each lineage records exactly 32 updates.

Metadata admission checks a complete fixed pilot, both final endpoints, and every fixed SELECT/MEASUREMENT_EVAL method/goal/layout cell. It intentionally permits terminal negative or unresolved online outcomes; it is not a positive-reward gate. This particular pilot has all 96 rows complete with no caps. Its IL/RL policies STOP on every online task and score 0/16 held-out contacts each, versus SEARCH 12/16. That negative comparison remains visible in the candidate's result description. Eight layouts with paired surface/deep goals are the held-out split unit; the 64 method episodes do not expand the independent sample size.

The worker subsequently decodes and validates **both** exact final checkpoint files, reconstructs the final-freeze record, compares its complete semantic identity to the published freeze, then loads the chosen final policy. This verifies the training-lineage and parameter fields before task construction. The metadata checker alone is not represented as full checkpoint decoding or signed training authenticity.

## Role and verification controls

Source review confirms that both the desktop API and worker reject any layout outside TRAIN/SELECT. The worker's role gate precedes checkpoint admission and task construction. The availability response marks held-out layouts noninteractive. No renderer-supplied checkpoint path, role, model or task arrays are accepted by this entry point.

The independent standard-library audit admitted the exact saved negative pilot using a disposable root and hardlinks to its fixed artifacts; it did not decode weights. Four extracted early-guard checks (STOP, SEARCH, IL, RL) each refused a held-out layout before any later controller/publication operation, with zero sentinel calls and no attempt created. The corrected generated negative-outcome test uses the actual nested `metrics` keys. All five portable metadata tests passed independently in 0.01 s against the staged verifier. These are metadata/control checks, not canonical integration or inference tests; root retains that next step.

Independent artifacts:

- `audit_publication.py`: SHA-256 `31383a79b4455e4658366a25e7c14378776b91e77ffd70c8b43624597b7b429b`.
- `publication-audit.json`: SHA-256 `0ad70a23cba1ac4d886679de54c728d54b441924f0ebf25ef3c7c7ae5e13b62d`.

The earlier test/map/patch pins were withdrawn because the test originally placed outcome fields at the wrong JSON level. Only the corrected pins above are reviewed for promotion. Manifest and production source hashes remained unchanged during that repair.
