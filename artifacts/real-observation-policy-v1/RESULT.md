# Bounded historical-execution exclusions

October 8, 2026. Scope: enforce known October 6 policy exclusions at selected
active APIs, CLI execution branches and cached-model support consumption.

`src/resectionlab/data_policy.py` retains explicit reason codes. Legacy policy
training, known synthetic/procedural checkpoint routes, two frozen-runtime
diagnostics, SynthStrip inference/acquisition/CLI, synthetic demo generation and
desktop training requests refuse. Cached brain support checks reject the two
pinned SynthStrip hashes and unresolved model ancestry. Historical files are
unchanged; raw anatomy and plain result metadata remain inspectable.

Validation command:

```sh
.venv/bin/python -m pytest tests/test_real_observation_policy.py -q
```

Final observed result: **42 passed in 1.51 seconds**. These are control-flow,
protocol, saved-metadata and hash checks; no generated patient fixture, real
patient processing, model evaluation, learning or simulation was performed.

Review history:

1. Initial 18 checks passed on the first narrow guards.
2. Independent read-only review found cached-mask reuse, pre-refusal checkpoint
   deserialization, archived-runtime diagnostics and CLI side effects. These
   paths were repaired; the expanded 40 checks passed.
3. Second source review found the population/procedural failure logger wrapped
   outside the refusal decorator. The existing tests missed output-directory
   effects. Refusal now wraps the logger; two destination-absent checks were
   added, giving the final 42-pass result above.

Limits: source/weight admission remains incomplete; this does not certify all
imports, frozen executable copies or cached cases. Existing synthetic patient
test suites were not run. Checkpoint-dependent legacy replay now refuses;
plain saved JSON is retained. No new compliant learner or physical-fidelity
claim follows from this milestone.
