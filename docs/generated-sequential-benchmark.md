# Generated finite-skill sequential benchmark

This is a bounded **post-access continuation** over authoritative SurgicalEngine
transitions. Earlier access costs remain unknown. Its results cannot be added
to the separate desktop opening demonstration or counted as patient operations.

`surgical_sequential_task.COMMON_CONFIG` versions the common endpoint:512
primitive attempts,240 modeled seconds,30 seconds per primitive,16 invalid
attempts and eight skill selections. STOP consumes an attempt and supplies no
motion, time or removal. Zero fluid/debris and complete exterior withdrawal are
required. Successful declared treatment retains the existing full-original-
target criterion: at least50% treatment and three original RAS octants each
contributing at least5% of the original target. Report100% removal separately.
Checked STOP below that goal is valid with residual; a compute/attempt/time
boundary is truncation, even when useful work occurred.

The inventory contains two exterior CUSA tip configurations, four independently
supplied target paths and STOP. Work expands to initial exterior placement for an
unposed tool, otherwise swept relocation, then work and withdrawal. Every
primitive outcome survives a later refusal or interruption. Canceled atomic
requests are computation charges without accepted physical outcomes, and have
separate counters. The task uses discrete compatible-cell removal; vascular,
thermal, continuum, drilling and CUSA-v3 dynamics are excluded from every method.
The existing primitive procedure adapter remains evaluation-only. A separate
reviewed admission applies solely to these finite generated skills.

Motor-cortex and corticospinal memberships precede planning, have source and
negative coverage bindings, and preserve target/structure overlaps. Direct
removal and cut-source incidence remain separate dimensions. Geometric encounter
and functional motor/language consequences are unavailable. The existing
population-prior FunctionalEvidence class is not relabelled as patient evidence.

All methods receive the same supplied anatomy/coverage, removal/tool state,
budgets, accepted history and inventory masks. The learned scorer uses28
documented deterministic spatial/state features; IDs, splits, pattern labels,
hashes and future branch outcomes are excluded. Candidate order is independently
shuffled. Skill IDs hash programs rather than rank them. V2 STOP executes a
checked withdrawal to the independently supplied exterior plane when required,
then checked STOP. Inserted/active/placed state and exterior distance are policy
features; the shared mask forbids instrument exchange while inserted. Greedy selects immediate
admissible target gain with the shared harm priority and STOP. Search adapts the
existing constrained Pareto/beam retention to the same inventory and replans
after checked prefixes under its original online ledger. Lookahead bounds
computation, not physical episode length. Inventory generation, discarded probes,
native requests, inference, replay and wall time are measured.

The cohort contains32 training,8 validation and16 held-out parent geometries;
development timing layouts are separate. Patterns include delayed preparation,
forbidden shortcuts, competing budgets and unavoidable residual. Related harm
interventions stay with their parent. Held-out outcomes never select checkpoints.
One32-unit candidate MLP and a separate32-unit value baseline receive episodic
reward-driven actor-critic updates. There is no imitation warm start. Returns
discount by0.997 per accepted modeled second. The engineering reward is accepted
target/27 minus accepted neural/27,0.05 bone/27,0.01 per modeled second and0.01 per
completed primitive attempt. Hard constraints remain independent of reward.
The campaign is capped at three seeds,20,000 requested transitions and two wall
hours including validation. All eight validation cases must finish before a
checkpoint can replace the incumbent. An untouched fallback is explicitly
untrained and cannot be labelled RL.

The preserved v1 diagnostic stopped after64 seed17 updates because refused work
could leave an accepted exterior move flagged as inserted without a recovery
skill. It never evaluated held-out outcomes. V2 repairs that shared interface
with real withdrawal and retained costs. The formal study freezes six epochs
(192 training episodes per seed), unchanged width/optimizer/reward and all three
seeds. Its200-transition pilot passed. The original11:50:07UTC campaign origin
and20,000-call limit continue across the repair;13:50:07UTC is the absolute
training cutoff.681 prior requests are measured, with at most704 additional
requests in the interrupted validation block;1,385 are conservatively charged.
The final records distinguish that upper bound from measured requests. Per-case
validation files now preserve a block interrupted before checkpoint selection.

Source classification is shared by policy, planning and new-episode admission.
Patient/source markers take precedence even with freshly rebound policy hashes.
Consequence selection applies declared bounds, maximum treatment, componentwise
neural/structure effects and supported exposures, then preparation, bone and
time. Incomparable effects produce unresolved STOP. Beam retention reserves
distinct harm vectors before duplicate burden variants. Unknown required
exposure remains unavailable.

Design rationale follows neurological-function priority in
[EANO](https://www.nature.com/articles/s41571-020-00447-z), reward-independent
[shielding](https://arxiv.org/abs/1708.08611), and
[separate reward/constraint objectives](https://proceedings.mlr.press/v70/achiam17a.html).
The engine guards retain their stated engineering scope; these references do
not supply clinical numerical thresholds or formal guarantees for this model.
Anatomical position remains separate from function, consistent with the
[tractography limitations examined by Voets et al.](https://pubmed.ncbi.nlm.nih.gov/34196915/).

From a fresh checkout with the installed project/RL dependencies:

```sh
PYTHONPATH=src .venv/bin/python scripts/run_sequential_milestone.py --generated-only --stage pilot --output testcases/overnight-sequential-v1/reproduction
PYTHONPATH=src .venv/bin/python scripts/run_sequential_milestone.py --generated-only --stage protocol --output testcases/overnight-sequential-v1/reproduction
cp docs/sequential-milestone-protocol.json testcases/overnight-sequential-v1/reproduction/protocol.json
PYTHONPATH=src .venv/bin/python scripts/run_sequential_milestone.py --generated-only --stage train --training-admission-review docs/sequential-training-admission-review-v2.json --output testcases/overnight-sequential-v1/reproduction
PYTHONPATH=src .venv/bin/python scripts/run_sequential_milestone.py --generated-only --stage evaluate --output testcases/overnight-sequential-v1/reproduction
```

`--stage controls` runs greedy/search without checkpoints. Every stage writes a
failure summary. Frozen evaluation retains all case/method/seed rows including
failures and timeouts; replay disagreement is an unmet criterion. Bulk evidence
stays in ignored testcases; the tracked delivery manifest and case-level CSV
preserve protocol, seeds, source/config/checkpoint hashes, commands, artifacts
and limitations. An optional acquired-packet path is inspection-only.
