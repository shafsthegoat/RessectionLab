# Prospective pilot: one RAW update on the native axis model

This is a **pipeline validation** whose caps have been accepted for implementation.
It still requires a tested runner, frozen source and an execution release. This
document authorizes no patient training. Its prospective declaration is
[`native-axis-raw-update-pilot-v1.json`](../manifests/experiments/native-axis-raw-update-pilot-v1.json).
The completed public preflight and every earlier attempt remain unchanged.

The question is whether one real actor update, its complete before/after
selection panels, durable transition accounting and independent geometry checks
can finish together on the existing axis model. A single seed and update cannot
establish learning efficacy, superiority to search, convergence or clinical use.

## Fixed experiment

Use the exact V2 patient, source cells, direct-native configuration, axis proposal
rule, ordering, fallback rule, three-cut horizon, tools, access, physical reward,
partial-contact cost and deterministic world manifests. Pin the V2 decision,
native and proposal hashes in the declaration. This is the expanded axis action model;
the old fixed-catalog native studies remain separate. Retain RAW 15+6 inputs and
the existing 16-hidden-unit actor and critic. No population initialization,
feature transformation, new proposals or reward changes enter this pilot.

Initialize scratch with seed 11 and require the initial full policy hash to match
the V2 untrained policy. Run, in order:

1. The whole two-world selection panel with unchanged initial weights.
2. Two complete stochastic optimization episodes, using the first two seeds of
   the existing three-seed optimization manifest in its recorded order.
3. Exactly one Adam update through the existing REINFORCE loss.
4. The whole same two-world selection panel with the updated weights.
5. Freeze initial/latest/selected weights and all six completed episode histories,
   then independently check their native histories. Reuse a geometry certificate
   only for identical source, model and full history; retain six audit receipts.

Configuration is seed 11, hidden width 16, episodes per update 2, one gradient
step, checkpoint interval 1, learner episode cap 4, and optimization-transition
cap 8. The physical adapter terminates after at most three cuts; STOP can end an
episode earlier. Thus two completed optimization episodes use at most six
transitions, while eight prevents the generic remaining-budget STOP rule from
truncating the intended two-episode batch. There are at most 18 transitions across
all six episodes. These are ceilings, not required cut counts. Learning rate
0.003, gamma 1, entropy weight 0.01, value weight 0.5 and joint gradient clip 5
stay unchanged. No retry, resume, cap extension or additional update is allowed.

The generic learner also makes one factory instance and an optimization-seed
reset to inspect shape and model during setup. Preserve this zero-transition
record explicitly; it supplies neither a score nor an optimization episode.
Its clone/reset cost belongs to separately reported initialization. Do not
mistake its unfinished-at-return accounting label for a seventh learning episode,
or use this exception to allow incomplete decision-bearing episodes.

The existing earliest-wins-ties selection rule remains. An updated policy can
score worse or tie and leave the initial policy selected. Preserve that outcome.
Report the previously measured greedy return only as historical context; do not
present this pilot as a matched-cost search comparison or rerun search here.

## Engineering cost allowance

The completed [V2 report](../artifacts/preflight/native-axis-v2/RESULT.md) measured
48.1939177 seconds for two untrained RAW selection episodes and their resets,
including 0.2141222 seconds of episode JSON export. One initial integrity-checked
clone cost 0.2930661 seconds; cold construction cost 8.0465926 seconds. The six
required episodes give the illustrative calculation:

`3 × 48.1939177 + 6 × 0.2930661 = 146.3401496 seconds`.

A **300-second online allowance** leaves 153.6598504 seconds
above that illustration. This is an engineering allowance, not a measured upper
bound, confidence interval or prediction of gradient throughput. The preflight
used one repeatedly reset simulator and extra inventory/observation reads. The
generic learner uses previous transition observations and a factory per episode.
The accounting wrapper adds integrity scans, history copies, durable exports and
verification. Sampled paths and post-update choices may have different costs.
The illustration neither proves a minimum nor guarantees completion in 300s.

The online clock must include the full initial panel, both optimization episodes,
the update, the full later panel, per-episode clone/factory/reset/inventory work,
decision traces and accounting at their actual call sites. Follow the common
learner's saved timing contract: initialization occurs separately; intermediate
checkpoint exports fall inside elapsed wall time, while the final checkpoint
export has its own measurement. Report every setup and export cost alongside
the online elapsed value, without silently charging it to another arm.

Use the preflight's 600-second whole-worker allowance, 6-GiB
process RSS ceiling and 15-second hard-termination grace. Whole-worker time also
covers case/source preparation, initial cold template, policy/optimizer setup,
independent audits and final publication. Source-copy/launcher time is separate
and also reported. These are engineering integration limits. Cooperative calls can
overrun a deadline; record the actual overshoot and the last durable operation.
No successful pilot is inferred from a process that is killed or loses exports.

## Required decision trace seam

`AxisTrainingAccounting` at commit `f25b13e` supplies authenticated executed and
returned transitions, native commits and committed-but-unreturned cuts. It does
not see actual policy logits. The common learner currently lacks a decision
observer. A minimal supported observer is therefore an implementation gate,
subject to independent review before execution. Its separate owner implements
the seam; this declaration does not change the underlying loss or action model.

Propose an optional observer in the common optimization and deterministic rollout
paths, invoked after the existing forward/action decision and before `step()`.
Pass detached copies of the observation and already-computed outputs, never a
live policy, simulator or autograd tensor. Persist role, panel/update/episode/step,
seed, ordered action IDs, source features, actual float32 policy inputs, transformed
actor inputs, action mask, masked logits, value, chosen row and ID, and whether
STOP was forced. The observer encodes masked negative infinity as the string
`-inf` alongside the mask. For a forced selection
STOP where existing code performs no forward pass, record outputs as absent with
that reason; do not run another forward merely to fill the trace.

The runner joins each record to the accounting model identity and the preserved
initial or latest checkpoint/profile identity by its update index. Those are
joined provenance, not fields captured inside the forward. If probabilities are
calculated afterward from the recorded masked logits, label them as derived and
state the calculation; do not describe them as captured distribution tensors.

The observer must make no extra policy forward, random draw, simulator read,
reset or transition. A durable trace entry precedes the attempted step and joins
to exactly one accounting outcome by episode and decision order. Retain trace
entries for precommit failures, postcommit failures and attempted decisions that
never return. Observer/export errors fail before the attempted transition. The
runner must not infer execution from a trace alone or reconstruct later logits
and describe them as online observations. Logging cost belongs to the pilot.

## Success and publication gates

Keep separate statuses for completed execution, a verified actor update, valid
selection and independent geometry. Require one finite update from exactly two
complete optimization episodes, a finite positive actor gradient norm and an
actual actor parameter hash change to claim a policy update. An optimizer step
that changes only the critic is a retained negative result. Require exactly two
complete selection rows at update indices 0 and 1, each with both prescribed
worlds and the matching checkpoint hash. Partial panels cannot rank a policy.

Candidate publication additionally requires successful worker and launcher
completion, successful final exports, complete authenticated accounting, no
committed-but-unreturned transition, exact agreement with generic returned-step
counters, complete joined decision traces, and accepted independent geometry
receipts. Freeze identities before audits. Even an otherwise valid initial
checkpoint is not published as a successful pilot candidate after a failed
update, incomplete later panel or receipt/audit failure. The accounting receipt
itself retains `candidate_eligible: false`; only the runner's final validated
record can grant eligibility. Keep all failure receipts and partial histories.

Before release, the runner needs tiny tests for observer-on/off tensor and action
equivalence; mask/ID ordering and no extra forwards/random draws; an actual
two-episode/one-actor-update integration; complete initial/later panel gates;
precommit and real postcommit interruption; trace/receipt disk failures; source
and world mismatch rejection before Adam; and worker/export/geometry failure
blocking publication. Include mocked-clock checks for initial-selection cost
and the total allowance. The orchestrator runs one final integrated regression
suite from exact preserved source after the focused observer, runner and
adversarial checks. The independent V2 artifact audit is now complete:
[`verification.json`](../artifacts/native-axis-v2-result-audit/verification.json)
records 25 passing adversarial checks. Its completion does not release this pilot.

This remains one previously studied structural mirror-derived patient with
unreviewed support, hypothetical access and missing motor/language evidence.
Two deterministic seeds do not supply independent patients or uncertainty
samples. No final-evaluation or stress world is opened, and no clinical
probability, clinical safety, efficacy or resumability is claimed.
