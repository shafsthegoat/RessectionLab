# Prepared TRAIN planner comparison

Version 2 is a prospective zero-update comparison. The first V1 execution is
preserved separately with all 12 arms rejected at the durable-history boundary:
live native metrics contain tuples while JSON reloads sequences as lists. The
repair compares canonical JSON for the entire saved, supplied and recaptured
metrics, preserving all fields and exact finite numeric values. No geometry,
reward, patient, model, action or budget changes accompany that repair. The new
`scripts/compare_prepared_training_planners.py` attempts TRAIN PAT05, PAT22, PAT25
and PAT28. It retains PAT16 and PAT20 and their original support-conflict receipts
in the six-patient denominator without decoding them. No SELECT or unopened case
is eligible. Existing runs and the default shortest-exit constructor stay intact.

The runner authenticates the retained source/access preparation. PAT05 uses the
separate adapter that verifies all 20 historical grid fields and rederives the
shared access rule against the original metadata. Other attempted cases use the
existing frozen-preparation loader. `prepare_native_access` screens the same six
exits, selects any statically admissible entry by distance/axis/sign, and supplies
one selected source. Its selection does not use reward. The runner verifies that
source inputs and tools differ only in access, then constructs one complete
selected initial inventory with the unchanged three-decision horizon and reward.
A selected inventory with no legal non-STOP action remains the result. There is
no fallback or alternative search selected from planner outcomes.

STOP, the fixed PAT05 visited-imitation model, and observed greedy search start
from independent mutable clones of that identical selected state. The checkpoint
is bound to its completed training byte inventory, architecture and parameter
hash `74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e`.
No optimizer, gradient or adaptation is constructed. Search uses the full
permitted nominal annotation; the CNN receives its declared 64-cube view.
Shared source/task does not mean identical representations. Selected-view
coverage and proposal dispositions remain in the preparation receipt.

Each arm has one uninterrupted cooperative 90-second planning/execution clock
and at most 468 actual native preview entries. Clone work, policy decisions,
search planning, actual replay, successor inventories and durable terminal export
occur inside that guard. Search action-score counts are reported separately from
native previews. The existing episode writer persists attempted and committed
steps, including committed-but-unreturned interruptions. Before independent
audit, the callback verifies that the saved terminal metrics equal the actual
task metrics, writes a separately retained terminal receipt, then exits the
execution phase and guard. Audit runs with preview instrumentation restored.
Post-audit guards still enforce the whole worker envelope and never query an
inactive arm guard.

Shared original preparation, static ingress screening, selected initial inventory
and independent audits are reported separately. The proposed outer per-patient
envelope is 600 seconds, 6 GiB and one CPU thread using the existing supervisor;
that setting requires the root's final declaration/release. Its sampled RSS covers
the worker and may miss transient peaks. The arm clock is cooperative and cannot
forcibly interrupt an in-flight native/C operation. There is no kernel-enforced
90-second claim or inferred cache-hit count.

A complete comparison requires all three named arms, unchanged initial task and
policy identities, successful preview/time accounting, complete terminal history,
accepted independent replay, source/checkpoint closure after export, and a
successful parent supervisor. Partial planning prefixes, incomplete replay,
audit rejection, source mutation, budget failure or export failure retain null
outcomes, never a zero. A complete STOP-only episode is a valid zero result but
provides no evidence of useful tissue access. All six patients remain in the
summary even when preparation or execution fails.

The default CLI validates only the prospective metadata. Declaration creation
captures tracked package files from the exact Git root plus explicit script
dependencies. Runtime validation uses that frozen path set and requires every
actually imported local source to be bound, so an isolated archive never discovers
an enclosing checkout. The new runner, its tests and this document are the only
files in this integration slice. Most offline controls use protocol doubles. A regression additionally uses the
actual tiny analytical NativeSpatialTask, real episode writer and PlanningBudget,
and independent native audit for STOP and non-STOP histories. A fixed test-only
legal-action policy performs no learning; no public patient or stored checkpoint
is loaded. Both positive boundary tests failed against preserved V1 bytes before
the repair, while a deliberately changed saved scalar remained rejected.

The clinical unknowns are unchanged: provisional support and hypothetical access,
limited proposal coverage, missing functional/vascular evidence and unvalidated
tissue mechanics. Geometric non-target volume is not neurological harm. This
comparison tests a preparation change with a frozen imitation policy; it is
neither an RL gain nor held-out clinical efficacy.
