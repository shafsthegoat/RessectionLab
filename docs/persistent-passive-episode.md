# Persistent passive tool episode

`resectionlab.persistent_passive` adds an opt-in generated task on the existing
native geometry engine. A tool moves along a fixed axis into a declared empty
tunnel, remains there during a generated status observation, and explicitly
returns to its access pose. In paired mode, a second tool starts inside the
tunnel and must separately withdraw before STOP. Both tools remain represented
at their recorded access poses after withdrawal; withdrawal does not mean absence.

The fixed fixture is an analytic support shell with a pre-existing 3×3 mm void.
At the held position, the complete 6 mm shaft and tip fit inside that void.
No action creates this cavity, removes tissue or earns contact/removal credit.
The five paired actions produce six authoritative boundary snapshots, including
the initial state. The generated clock totals four seconds; those durations are
declared test inputs, not measured operative time.

```python
from resectionlab.persistent_passive import run_passive_development_episode

episode = run_passive_development_episode(paired=True)
```

The task checks the full swept tool against tissue, hard exclusions, unknown
space when declared, and the other recorded tool. An independent checker repeats
tissue/access checks; tool–tool clearance uses the primary checker only. Fresh
task replay verifies state and frame consistency using the same implementation.
These checks establish software behavior under generated assumptions, not
physical or clinical validity.

State seals bind poses, activation, clock, history and generated observation
input. Changed state or unsupported actions are refused. Cancellation before
commit preserves state; cancellation after commit reports the durable transition.
The v1 observation hides its generated status until acquisition. V1 changes the
displayed result and STOP reason, while the opt-in v2 below can change whether
a further passive movement is offered. Neither is a trained policy.

`export_passive_episode` uses its own versioned schema. Existing aspiration/probe
strokes still insert and withdraw within each action. Their desktop replay cannot
be reused to depict retained instruments. The fixed paired v1 now has a separate desktop backend operation and source-bound
workspace save/reopen with fresh reconstruction. Renderer integration is pending.
Search, trained policies, patient execution and desktop v2 remain unsupported.
The second instrument's insertion is not modeled by the fixed episode.

The [Case2 observation result](../artifacts/resect-case2-sparse-update-v1/RESULT.md)
provides separate evidence for sparse image updating given six observed landmark
correspondences. It does not validate this generated action sequence, dense
tissue response or cutting forces. Joining these capabilities still requires an
explicit observation-to-planning-state update and matched decision evaluation.

## Observation-conditioned generated decision

```python
from resectionlab.persistent_passive import run_passive_decision_episode

episode = run_passive_decision_episode()
```

The v2 source is the same generated anatomy, with a public navigation waypoint
and a fixed held start. Acquiring a generated registration observation reveals
a navigation-to-source translation and a declared endpoint error radius. Before
reveal, alternate hidden measurements preserve the same public observations,
actions and movement history. After reveal, an eligible waypoint offers
`PASSIVE_ADVANCE`; missing information, excessive error, wrong fixed-axis
alignment, inadequate reach, uncertain image-domain coverage or full-tool
collision leave withdrawal available. A caller may also decline an eligible
advance. Both tools must withdraw before STOP.

This rule assumes an exact held start, fixed orientation and an endpoint inside
a generated Euclidean error ball. Inflated shaft/tip capsules conservatively
cover the uncertain path; separate affine-domain and access-depth/reach checks
cover constraints that radius inflation alone does not enforce. The bound is
not an empirically calibrated sensor uncertainty. No tissue is displaced or
removed; no anatomical warp, force or injury response is inferred. Export uses
`resectionlab.generated-persistent-passive-decision-episode.v2` and fresh replay.
See the [113-test integration and independent review](../artifacts/persistent-passive-decision-v2/INTEGRATION.md).
