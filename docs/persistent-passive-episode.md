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
The observation hides its generated status until acquisition. That status currently
changes the displayed result and STOP reason, not the physical choices: this is
not yet a useful information-gathering policy or a completed hybrid model.

`export_passive_episode` uses its own versioned schema. Existing aspiration/probe
strokes still insert and withdraw within each action. Their desktop replay cannot
be reused to depict retained instruments. The passive exporter is not admitted
to search, trained policies, patient execution, desktop hydration or workspace
save/reopen. Those consumers need a separate complete contract with fresh replay
and two-tool display before integration. The second instrument's insertion is
also not modeled by the current fixed episode.

The [Case2 observation result](../artifacts/resect-case2-sparse-update-v1/RESULT.md)
provides separate evidence for sparse image updating given six observed landmark
correspondences. It does not validate this generated action sequence, dense
tissue response or cutting forces. Joining these capabilities still requires an
explicit observation-to-planning-state update and matched decision evaluation.
