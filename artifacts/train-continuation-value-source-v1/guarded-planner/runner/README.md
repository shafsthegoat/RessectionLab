One fixed TRAIN logged-action continuation comparison
===================================================

Root alone freezes and runs this candidate. Source preparation performs no image,
checkpoint, native or optimizer execution. Four original TRAIN sources,36 completed
traces and497 logged states are immutable inputs. The four teacher traces in the
accepted RL8 audit are the already replayed same29 fixed GREEDY teacher states.
Each original trace receives one authoritative fixed-action execution and an
independent full-tool/outcome audit, then existing planning_clone/admit_collected_trace
creates the exact nominal training trace. Original context/behavior seals remain
historical provenance. Every observation/catalog/mask/row/full history must match.

The loaded IL64 encoder, actor, STOP and critic remain frozen. One cloned actor MLP
fits logged Q_i=r_i+max(0,Q_(i+1)), STOP0, for exactly64 updates. This preserves the
current negative action cost. No negative-preparation-success trajectory exists in
this corpus; the claim is productive continuation ranking, not opening credit/Q*.
Same-state all-candidate features require one frozen forward; only chosen logged
rows train the head. No new training trajectory, artificial label or held-out role.

Fresh full-inventory GREEDY is the primary comparator, never called beam. The
separate canonical guarded Q planner uses the identical certified inventory and
actual immediate nominal reward: positive candidates ranked by positive Q with
original-order ties, otherwise exact greedy fallback; STOP iff none positive.
The negative-action guard is hand-coded, not learned. Frozen IL64 and scratchRL8
greedy endpoints are diagnostics. All four routes get fresh selected-action replay
and the existing independent geometry/outcome checker.

Useful gain requires all16 complete/certified routes, exact world identities, no
casewise target/return loss or outside/contact increase versus fresh GREEDY, plus
strictly fewer aggregate planning previews/transitions or shorter path/fewer cuts.
Ties are negative. Full setup/fit/forward/replay costs are reported separately and
never amortized away or described as a proven speedup. Initial/final casewise MSE
is descriptive, not the success criterion.

Forecast and hard limits: single CPU thread,3600s worker+30s cleanup,3GiB group RSS,
100000 previews,2048 frozen forwards,64 head updates. Saved RL collection+replay
cost2497s included two native executions per trace; this reconstruction uses one
plus checker. Estimated reconstruction1300–1800s, frozen endpoints~700s, fresh
GREEDY/Q~200–600s and tiny head fit. One attempt only; cap failure stays negative.
Total stage8MiB, output6MiB (source/audit/cache reserve2MiB), head file<=128KiB.
No full base weights, decoded source arrays or repeated bulk histories copied.
Full new ordered actions and per-step reward/termination/source/observation hashes
permit fresh replay; immutable original traces remain referenced by digest.

Existing historical IL context budgets are retained solely for source/world
admission. They do not enlarge the stricter actual caps above or execute old CE.

Root commands (fresh cache paths must not already exist):

  .venv/bin/python -I -B -X pycache_prefix=/tmp/resectionlab-q-controls-cache -m pytest -q build/train-continuation-value-v1/runner/test_runner.py
  .venv/bin/python -I -B build/train-continuation-value-v1/runner/freeze_sources.py --release
  .venv/bin/python -I -B -X pycache_prefix=/tmp/resectionlab-q-parent-cache build/train-continuation-value-v1/runner/run_owned.py --source-index-sha256 <printed-SHA> --execute

No execution has occurred during preparation. Freeze follows the root's focused
model/planner/runner controls and canonical source commit. Frozen source-index
publication is exclusive; rerunning requires a new explicit root decision.
