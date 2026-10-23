# Prepared patient rescue task

The experimental adapter in `scripts/patient_rescue_env.py` defines a separate
task, `resectionlab.prepared-patient-rescue.v1`. It starts after a recorded vessel
injury and native control cleanup. Historical actions retain their original
radii, damage and elapsed time; they are privileged task preparation, **not an
admitted raw-RL episode**. The ordinary raw environment and checkpoint contract
remain unchanged.

`prepare_patient_rescue_task.py` authors fixed surface contacts and validates
each entry/coagulation/inactive-withdrawal route through public actions. It then
checks a complete scripted rescue, including aspiration and five seconds of
stable observation with controls withdrawn. It saves a source-bound manifest,
native branches and exact suffix replay evidence under a new `testcases/`
directory. Geometry diagnostics used to author those waypoints are privileged.
No fitting starts before that baseline passes.

`from_manifest()` publicly reconstructs the complete setup using the protected
engine, measuring actual historical near-contact and strain ledgers. Reset
clones that verified state and retains its cumulative physical burdens. Suffix
limits allow at most 60 simulated seconds, 128 attempted primitives, twelve
option decisions, 0.1 mm³ additional non-target removal, 0.1 mm² additional
vessel damage and 10 mm³ additional modeled bleeding. Existing debris from the
archival access remains an explicit entry burden; this is a wound-fluid cleanup
task. Imported raw-RL checkpoints cannot initialize it.

The six choices are three fixed coagulation contacts, aspiration at the declared
pool, waiting and cleanup verification. Each operative option exchanges tools
when needed, executes catalog-sized nominal skills, then withdraws twice along
the validated line to reach exterior space before another line is chosen. There
is no active-energy exit, feasibility preview, hidden action mask or automatic
choice of the next treatment. Partial/refused options end the episode rather
than invoking unreviewed recovery.

The policy receives sixty values: current published local samples and slot
controls, selected tool settings, remaining budgets, its own option counts and
verification time, and four declared modeled field sensors (retained fluid,
active outlets, outlet rate and emitted volume since entry). Fixed waypoints
are common task inputs. Full grids, face doses, snapshots, reward diagnostics
and future outcomes stay outside policy inference. These are simulator sensors,
not acquired physiological measurements; the observation is partial.

Success requires zero active outlets, retained fluid including reservoirs at or
below 0.1 mm³, inactive/withdrawn/released controls, no additional debris, and
five actual simulated inspection seconds without renewed emission or fluid
increase. Reward retains cumulative engineering injury costs and adds a declared
outlet/fluid potential difference and stable-cleanup bonus. It is uncalibrated.

Install the optional trainer with `pip install -e '.[rl-training]'`, then run:

```sh
.venv/bin/python scripts/train_patient_rescue.py \
  --manifest /absolute/path/under/testcases/task-manifest.json \
  --output testcases/outputs/new-patient-rescue-learning-run
```

The bounded comparison uses Stable Baselines3 PPO, seeds 17 and 29, and 64 actual
option transitions per seed for both PPO and search. Scripted/untrained
assessments have no optimization samples. Search uses a permutation prior over
all six options; its exploration procedure differs from PPO. All assessments
share the same prepared world and decision/time/primitive limits. Seeds change
policy initialization and search sampling, not the physical anatomy. There is
no held-out patient or generalization claim.

Every episode retains native actions, attempted RL transitions, observation
hashes, nominal proposals, rewards and timing. Actual public engine calls are
counted separately from proposals refused before execution. Evaluation replay
checks the observation/request history as well as physical state and rewards.
Native assessment branches preserve all earlier workspace branches and open in
Electron. **Patient rescue experiment · PPO and baselines** displays the saved
report and opens a selected trajectory. Only a canonically replayed matching
endpoint receives the endpoint-verified label; that label does not certify
policy inference or clinical outcomes.

This is a small development task over one authored injury. It exercises order,
coverage, ongoing bleeding and delayed cleanup, while leaving access and tumor
resection to future tasks. It does not qualify a surgical policy.
