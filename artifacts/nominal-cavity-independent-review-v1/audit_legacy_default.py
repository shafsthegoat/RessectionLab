"""Compare default analytic behavior to an exact committed pre-provider module.

Requires the Git object below, current package imports, and NumPy/SciPy. No
patient arrays, learning, timing comparison, or performance claim is involved.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from resectionlab.native_spatial_task import make_native_opening_task

BASE = "3d9278de4735640150e4d9b83ae0ec1d88c3d407"
PATH = "src/resectionlab/native_spatial_task.py"
root = Path(__file__).resolve().parents[2]
source = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"], cwd=root)
module = types.ModuleType("resectionlab._review_committed_fixed_spatial")
module.__package__ = "resectionlab"
sys.modules[module.__name__] = module
exec(compile(source, f"{BASE}:{PATH}", "exec"), module.__dict__)
before, after = module.make_native_opening_task(), make_native_opening_task()
checks = []
for stage in range(3):
    old_inventory, new_inventory = before.candidate_inventory(), after.candidate_inventory()
    old_observation, new_observation = before.observation(), after.observation()
    assert before.case.source_hash == after.case.source_hash
    assert before.decision_model_hash == after.decision_model_hash
    assert old_observation.fingerprint == new_observation.fingerprint
    assert old_observation.action_ids == new_observation.action_ids
    # Added metadata are allowed; all historical inventory fields retain values.
    assert old_inventory == {key: new_inventory[key] for key in old_inventory}
    assert before._engine.history == after._engine.history
    checks.append({"stage": stage, "source_hash": before.case.source_hash,
        "decision_model_hash": before.decision_model_hash,
        "observation_fingerprint": old_observation.fingerprint,
        "legal_actions": len(old_observation.action_ids), "history_length": len(before._engine.history),
        "all_historical_inventory_fields_equal": True})
    if stage < 2:
        tool = before.case.tools[stage].tool_id
        voxel = [4, 4, 1 if stage == 0 else 5]
        action = next(row["action_id"] for row in old_inventory["ledger"]
            if row["tool_id"] == tool and row["voxel"] == voxel and row["feasible"])
        old_step, new_step = before.step(action), after.step(action)
        assert old_step.reward == new_step.reward and old_step.info == new_step.info
assert before.independent_geometry_check().feasible and after.independent_geometry_check().feasible
result = {"status": "passed", "baseline_commit": BASE,
    "baseline_module_sha256": hashlib.sha256(source).hexdigest(),
    "current_module_sha256": hashlib.sha256((root / PATH).read_bytes()).hexdigest(),
    "checks": checks, "scope": "Two analytic transitions, exact historical default contract; no patient or model execution."}
(Path(__file__).parent / "legacy-default-equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "states_compared": len(checks)}))
