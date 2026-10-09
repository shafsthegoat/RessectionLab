"""Saved-only audit: stdlib/NumPy array reading; no policy/native imports."""
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "build/working-length-reverse-diagnostic-v1"
RUN = BASE / "attempt-01"
OUT = Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def close(a, b): assert math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-13), (a,b)
def array_digest(array):
    encoded = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded + array.tobytes(order="C")).hexdigest()
def semantic(value):
    return "sha256:" + hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def probability(values):
    high = max(values)
    return math.exp(values[0]-high) / math.fsum(math.exp(x-high) for x in values)

expected_declaration = "8463bbf4e244498cc1df6ca2a92df2ed981f64c9622f158aaf7f62e50bd01204"
assert sha(RUN / "declaration-input.json") == expected_declaration
declaration = read(RUN / "declaration-input.json")
index = read(RUN / "output-sha256.json")
files = {str(p.relative_to(RUN)):p for p in RUN.rglob("*") if p.is_file()}
assert set(files)-{"output-sha256.json"} == set(index)
assert all(sha(files[name]) == value for name,value in index.items())
for group in ("source_sha256", "input_sha256"):
    for name,value in declaration[group].items():
        assert sha(RUN / "source-and-input-snapshot" / name) == value
        assert sha(ROOT / name) == value
selected = declaration["checkpoint_lineage"]["frozen_checkpoints"]["RL"]
checkpoint = ROOT / selected["path"]
assert sha(checkpoint) == selected["file_sha256"]
prior_index = read(checkpoint.parent / "output-sha256.json")
assert prior_index[checkpoint.name] == selected["file_sha256"]
assert prior_index["declaration-input.json"] == selected["declaration_sha256"]
result, identity, supervisor = [read(RUN/name) for name in ("result.json", "input-identity.json", "supervisor.json")]
suite = read(RUN / "source-and-input-snapshot" / declaration["saved_suite"])
saved = suite["methods"]["RL"]["details"]["decisions"][0]
assert result["status"] == supervisor["status"] == "complete"
assert supervisor["returncode"] == 0 and supervisor["worker_termination_confirmed"]
assert supervisor["cleanup_errors"] == [] and supervisor["unresolved_worker_pid"] is None
assert supervisor["termination_reason"] is None and not supervisor["timed_out"] and not supervisor["automatic_retry"]
assert supervisor["declaration_sha256"] == expected_declaration
assert supervisor["seconds"] < 20. and supervisor["sampled_peak_rss_bytes"] <= 1073741824
assert result["forward_calls"] == {"attempted":1,"completed":1}
for name in ("executed_actions", "search_calls", "optimizer_updates", "patient_files_opened"):
    assert result[name] == 0
assert not result["planning_efficacy_measured"] and not result["critic_used"]
assert not identity["input_geometry_certified"] and not identity["strategy_execution_permitted"]
assert result["checkpoint_parameter_hash"] == selected["parameter_hash"]
assert result["same_other_preprocessing_tensors"] is True
assert (RUN / "worker.log").stat().st_size == 0
assert result["original_observation_hash"] == identity["original_hash"] == saved["observation_hash"] == declaration["expected_root_observation_hash"]
assert result["legal_action_ids"] == identity["action_ids"] == saved["action_ids"]
assert result["saved_actual_logits"] == saved["legal_logits"]

array_names = ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm", "action_geometry", "state_features", "action_mask")
with np.load(RUN / "generated-permitted-inputs.npz", allow_pickle=False) as archive:
    assert set(archive.files) == {"original_"+name for name in array_names} | {"changed_action_geometry"}
    original = {name: archive["original_"+name].copy() for name in array_names}
    changed_geometry = archive["changed_action_geometry"].copy()
assert original["action_geometry"].shape == changed_geometry.shape == (8,16)
assert original["action_geometry"].dtype == changed_geometry.dtype
assert np.argwhere(original["action_geometry"] != changed_geometry).tolist() == [[i,12] for i in range(1,8)]
assert np.all(original["action_geometry"][1:,12] == 120.)
expected_lengths = [declaration["lengths_mm"][name] for name in identity["action_tool_ids"][1:]]
assert changed_geometry[1:,12].tolist() == expected_lengths
assert original["action_mask"].dtype == bool and original["action_mask"].all()
assert original["state_features"][:2].tolist() == [0.,2.]
assert original["image_channels"].shape == (6,9,9,7)
for name,array in original.items():
    assert array_digest(array) == identity["original_array_sha256"][name]
    expected = changed_geometry if name == "action_geometry" else array
    assert array_digest(expected) == identity["changed_array_sha256"][name]
# Rebuild fingerprints directly from frozen literal schema constants, without
# importing the DTO, policy or native task modules.
source = RUN / "source-and-input-snapshot/src/resectionlab/spatial_observations.py"
names = {"SPATIAL_OBSERVATION_VERSION", "CHANNEL_NAMES", "ACTION_GEOMETRY_NAMES", "STATE_FEATURE_NAMES"}
constants = {}
for node in ast.parse(source.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
        constants[node.targets[0].id] = ast.literal_eval(node.value)
for label in ("original", "changed"):
    payload = {"version":constants["SPATIAL_OBSERVATION_VERSION"],"track":"inference_only",
        "channels":constants["CHANNEL_NAMES"],"action_geometry":constants["ACTION_GEOMETRY_NAMES"],
        "state":constants["STATE_FEATURE_NAMES"],"arrays":identity[label+"_array_sha256"],
        "action_ids":identity["action_ids"],"action_tool_ids":identity["action_tool_ids"],
        "provenance":identity["channel_provenance"]}
    expected = identity["original_hash"] if label == "original" else identity["intervention_hash"]
    assert semantic(payload) == expected
assert result["intervention_observation_hash"] == identity["intervention_hash"] != identity["original_hash"]

old, new = result["saved_actual_logits"], result["diagnostic_logits"]
assert len(old) == len(new) == 8 and all(math.isfinite(value) for value in old+new)
assert old[0] == new[0] == -1.811178207397461
assert result["logit_changes"] == [b-a for a,b in zip(old,new)]
close(result["saved_stop_minus_best_movement"], old[0]-max(old[1:]))
close(result["stop_minus_best_movement"], new[0]-max(new[1:]))
close(result["stop_softmax"], probability(new))
best = max(range(len(new)), key=new.__getitem__)
assert result["highest_ranked_id_not_executed"] == identity["action_ids"][best]
decomp = read(RUN / "source-and-input-snapshot/artifacts/matched-estimate-methods-v1/logit-decomposition.json")
common = []
for row in decomp["methods"]["RL"]["common_actions"]:
    index_in_input = identity["action_ids"].index(row["actual_120mm"]["action_id"])
    assert new[index_in_input] == row["original"]["logit"]
    assert old[index_in_input] == row["actual_120mm"]["logit"]
    common.append({"tool_id":row["tool_id"],"endpoint":row["voxel"],"restored_logit":new[index_in_input]})
assert len(common) == 4
phases = result["reconstruction_native_previews"]["phases"]
assert set(phases) == {"generated_fixture", "actual_120mm_observation"}
assert sum(row["started"] for row in phases.values()) == 24 <= declaration["max_reconstruction_previews"]
for row in phases.values():
    assert row["started"] == row["returned"] == row["feasible"]+row["rejected"] == 12
    assert row["raised"] == 0
assert result["reconstruction_seconds"] < result["elapsed_seconds"] < supervisor["seconds"]
assert all(sha(files[name]) == value for name,value in index.items())
audit = {"status":"PASS_saved_reverse_descriptor_diagnostic","declaration_sha256":expected_declaration,
    "output_index_sha256":sha(RUN/"output-sha256.json"),"indexed_files":len(index),
    "finalized_files":len(files),"finalized_bytes":sum(p.stat().st_size for p in files.values()),
    "source_bindings_checked":len(declaration["source_sha256"]),"input_bindings_checked":len(declaration["input_sha256"]),
    "checkpoint_file_sha256":sha(checkpoint),"checkpoint_parameter_hash":result["checkpoint_parameter_hash"],
    "exact_changed_array_coordinates":[[i,12] for i in range(1,8)],"changed_lengths_mm":expected_lengths,
    "original_observation_hash":identity["original_hash"],"intervention_observation_hash":identity["intervention_hash"],
    "both_observation_hashes_recomputed_from_saved_arrays_and_metadata":True,
    "all_other_saved_array_hashes_unchanged":True,"forward_calls":result["forward_calls"],
    "executed_actions":0,"search_calls":0,"optimizer_updates":0,"patient_files_opened":0,
    "original_stop_minus_best_movement":result["saved_stop_minus_best_movement"],
    "diagnostic_stop_minus_best_movement":result["stop_minus_best_movement"],
    "original_stop_softmax":probability(old),"diagnostic_stop_softmax":probability(new),
    "stop_logit_unchanged":old[0],"highest_ranked_id_not_executed":result["highest_ranked_id_not_executed"],
    "four_common_movement_scores_exactly_restore_original_logits":common,
    "actual_attempt_reconstruction_previews":24,"actual_attempt_reconstruction_seconds":result["reconstruction_seconds"],
    "actual_worker_seconds":result["elapsed_seconds"],"supervisor":supervisor,
    "review_new_checkpoint_decodes":0,"review_new_model_forwards":0,"review_new_native_previews":0,
    "limits":["DTO descriptor intervention only; altered actions were uncertified and not executed", "score sensitivity is not strategy efficacy or generalized policy repair", "one familiar fully observed generated fixture, not limited-input patient transfer", "runtime preprocessing and post-forward state checks are source-bound receipts, not repeated model evaluation", "sampled worker RSS may miss transient peaks; source snapshots/postflight excluded from worker timer"]}
(OUT/"audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({"status":audit["status"],"audit_sha256":sha(OUT/"audit.json"),"files":len(files),"bytes":audit["finalized_bytes"],"stop_margin":result["stop_minus_best_movement"],"stop_softmax":result["stop_softmax"]},sort_keys=True))
