"""Read saved source/JSON evidence only; never import or execute the planner."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/pat05-real-spatial-profile-v2"
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
read = lambda path: json.loads(path.read_bytes())
receipt = read(BASE / "run/receipt.json")
outcome = read(BASE / "outcome.json")
declaration = read(BASE / "run/declaration-input.json")
binding = read(BASE / "archive-source-binding.json")
release = read(BASE / "execution-release.json")
supervisor = read(BASE / "run/supervisor.json")
evidence = read(BASE / "evidence-sha256.json")
before = {name: sha((BASE / name).read_bytes()) for name in evidence}
checks = {}


def check(name, condition):
    checks[name] = bool(condition)


def close(first, second, atol=1e-12):
    return bool(np.isclose(first, second, rtol=0, atol=atol))


check("all_saved_evidence_hashes", before == evidence)
check("run_output_manifest", all(sha((BASE / "run" / name).read_bytes()) == digest
    for name, digest in read(BASE / "run/output-sha256.json").items()))
check("release_binds_archive_preparation", release["archive_source_binding_sha256"] == before["archive-source-binding.json"])
check("one_zero_update_attempt_released", release["attempts_authorized"] == 1 and release["optimizer_updates_authorized"] == 0)
check("exact_declaration_bytes", all(sha((BASE / "run" / name).read_bytes()) == declaration_sha
    for name in ("declaration-input.json", "declaration.json")
    for declaration_sha in (binding["declaration_sha256"], release["declaration_sha256"], supervisor["declaration_sha256"])))
archive_path = Path(binding["source_archive"])
check("source_archive_bytes", sha(archive_path.read_bytes()) == binding["archive_sha256"] == release["source_archive_sha256"])
with tarfile.open(archive_path) as archive:
    archived = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
source_map = declaration["source_sha256"]
numerical_members = {name for name in archived if name == "scripts/preflight_real_spatial_policy.py"
                     or name.startswith("src/resectionlab/") and name.endswith(".py")}
check("closed_archive_numerical_inventory", numerical_members == set(source_map) and len(source_map) == 53)
check("archived_source_bindings", all(sha(archived[name]) == digest for name, digest in source_map.items()))
check("frozen_source_still_matches", all(sha((Path(binding["frozen_source_root"]) / name).read_bytes()) == digest
    for name, digest in source_map.items()))
check("commit_source_bindings", all(sha(subprocess.check_output(["git", "show", binding["source_commit"] + ":" + name], cwd=ROOT)) == digest
    for name, digest in source_map.items()))
check("runtime_source_bindings", receipt["source_sha256"] == binding["source_sha256"] == source_map)
check("archived_declaration", archived["manifests/experiments/pat05-real-spatial-profile-v2.json"] == (BASE / "run/declaration-input.json").read_bytes())
prior = json.loads(archived["manifests/experiments/pat05-real-spatial-profile-v1.json"])
changed_shared_fields = sorted(key for key in prior.keys() & declaration.keys() if prior[key] != declaration[key])
check("v1_common_settings_only_declared_changes", changed_shared_fields == ["adapter_options", "source_sha256", "study_id"])
check("adapter_only_explicit_grid_option", declaration["adapter_options"] == {
    **prior["adapter_options"], "native_grid_reconciliation": "orthogonal_roundoff_1e-6mm"})
source_changes = sorted(name for name in source_map if source_map[name] != prior["source_sha256"].get(name))
check("only_reviewed_three_numerical_modules_changed", source_changes == [
    "scripts/preflight_real_spatial_policy.py", "src/resectionlab/native_spatial_task.py", "src/resectionlab/spatial_policy_diagnostics.py"])
grid = receipt["initial_task_metrics"]["native_grid_reconciliation"]
check("eight_runtime_grid_fields", len(declaration["expected_native_grid_binding"]) == 8 and all(
    grid.get(name) == value for name, value in declaration["expected_native_grid_binding"].items()))
ep, = receipt["episodes"]
metrics, history, decisions = ep["metrics"], ep["metrics"]["history"], ep["decisions"]
check("grid_frozen_in_history", grid == metrics["native_grid_reconciliation"] and all(
    row["native_affine"] == grid["derived_affine_ras_mm"] for row in history))
check("source_support_and_normalization_bindings", all(grid[key] == receipt["initial_task_metrics"]["intensity_normalization"][key]
    for key in ("source_image_hash", "support_hash")))
check("three_fixed_actions_three_forwards_zero_updates", receipt["optimizer_updates"] == 0 and
    len(history) == len(decisions) == ep["committed_transition_count"] == ep["rollout_forward_calls"] == 3 and
    ep["profile_action_rule"] == "first_legal_inventory_row; not a search teacher")
check("recorded_frozen_parameter_identity", receipt["initial_parameter_hash"] == ep["parameter_hash"] and
    "learning" not in receipt and "updated_parameter_hash" not in receipt and not (BASE / "run/latest-policy.pt").exists())
initial = receipt["initial_candidate_inventory"]
inventories = [initial, *(row["after_candidate_inventory"] for row in decisions)]
check("initial36_complete_local_slots", all(initial[key] == 36 for key in ("declared_slots", "evaluated_slots", "accepted_count"))
    and len(initial["ledger"]) == 36 and initial["omitted_count"] == 0 and initial["complete"])
check("initial_pairs_unique", len({(tuple(row["voxel"]), row["tool_id"]) for row in initial["ledger"]}) == 36)
check("fixed_actions_match_first_legal_rows", all(decision["action_id"] == next(row["action_id"] for row in inventories[i]["ledger"] if row["feasible"])
    == history[i]["action_id"] for i, decision in enumerate(decisions)))
check("executed_actions_are_not_policy_choices", all(row["action_id"] != row["policy_selected_action_id"] for row in decisions))
check("inventory_counts_match_recorded_decisions", all(row["candidate_count_including_stop"] == inventories[i]["accepted_count"] + 1
    for i, row in enumerate(decisions)))
check("terminal_inventory_not_omitted", inventories[-1]["terminated_slots"] == 36 and inventories[-1]["evaluated_slots"] == 0
    and inventories[-1]["omitted_count"] == 0 and inventories[-1]["complete"])
phases = receipt["native_preview_cost"]["phases"]
totals = {key: sum(row[key] for row in phases.values()) for key in
    ("started", "returned", "feasible", "rejected", "raised", "returned_microsteps")}
check("144_process_previews", totals == {"started": 144, "returned": 144, "feasible": 138,
    "rejected": 6, "raised": 0, "returned_microsteps": 3552})
check("fresh_and_next_state_preview_accounting", phases["initial_task_preparation"]["started"] == 36 and
    phases["untrained_fixed_inventory_profile"]["started"] == 36 + inventories[1]["evaluated_slots"] + inventories[2]["evaluated_slots"]
    and phases["untrained_fixed_inventory_profile"]["rejected"] == inventories[1]["rejected_count"] + inventories[2]["rejected_count"])
coverage = receipt["initial_observation_coverage"]
check("target_crop_arithmetic", coverage["nominal_target_mass_in_crop"] == 2786 and
    coverage["nominal_target_mass_total"] == 11437 and close(coverage["nominal_target_mass_fraction_visible"], 2786/11437))
check("coverage_state_chaining", coverage == decisions[0]["before_observation_coverage"] and all(
    decisions[i]["after_observation_coverage"] == decisions[i+1]["before_observation_coverage"] for i in range(2)))
check("initial_rays_all_visible", coverage["sample_coverage_counts"] == {"full": 36, "none": 0, "partial": 0} and all(
    row["ray_samples_inside"] == row["ray_samples"] == 5 and row["straight_segment_fraction_inside"] == 1. for row in coverage["actions"]))
axial = receipt["nominal_depth_coverage"]
check("axial_target_center_fraction_arithmetic", axial["nominal_target_centers_beyond_axial_upper_bound"] == 10558 and
    axial["nominal_target_centers_total"] == 11437 and close(outcome["fraction_target_centers_beyond_axial_catalog_upper_bound"], 10558/11437))
check("axial_distal_capsule_bound", close(axial["distal_tip_capsule_axial_upper_bound_mm"],
    axial["endpoint_depth_range_mm"][1] + max(tool["tip_radius_mm"] for tool in declaration["tools"])))
removed, contacted, per_action = set(), set(), []
affine = np.asarray(grid["derived_affine_ras_mm"])
voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
prior_tool = None
for row in history:
    cells = {tuple(p) for p in row["removed_indices_native"]}
    contacts = {tuple(p) for p in row["contact_indices_native"]}
    check("disjoint_new_cells_"+str(len(per_action)), not (removed & cells) and len(cells) == len(row["removed_indices_native"]))
    micro_cells = {tuple(p) for m in row["microsteps"] for p in m["removed_indices_native"]}
    check("microstep_removal_union_"+str(len(per_action)), micro_cells == cells)
    check("cell_volume_"+str(len(per_action)), close(row["removed_volume_mm3"], len(cells)*voxel_volume))
    path = 2*float(np.linalg.norm(np.asarray(row["tip_mm"])-row["entry_mm"]))
    weights = declaration["objective"]
    reward = weights["target_per_mm3"]*row["target_removed_mm3"]-weights["normal_per_mm3"]*row["normal_removed_mm3"] \
        -weights["action_cost"]-weights["motion_per_mm"]*path-weights["tool_change_cost"]*(prior_tool is not None and prior_tool != row["tool_id"])
    check("reward_and_full_path_"+str(len(per_action)), close(row["reward"], reward) and close(row["complete_tool_path_length_mm"], path))
    removed |= cells; contacted |= contacts; prior_tool = row["tool_id"]
    per_action.append({"removed_cells":len(cells),"contacted_cells":len(contacts),"microsteps":len(row["microsteps"]),
        "tool_id":row["tool_id"],"target_removed_mm3":row["target_removed_mm3"],"normal_removed_mm3":row["normal_removed_mm3"],"reward":row["reward"]})
check("raw_volume_contact_accounting", len(removed)==26 and len(contacted)==104 and len(contacted-removed)==78 and
    close(metrics["simulated_removed_volume_mm3"],26*voxel_volume) and close(metrics["currently_retained_contacted_tissue_upper_bound_mm3"],78*voxel_volume)
    and metrics["target_removed_mm3"]==0 and close(metrics["total_reward"],sum(row["reward"] for row in history)))
audit = ep["independent_geometry_check"]
check("saved_full_history_audit", audit["feasible"] and not audit["failures"] and audit["action_count"]==3 and
    audit["complete_tool_checked"] and audit["frontier_checked"] and audit["unsupported_source_tissue_volume_mm3"]==0 and
    audit["source_case_hash"]==metrics["source_hash"] and close(audit["contained_source_tissue_volume_mm3"],26*voxel_volume))
check("clinical_unknowns_preserved", metrics["clinical_deficit_probability"] is None and
    metrics["functional_evidence_available"]=={"motor":False,"language":False} and metrics["partial_contact_weight"]==0)
timing=outcome["timing"]
preview_seconds=sum(row["seconds"] for row in phases.values())
check("timing_totals", close(timing["all_preview_seconds_nested_in_preparation_and_episode"],preview_seconds) and
    close(timing["episode_policy_forward_seconds"],sum(row["forward_seconds"] for row in decisions)) and
    close(timing["preview_fraction_of_worker_elapsed"],preview_seconds/receipt["elapsed_seconds"]))
check("timing_residual_accounting", close(timing["worker_residual_outside_loading_preparation_episode_audit_seconds"],
    receipt["elapsed_seconds"]-receipt["input_seconds"]-receipt["preparation_seconds"]-ep["online_seconds"]-ep["independent_audit_seconds"]) and
    close(timing["parent_import_startup_shutdown_residual_seconds"],supervisor["seconds"]-receipt["elapsed_seconds"]))
check("completed_within_declared_resources", supervisor["returncode"]==0 and supervisor["status"]==receipt["status"]=="complete" and
    not supervisor["timed_out"] and supervisor["termination_reason"] is None and supervisor["seconds"]<600 and
    max(supervisor["sampled_peak_rss_bytes"],receipt["peak_rss_bytes"])<6*1024**3)
check("evidence_unchanged_during_audit", before == {name:sha((BASE/name).read_bytes()) for name in evidence})
result={"schema":"saved-pat05-v2-independent-audit-v1","status":"passed" if all(checks.values()) else "findings",
    "checks":checks,"source_commit":binding["source_commit"],"evidence_sha256":before,
    "v1_common_field_changes":changed_shared_fields,"v1_numerical_source_changes":source_changes,
    "process_preview_totals":totals,"per_action":per_action,"committed_microsteps":sum(r["microsteps"] for r in per_action),
    "volume_mm3":{"voxel":voxel_volume,"geometric_removed":26*voxel_volume,"normal_per_action_sum":sum(r["normal_removed_mm3"] for r in history),
        "normal_aggregate_raw":metrics["normal_removed_mm3"],"aggregate_minus_per_action":metrics["normal_removed_mm3"]-sum(r["normal_removed_mm3"] for r in history)},
    "timing":timing,"target_center_axial_fraction":10558/11437,
    "interpretation":"Engineering profile of three fixed first-legal actions; no learned policy performance. The 92.314% figure describes target voxel centers beyond a one-dimensional axial bound, not a measured percentage of unreachable complete-tool tissue volume.",
    "limits":["No patient bundle/header/image loaded or task/model/geometry re-executed; target-center counts and saved audit are checked for recorded consistency, not recomputed from anatomy.",
        "Frozen parameter hashes are recorded equality; no checkpoint tensors were created or independently replayed.",
        "Observed aggregate normal-volume rounding differs from exact geometric/per-action accounting by about 6.82e-7 mm³; raw values are retained.",
        "Preview time is nested within construction/episode totals; it must not be added again. 3552 preview microsteps are distinct from 16 committed microsteps.",
        "Archive integrity is verified while the local ignored source archive is present. Source code is recoverable from committed Git; original patient bundle remains separately acquired and ignored."],
    "patient_file_loaded":False,"profile_rerun":False}
(OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":result["status"],"passed":sum(checks.values()),"failed":[name for name,value in checks.items() if not value],
    "volume_mm3":result["volume_mm3"],"committed_microsteps":result["committed_microsteps"]},indent=2))
if not all(checks.values()):
    raise SystemExit(1)
