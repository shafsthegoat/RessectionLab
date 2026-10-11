"""Stdlib-only saved result audit; does not import or execute the simulator."""
from collections import Counter
from hashlib import sha256
import json
from math import isclose, pi
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def digest(value):
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_record(name):
    raw = (HERE / name).read_bytes()
    return {"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def near(a, b):
    assert isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)


def main():
    spec = read("declaration.json")
    report = read("run-01/report.json")
    launcher = read("run-01/launcher.json")
    prospective = read("run-01/prospective-source.json")
    release = read("root-release.json")
    dispatch = read("dispatch.json")
    controls = read("controls-01.json")
    assert report["source"] == prospective["source"]
    assert spec == prospective["declaration"]
    assert digest(spec) == report["declaration_hash"]
    assert file_record("root-release.json")["sha256"] == dispatch["root_release_sha256"]
    assert release["head"] == dispatch["head"] == controls["head"]
    for name, expected in release["files"].items():
        assert file_record(name)["sha256"] == expected
    assert controls["returncode"] == 0 and controls["source_unchanged"]
    assert controls["child_reaped"] and not controls["timed_out"]
    assert file_record("controls-01.log")["sha256"] == controls["log_sha256"]
    assert controls["source_before"] == controls["source_after"]
    for name, pin in controls["source_before"].items():
        assert prospective["source"]["files"][name] == pin
    assert report["source_unchanged"] and launcher["source_unchanged"]
    assert report["status"] == launcher["status"] == "completed"
    assert launcher["worker_exit_code"] == 0 and launcher["parent_stop_reason"] is None
    assert launcher["numerical_rows_complete"] and launcher["independent_audits_complete"]
    assert report["human_patients"] == report["training_updates"] == 0
    assert report["numerical_threads"] == spec["caps"]["threads"] == 1
    assert report["peak_worker_rss_bytes"] < spec["caps"]["worker_peak_rss_bytes"]
    assert report["worker_seconds"] < spec["caps"]["cooperative_worker_seconds"]
    assert launcher["parent_wall_seconds"] < spec["caps"]["parent_wall_seconds"]
    expected = [f"{g['name']}-{h:g}mm" for h in spec["spacings_mm"] for g in spec["grids"]]
    rows = report["rows"]
    assert expected == report["planned_rows"] == [r["row_id"] for r in rows]
    assert len(rows) == report["completed_rows"] == report["expected_rows"] == 18
    assert report["missing_rows"] == []
    assert report["planned_action_slots"] == spec["expected_action_slots"] == 180

    totals = Counter()
    summary = []
    anatomy_by_sampling = {}
    tool_actions = {}
    for row in rows:
        tool = row["fixed_tool"]
        assert tool == next(t for t in spec["tools"] if t["tool_id"] == row["grid"]["tool_id"])
        step = spec["preparation_lattice_mm"][tool["tool_id"]]
        physical = [{"index": i + 1, "kind": "side_preparation", "tool_id": tool["tool_id"],
                     "entry_mm": [step*x, step*y, spec["entry_mm"][2]],
                     "tip_mm": [step*x, step*y, spec["preparation_tip_z_mm"]]}
                    for i, (x, y) in enumerate(spec["preparation_order_lattice_units"])]
        physical.append({"index": 10, "kind": "target_stroke", "tool_id": tool["tool_id"],
                         "entry_mm": spec["entry_mm"], "tip_mm": [0., 0., spec["target_tip_z_mm"]]})
        assert digest(physical) == row["physical_actions_sha256"]
        tool_actions.setdefault(tool["tool_id"], row["physical_actions_sha256"])
        assert tool_actions[tool["tool_id"]] == row["physical_actions_sha256"]
        key = (row["spacing_mm"], row["grid"]["sampling"])
        anatomy = (row["source_hash"], row["shape"], row["affine"])
        anatomy_by_sampling.setdefault(key, anatomy)
        assert anatomy_by_sampling[key] == anatomy
        actions = row["actions"]
        assert len(actions) == row["planned_actions"] == 10
        for action, planned in zip(actions, physical):
            assert {k: action[k] for k in planned} == planned
        accepted = [a for a in actions if a["status"] == "accepted"]
        rejected = [a for a in actions if a["status"] == "rejected"]
        unexecuted = [a for a in actions if a["status"] == "unexecuted_prior_refusal"]
        assert len(rejected) <= 1
        assert [a["index"] for a in accepted] == list(range(1, len(accepted) + 1))
        assert not rejected or rejected[0]["index"] == len(accepted) + 1
        assert not unexecuted or [a["index"] for a in unexecuted] == list(range(len(accepted) + 2, 11))
        attempted = accepted + rejected
        for prev, current in zip(attempted, attempted[1:]):
            assert prev["result_state_hash"] == current["source_state_hash"]
        for a in rejected:
            assert a["result_state_hash"] == a["source_state_hash"]
            assert a["removed_source_cell_mm3"] == a["removed_physical_mm3"] == a["removed_physical_target_mm3"] == 0
        for a in accepted:
            if a["kind"] == "side_preparation":
                near(a["removed_physical_target_mm3"], 0)
        native = row["native"]
        assert native["complete_sequence"] == native["feasible"] == (len(accepted) == 10)
        assert len(accepted) == native["accepted_actions"]
        assert row["native_calls"] == len(attempted)
        assert native["partial_cells_kept_occupied"]
        for field in ("removed_source_cell_mm3", "removed_physical_mm3", "removed_physical_target_mm3"):
            near(sum(a[field] for a in accepted), native[field])
        near(sum(a["frozen_reward"] for a in accepted), native["frozen_reward_total"])
        near(native["removed_source_cell_mm3"] - native["removed_physical_mm3"], native["removed_outer_voxelization_excess_mm3"])
        bounds = row["static_not_executable"]
        assert native["removed_physical_mm3"] <= bounds["contained_physical_lower_mm3"] + 1e-8
        assert bounds["contained_physical_lower_mm3"] <= bounds["contact_physical_upper_mm3"]
        if native["complete_sequence"]:
            near(native["removed_physical_mm3"], bounds["contained_physical_lower_mm3"])
        audit = row["independent_audit"]
        if accepted:
            assert audit["status"] == "completed"
            cert = audit["certificate"]
            assert cert["feasible"] and cert["complete_tool_checked"] and cert["frontier_checked"]
            assert cert["failures"] == [] and cert["unsupported_source_tissue_volume_mm3"] == 0
            assert cert["action_count"] == len(accepted) and cert["source_case_hash"] == row["source_hash"]
            near(cert["claimed_source_tissue_volume_mm3"], native["removed_source_cell_mm3"])
            near(cert["contained_source_tissue_volume_mm3"], native["removed_source_cell_mm3"])
        else:
            assert audit["status"] == "not_applicable_rejected_no_committed_history"
        totals.update(accepted_actions=len(accepted), rejected_actions=len(rejected),
                      unexecuted_actions=len(unexecuted), native_calls=len(attempted),
                      complete_sequences=int(native["complete_sequence"]), audited_prefixes=int(bool(accepted)),
                      empty_prefixes=int(not accepted))
        fail = rejected[0] if rejected else None
        summary.append({"row_id": row["row_id"], "spacing_mm": row["spacing_mm"],
            "sampling": row["grid"]["sampling"], "tool": tool["tool_id"],
            "complete_sequence": native["complete_sequence"], "accepted_actions": len(accepted),
            "first_failure": None if not fail else {k: fail[k] for k in ("index", "kind", "reason", "failure_tip_mm")},
            "source_credit_mm3": native["removed_source_cell_mm3"],
            "physical_removed_mm3": native["removed_physical_mm3"],
            "physical_target_mm3": native["removed_physical_target_mm3"],
            "outer_excess_mm3": native["removed_outer_voxelization_excess_mm3"],
            "frozen_reward": native["frozen_reward_total"],
            "planned_static_lower_mm3": bounds["contained_physical_lower_mm3"],
            "planned_static_upper_mm3": bounds["contact_physical_upper_mm3"],
            "audit_status": audit["status"], "audit_seconds": audit.get("seconds", 0)})
    counts = report["native_call_costs_including_incomplete_row"]
    assert totals["native_calls"] == counts["native_strokes_started"] == counts["native_strokes_returned"] == 137
    assert totals["complete_sequences"] == report["accepted_complete_sequences"] == report["verified_complete_sequences"] == 11
    assert totals["accepted_actions"] + totals["rejected_actions"] + totals["unexecuted_actions"] == 180
    by_spacing = [{"spacing_mm": h, "complete": sum(r["native"]["complete_sequence"] for r in rows if r["spacing_mm"] == h),
                   "denominator": 6} for h in spec["spacings_mm"]]
    analytic = []
    for tool in spec["tools"]:
        radius = tool["tip_radius_mm"]
        height = spec["target_tip_z_mm"] + radius - spec["target_start_z_mm"]
        exact_target = pi * height**2 * (radius - height/3)
        fine = [r for r in summary if r["tool"] == tool["tool_id"] and r["spacing_mm"] == .25]
        analytic.append({"tool": tool["tool_id"], "spherical_cap_height_mm": height,
            "exact_fixed_continuous_target_mm3": exact_target,
            "fine_grid_target_shortfall_percent": {r["sampling"]: 100*(1-r["physical_target_mm3"]/exact_target) for r in fine}})
    output = {"status": "saved_only_checks_passed", "scope": "Stdlib scalar/metadata joins, no geometry rerun or arrays",
        "head": release["head"], "sources": [file_record(n) for n in ("run-01/report.json", "run-01/launcher.json", "run-01/prospective-source.json", "root-release.json", "dispatch.json", "controls-01.json")],
        "counts": dict(totals), "by_spacing": by_spacing, "action_hashes_by_tool": tool_actions,
        "invariants": ["exact declared action sequence per tool across all grids", "matched anatomy within tool pair", "declared source/release/control identity", "state chains and refused-action immutability", "action sums equal native volume and reward totals", "zero preparation target removal", "all nonempty prefixes have accepted full-tool/frontier independent certificates", "all complete removals equal static contained-cell union", "all rows and action slots accounted"],
        "costs": {"parent_seconds": launcher["parent_wall_seconds"], "worker_seconds": report["worker_seconds"], "peak_worker_rss_bytes": report["peak_worker_rss_bytes"], "native_seconds": counts["native_seconds"], "independent_audit_seconds": sum(r["audit_seconds"] for r in summary)},
        "posthoc_analytic_reference": {"formula": "pi*h^2*(r-h/3), h=target_tip_z+tip_radius-target_start_z", "basis": "Only final distal spherical cap reaches the frozen target plane; all preparation caps end below it", "tools": analytic},
        "rows": summary}
    (HERE / "saved-audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": output["status"], "counts": output["counts"], "by_spacing": by_spacing,
                      "audit_file": file_record("saved-audit.json")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
