"""Read-only audit of saved native frontier probe; no simulator/geometry execution."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_fill_holes

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "artifacts/native-frontier-expansion-v1/experiment-2"
STUDY = ROOT / "artifacts/learning/procedural-native-to-ucsf-v2"
sys.path.insert(0, str(STUDY / "frozen-source/src"))
from resectionlab.imaging import load_case


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8), (a, b)


declaration = read(EXPERIMENT / "declaration.json")
report = read(EXPERIMENT / "report.json")
snapshot = read(STUDY / "worker-source.json")
assert declaration["frozen_runtime_hash"] == snapshot["numerical_runtime_content_hash"]
for name, expected in snapshot["numerical_runtime_sha256"].items():
    assert hashlib.sha256((STUDY / "frozen-source" / name).read_bytes()).hexdigest() == expected
script_bytes = (EXPERIMENT / "experiment-script.py").read_bytes()
assert hashlib.sha256(script_bytes).hexdigest() == declaration["script_sha256"]
assert script_bytes == (ROOT / "scripts/probe_native_frontier.py").read_bytes()
assert digest({k: v for k, v in declaration.items() if k != "proposal_model_hash"}) == declaration["proposal_model_hash"]
assert digest(declaration) == report["declaration_hash"]
assert declaration["caps_per_inventory"] == {"max_actions": 3, "max_preview_checks": 128, "max_search_seconds": 45.}
assert declaration["new_action_model"] and declaration["gradient_steps"] == 0
assert declaration["world_partitions_opened"] == []
assert declaration["final_worlds_used"] is declaration["stress_worlds_used"] is False
assert len(declaration["rule"]["offsets_source_voxels"]) == 13

compression = read(EXPERIMENT / "candidate-compression.json")
candidates = {}
for row in compression["files"]:
    compressed = (EXPERIMENT / row["gzip_file"]).read_bytes()
    raw = gzip.decompress(compressed)
    assert len(compressed) == row["gzip_bytes"] and len(raw) == row["raw_bytes"]
    assert hashlib.sha256(compressed).hexdigest() == row["gzip_sha256"]
    assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
    if (EXPERIMENT / row["raw_file"]).exists():
        assert raw == (EXPERIMENT / row["raw_file"]).read_bytes()
    candidate = json.loads(raw)
    assert digest({k: v for k, v in candidate.items() if k != "candidate_hash"}) == candidate["candidate_hash"]
    candidates[row["raw_file"].removesuffix("-candidate.json")] = candidate

case = load_case(STUDY / "source-case.ressectionlab")
assert case.semantic_hash == declaration["source_hash"]
labels = np.zeros(case.mri.shape, np.int16)
for label, name in enumerate(sorted(case.compartments), 1):
    mask = case.compartments[name]
    assert not np.any(labels[mask])
    labels[mask] = label
assert case.brain_mask is None and case.frame == "RAS+"
tissue = binary_fill_holes(np.asarray(case.mri) != 0) | (labels > 0)
tools = {tool["tool_id"]: tool for tool in declaration["tools"]}
weights = declaration["reward"]
partial_weight = declaration["partial_contact_weight"]


def predicted_inventory(label_field, remaining, affine, access):
    """Rebuild the finite proposal list independently, with no feasibility calls."""
    basis = affine[:3, :3]
    origin = (np.linalg.inv(affine) @ np.r_[access["center_mm"], 1.])[:3]
    normal = np.asarray(access["normal_inward"])
    orientation = (basis / np.linalg.norm(basis, axis=0)).T @ normal
    axis = int(np.argmax(np.abs(orientation)))
    direction = int(np.sign(orientation[axis]))
    transverse = [i for i in range(3) if i != axis]
    rays, omitted = [], []
    for offset in declaration["rule"]["offsets_source_voxels"]:
        entry_index = origin.copy()
        entry_index[transverse] = np.rint(origin[transverse]) + offset
        assert np.all(entry_index[transverse] >= 0)
        assert np.all(entry_index[transverse] < np.asarray(label_field.shape)[transverse])
        selector = np.rint(entry_index).astype(int).tolist()
        selector[axis] = slice(None)
        available = np.flatnonzero((label_field[tuple(selector)] > 0) & remaining[tuple(selector)])
        available = sorted((int(i) for i in available if (i - origin[axis]) * direction > 0),
                           key=lambda i: (i - origin[axis]) * direction)
        if not available:
            omitted.append({"offset": offset, "reason": "NO_REMAINING_TARGET_IN_COLUMN"})
            continue
        entry = (affine @ np.r_[entry_index, 1.])[:3]
        for tool_id, tool in tools.items():
            if np.linalg.norm(entry - access["center_mm"]) + max(tool["tip_radius_mm"], tool["shaft_radius_mm"]) > access["radius_mm"] + 1e-9:
                omitted.append({"offset": offset, "tool_id": tool_id, "reason": "FULL_TOOL_APERTURE_PREFILTER"})
                continue
            ends = []
            for depth in dict.fromkeys((available[-1], available[0])):
                index = entry_index.copy()
                index[axis] = depth
                ends.append((affine @ np.r_[index, 1.])[:3].tolist())
            rays.append((f"axis:{offset[0]}:{offset[1]}:{tool_id}", entry.tolist(), ends))
    return rays, omitted


def check(name, row, history, label_field, tissue_field, affine, access, expanded=True, audit=None):
    volume = abs(float(np.linalg.det(affine[:3, :3])))
    removed, contacts, cumulative_partial = set(), set(), set()
    score = 0.
    previous_tool = None
    remaining = tissue_field.copy()
    denominator = []
    assert row["preview_checks"] == len(row["attempts"]) <= 128
    assert row["search_seconds"] < 45.
    assert row["accepted_preview_count"] == sum(a["feasible"] for a in row["attempts"])
    assert row["rejection_counts"] == dict(Counter(a["reason"] for a in row["attempts"] if not a["feasible"]))
    assert len(history) == len(row["committed_actions"]) <= 3
    for round_index, round_info in enumerate(row["rounds"]):
        attempts = [a for a in row["attempts"] if a["round"] == round_index]
        for a in attempts:
            if a["feasible"]:
                expected = (weights["target_per_mm3"] * a["target_mm3"] - weights["normal_per_mm3"] * a["normal_mm3"]
                            - partial_weight * weights["normal_per_mm3"] * a["partial_normal_contact_mm3"] - weights["action_cost"]
                            - weights["motion_per_mm"] * a["motion_mm"]
                            - weights["tool_change_cost"] * int(previous_tool is not None and a["tool_id"] != previous_tool))
                close(expected, a["nominal_increment"])
        if expanded:
            rays, omitted = predicted_inventory(label_field, remaining, affine, access)
            assert len(rays) == round_info["ray_count"] and omitted == round_info["omitted"]
            checked_rays = []
            cursor = 0
            for ray_id, entry, endpoints in rays:
                if cursor == len(attempts):
                    break
                checked_rays.append(ray_id)
                for endpoint_index, endpoint in enumerate(endpoints):
                    if cursor == len(attempts):
                        break
                    attempt = attempts[cursor]
                    assert attempt["ray_id"] == ray_id and attempt["entry_mm"] == entry and attempt["tip_mm"] == endpoint
                    assert attempt["fallback"] == (endpoint_index > 0)
                    cursor += 1
                    if attempt["feasible"]:
                        break
            assert cursor == len(attempts)
            denominator.append({"round": round_index, "generated_rays": len(rays), "checked_rays": len(checked_rays),
                                "unchecked_rays": len(rays) - len(checked_rays), "preview_checks": len(attempts), "omissions": omitted})
        if round_index >= len(history):
            if row["termination"] == "no_positive_legal_proposal":
                assert not any(a["feasible"] and a["nominal_increment"] > 0 for a in attempts)
            continue
        chosen = row["committed_actions"][round_index]
        assert chosen == max((a for a in attempts if a["feasible"]), key=lambda a: (a["nominal_increment"], a["ray_id"]))
        h = history[round_index]
        assert all(h[key] == chosen[key] for key in ("entry_mm", "tip_mm", "tool_id"))
        assert h["native_affine"] == affine.tolist() and h["source_shape"] == list(label_field.shape)
        cells = {tuple(c) for c in h["removed_indices_native"]}
        touch = {tuple(c) for c in h["contact_indices_native"]}
        assert len(cells) == len(h["removed_indices_native"]) and not cells & removed and cells <= touch
        assert all(tissue_field[c] for c in cells)
        micro_removed = set()
        prior_tip = np.asarray(h["entry_mm"])
        corner_offsets = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) @ affine[:3, :3].T
        for micro in h["microsteps"]:
            np.testing.assert_allclose(prior_tip, micro["tip_start_mm"], atol=1e-8)
            microcells = {tuple(c) for c in micro["removed_indices_native"]}
            assert not microcells & (removed | micro_removed)
            if microcells:
                start, end = np.asarray(micro["active_stroke_start_mm"]), np.asarray(micro["active_stroke_end_mm"])
                vertices = np.asarray(list(microcells)) @ affine[:3, :3].T + affine[:3, 3]
                vertices = vertices[:, None, :] + corner_offsets[None, :, :]
                segment = end - start
                fractions = np.clip(np.sum((vertices - start) * segment, axis=-1) / np.dot(segment, segment), 0, 1)
                assert np.all(np.sum((vertices - start - fractions[..., None] * segment) ** 2, axis=-1) <= micro["active_radius_mm"] ** 2 + 1e-9)
            micro_removed |= microcells
            prior_tip = np.asarray(micro["tip_end_mm"])
        assert micro_removed == cells
        newpartial = {c for c in touch - cells - removed - contacts if tissue_field[c] and label_field[c] == 0}
        target = sum(label_field[c] > 0 for c in cells) * volume
        normal = len(cells) * volume - target
        close(target, chosen["target_mm3"])
        close(normal, chosen["normal_mm3"])
        close(len(newpartial) * volume, chosen["partial_normal_contact_mm3"])
        close(len(cells) * volume, h["removed_volume_mm3"])
        score += chosen["nominal_increment"]
        removed |= cells
        contacts |= touch
        cumulative_partial |= newpartial
        remaining[tuple(np.asarray(list(cells)).T)] = False
        previous_tool = h["tool_id"]
    target = sum(label_field[c] > 0 for c in removed) * volume
    normal = len(removed) * volume - target
    close(target, row["metrics"]["simulated_removed_target_mm3"])
    close(normal, row["metrics"]["simulated_removed_normal_mm3"])
    close(np.count_nonzero(label_field) * volume - target, row["metrics"]["modeled_residual_target_mm3"])
    close(len(cumulative_partial) * volume, row["cumulative_partial_normal_contact_mm3"])
    close(score, row["nominal_return"])
    if audit is not None:
        assert audit["feasible"] and audit["complete_tool_checked"] and audit["frontier_checked"]
        assert audit["checker_version"] == "independent-native-sequence-v2"
        assert audit["unsupported_source_tissue_volume_mm3"] == 0
        close(len(removed) * volume, audit["claimed_source_tissue_volume_mm3"])
        close(len(removed) * volume, audit["contained_source_tissue_volume_mm3"])
    if name == "barrier":
        assert all(c[2] < 12 for c in removed)
        assert denominator[-1]["unchecked_rays"] == 14 and row["termination"] == "preview_cap"
    return {"target_mm3": float(target), "normal_mm3": float(normal), "cumulative_partial_normal_mm3": len(cumulative_partial) * volume,
            "nominal_return": score, "cuts": len(history), "termination": row["termination"], "denominator": denominator}


results = {}
candidate = candidates["patient-expanded"]
assert candidate["case_hash"] == declaration["source_hash"] and candidate["engine_hash"] == declaration["unchanged_engine_hash"]
assert candidate["candidate_hash"] == report["expanded"]["independent_audit"]["candidate_hash"]
results["expanded"] = check("expanded", report["expanded"], candidate["history"], labels, tissue, case.affine,
                            declaration["access"], audit=report["expanded"]["independent_audit"])
for name in ("solid", "barrier"):
    synthetic_labels = np.zeros((17, 17, 20), np.int16)
    synthetic_labels[3:14, 3:14, 5:19] = 1
    candidate = candidates[f"frontier-{name}"]
    row = report["phantoms"][name]
    assert candidate["candidate_hash"] == row["independent_audit"]["candidate_hash"]
    results[name] = check(name, row, candidate["history"], synthetic_labels, np.ones_like(synthetic_labels, dtype=bool), np.eye(4),
                          {"center_mm": [8, 8, -.5], "normal_inward": [0, 0, 1], "radius_mm": 6.}, audit=row["independent_audit"])

prior_status = read(STUDY / "comparison/status.json")
prior_audit_dir = STUDY / "comparison" / prior_status["geometry_validation_run_id"]
prior_replay = read(prior_audit_dir / "native-history-replay.json")["SEARCH"]["metrics"]
prior_certificate = read(prior_audit_dir / "native-history-audit.json")["SEARCH"]
assert prior_replay["native_engine_config_hash"] == declaration["unchanged_engine_hash"]
results["fixed"] = check("fixed", report["fixed"], prior_replay["history"], labels, tissue, case.affine,
                        declaration["access"], expanded=False, audit=prior_certificate)
failed_note = (EXPERIMENT.parent / "experiment-1/FAILED_HARNESS.txt").read_text()
assert "numerical_runtime_hash" in failed_note and "before" in failed_note.lower()
receipt = {
    "status": "passed_with_declared_prototype_limits", "training_or_geometry_reexecuted": False,
    "proposal_model_hash": declaration["proposal_model_hash"], "declaration_hash": report["declaration_hash"],
    "source_runtime_hash": declaration["frozen_runtime_hash"], "numerical_source_files_verified": len(snapshot["numerical_runtime_sha256"]),
    "compressed_histories_byte_verified": 3, "results": results,
    "fixed_audit_binding": "exact two saved entries/targets/tools and engine hash match prior v2 SEARCH independently certified source-cell history; no fresh fixed audit claimed",
    "causal_checker_review": "frozen evaluation.py lines959-968 tests swept shaft against remaining BEFORE microstep removal; no future-cavity borrowing",
    "world_scope": "no optimization/selection/final/stress panel opened; factory construction can initialize a deterministic nominal world object",
    "limits": ["post hoc new action model, not learning improvement or independent patient generalization", "orthogonal source grid, axis-aligned access and integer transverse origin intended; prototype np.isclose/allclose default relative tolerance accepts small deviations; actual inputs exactly aligned/integer", "out-of-grid offset silently skipped in prototype; all13 offsets in bounds for these runs", "barrier third round14 rays untested at128-preview cap", "unchosen preview feasibility not individually re-certified; independent certificates cover committed histories", "estimated source brain envelope, hypothetical access, no validated functional evidence or mechanics"],
}
(Path(__file__).parent / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": receipt["status"], "results": {k: {f: v for f, v in row.items() if f != "denominator"} for k, row in results.items()}}, indent=2))
