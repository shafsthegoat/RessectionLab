"""Report the completed six-exit diagnostic using saved JSON only."""
from collections import Counter
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE / "run-01"


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    receipt, receipt_hash = read(RUN / "receipt.json")
    supervisor, supervisor_hash = read(RUN / "supervisor.json")
    declaration, declaration_hash = read(RUN / "declaration-input.json")
    deviation, deviation_hash = read(HERE / "launch-deviation.json")
    preservation, preservation_hash = read(HERE / "post-run-source-receipt.json")
    assert receipt["status"] == supervisor["status"] == "complete"
    assert receipt["all_six_initial_inventories_complete"]
    assert receipt["executed_transitions"] == receipt["commits"] == receipt["optimizer_updates"] == 0
    assert not receipt["replacement_access_selected"]
    assert declaration_hash == supervisor["declaration_sha256"] == deviation["manifest_sha256"]
    assert not deviation["pre_run_source_commit_completed"]
    assert not deviation["pre_run_source_archive_created"]
    assert not preservation["archived_before_run"] and preservation["all_bytes_match_prospective_manifest"]
    assert preservation["manifest_sha256"] == declaration_hash
    assert preservation["launch_deviation_sha256"] == deviation_hash
    rows = []
    for index, result in enumerate(receipt["exits"]):
        access, inventory, traces = result["exit"], result["initial_inventory"], result["trace"]
        assert result["status"] == "complete" and access["exit_index"] == index
        assert inventory["complete"] and inventory["ledger_complete"]
        assert result["preview_calls"] == len(traces) == inventory["emitted_count"] == inventory["declared_slots"] == 78
        assert inventory["omitted_count"] == inventory["duplicate_count"] == inventory["unavailable_count"] == 0
        counts = Counter(row["reason"] for row in traces)
        assert counts == result["proposal_coverage"]["preview_dispositions"]
        actor, accepted = result["actor_coverage"], result["proposal_coverage"]["accepted_envelope"]
        axis_label = "xyz"[access["axis"]] + ("−" if access["outward_sign"] == -1 else "+")
        failed = [row for row in traces if row["reason"] == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"]
        assert all(row["failure_step_index"] == row["completed_preview_microsteps"] == 0
                   and row["failure_insertion_fraction"] == 0 for row in failed)
        rows.append({
            "exit_index": index, "source_axis_outward_label": axis_label,
            "original_selected": access["selected_original"],
            "representative_to_exit_distance_mm": access["distance_mm"],
            "declared_slots": 78, "emitted_previews": 78,
            "accepted": counts.get("NATIVE_CONNECTED_STROKE", 0),
            "shaft_rejected": len(failed), "working_reach_rejected": counts.get("HARD_GEOMETRY:WORKING_REACH", 0),
            "all_shaft_rejections_at_entry": True if failed else None,
            "local_outside_neighbor": result["exterior_neighbor"],
            "nominal_target_cells": actor["nominal_target_positive_voxels_total"],
            "nominal_target_cells_in_actor_crop": actor["nominal_target_positive_voxels_in_crop"],
            "nominal_target_mass_fraction_in_actor_crop": actor["nominal_target_mass_fraction_visible"],
            "target_centers_outside_initial_accepted_aabb": accepted["target_centers_outside_aabb"],
            "elapsed_seconds": result["elapsed_seconds"],
            "source_hash": result["source_hash"], "decision_model_hash": result["decision_model_hash"],
        })
    assert [row["accepted"] for row in rows] == [0, 78, 62, 58, 78, 72]
    original = receipt["exits"][0]
    assert original["original_inventory_exactly_reproduced"]
    central_ids = {row["action_id"] for row in original["initial_inventory"]["emitted"]
                   if row["column_index"] == 0 and row["family"] == "exposed_opening"}
    original_trace = [row for row in original["trace"] if row["action_id"] in central_ids]
    sources = declaration["source_sha256"]
    source_map_digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary = {
        "version": "pat25-six-existing-exits-saved-result-v1", "subject": receipt["subject"],
        "role": "TRAIN", "scope": "Initial uncommitted preview diagnostics; no policy comparison or replacement-access selection",
        "input_sha256": {"run-01/receipt.json": receipt_hash, "run-01/supervisor.json": supervisor_hash,
                         "run-01/declaration-input.json": declaration_hash, "launch-deviation.json": deviation_hash,
                         "post-run-source-receipt.json": preservation_hash},
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bound_source_count": len(sources), "bound_source_map_canonical_json_sha256": source_map_digest,
        "diagnostic_script_sha256": sources["scripts/diagnose_pat25_access.py"],
        "original_bundle_sha256": declaration["member"]["case_bundle_sha256"],
        "original_preparation_sha256": receipt["original_preparation_sha256"],
        "original_inventory_exactly_reproduced": True, "total_previews": receipt["total_previews"],
        "executed_transitions": 0, "commits": 0, "optimizer_updates": 0,
        "replacement_access_selected": False, "clinical_deficit_probability": None,
        "resource_settings": receipt["settings"], "worker_seconds": receipt["elapsed_seconds"],
        "supervisor_seconds": supervisor["seconds"], "worker_peak_rss_bytes": receipt["peak_rss_bytes"],
        "supervisor_sampled_peak_rss_bytes": supervisor["sampled_peak_rss_bytes"],
        "supervisor_rss_samples": supervisor["rss_samples"], "automatic_retry": supervisor["automatic_retry"],
        "pre_run_source_commit_completed": False, "pre_run_source_archive_created": False,
        "pre_run_release_file_created": False,
        "post_run_source_preservation": preservation,
        "provenance_scope": "Prospective manifest/source-hash admission checks and saved independent review; pre-run commit/archive/release block failed, documented separately",
        "original_central_opening_failure_traces": original_trace, "exits": rows,
        "limits": ["Estimated unreviewed support and hypothetical access; no cortical or clinical permission",
                   "Preview acceptance does not establish target removal or complete resection",
                   "Local exterior-connected neighbor does not establish whole-shaft clearance",
                   "AABB exclusions are current optimistic bounding-box diagnostics, not removed fractions",
                   "Alternative accesses change actor crops; y− and z+ have zero target visible",
                   "No cuts, independent executed-trajectory replay, training or new patient-role access"],
    }
    (HERE / "compact-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    colors = ("#318680", "#BC6A5D", "#DCB460")
    fields = ("accepted", "shaft_rejected", "working_reach_rejected")
    names = ("Passed native preview", "Shaft collision at entry", "Working reach exceeded")
    for i, row in enumerate(rows):
        start = 0
        for field, color, name in zip(fields, colors, names):
            value = row[field]
            ax.barh(i, value, left=start, height=.62, color=color, label=name if i == 0 else None)
            if value:
                ax.text(start + value / 2, i, str(value), ha="center", va="center",
                        color="white" if field != "working_reach_rejected" else "#302B20", fontsize=10)
            start += value
        ax.text(81, i, f"{row['nominal_target_mass_fraction_in_actor_crop']:.2%}", va="center", color="#364348")
    ax.set_yticks(range(6), [row["source_axis_outward_label"] + ("  original" if row["original_selected"] else "") for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 92)
    ax.set_xticks([0, 26, 52, 78])
    ax.set_xlabel("Initial previews per hypothetical exit (78 each)", loc="left")
    ax.set_ylabel("Outward source-axis exit")
    ax.text(81, -.65, "Target in\nactor crop", ha="left", va="bottom", fontsize=9, color="#364348")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C4CDCF")
    ax.tick_params(axis="both", length=0)
    # Explicit handles keep zero-valued first-row categories in the legend.
    from matplotlib.patches import Patch
    fig.legend([Patch(color=color) for color in colors], names, loc="lower left", bbox_to_anchor=(.12, .04), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("PAT25: initial geometry outcomes at six existing exits", x=.12, y=.98, ha="left", fontsize=14, weight="bold")
    fig.text(.12, .91, "Same support and two generic tools · no executed cuts · no replacement access selected", color="#58656A", fontsize=10)
    fig.text(.12, .01, "Source-axis signs are not anatomical orientation labels. Preview acceptance is not clinical clearance.", fontsize=8, color="#58656A")
    fig.subplots_adjust(left=.12, right=.96, top=.80, bottom=.23)
    fig.savefig(HERE / "comparison.png", dpi=150, facecolor="white")
    fig.savefig(HERE / "comparison.svg", facecolor="white")
    plt.close(fig)
    print(json.dumps({"total_previews": receipt["total_previews"], "accepted_by_exit": [row["accepted"] for row in rows],
                      "manifest_sha256": declaration_hash, "deviation_sha256": deviation_hash}, indent=2))


if __name__ == "__main__":
    main()
