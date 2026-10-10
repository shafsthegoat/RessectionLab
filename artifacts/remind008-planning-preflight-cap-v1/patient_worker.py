"""One owned ReMIND-008 public-input scale preflight; no private loader."""
import argparse
import cProfile
from contextlib import contextmanager
import hashlib
import json
import signal
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
PUBLIC_MANIFEST = "build/remind-planning-qc-v1/ReMIND-008-public-inputs-v2.json"
PUBLIC_MANIFEST_SHA = "57b9588c2a72874cae1cd6555fcd58cfada543ac63cda7f1e6b41fefa5b2163b"
ARRAY_KEYS = ("image", "supplied_support", "supplied_whole_tumor", "whole_tumor_domain")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    # The lease check imports no tensor/model runtime and must precede those imports.
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    if sha(args.release) != args.release_sha256: raise ValueError("Root release changed")
    release = json.loads(args.release.read_text())
    required = {
        "public_geometry_and_source_linkage_checked": True,
        "native_image_and_supplied_support_domain_fully_covered": True,
        "supplied_region_partial_progress_accepted": True,
        "source_target_and_support_unchanged": True,
        "hypothetical_access_assumption_accepted": True,
        "clinical_or_anatomical_accuracy_claim": False,
    }
    if release["research_task_admission"] != required:
        raise ValueError("Root must explicitly admit the bounded supplied-region research condition")
    if release["public_manifest"] != {"path": PUBLIC_MANIFEST, "sha256": PUBLIC_MANIFEST_SHA}:
        raise ValueError("Exact public-only manifest required")
    output = ROOT / release["output"]; output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); phase = "before_public_imports"; events = []
    result = {"status": "started", "scope": "one_TRAIN_supplied_region_scale_preflight",
        "release_sha256": args.release_sha256, "private_array_reads": 0,
        "profile_enabled": release["profile"], "events": events}

    def progress(name, **details):
        nonlocal phase
        phase = name
        row = {"phase": name, "elapsed_seconds": time.perf_counter()-start, **details}
        events.append(row)
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
        temporary.replace(output / "progress.json")
        with (output / "phase-events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()

    profiler = cProfile.Profile() if release["profile"] else None
    if profiler is not None: profiler.enable()
    original_scope = None
    def deadline(signum, frame):
        raise TimeoutError("Fixed120s whole-worker cooperative deadline; preserve failure/profile")
    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120.)
    try:
        progress("before_public_imports")
        import numpy as np
        from resectionlab.core import array_digest, semantic_digest, thaw_json
        from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
        from resectionlab.native_proposals import NominalCavityProposalConfig, SUPPLIED_GOAL_REGION
        from resectionlab.native_spatial_task import NativeSpatialCase
        from resectionlab.patient_planning_admission import VERSION, COHORT_SHA256, QC_SCOPE, make_patient_planning_task
        from resectionlab.patient_planning_learning import PREFLIGHT_PROTOCOL
        import resectionlab.patient_planning_preflight as preflight

        # Persist canonical phase boundaries without changing their instrumentation.
        original_scope = preflight._CallCosts.scope
        @contextmanager
        def scope(meter, name):
            progress("phase_start:"+name)
            with original_scope(meter, name): yield
            progress("phase_complete:"+name)
        preflight._CallCosts.scope = scope
        limits = release["limits"]
        if limits != {"max_steps": 6, "max_optimizer_updates": 2,
                "search": {"max_calls": 96, "beam_width": 2, "seconds": 60},
                "max_native_previews": 2048, "max_policy_forwards": 40,
                "worker_seconds": 120, "memory_bytes": 3221225472, "threads": 1}:
            raise ValueError("One fixed first-scale budget required; no sweep")
        protocol = None

        def prepare_public_source():
            nonlocal protocol
            progress("before_public_array_loading")
            manifest_path = ROOT / PUBLIC_MANIFEST
            if sha(manifest_path) != PUBLIC_MANIFEST_SHA: raise ValueError("Public manifest changed")
            manifest = json.loads(manifest_path.read_text())
            if (manifest["patient_id"] != "ReMIND-008" or manifest["role"] != "TRAIN"
                    or manifest["private_evaluation_files_included"] is not False
                    or set(manifest["input_files"]) != set(ARRAY_KEYS)):
                raise ValueError("Exact four public input arrays required")
            arrays = {}
            for key in ARRAY_KEYS:
                record = manifest["input_files"][key]; path = Path(record["path"])
                if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
                    raise ValueError("Public input changed: "+key)
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                if list(array.shape) != manifest["shape_xyz"] or str(array.dtype) != record["dtype"]:
                    raise ValueError("Public array layout changed: "+key)
                arrays[key] = array
                progress("public_array_loaded:"+key, bytes=record["bytes"])
            image, support, target, domain = (arrays[key] for key in ARRAY_KEYS)
            if (not all(np.isin(a, (0, 1)).all() for a in (support, target, domain))
                    or np.any((target != 0) & (domain == 0))):
                raise ValueError("Supplied label/domain changed; no filling or clipping")
            positive = np.argwhere(target != 0)
            unsupported = int(np.count_nonzero((target != 0) & (support == 0)))
            if len(positive) != 18509 or unsupported != 6394:
                raise ValueError("Qualified public target/support consistency differs")

            # Fixed source-axis0, hemisphere-side rule; no route/outcome ranking.
            # Both target and support are explicitly supplied public inputs.
            center = positive.mean(axis=0)
            occupied_x = np.flatnonzero(np.any(support != 0, axis=(1, 2)))
            positive_side = center[0] >= (occupied_x[0]+occupied_x[-1])/2
            sign = -1 if positive_side else 1
            transverse = np.rint(center[1:]).astype(int)
            column = np.flatnonzero(support[:, transverse[0], transverse[1]])
            if not len(column): raise ValueError("Fixed public access column contains no estimated support; no replacement")
            first = int(column[-1] if positive_side else column[0])
            access_index = np.array((first-sign*.5, *transverse), np.float64)
            affine = np.asarray(manifest["affine_ras_mm"], np.float64)
            axis = affine[:3, 0]/np.linalg.norm(affine[:3, 0])
            access = AccessWindow(affine[:3, :3]@access_index+affine[:3, 3], sign*axis,
                                  6., "fixed-public-target-side-axis0-v1")
            derived = {"rule": "fixed_source_axis0_target_centroid_side_of_support_extent",
                "target_centroid_source_voxels": center.tolist(), "access_index": access_index.tolist(),
                "access_side": "positive_source_axis0" if positive_side else "negative_source_axis0",
                "support_extent_axis0": [int(occupied_x[0]), int(occupied_x[-1])],
                "transverse_rounding": "numpy_rint_ties_to_even",
                "source_voxel_entry_offset": .5, "radius_mm": 6.,
                "route_search": False, "private_reference_used": False,
                "public_target_positive_voxels": len(positive), "unsupported_target_positive_voxels": unsupported,
                "task_condition": "PARTIAL_TARGET_PROGRESS",
                "annotation_domain_hash": array_digest(domain),
                "annotation_domain_meaning": "outside source annotation domain remains unknown anatomy",
                "goal_membership_meaning": "exact supplied positive region; outside this region is not normal anatomy truth",
                "estimated_support_zeros": "simulation occupancy assumption, not certified empty anatomy",
                "tools": "existing geometry.GENERIC_TOOLS; generic research geometry, not device validation"}
            write(output / "public-task-derivation.json", derived)
            progress("before_native_case_constructor", native_shape=list(image.shape), actor_crop=[64,64,64])
            source = NativeSpatialCase(image, support, target, affine, access, GENERIC_TOOLS,
                track="annotation_assisted", support_source_kind="supplied_annotation",
                support_derivation="unchanged supplied automatic Brainlab cerebrum estimate; zeros are not confirmed free anatomy",
                nominal_target=target, target_source_kind="supplied_annotation",
                target_derivation="unchanged supplied manual whole-tumor region; partial supported progress only; outside region not normal-anatomy truth",
                crop_shape=(64,64,64), support_provenance={"public_manifest_sha256": PUBLIC_MANIFEST_SHA,
                    "supplied_support_file_sha256": manifest["input_files"]["supplied_support"]["sha256"],
                    "supplied_target_file_sha256": manifest["input_files"]["supplied_whole_tumor"]["sha256"],
                    "target_domain_file_sha256": manifest["input_files"]["whole_tumor_domain"]["sha256"],
                    "target_domain_hash": array_digest(domain), "public_access_rule": derived,
                    "occupancy_unchanged": True, "target_unchanged": True},
                intensity_normalization="support_percentile_1_99",
                native_grid_reconciliation="orthogonal_roundoff_1e-6mm",
                proposal_mode="nominal_cavity_v1", proposal_config=NominalCavityProposalConfig(),
                target_semantics=SUPPLIED_GOAL_REGION)
            progress("native_case_constructed", source_hash=source.source_hash)
            files = manifest["input_files"]
            binding = {"version": VERSION, "evidence_domain": "acquired_patient",
                "subject": "ReMIND-008", "patient_group": manifest["patient_group"], "cohort_sha256": COHORT_SHA256,
                "t1_source_sha256": files["image"]["sha256"],
                "support_source_sha256": files["supplied_support"]["sha256"],
                "target_source_sha256": files["supplied_whole_tumor"]["sha256"],
                "image_array_hash": array_digest(source.structural_intensity),
                "support_array_hash": array_digest(source.observed_support),
                "target_array_hash": array_digest(source.nominal_target), "affine_array_hash": array_digest(source.affine_ras_mm),
                "source_hash": source.source_hash, "availability_basis": "retrospective_annotation_assisted_source_preop",
                "acquired_at": None, "annotation_available_at": None,
                "support_semantics": "source_automatic_Brainlab_cerebrum_annotation",
                "target_semantics": "source_manual_whole_tumor_annotation"}
            qc = {"version": VERSION, "evidence_domain": "acquired_patient", "subject": "ReMIND-008",
                "public_source_binding_hash": semantic_digest(binding), "status": "pass", "scope": QC_SCOPE,
                "source_linkage_checked": True, "frame_geometry_checked": True, "coverage_checked": True,
                "annotation_meaning_checked": True, "public_support_assumption": True,
                "native_domain_fully_covered": True, "hypothetical_access_assumption": True,
                "evidence_record_sha256": args.release_sha256}
            protocol = {"version": VERSION, "scope": "patient_native_planning_experiment",
                "subject": "ReMIND-008", "role": "TRAIN", "evidence_domain": "acquired_patient",
                "public_source_binding_hash": semantic_digest(binding), "qc_receipt_hash": semantic_digest(qc),
                **limits, "initialization": "fresh_seeded_shared_initialization", "private_reference_used": False,
                "clinical_claim": False, "split_changes": False, "runtime_release_sha256": args.release_sha256,
                "learning_protocol_hash": semantic_digest(PREFLIGHT_PROTOCOL)}
            write(output / "admitted-public-bindings.json", {"source_binding": binding, "qc": qc,
                "protocol": protocol, "supplied_goal_extent": thaw_json(source._supplied_goal_extent),
                "native_grid_reconciliation": thaw_json(source._grid_record),
                "intensity_normalization": thaw_json(source._normalization_record)})
            return source, binding, qc, protocol

        # Source preparation does not build a task/inventory. Count any native
        # preview dispatch nevertheless, before installing the canonical guard.
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
        preparation_started = time.perf_counter()
        with NativePreviewProfiler(NativeResectionEngine) as preparation_meter:
            source, binding, qc, protocol = prepare_public_source()
        prepared_previews = sum(row["started"] for row in preparation_meter.records.values())
        result["public_preparation_native_preview_calls"] = prepared_previews
        result["public_source_preparation_seconds"] = time.perf_counter()-preparation_started
        if prepared_previews != 0:
            raise RuntimeError("Unexpected preview before the monitored native task factory")

        def public_task_factory():
            progress("before_native_task_constructor_and_initial_inventory")
            task, context = make_patient_planning_task(source,
                cohort_bytes=(ROOT/"manifests/experiments/remind-component-cohort-v1.json").read_bytes(),
                source_binding=binding, qc_receipt=qc, protocol=protocol)
            write(output / "initial-inventory.json", task.candidate_inventory())
            progress("public_task_and_inventory_ready", legal_nonstop_actions=len(task.observation().action_ids)-1)
            return task, context

        child = preflight.run_patient_preflight(public_task_factory, protocol=protocol, output=output/"preflight")
        result.update(status="complete_preflight_not_population_evidence",
            canonical_result_sha256=sha(output/"preflight/result.json"),
            optimizer_updates=child["optimizer_updates"], policy_forwards=child["total_policy_forward_calls"],
            native_budget=child["native_budget"], costs=child["costs"],
            supplied_goal_extent=thaw_json(source._supplied_goal_extent),
            postseal_private_evaluation_performed=False)
        progress("complete_public_preflight")
    except BaseException as error:
        result.update(status="failed_or_unresolved", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        if original_scope is not None: preflight._CallCosts.scope = original_scope
        if profiler is not None:
            profiler.disable(); profiler.dump_stats(str(output / "profile.pstats"))
        result.update(last_phase=phase, complete_profiled_wall_seconds=time.perf_counter()-start,
            profile_scope="includes imports, public loader, native inventory, training and replay; not an unprofiled timing benchmark")
        write(output / "result.json", result)


if __name__ == "__main__": main()
