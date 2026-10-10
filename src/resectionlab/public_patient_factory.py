"""Public-only factory for frozen TRAIN and explicitly requested SELECT cases.

Reuses the executed first-case access/tools/context condition. The owning runner
binds runtime limits and authenticates saved QC; this does not run a simulation.
No original DICOM, private annotation or evaluator loader is present.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.patient_planning_admission import (VERSION, COHORT_SHA256, QC_SCOPE,
    PARTIAL_DOMAIN_TRAIN_SUBJECTS, PARTIAL_DOMAIN_UNION_OCCUPANCY, PARTIAL_DOMAIN_QC_SCOPE,
    PARTIAL_DOMAIN_SELECT_QC_SCOPE)
TRAIN_SUBJECTS = ("ReMIND-008", "ReMIND-010", "ReMIND-020", "ReMIND-025")
SELECT_SUBJECTS = ("ReMIND-013", "ReMIND-037")
ARRAY_KEYS = ("image", "supplied_support", "supplied_whole_tumor", "whole_tumor_domain")
PARTIAL_DOMAIN_ARRAY_KEYS = (*ARRAY_KEYS, "supplied_support_domain")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")

def load_public_manifest(path, expected_sha256, cohort_bytes, *, expected_role="TRAIN", partial_domain=False):
    if type(partial_domain) is not bool or (partial_domain and expected_role not in ("TRAIN", "SELECT")):
        raise ValueError("Partial source domain requires an explicit frozen TRAIN or SELECT role")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Public manifest changed")
    if type(cohort_bytes) is not bytes or hashlib.sha256(cohort_bytes).hexdigest() != COHORT_SHA256:
        raise ValueError("Original frozen cohort bytes required")
    manifest = json.loads(raw)
    subject = manifest["patient_id"]
    members = [m for m in json.loads(cohort_bytes)["members"] if m["subject"] == subject]
    subjects = ((PARTIAL_DOMAIN_TRAIN_SUBJECTS if expected_role == "TRAIN" else ("ReMIND-013",)) if partial_domain else
        {"TRAIN": TRAIN_SUBJECTS, "SELECT": SELECT_SUBJECTS}.get(expected_role, ()))
    keys = PARTIAL_DOMAIN_ARRAY_KEYS if partial_domain else ARRAY_KEYS
    if (subject not in subjects or len(members) != 1 or members[0]["role"] != expected_role
            or manifest["role"] != expected_role or manifest["patient_group"] != members[0]["patient_group"]
            or manifest["private_evaluation_files_included"] is not False
            or set(manifest["input_files"]) != set(keys)
            or manifest.get("task_condition", "PARTIAL_TARGET_PROGRESS") != "PARTIAL_TARGET_PROGRESS"):
        raise ValueError("Exact five public arrays and fixed TRAIN partial-domain condition required" if partial_domain else
            "Exact four public arrays and frozen requested role partial-target condition required")
    if partial_domain and expected_role == "SELECT" and manifest.get("schema") != "remind-fixed-SELECT-partial-domain-public-inputs-v1":
        raise ValueError("Explicit five-array partial-domain SELECT013 manifest required")
    if expected_role == "SELECT" and manifest.get("public_only") is not True:
        raise ValueError("SELECT requires public-only qualification")
    if partial_domain and (manifest.get("source_domain_condition") != PARTIAL_DOMAIN_UNION_OCCUPANCY
            or manifest.get("public_only") is not True or manifest.get("training_admitted") is not False):
        raise ValueError("Explicit partial-domain public-only nontraining condition required")
    if subject != "ReMIND-008" and (
            manifest.get("public_support_domain_fully_covered") is not (not partial_domain)
            or manifest["source_bindings"]["cohort_sha256"] != COHORT_SHA256
            or manifest["reindex_policy"]["source_MR_samples_preserved"] is not True
            or manifest["reindex_policy"]["original_MR_affine_overwritten"] is not False):
        raise ValueError("Public coverage and explicit original MRI provenance required")
    return manifest


def check_public_labels(support, target, domain, manifest):
    if (not all(np.isin(a, (0, 1)).all() for a in (support, target, domain))
            or np.any((target != 0) & (domain == 0))):
        raise ValueError("Supplied label/domain changed; no filling or clipping")
    if not np.any(support):
        raise ValueError("Supplied support is empty; fixed public access is unavailable")
    positive = np.argwhere(target != 0)
    unsupported = int(np.count_nonzero((target != 0) & (support == 0)))
    expected = manifest["public_target_support_consistency"]
    if (not len(positive) or len(positive) != expected["whole_tumor_positive_voxels"]
            or unsupported != expected["whole_tumor_positive_outside_supplied_support"]):
        raise ValueError("Qualified public target/support consistency differs")
    return positive, unsupported


def prepare_public_source(output, original_release, original_release_sha256, expected_protocol, progress,
        *, public_manifest_path, public_manifest_sha256, cohort_bytes,
        learning_protocol_hash=None, proposal_config=None, public_target_context_variant=None,
        expected_role="TRAIN", checkpoint_lineage=None, occupancy_condition="raw_cerebrum_baseline",
        occupancy_learning_protocol=None, occupancy_inference_protocol=None, post_exposure_condition=None,
        partial_domain_inputs=False):
    """Build one public source and admission inputs; caller owns supervised use.

    expected_protocol may be None for a first construction; then a learning
    protocol hash is required explicitly. A supplied baseline source hash and
    expected protocol still demand exact reconstruction, preserving the 008 path.
    The default access, tools, proposals and 64-cube context are unchanged.
    """
    from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
    from resectionlab.native_proposals import NominalCavityProposalConfig, SUPPLIED_GOAL_REGION
    from resectionlab.native_spatial_task import (NativeSpatialCase, RAW_CEREBRUM_OCCUPANCY,
        SUPPLIED_TUMOR_UNION_OCCUPANCY, DERIVED_OCCUPANCY_SOURCE_KIND)
    release = original_release
    limits = release["limits"]
    partial_domain = occupancy_condition == PARTIAL_DOMAIN_UNION_OCCUPANCY
    if occupancy_condition not in (RAW_CEREBRUM_OCCUPANCY, SUPPLIED_TUMOR_UNION_OCCUPANCY, PARTIAL_DOMAIN_UNION_OCCUPANCY):
        raise ValueError("Unknown explicit occupancy condition")
    derived_occupancy = occupancy_condition in (SUPPLIED_TUMOR_UNION_OCCUPANCY, PARTIAL_DOMAIN_UNION_OCCUPANCY)
    occupancy_learning = occupancy_learning_protocol is not None
    occupancy_inference = occupancy_inference_protocol is not None
    post_inference = occupancy_inference and post_exposure_condition is not None
    if type(partial_domain_inputs) is not bool or (partial_domain_inputs and not (
            partial_domain and post_inference and expected_role == "SELECT")):
        raise ValueError("Supplied partial-domain input is an explicit post-exposure SELECT013 option")
    partial_inputs = partial_domain and (not post_inference or partial_domain_inputs)
    derived_full_domain = post_inference and not partial_inputs
    if occupancy_inference and (occupancy_condition != (PARTIAL_DOMAIN_UNION_OCCUPANCY if post_inference else SUPPLIED_TUMOR_UNION_OCCUPANCY)
            or expected_role != "SELECT" or checkpoint_lineage is None or occupancy_learning
            or type(limits.get("max_optimizer_updates")) is not int or limits["max_optimizer_updates"] != 0
            or type(limits.get("max_policy_forwards")) is not int or limits["max_policy_forwards"] <= 0):
        raise ValueError("Union inference requires fixed SELECT013 frozen weights and zero updates")
    if post_exposure_condition is not None:
        from resectionlab.post_exposure import VERSION as POST_EXPOSURE_VERSION
        if (post_exposure_condition != POST_EXPOSURE_VERSION or not partial_domain
                or not (post_inference and expected_role == "SELECT" and checkpoint_lineage is not None
                        or not occupancy_inference and expected_role == "TRAIN" and checkpoint_lineage is None)):
            raise ValueError("Post-exposure is a separate fixed-four partial-domain TRAIN search condition")
    if occupancy_learning and (occupancy_condition not in (SUPPLIED_TUMOR_UNION_OCCUPANCY, PARTIAL_DOMAIN_UNION_OCCUPANCY)
            or partial_domain and post_exposure_condition is None
            or expected_role != "TRAIN" or checkpoint_lineage is not None):
        raise ValueError("Union learning is restricted to the original full-coverage TRAIN condition")
    if derived_occupancy and not occupancy_learning and not occupancy_inference and (expected_role != "TRAIN" or checkpoint_lineage is not None
            or type(limits.get("max_optimizer_updates")) is not int or limits["max_optimizer_updates"] != 0
            or type(limits.get("max_policy_forwards")) is not int or limits["max_policy_forwards"] != 0
            or public_target_context_variant is None):
        raise ValueError("Derived occupancy is TRAIN search-only with zero model and optimizer budgets and the public target/domain context")
    if expected_protocol is not None:
        if learning_protocol_hash is not None and learning_protocol_hash != expected_protocol["learning_protocol_hash"]:
            raise ValueError("Conflicting learning protocol hashes")
        learning_protocol_hash = expected_protocol["learning_protocol_hash"]
    token = learning_protocol_hash.removeprefix("sha256:") if isinstance(learning_protocol_hash, str) else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Explicit learning protocol hash required")
    if occupancy_learning:
        from resectionlab.patient_planning_cohort_spec import (validate_union_obstruction_learning,
            validate_post_exposure_learning)
        validator = validate_post_exposure_learning if partial_domain else validate_union_obstruction_learning
        occupancy_learning_protocol = validator(occupancy_learning_protocol,
            learning_protocol_hash=learning_protocol_hash, proposal_config=proposal_config,
            max_steps=limits.get("max_steps"),
            **({'post_exposure_condition':post_exposure_condition} if partial_domain else {}))
        if (public_target_context_variant != occupancy_learning_protocol["public_target_context_variant"]
                or type(limits.get("max_optimizer_updates")) is not int
                or limits["max_optimizer_updates"] < 2*occupancy_learning_protocol["updates_per_method"]
                or type(limits.get("max_policy_forwards")) is not int or limits["max_policy_forwards"] <= 0):
            raise ValueError("Union learning requires the existing public context and positive declared learning budgets")
    if occupancy_inference:
        from resectionlab.patient_planning_admission import (validate_union_select013_inference,
            validate_post_exposure_select013_inference)
        validator = validate_post_exposure_select013_inference if post_inference else validate_union_select013_inference
        occupancy_inference_protocol = validator(occupancy_inference_protocol,
            checkpoint_lineage=checkpoint_lineage, learning_protocol_hash=learning_protocol_hash,
            proposal_config=proposal_config, max_steps=limits.get("max_steps"), search=limits.get("search"),
            public_target_context_variant=public_target_context_variant,
            **({"post_exposure_condition": post_exposure_condition} if post_inference else {}))
    if expected_role == "TRAIN":
        if checkpoint_lineage is not None:
            raise ValueError("TRAIN factory does not accept SELECT checkpoint lineage")
    elif expected_role == "SELECT":
        from resectionlab.patient_planning_admission import validate_select_checkpoint_lineage
        if (set(limits) != {"max_steps", "max_optimizer_updates", "max_native_previews", "max_policy_forwards",
                           "worker_seconds", "memory_bytes", "threads", "search"}
                or type(limits["max_optimizer_updates"]) is not int or limits["max_optimizer_updates"] != 0):
            raise ValueError("SELECT requires an explicit zero-update runtime budget")
        checkpoint_lineage = thaw_json(validate_select_checkpoint_lineage(
            checkpoint_lineage, learning_protocol_hash=learning_protocol_hash,
            **({"learning_protocol": occupancy_inference_protocol} if post_inference else {})))
    else:
        raise ValueError("Only frozen TRAIN or SELECT public construction is supported")
    progress("before_public_array_loading")
    manifest = load_public_manifest(public_manifest_path, public_manifest_sha256, cohort_bytes,
        expected_role=expected_role, partial_domain=partial_inputs)
    subject = manifest["patient_id"]
    if occupancy_inference:
        from resectionlab.patient_planning_admission import UNION_SELECT_SUBJECT
        if subject != UNION_SELECT_SUBJECT:
            raise ValueError("This explicit frozen union transfer admits SELECT013 only; 037 held and EVAL closed")
    full_domain_derivation = None
    if derived_full_domain:
        from resectionlab.patient_planning_admission import qualified_full_support_domain
        full_domain_derivation = qualified_full_support_domain(manifest, public_manifest_sha256)
    arrays = {}
    input_keys = PARTIAL_DOMAIN_ARRAY_KEYS if partial_inputs else ARRAY_KEYS
    for key in input_keys:
        record = manifest["input_files"][key]; path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Public input changed: "+key)
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if list(array.shape) != manifest["shape_xyz"] or str(array.dtype) != record["dtype"]:
            raise ValueError("Public array layout changed: "+key)
        arrays[key] = array
        progress("public_array_loaded:"+key, bytes=record["bytes"])
    image, support, target, domain = (arrays[key] for key in ARRAY_KEYS)
    positive, unsupported = check_public_labels(support, target, domain, manifest)
    support_domain = np.ones(image.shape, dtype=bool) if derived_full_domain else None
    if partial_inputs:
        values = arrays["supplied_support_domain"]
        if not np.isin(values, (0, 1)).all(): raise ValueError("Source support domain must be binary")
        support_domain = np.asarray(values, bool)
        if not support_domain.any() or np.any((support != 0) & ~support_domain):
            raise ValueError("Source support positives must stay within unchanged source coverage")
        counts = {"support_domain": int(support_domain.sum()),
            "target_in_known_support_positive": int(np.count_nonzero((target != 0) & support_domain & (support != 0))),
            "target_in_known_support_zero": int(np.count_nonzero((target != 0) & support_domain & (support == 0))),
            "target_in_unknown_support_domain": int(np.count_nonzero((target != 0) & ~support_domain))}
        if counts != manifest["public_source_domain_counts"]:
            raise ValueError("Source-domain/full-target partition differs from saved qualification")

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
    # Strict defaults retain the historical raw-S access exactly.
    occupancy = np.logical_or(support, target) if derived_occupancy else support
    post_start = None
    if post_exposure_condition is not None:
        from resectionlab.native_spatial_task import reconcile_native_grid_roundoff
        from resectionlab.post_exposure import prepare_post_exposure
        native_affine, _ = reconcile_native_grid_roundoff(affine, image.shape)
        access, post_start = prepare_post_exposure(support=support, target=target,
            support_domain=support_domain, affine=native_affine, previous_access=access)
        derived = {**derived, "strict_baseline_access_unchanged_in_original_condition": True,
            "post_exposure": thaw_json(post_start.record)}
    write(output / "public-task-derivation.json", derived)
    progress("before_native_case_constructor", native_shape=list(image.shape), actor_crop=[64,64,64])
    source = NativeSpatialCase(image, occupancy, target, affine, access, GENERIC_TOOLS,
        track="annotation_assisted", support_source_kind=(DERIVED_OCCUPANCY_SOURCE_KIND if derived_occupancy else "supplied_annotation"),
        support_derivation=("explicit simulated S union T rigid-cell occupancy assumption; original automatic cerebrum and manual tumor retained; not validated material or corrected anatomy"
            if derived_occupancy else "unchanged supplied automatic Brainlab cerebrum estimate; zeros are not confirmed free anatomy"),
        nominal_target=target, target_source_kind="supplied_annotation",
        target_derivation=("unchanged supplied manual whole-tumor region; partial supported progress only; outside region not normal-anatomy truth"
            if subject == "ReMIND-008" else "unchanged supplied NN-derived manual whole-tumor region on declared planning grid; partial supported progress only; source losses retained in public manifest"),
        crop_shape=(64,64,64), support_provenance={"public_manifest_sha256": public_manifest_sha256,
            "supplied_support_file_sha256": manifest["input_files"]["supplied_support"]["sha256"],
            "supplied_target_file_sha256": manifest["input_files"]["supplied_whole_tumor"]["sha256"],
            "target_domain_file_sha256": manifest["input_files"]["whole_tumor_domain"]["sha256"],
            "target_domain_hash": array_digest(domain), "public_access_rule": derived,
            "occupancy_unchanged": not derived_occupancy or unsupported == 0, "target_unchanged": True,
            **({"occupancy_condition": occupancy_condition, "raw_support_array_hash": array_digest(np.asarray(support, bool)),
                "target_domain_binary_hash": array_digest(np.asarray(domain, bool)),
                "source_QC_scope": "unchanged supplied annotations and frame; derived occupancy is not anatomically validated"}
               if derived_occupancy else {}),
            **({"support_domain_file_sha256": manifest["input_files"]["supplied_support_domain"]["sha256"],
                "support_domain_binary_hash": array_digest(support_domain),
                "partial_source_domain_preserved": True} if partial_inputs else {}),
            **({"support_domain_derivation": full_domain_derivation,
                "support_domain_binary_hash": array_digest(support_domain)} if derived_full_domain else {}),
            **({} if subject == "ReMIND-008" else {"source_MR_crop_affine_ras_mm": manifest["source_MR_crop_affine_ras_mm"],
                "explicit_planning_grid": manifest["reindex_policy"], "public_source_bindings": manifest["source_bindings"],
                "public_label_resampling": manifest["public_label_resampling"]})},
        intensity_normalization="support_percentile_1_99",
        native_grid_reconciliation="orthogonal_roundoff_1e-6mm",
        proposal_mode="nominal_cavity_v1", proposal_config=NominalCavityProposalConfig() if proposal_config is None else proposal_config,
        target_semantics=SUPPLIED_GOAL_REGION,
        **({"occupancy_source_support": support} if derived_occupancy else {}),
        **({"support_domain": support_domain} if partial_domain else {}),
        **({"post_exposure": post_start} if post_start is not None else {}),
        **({} if public_target_context_variant is None else {
            'public_target_context_variant':public_target_context_variant,
            'public_target_domain':np.asarray(domain,dtype=bool)}))
    if "baseline_public_source_hash" in release and source.source_hash != release["baseline_public_source_hash"]:
        raise ValueError("Exact supplied baseline public world changed")
    progress("native_case_constructed", source_hash=source.source_hash)
    files = manifest["input_files"]
    binding = {"version": VERSION, "evidence_domain": "acquired_patient",
        "subject": subject, "patient_group": manifest["patient_group"], "cohort_sha256": COHORT_SHA256,
        "t1_source_sha256": files["image"]["sha256"],
        "support_source_sha256": files["supplied_support"]["sha256"],
        "target_source_sha256": files["supplied_whole_tumor"]["sha256"],
        "image_array_hash": array_digest(source.structural_intensity),
        "support_array_hash": array_digest(source.observed_support),
        "target_array_hash": array_digest(source.nominal_target), "affine_array_hash": array_digest(source.affine_ras_mm),
        "source_hash": source.source_hash, "availability_basis": "retrospective_annotation_assisted_source_preop",
        "acquired_at": None, "annotation_available_at": None,
        "support_semantics": ("derived_simulated_S_union_T_occupancy_assumption" if derived_occupancy
            else "source_automatic_Brainlab_cerebrum_annotation"),
        "target_semantics": "source_manual_whole_tumor_annotation",
        **({"occupancy_derivation": thaw_json(source._occupancy_derivation),
            "target_domain_source_sha256": files["whole_tumor_domain"]["sha256"],
            "target_domain_binary_hash": array_digest(np.asarray(domain, bool))} if derived_occupancy else {}),
        **({"support_domain_source_sha256": files["supplied_support_domain"]["sha256"],
            "support_domain_binary_hash": array_digest(support_domain),
            "source_and_simulated_domains": thaw_json(source._domain_record)} if partial_inputs else {}),
        **({"support_domain_derivation": full_domain_derivation,
            "support_domain_binary_hash": array_digest(support_domain),
            "source_and_simulated_domains": thaw_json(source._domain_record)} if derived_full_domain else {}),
        **({"post_exposure": thaw_json(post_start.record)} if post_start is not None else {})}
    qc = {"version": VERSION, "evidence_domain": "acquired_patient", "subject": subject,
        "public_source_binding_hash": semantic_digest(binding), "status": "pass",
        "scope": (PARTIAL_DOMAIN_SELECT_QC_SCOPE if post_inference else PARTIAL_DOMAIN_QC_SCOPE) if partial_inputs else QC_SCOPE,
        "source_linkage_checked": True, "frame_geometry_checked": True, "coverage_checked": True,
        "annotation_meaning_checked": True, "public_support_assumption": True,
        "native_domain_fully_covered": not partial_inputs, "hypothetical_access_assumption": True,
        "evidence_record_sha256": manifest.get("source_bindings", {}).get("saved_array_review_sha256", original_release_sha256),
        **({"derived_occupancy_anatomically_validated": False} if derived_occupancy else {}),
        **({"partial_source_domain_preserved": True} if partial_inputs else {}),
        **({"derived_full_source_domain_preserved": True} if derived_full_domain else {})}
    protocol = {"version": VERSION, "scope": "patient_native_planning_experiment",
        "subject": subject, "role": expected_role, "evidence_domain": "acquired_patient",
        "public_source_binding_hash": semantic_digest(binding), "qc_receipt_hash": semantic_digest(qc),
        **limits, "initialization": ("public_world_search_only" if derived_occupancy and not occupancy_learning and not occupancy_inference else
            "fresh_seeded_shared_initialization" if expected_role == "TRAIN" else "frozen_TRAIN_checkpoint_reload"), "private_reference_used": False,
        "clinical_claim": False, "split_changes": False, "runtime_release_sha256": original_release_sha256,
        "learning_protocol_hash": learning_protocol_hash,
        **({} if expected_role == "TRAIN" else {"checkpoint_lineage": checkpoint_lineage}),
        **({"occupancy_condition": occupancy_condition} if derived_occupancy else {}),
        **({"occupancy_learning_protocol": thaw_json(occupancy_learning_protocol)} if occupancy_learning else {}),
        **({"occupancy_inference_protocol": thaw_json(occupancy_inference_protocol)} if occupancy_inference else {}),
        **({"post_exposure_condition_hash": post_start.fingerprint} if post_start is not None else {})}
    if derived_occupancy:
        write(output / "derived-occupancy-assumption.json", {"condition": occupancy_condition,
            "derivation": thaw_json(source._occupancy_derivation),
            "raw_public_files": {key: {"sha256": files[key]["sha256"], "bytes": files[key]["bytes"]}
                for key in input_keys},
            **({"source_and_simulated_domains": thaw_json(source._domain_record)} if partial_domain else {}),
            "target_domain_hash": array_digest(domain), "access_derived_from": ("raw S and unchanged T" if post_start is None else
                "same original side/centroid; full O-cell face across fixed disc; separately declared post-exposure condition"),
            **({"post_exposure": thaw_json(post_start.record)} if post_start is not None else {}),
            "public_access_rule": derived, "source_QC_validates_derived_material": False,
            "policy_comparison_permitted": occupancy_learning or occupancy_inference,
            **({"comparison_scope": ("same_declared_post_exposure_world_SELECT013_frozen_inference_only" if post_inference
                else "same_declared_union_world_SELECT013_frozen_inference_only")} if occupancy_inference else {}),
            **({"comparison_scope": ("same_declared_post_exposure_world_new_four_TRAIN_only" if partial_domain
                else "same_declared_union_world_fixed_four_TRAIN_only")} if occupancy_learning else {}),
            "normalization_coupling": "support_percentile_1_99 uses this condition occupancy; compare recorded bounds across arms",
            "intensity_normalization": thaw_json(source._normalization_record)})
    write(output / "admitted-public-bindings.json", {"source_binding": binding, "qc": qc,
        "protocol": protocol, "supplied_goal_extent": thaw_json(source._supplied_goal_extent),
        "native_grid_reconciliation": thaw_json(source._grid_record),
        "intensity_normalization": thaw_json(source._normalization_record)})
    if expected_protocol is not None and protocol != expected_protocol: raise ValueError("Reconstructed original protocol differs")
    return source, binding, qc, protocol
