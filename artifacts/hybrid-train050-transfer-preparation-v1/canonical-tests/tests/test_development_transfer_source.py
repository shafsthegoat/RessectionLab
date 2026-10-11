"""Generated source-admission controls; no acquired inputs or model parameters.

The named people exercise metadata joins only. Small invented arrays and opaque
invented model/QC digests do not establish patient eligibility or actual lineage.
The experiment's direct-loader/worker controls live outside this canonical suite.
"""
import copy
import json

import numpy as np
import pytest

from resectionlab import patient_planning_admission as admission
from resectionlab import public_patient_factory as factory
from resectionlab.core import array_digest, semantic_digest
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.post_exposure import VERSION as EXPOSURE
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
from test_paired_train_occupancy import fixture as original_fixture


def fixture(tmp_path, subject="ReMIND-050", role="TRAIN"):
    inputs, _ = original_fixture(tmp_path, subject=subject, role=role)
    path = inputs["public_manifest_path"]
    manifest = json.loads(path.read_bytes())
    image = np.arange(343, dtype=np.float32).reshape(7, 7, 7)
    support = np.zeros(image.shape, np.uint8); support[1:4, 2:5, 2:5] = 1
    target = np.zeros_like(support); target[3:5, 3, 3] = 1
    target_domain = np.ones_like(support); target_domain[0, 0, 0] = 0
    support_domain = target_domain.copy(); support_domain[4, 3, 3] = 0
    arrays = (image, support, target, target_domain, support_domain)
    for key, array in zip(factory.PARTIAL_DOMAIN_ARRAY_KEYS, arrays):
        file = tmp_path / (key + ".npy")
        np.save(file, array, allow_pickle=False)
        manifest["input_files"][key] = {"path": str(file), "sha256": factory.sha(file),
            "bytes": file.stat().st_size, "dtype": str(array.dtype)}
    manifest.update(schema="remind-fixed-TRAIN-partial-domain-public-inputs-v1",
        shape_xyz=list(image.shape), public_support_domain_fully_covered=False,
        source_domain_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        training_admitted=False, public_source_domain_counts={"support_domain": int(support_domain.sum()),
            "target_in_known_support_positive": 1, "target_in_known_support_zero": 0,
            "target_in_unknown_support_domain": 1})
    manifest["source_bindings"]["intended_use_review_sha256"] = "e" * 64
    path.write_text(json.dumps(manifest))
    inputs["public_manifest_sha256"] = factory.sha(path)
    record = {"version": admission.DEVELOPMENT_TRANSFER_VERSION, "subject": subject,
        "role": role, "scope": "development_transfer_not_heldout_SELECT",
        "methods": ["balanced_ridge_top8", "exhaustive"], "K": 8, "max_steps": 24,
        "refits": 0, "optimizer_updates": 0, "neural_forwards": 0,
        "training_subjects": ["ReMIND-002", "ReMIND-015", "ReMIND-018", "ReMIND-045"],
        "public_manifest_sha256": inputs["public_manifest_sha256"],
        "intended_use_review_sha256": "e" * 64, "frozen_ridge_sha256": "3" * 64,
        "frozen_ridge_fingerprint": "sha256:" + "4" * 64, "training_result_sha256": "5" * 64}
    return inputs, record, arrays


def build(output, inputs, record, *, explicit=True, **changed_limits):
    output.mkdir()
    limits = {"max_steps": 24, "max_optimizer_updates": 0, "max_native_previews": 5000,
        "max_policy_forwards": 0, "worker_seconds": 20, "memory_bytes": 512 * 1024**2,
        "threads": 1, "search": {"max_calls": 1, "beam_width": 1, "seconds": 1}}
    limits.update(changed_limits)
    options = {"development_transfer": record} if explicit else {}
    return factory.prepare_public_source(output, {"limits": limits}, "b" * 64, None,
        lambda *a, **k: None, **inputs, learning_protocol_hash=semantic_digest(record),
        proposal_config=NominalCavityProposalConfig(max_candidates=120,
            intermediate_opening_mm=1., tool_footprint_opening=True),
        public_target_context_variant=TARGET_CONTEXT,
        occupancy_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        post_exposure_condition=EXPOSURE, **options)


def arguments(built, inputs):
    case, source, qc, protocol = built
    return case, {"cohort_bytes": inputs["cohort_bytes"], "source_binding": source,
        "qc_receipt": qc, "protocol": protocol}


def rebind(args):
    source_hash = semantic_digest(args["source_binding"])
    args["qc_receipt"]["public_source_binding_hash"] = source_hash
    args["protocol"]["public_source_binding_hash"] = source_hash
    args["protocol"]["qc_receipt_hash"] = semantic_digest(args["qc_receipt"])


def forbidden(*a, **k):
    pytest.fail("Source-only refusal/validation must not open arrays or construct a native inventory")


def test_default_050_stays_closed_before_array_loading(tmp_path, monkeypatch):
    inputs, record, _ = fixture(tmp_path)
    monkeypatch.setattr(factory.np, "load", forbidden)
    with pytest.raises(ValueError):
        factory.load_public_manifest(inputs["public_manifest_path"], inputs["public_manifest_sha256"],
            inputs["cohort_bytes"], partial_domain=True)
    with pytest.raises(ValueError):
        build(tmp_path / "default", inputs, record, explicit=False)


def test_explicit_bound_TRAIN_source_preserves_unknown_full_target_without_inventory(tmp_path, monkeypatch):
    inputs, record, arrays = fixture(tmp_path)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    monkeypatch.setattr(admission, "NativeSpatialTask", forbidden)
    case, args = arguments(build(tmp_path / "source", inputs, record), inputs)
    validated = admission.validate_patient_planning_source(case, **args)
    assert validated[4]["subject"] == "ReMIND-050" and validated[4]["role"] == "TRAIN"
    assert args["protocol"]["max_optimizer_updates"] == args["protocol"]["max_policy_forwards"] == 0
    assert args["protocol"]["development_transfer"] == record
    np.testing.assert_array_equal(case.occupancy_source_support, arrays[1])
    np.testing.assert_array_equal(case.nominal_target, arrays[2])
    np.testing.assert_array_equal(case.support_domain, arrays[4])
    assert case._supplied_goal_extent["full_region_positive_voxels"] == 2
    assert not case.support_domain[4, 3, 3] and case._interaction_domain[4, 3, 3]
    assert not case._interaction_domain[0, 0, 0]  # unknown outside T remains unknown
    assert args["qc_receipt"]["partial_source_domain_preserved"] is True
    assert args["qc_receipt"]["native_domain_fully_covered"] is False
    assert "checkpoint_lineage" not in args["protocol"] and "occupancy_learning_protocol" not in args["protocol"]


@pytest.mark.parametrize("subject,role", [("ReMIND-002", "TRAIN"), ("ReMIND-013", "SELECT"),
    ("ReMIND-013", "TRAIN"), ("ReMIND-067", "TRAIN")])
def test_fitted_or_wrong_role_person_refused_before_array_loading(tmp_path, monkeypatch, subject, role):
    inputs, record, _ = fixture(tmp_path, subject, role)
    monkeypatch.setattr(factory.np, "load", forbidden)
    with pytest.raises(ValueError):
        build(tmp_path / "refused", inputs, record)


@pytest.mark.parametrize("field", ["public_manifest_sha256", "intended_use_review_sha256"])
def test_source_and_intended_use_hash_mismatch_refused_before_array_loading(tmp_path, monkeypatch, field):
    inputs, record, _ = fixture(tmp_path)
    record[field] = "a" * 64
    monkeypatch.setattr(factory.np, "load", forbidden)
    with pytest.raises(ValueError):
        build(tmp_path / "refused", inputs, record)


@pytest.mark.parametrize("field", ["public_manifest_sha256", "intended_use_review_sha256",
    "frozen_ridge_sha256", "frozen_ridge_fingerprint", "training_result_sha256"])
def test_pending_binding_refused_as_metadata_only(tmp_path, field):
    _, record, _ = fixture(tmp_path)
    record[field] = None
    with pytest.raises(ValueError):
        admission.validate_development_transfer(record)


@pytest.mark.parametrize("field", ["max_optimizer_updates", "max_policy_forwards"])
@pytest.mark.parametrize("value", [1, False])
def test_nonzero_or_boolean_learning_budget_refused_before_arrays(tmp_path, monkeypatch, field, value):
    inputs, record, _ = fixture(tmp_path)
    monkeypatch.setattr(factory.np, "load", forbidden)
    with pytest.raises(ValueError):
        build(tmp_path / "refused", inputs, record, **{field: value})


@pytest.mark.parametrize("mutation", ["Ds_hash", "domain_partition", "unknown_declared_known", "transfer_protocol_join", "private_reference"])
def test_rebound_domain_or_information_contract_is_refused(tmp_path, monkeypatch, mutation):
    inputs, record, _ = fixture(tmp_path)
    case, args = arguments(build(tmp_path / "source", inputs, record), inputs)
    args = copy.deepcopy(args)
    if mutation == "Ds_hash":
        args["source_binding"]["support_domain_binary_hash"] = array_digest(np.ones(case.support_domain.shape, bool))
    elif mutation == "domain_partition":
        args["source_binding"]["source_and_simulated_domains"]["target_in_unknown_support_voxels"] = 0
    elif mutation == "unknown_declared_known":
        args["qc_receipt"]["native_domain_fully_covered"] = True
    elif mutation == "transfer_protocol_join":
        args["protocol"]["development_transfer"]["frozen_ridge_sha256"] = "a" * 64
    else:
        args["protocol"]["private_reference_used"] = True
    rebind(args)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    with pytest.raises(ValueError):
        admission.validate_patient_planning_source(case, **args)


def test_old_generated_default_record_still_has_no_transfer_option():
    from test_patient_planning_admission import fixture_contract
    case, args = fixture_contract()
    source, qc, protocol, *_ = admission.validate_patient_planning_source(case, **args)
    assert source == args["source_binding"] and qc == args["qc_receipt"] and protocol == args["protocol"]
    assert "development_transfer" not in protocol


@pytest.mark.parametrize("field", ["refits", "optimizer_updates", "neural_forwards"])
def test_transfer_record_never_authorizes_learning(tmp_path, monkeypatch, field):
    inputs, record, _ = fixture(tmp_path)
    record[field] = 1
    monkeypatch.setattr(factory.np, "load", forbidden)
    with pytest.raises(ValueError):
        build(tmp_path / "refused", inputs, record)


def test_support_positive_cannot_be_relabelled_unknown_even_with_rehashed_manifest(tmp_path, monkeypatch):
    inputs, record, arrays = fixture(tmp_path)
    manifest = json.loads(inputs["public_manifest_path"].read_bytes())
    domain = arrays[4].copy(); domain[2, 3, 3] = 0
    file = tmp_path / "supplied_support_domain.npy"
    np.save(file, domain, allow_pickle=False)
    manifest["input_files"]["supplied_support_domain"]["sha256"] = factory.sha(file)
    manifest["public_source_domain_counts"]["support_domain"] = int(domain.sum())
    inputs["public_manifest_path"].write_text(json.dumps(manifest))
    inputs["public_manifest_sha256"] = factory.sha(inputs["public_manifest_path"])
    record["public_manifest_sha256"] = inputs["public_manifest_sha256"]
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    with pytest.raises(ValueError, match="support positives"):
        build(tmp_path / "refused", inputs, record)
