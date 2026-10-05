"""Analytic syntax/isolation controls; no patient observations or training."""
from dataclasses import FrozenInstanceError, replace
import hashlib

import numpy as np
import pytest

from resectionlab import mechanics_landmarks as lm


POINTS = np.array([
    [0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2],
    [2, 2, 0], [2, 0, 2], [0, 2, 2], [2, 2, 2],
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
], dtype=float)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tag(points=POINTS, destination=None, *, suffix=""):
    destination = points + [0.25, -0.5, 1] if destination is None else destination
    rows = [" ".join(map(str, (*a, *b))) + suffix for a, b in zip(points, destination, strict=True)]
    return ("MNI Tag Point File\nVolumes = 2;\n% source fixture only\nPoints =\n"
            + "\n".join(rows) + "\n;\n").encode("ascii")


def bind(payload, *, role=lm.DISPLACEMENT_ROLE, verified=True, transform=None):
    matrix = np.eye(4) if transform is None else transform
    qc = lm.LandmarkFrameQC("1" * 64, "2" * 64, matrix, matrix,
        "3" * 64, "analytic explicit frame declaration") if verified else None
    return lm.LandmarkPairBinding("analytic_not_patient", role, digest(payload), "1" * 64, "2" * 64, qc)


def prepare(payload=None, **kwargs):
    payload = tag() if payload is None else payload
    binding = bind(payload, **kwargs)
    source = lm.parse_tag_sources(payload, binding)
    partition = lm.partition_displacement_sources(source)
    return payload, binding, source, partition


def freeze(forward, binding):
    return lm.freeze_landmark_model(forward, source_binding=binding,
        model_sha256="4" * 64, prediction_sha256="5" * 64)


def test_fps_uses_original_order_for_exact_ties_and_fixed_rank3_rule():
    _, _, source, partition = prepare()
    assert partition.boundary_ids == (1, 8, 2, 3, 4, 5)
    assert partition.validation_ids == (6, 7, 9, 10, 11, 12)
    assert source.frame == "world_mm_unverified"
    assert source.row_ids == tuple(range(1, 13))
    assert partition.rank_ratio > 1e-6
    assert partition.rule == "source_fps_six_rank3_v1"


@pytest.mark.parametrize("suffix", ["", ' "anatomy label"', ' 1.25 7 9 "tag%label"', " -0.5 -1 2"])
def test_documented_writer_subset_labels_and_auxiliary_fields_are_not_model_inputs(suffix):
    payload, binding, source, partition = prepare(tag(suffix=suffix))
    baseline, base_binding, _, base_partition = prepare()
    assert partition == base_partition
    forward = lm.build_forward_landmarks(payload, binding, partition)
    other = lm.build_forward_landmarks(baseline, base_binding, base_partition)
    assert forward.forward_hash == other.forward_hash
    assert binding.audit_hash != base_binding.audit_hash or suffix == ""
    np.testing.assert_array_equal(source.source_world_mm, POINTS)


def test_source_and_forward_pass_never_convert_sealed_destination_tokens(monkeypatch):
    _, _, _, partition = prepare()
    destination = (POINTS + 1).astype(object)
    for row in partition.validation_ids:
        destination[row - 1] = ["SEALED", "nan", "1e999"]
    payload = tag(destination=destination)
    binding = bind(payload)
    conversions = []
    original = lm._coordinate_tokens

    def observed(tokens):
        conversions.append(tuple(tokens))
        return original(tokens)

    monkeypatch.setattr(lm, "_coordinate_tokens", observed)
    source = lm.parse_tag_sources(payload, binding)
    actual_partition = lm.partition_displacement_sources(source)
    forward = lm.build_forward_landmarks(payload, binding, actual_partition)
    assert not any("SEALED" in value for value in conversions)
    frozen = freeze(forward, binding)
    with pytest.raises(ValueError, match="Accessed landmark"):
        lm.reveal_validation_landmarks(payload, binding, actual_partition, frozen)
    assert any("SEALED" in value for value in conversions)


def test_withheld_destination_mutation_preserves_forward_but_invalidates_audit_freeze():
    payload, binding, _, partition = prepare()
    forward = lm.build_forward_landmarks(payload, binding, partition)
    frozen = freeze(forward, binding)
    changed = POINTS + [0.25, -0.5, 1]
    changed[np.array(partition.validation_ids) - 1] += [17, -8, 9]
    next_payload, next_binding, _, next_partition = prepare(tag(destination=changed))
    next_forward = lm.build_forward_landmarks(next_payload, next_binding, next_partition)
    assert partition == next_partition
    assert forward.to_manifest() == next_forward.to_manifest()
    assert forward.forward_hash == next_forward.forward_hash
    assert binding.audit_hash != next_binding.audit_hash
    with pytest.raises(ValueError, match="Sealed source"):
        lm.reveal_validation_landmarks(next_payload, next_binding, next_partition, frozen)
    validation = lm.reveal_validation_landmarks(payload, binding, partition, frozen)
    assert validation.row_ids == partition.validation_ids
    np.testing.assert_array_equal(validation.observed_ras_mm,
        (POINTS + [0.25, -0.5, 1])[np.array(partition.validation_ids) - 1])


def test_forward_allowlist_contains_only_six_b_pairs_and_no_later_image_identity():
    payload, binding, _, partition = prepare()
    forward = lm.build_forward_landmarks(payload, binding, partition)
    manifest = forward.to_manifest()
    assert set(manifest) == {"schema_version", "role", "patient_group", "source_image_sha256",
        "partition_hash", "frame", "units", "boundary_ids", "source_ras_mm", "observed_ras_mm",
        "source_world_to_ras_mm", "destination_world_to_ras_mm"}
    assert len(manifest["source_ras_mm"]) == len(manifest["observed_ras_mm"]) == 6
    assert binding.tag_sha256 not in str(manifest)
    assert binding.destination_image_sha256 not in str(manifest)
    assert "not_tool_boundaries" in manifest["role"]
    with pytest.raises(TypeError):
        replace(forward, during_image_path="forbidden.nii.gz")


def test_no_world_axis_inference_and_qc_matches_image_pair():
    payload, binding, _, partition = prepare(verified=False)
    with pytest.raises(ValueError, match="WORLD_FRAME_UNVERIFIED"):
        lm.build_forward_landmarks(payload, binding, partition)
    qc = bind(payload).frame_qc
    with pytest.raises(ValueError, match="different images"):
        replace(binding, source_image_sha256="9" * 64, frame_qc=qc)


def test_explicit_physical_frame_transform_preserves_mm_and_vector_direction():
    matrix = np.array([[0, -1, 0, 12], [-1, 0, 0, 3], [0, 0, 1, -2], [0, 0, 0, 1.]])
    payload, binding, _, partition = prepare(transform=matrix)
    forward = lm.build_forward_landmarks(payload, binding, partition)
    chosen = POINTS[np.array(partition.boundary_ids) - 1]
    np.testing.assert_array_equal(forward.source_ras_mm, chosen @ matrix[:3, :3].T + matrix[:3, 3])
    np.testing.assert_array_equal(forward.displacement_ras_mm, np.tile([0.5, -0.25, 1], (6, 1)))
    scaled = matrix.copy()
    scaled[:3, :3] *= 2
    with pytest.raises(ValueError, match="rigid"):
        bind(payload, transform=scaled)


@pytest.mark.parametrize("points", [POINTS[:11], np.vstack([POINTS[:11], POINTS[0]]),
    np.column_stack([np.arange(12), np.arange(12) ** 2, np.zeros(12)])])
def test_frozen_eligibility_fails_without_destination_driven_rescue(points):
    payload = tag(points)
    with pytest.raises(ValueError, match="at least 12|rank3"):
        lm.partition_displacement_sources(lm.parse_tag_sources(payload, bind(payload)))


def test_registration_role_is_a_separate_input_type_and_pair():
    payload = tag()
    binding = bind(payload, role=lm.REGISTRATION_ROLE)
    sources = lm.parse_tag_sources(payload, binding)
    with pytest.raises(ValueError, match="before-US/during-US"):
        lm.partition_displacement_sources(sources)
    registration = lm.read_registration_landmarks(payload, binding)
    assert len(registration.source_ras_mm) == 12
    with pytest.raises(TypeError, match="forward observations"):
        freeze(registration, binding)
    with pytest.raises(ValueError, match="separate MRI"):
        lm.read_registration_landmarks(payload, bind(payload))


@pytest.mark.parametrize("change", [
    lambda b: b.replace(b"Volumes = 2", b"Volumes = 1"),
    lambda b: b.replace(b"MNI Tag Point File", b"Other format"),
    lambda b: b.rstrip()[:-1],
    lambda b: b + b"0 0 0 1 1 1\n",
    lambda b: b.replace(b"\n;", b"\n;;"),
    lambda b: b.replace(b"\n;", b"\nunterminated \"label\n;"),
])
def test_malformed_syntax_is_rejected_without_echoing_patient_tokens(change):
    payload = change(tag())
    with pytest.raises(ValueError):
        lm.parse_tag_sources(payload, bind(payload))


@pytest.mark.parametrize("suffix", [" 1 2", " 1.2 3.5 4", " unquoted_label", ' 1 2 3 4 "x"'])
def test_unsupported_suffixes_fail_closed(suffix):
    payload = tag(suffix=suffix)
    with pytest.raises(ValueError, match="auxiliary"):
        lm.parse_tag_sources(payload, bind(payload))


def test_source_corruption_partition_forgery_and_invalid_accessed_b_fail():
    payload, binding, _, partition = prepare()
    with pytest.raises(ValueError, match="declared source"):
        lm.parse_tag_sources(payload + b" ", binding)
    forged = replace(partition, boundary_ids=partition.boundary_ids[::-1])
    with pytest.raises(ValueError, match="Partition is stale"):
        lm.build_forward_landmarks(payload, binding, forged)
    destination = (POINTS + 1).astype(object)
    destination[partition.boundary_ids[0] - 1] = ["nan", "0", "0"]
    invalid = tag(destination=destination)
    with pytest.raises(ValueError, match="Accessed landmark"):
        lm.build_forward_landmarks(invalid, bind(invalid), partition)


def test_freeze_and_source_objects_are_immutable_and_detect_bypassed_mutation():
    payload, binding, source, partition = prepare()
    forward = lm.build_forward_landmarks(payload, binding, partition)
    frozen = freeze(forward, binding)
    for array in (source.source_world_mm, forward.source_ras_mm, forward.observed_ras_mm,
                  binding.frame_qc.source_world_to_ras_mm):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        frozen.model_sha256 = "9" * 64
    object.__setattr__(frozen, "prediction_sha256", "sha256:" + "9" * 64)
    with pytest.raises(ValueError, match="freeze was changed"):
        lm.reveal_validation_landmarks(payload, binding, partition, frozen)
    object.__setattr__(binding, "role", lm.REGISTRATION_ROLE)
    with pytest.raises(ValueError, match="binding was changed"):
        source.assert_intact()


def test_array_layout_and_forward_interpretation_mutations_are_detected():
    payload, binding, _, partition = prepare()
    forward = lm.build_forward_landmarks(payload, binding, partition)
    forward.source_ras_mm.strides = forward.source_ras_mm.strides[::-1]
    with pytest.raises(ValueError, match="Forward landmark input changed"):
        forward.to_manifest()
    forward = lm.build_forward_landmarks(payload, binding, partition)
    object.__setattr__(forward, "patient_group", "different")
    with pytest.raises(ValueError, match="Forward landmark input changed"):
        freeze(forward, binding)


def test_parser_limits_and_nonfinite_source_are_explicit(monkeypatch):
    payload = tag()
    monkeypatch.setattr(lm, "MAX_TAG_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="bounds"):
        lm.parse_tag_sources(payload, bind(payload))
    monkeypatch.setattr(lm, "MAX_TAG_BYTES", 1024 * 1024)
    invalid = payload.replace(b"0.0 0.0 0.0", b"nan 0.0 0.0", 1)
    with pytest.raises(ValueError, match="Accessed landmark"):
        lm.parse_tag_sources(invalid, bind(invalid))
    with pytest.raises(ValueError, match="immutable bytes"):
        lm.parse_tag_sources(bytearray(payload), bind(payload))


def test_model_freeze_requires_exact_forward_type_and_matching_audit():
    payload, binding, _, partition = prepare()
    forward = lm.build_forward_landmarks(payload, binding, partition)
    with pytest.raises(ValueError, match="does not match"):
        freeze(forward, replace(binding, patient_group="other"))
    with pytest.raises(ValueError, match="freeze is required"):
        lm.reveal_validation_landmarks(payload, binding, partition, None)
    with pytest.raises(TypeError, match="typed pair binding"):
        lm.parse_tag_sources(payload, {})
