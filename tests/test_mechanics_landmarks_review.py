"""Independent constructed-record controls; no patient files or solver runs."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from resectionlab import mechanics_landmarks as lm


POINTS = np.array([
    [0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4],
    [4, 4, 0], [4, 0, 4], [0, 4, 4], [4, 4, 4],
    [1, 1, 1], [1, 2, 1], [2, 1, 1], [1, 1, 2], [2, 2, 2],
], dtype=float)


def payload(points=POINTS, destinations=None):
    destinations = points + [0.5, -0.25, 2] if destinations is None else destinations
    rows = [" ".join(str(v) for v in (*a, *b))
            for a, b in zip(points, destinations, strict=True)]
    return ("MNI Tag Point File\nVolumes = 2;\nPoints =\n"
            + "\n".join(rows) + "\n;\n").encode("ascii")


def binding(data, *, role=lm.DISPLACEMENT_ROLE, verified=True,
            source_transform=None, destination_transform=None, destination_hash="b" * 64):
    source_transform = np.eye(4) if source_transform is None else source_transform
    destination_transform = np.eye(4) if destination_transform is None else destination_transform
    qc = lm.LandmarkFrameQC("a" * 64, destination_hash, source_transform,
        destination_transform, "c" * 64, "constructed axis contract") if verified else None
    return lm.LandmarkPairBinding("constructed_not_patient", role,
        hashlib.sha256(data).hexdigest(), "a" * 64, destination_hash, qc)


def prepared(data=None, **kwargs):
    data = payload() if data is None else data
    bound = binding(data, **kwargs)
    source = lm.parse_tag_sources(data, bound)
    partition = lm.partition_displacement_sources(source)
    return data, bound, source, partition


def frozen(data, bound, partition):
    forward = lm.build_forward_landmarks(data, bound, partition)
    freeze = lm.freeze_landmark_model(forward, source_binding=bound,
        model_sha256="d" * 64, prediction_sha256="e" * 64)
    return forward, freeze


def test_exact_fps_against_independent_scalar_oracle_and_all_remaining_v():
    _, _, source, partition = prepared()
    chosen = [0]
    for _ in range(5):
        remaining = [i for i in range(len(POINTS)) if i not in chosen]
        def distance(i):
            return min(sum((float(POINTS[i, k]) - float(POINTS[j, k])) ** 2
                           for k in range(3)) for j in chosen)
        chosen.append(max(remaining, key=lambda i: (distance(i), -i)))
    assert tuple(i + 1 for i in chosen) == partition.boundary_ids == (1, 8, 2, 3, 4, 5)
    assert partition.validation_ids == tuple(i + 1 for i in range(13) if i not in chosen)
    assert len(partition.validation_ids) == 7
    centered = POINTS[chosen] - POINTS[chosen].mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    assert partition.rank_ratio == pytest.approx(np.sqrt(eigenvalues[0] / eigenvalues[-1]), rel=1e-14)
    assert source.frame == "world_mm_unverified"
    _, _, _, twelve = prepared(payload(POINTS[:12]))
    assert len(twelve.boundary_ids) == len(twelve.validation_ids) == 6
    with pytest.raises(ValueError, match="at least 12"):
        prepared(payload(POINTS[:11]))


@pytest.mark.parametrize("ratio,allowed", [(1e-6, False), (np.nextafter(1e-6, 0), False),
                                          (np.nextafter(1e-6, 1), True)])
def test_strict_rank_threshold_controlled_singular_values(monkeypatch, ratio, allowed):
    # This isolates the exact comparison; the preceding test independently checks actual rank arithmetic.
    monkeypatch.setattr(lm.np.linalg, "svd", lambda *args, **kwargs: np.array([1., .5, ratio]))
    if allowed:
        assert prepared()[3].rank_ratio == ratio
    else:
        with pytest.raises(ValueError, match="rank3"):
            prepared()


def test_v_destinations_and_later_image_identity_cannot_change_allowed_forward():
    data, bound, source, partition = prepared()
    forward, freeze = frozen(data, bound, partition)
    targets = POINTS + [0.5, -0.25, 2]
    targets[np.array(partition.validation_ids) - 1] *= -173
    other_data, other_bound, other_source, other_partition = prepared(
        payload(destinations=targets), destination_hash="f" * 64)
    other_forward, _ = frozen(other_data, other_bound, other_partition)
    assert source.source_hash == other_source.source_hash
    assert partition == other_partition
    assert forward.to_manifest() == other_forward.to_manifest()
    assert bound.audit_hash != other_bound.audit_hash
    with pytest.raises(ValueError, match="Sealed source"):
        lm.reveal_validation_landmarks(other_data, other_bound, other_partition, freeze)
    assert len(forward.source_ras_mm) == len(forward.observed_ras_mm) == 6
    assert not ({"validation_ids", "tag_sha256", "destination_image_sha256", "image", "mask", "cavity"}
                & forward.to_manifest().keys())


@pytest.mark.parametrize("tamper", ["model", "prediction", "partition", "source_audit", "pair"])
def test_invalid_freeze_or_pair_rejected_before_any_v_numeric_conversion(monkeypatch, tamper):
    _, _, _, initial = prepared()
    targets = (POINTS + 1).astype(object)
    targets[np.array(initial.validation_ids) - 1] = ["WITHHELD", "nan", "1e999"]
    data, bound, _, partition = prepared(payload(destinations=targets))
    conversions = []
    original = lm._coordinate_tokens
    def instrumented(tokens):
        conversions.append(tuple(tokens))
        return original(tokens)
    monkeypatch.setattr(lm, "_coordinate_tokens", instrumented)
    _, freeze = frozen(data, bound, partition)
    assert not any("WITHHELD" in row for row in conversions)
    if tamper in {"model", "prediction", "source_audit"}:
        field = {"model": "model_sha256", "prediction": "prediction_sha256",
                 "source_audit": "source_audit_hash"}[tamper]
        object.__setattr__(freeze, field, "sha256:" + "9" * 64)
    elif tamper == "partition":
        partition = replace(partition, boundary_ids=partition.boundary_ids[::-1])
    else:
        bound = replace(bound, patient_group="different_constructed_group")
    with pytest.raises(ValueError):
        lm.reveal_validation_landmarks(data, bound, partition, freeze)
    assert not any("WITHHELD" in row for row in conversions)


def test_only_explicit_valid_reveal_converts_withheld_destinations(monkeypatch):
    data, bound, _, partition = prepared()
    forward, freeze = frozen(data, bound, partition)
    conversions = []
    original = lm._coordinate_tokens
    def instrumented(tokens):
        conversions.append(tuple(tokens))
        return original(tokens)
    monkeypatch.setattr(lm, "_coordinate_tokens", instrumented)
    revealed = lm.reveal_validation_landmarks(data, bound, partition, freeze)
    chosen = np.array(partition.validation_ids) - 1
    assert revealed.row_ids == partition.validation_ids
    np.testing.assert_array_equal(revealed.source_ras_mm, POINTS[chosen])
    np.testing.assert_array_equal(revealed.observed_ras_mm, (POINTS + [.5, -.25, 2])[chosen])
    assert revealed.freeze_hash == freeze.freeze_hash
    assert len(revealed.row_ids) + len(forward.boundary_ids) == len(POINTS)
    assert set(revealed.row_ids).isdisjoint(forward.boundary_ids)
    for row in (POINTS + [.5, -.25, 2])[chosen]:
        assert tuple(map(str, row)) in conversions


def test_independent_source_and_destination_frame_transforms_define_displacement():
    source = np.array([[-1, 0, 0, 4], [0, -1, 0, 7], [0, 0, 1, -2], [0, 0, 0, 1.]])
    target = np.array([[0, -1, 0, 6], [1, 0, 0, -3], [0, 0, 1, 8], [0, 0, 0, 1.]])
    data, bound, _, partition = prepared(source_transform=source, destination_transform=target)
    forward, _ = frozen(data, bound, partition)
    ids = np.array(partition.boundary_ids) - 1
    expected_source = POINTS[ids] @ source[:3, :3].T + source[:3, 3]
    expected_target = (POINTS + [.5, -.25, 2])[ids] @ target[:3, :3].T + target[:3, 3]
    np.testing.assert_array_equal(forward.displacement_ras_mm, expected_target - expected_source)
    for array in (forward.source_ras_mm, forward.observed_ras_mm, bound.frame_qc.source_world_to_ras_mm):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    bad = source.copy(); bad[0, 0] = -1.01
    with pytest.raises(ValueError, match="rigid"):
        prepared(source_transform=bad)


def test_unverified_frame_and_separate_registration_role_cannot_grant_displacement_access():
    data, bound, _, partition = prepared(verified=False)
    with pytest.raises(ValueError, match="WORLD_FRAME_UNVERIFIED"):
        lm.build_forward_landmarks(data, bound, partition)
    baseline = binding(data, role=lm.REGISTRATION_ROLE)
    with pytest.raises(ValueError, match="before-US/during-US"):
        lm.build_forward_landmarks(data, baseline, partition)
    registration = lm.read_registration_landmarks(data, baseline)
    assert len(registration.source_ras_mm) == 13  # all permitted baseline pairs
    with pytest.raises(TypeError, match="forward observations"):
        lm.freeze_landmark_model(registration, source_binding=baseline,
            model_sha256="d" * 64, prediction_sha256="e" * 64)
    with pytest.raises(ValueError, match="WORLD_FRAME_UNVERIFIED"):
        lm.read_registration_landmarks(data, binding(data, role=lm.REGISTRATION_ROLE, verified=False))


def test_source_coordinate_corruption_cannot_reuse_old_partition():
    data, bound, _, partition = prepared()
    changed = POINTS.copy()
    changed[partition.validation_ids[-1] - 1] += [.125, .25, .5]
    other = payload(changed)
    with pytest.raises(ValueError, match="Partition is stale"):
        lm.build_forward_landmarks(other, binding(other), partition)
    duplicate = POINTS.copy(); duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="unique source"):
        prepared(payload(duplicate))
    nonfinite = POINTS.astype(object); nonfinite[-1, 0] = "1e999"
    with pytest.raises(ValueError, match="finite real"):
        prepared(payload(nonfinite, destinations=POINTS))
