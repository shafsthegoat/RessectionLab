"""Synthetic exact-cover cache tests; production native code remains unchanged."""
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from resectionlab import geometry, native_resection as native
from resectionlab.evaluation import independent_check_native_history
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache, inject_native_capsule_cache


def fixture(affine=None, **kwargs):
    affine = np.eye(4) if affine is None else np.asarray(affine, float)
    tissue = np.zeros((9, 9, 10), bool)
    tissue[1:8, 1:8, 2:9] = True
    labels = np.zeros(tissue.shape, np.int16)
    labels[3:6, 3:6, 4:8] = 1
    entry = affine[:3, :3] @ [4, 4, 1.5] + affine[:3, 3]
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    config = native.NativeResectionConfig(tissue, labels, affine,
        geometry.AccessWindow(entry, normal, 6), native.NATIVE_GENERIC_TOOLS,
        "synthetic-cache-source", "synthetic explicit native tissue", **kwargs)
    return config, affine[:3, :3] @ [4, 4, 7] + affine[:3, 3]


def encoded(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if is_dataclass(item):
            return {field.name: getattr(item, field.name) for field in fields(item)}
        raise TypeError(type(item).__name__)
    return json.dumps(value, default=convert, sort_keys=True, allow_nan=False)


def state(engine):
    return (engine.state_hash, encoded(engine.history),
            *(getattr(engine, name).tobytes() for name in
              ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask")))


def test_cold_warm_and_same_cavity_clone_reset_results_exact_and_independently_checked():
    config, tip = fixture()
    reference = native.NativeResectionEngine(config)
    expected = reference.execute_stroke("native-wide-aspiration", tip)
    assert expected.feasible
    expected_state = state(reference)
    cache = ExactCapsuleCoverCache(config)
    engine = native.NativeResectionEngine(config)
    with inject_native_capsule_cache(cache):
        cold = engine.preview_stroke("native-wide-aspiration", tip)
        assert encoded(cold) == encoded(expected)
        clone = engine.clone()
        warm = clone.preview_stroke("native-wide-aspiration", tip)
        assert encoded(warm) == encoded(expected)
        clone.commit_preview(warm)
        assert state(clone) == expected_state
        assert not engine.removed_mask.any()
        engine.commit_preview(cold)
        assert state(engine) == expected_state
        again = engine.preview_stroke("native-wide-aspiration", tip)
        assert not again.feasible and "NO_NEW" in again.reason
        engine.reset()
        repeated = engine.execute_stroke("native-wide-aspiration", tip)
        assert encoded(repeated) == encoded(expected) and state(engine) == expected_state
    assert native.capsule_voxel_indices is geometry.capsule_voxel_indices
    assert cache.stats()["hits"] > 0
    case = SimpleNamespace(mri=np.zeros(config.tissue_mask.shape), affine=config.affine,
                           frame="RAS+", semantic_hash=config.source_hash)
    first = independent_check_native_history(case, config.tools, reference.history,
        tissue_mask=config.tissue_mask, access=config.access, hard_exclusion=config.hard_exclusion)
    second = independent_check_native_history(case, config.tools, engine.history,
        tissue_mask=config.tissue_mask, access=config.access, hard_exclusion=config.hard_exclusion)
    assert first.feasible and second.feasible
    assert first.to_dict() == second.to_dict()
    # Disabling the injection reproduces the complete reference receipt.
    reversal = native.NativeResectionEngine(config)
    assert encoded(reversal.execute_stroke("native-wide-aspiration", tip)) == encoded(expected)
    assert state(reversal) == expected_state


@pytest.mark.parametrize("limits", [dict(max_payload_bytes=0), dict(max_entries=0),
                                  dict(max_payload_bytes=1), dict(max_entries=1)])
def test_zero_oversize_and_forced_eviction_preserve_results(limits):
    config, _ = fixture()
    cache = ExactCapsuleCoverCache(config, **limits)
    scene = native.NativeResectionEngine(config)._cell_scene
    for tip in ([4, 4, 5], [4, 4, 6], [4, 4, 5]):
        result = cache.query(scene, [4, 4, 2], tip, 1.25)
        np.testing.assert_array_equal(result, geometry.capsule_voxel_indices(scene, [4, 4, 2], tip, 1.25))
    stats = cache.stats()
    assert stats["retained_payload_bytes"] <= stats["max_payload_bytes"]
    assert stats["entries"] <= stats["max_entries"]
    if limits == {"max_entries": 1}:
        assert stats["evictions"] == 2
    else:
        assert stats["bypasses"] == 3 and stats["entries"] == 0


def test_empty_results_still_bounded_by_entry_cap_and_arrays_cannot_mutate():
    config, _ = fixture()
    cache = ExactCapsuleCoverCache(config, max_entries=2)
    scene = native.NativeResectionEngine(config)._cell_scene
    for offset in range(5):
        cache.query(scene, [100 + offset, 100, 100], [101 + offset, 100, 100], 0.)
    assert cache.stats()["entries"] == 2 and cache.stats()["evictions"] == 3
    result = cache.query(scene, [4, 4, 2], [4, 4, 5], 1.25)
    for change in (lambda: result.setflags(write=True), lambda: setattr(result, "shape", (result.size,)),
                   lambda: setattr(result, "dtype", np.int32)):
        with pytest.raises(ValueError):
            change()
    with pytest.raises(AttributeError):
        cache.namespace = "another-source"


@pytest.mark.parametrize("angle,mirror", [(0., 1), (.413, 1), (.413, -1)])
def test_oblique_mirrored_and_tangent_queries_equal_reference(angle, mirror):
    affine = np.eye(4)
    affine[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.diag([mirror, 1.5, .8])
    affine[:3, 3] = [11, -7, 4]
    config, _ = fixture(affine)
    cache, scene = ExactCapsuleCoverCache(config), native.NativeResectionEngine(config)._cell_scene
    points = np.array([[4, 4, 2], [4, 4, 5]]) @ affine[:3, :3].T + affine[:3, 3]
    for radius in (0., .5, np.nextafter(.5, 0.), np.nextafter(.5, 1.)):
        expected = geometry.capsule_voxel_indices(scene, *points, radius)
        for _ in range(2):
            np.testing.assert_array_equal(cache.query(scene, *points, radius), expected)
    assert cache.stats()["hits"] == 4 and cache.stats()["misses"] == 4


def test_source_affine_radius_and_mode_namespaces_never_alias():
    config, _ = fixture()
    cache, scene = ExactCapsuleCoverCache(config), native.NativeResectionEngine(config)._cell_scene
    start, end = np.array([4., 4., 2.]), np.array([4., 4., 5.])
    cache.query(scene, start, end, .5)
    end[2] = np.nextafter(end[2], np.inf)
    cache.query(scene, start, end, .5)
    cache.query(scene, start, end, np.nextafter(.5, np.inf))
    assert cache.stats()["misses"] == 3
    assert ExactCapsuleCoverCache(replace(config, source_hash="other-source")).namespace != cache.namespace
    moved = config.affine.copy()
    moved[0, 3] += 1
    other, _ = fixture(moved)
    assert ExactCapsuleCoverCache(other).namespace != cache.namespace
    with pytest.raises(RuntimeError):
        cache.query(native.NativeResectionEngine(other)._cell_scene, start, end, .5)
    object.__setattr__(scene, "_orthogonal_spacing", None)
    with pytest.raises(RuntimeError):
        cache.query(scene, start, end, .5)


def test_context_restores_on_exception_rejects_nesting_and_foreign_patch(monkeypatch):
    config, _ = fixture()
    cache = ExactCapsuleCoverCache(config)
    with pytest.raises(LookupError):
        with inject_native_capsule_cache(cache):
            with pytest.raises(RuntimeError):
                with inject_native_capsule_cache(cache):
                    pass
            raise LookupError("deliberate probe failure")
    assert native.capsule_voxel_indices is geometry.capsule_voxel_indices
    foreign = lambda *args: None
    monkeypatch.setattr(native, "capsule_voxel_indices", foreign)
    with pytest.raises(RuntimeError):
        with inject_native_capsule_cache(cache):
            pass
    assert native.capsule_voxel_indices is foreign


@pytest.mark.parametrize("name", ["world_to_voxel", "voxel_to_world"])
def test_instance_coordinate_method_replacement_cannot_bypass_scene_frame_guard(name):
    config, _ = fixture()
    cache, scene = ExactCapsuleCoverCache(config), native.NativeResectionEngine(config)._cell_scene
    cache.query(scene, [4, 4, 2], [4, 4, 5], 1.25)
    object.__setattr__(scene, name, lambda points: np.zeros_like(points))
    with pytest.raises(RuntimeError):
        cache.query(scene, [4, 4, 2], [4, 4, 5], 1.25)


def test_changed_frozen_source_is_rejected_before_and_after_preparation():
    config, _ = fixture()
    cache = ExactCapsuleCoverCache(config)
    scene = native.NativeResectionEngine(config)._cell_scene
    changed = config.target_labels.copy()
    changed[4, 4, 4] = 0
    object.__setattr__(config, "target_labels", geometry._immutable(changed))
    with pytest.raises(RuntimeError):
        cache.query(scene, [4, 4, 2], [4, 4, 5], 1.25)
    with pytest.raises(ValueError):
        ExactCapsuleCoverCache(config)


@pytest.mark.parametrize("changed", ["source", "scene"])
def test_input_conversion_cannot_change_source_after_guards_before_warm_hit(changed):
    config, _ = fixture()
    cache = ExactCapsuleCoverCache(config)
    scene = native.NativeResectionEngine(config)._cell_scene
    cache.query(scene, [4, 4, 2], [4, 4, 5], 1.25)

    class MutatingStart:
        def __array__(self, dtype=None, copy=None):
            if changed == "source":
                labels = config.target_labels.copy()
                labels[4, 4, 4] = 0
                object.__setattr__(config, "target_labels", geometry._immutable(labels))
            else:
                inverse = scene._inverse.copy()
                inverse[0, 3] += 1.
                object.__setattr__(scene, "_inverse", geometry._immutable(inverse))
            return np.array([4., 4., 2.], dtype=dtype)

    with pytest.raises(RuntimeError):
        cache.query(scene, MutatingStart(), [4, 4, 5], 1.25)


@pytest.mark.parametrize("step", [.5, .25, .1])
def test_warm_cache_preserves_short_tip_prior_cavity_shaft_rejection(step):
    tissue = np.zeros((5, 5, 10), bool)
    tissue[2, 2, 5] = True
    tool = geometry.ToolGeometry("short-tip", .9, .1, 10, tip_length_mm=.1)
    config = native.NativeResectionConfig(tissue, tissue.astype(np.int16), np.eye(4),
        geometry.AccessWindow([2, 2, 4.49], [0, 0, 1], 2), (tool,), "short-tip-source",
        "single native source cell", max_tip_step_mm=step)
    reference = native.NativeResectionEngine(config).preview_stroke(tool.tool_id, [2, 2, 5.49])
    assert not reference.feasible and reference.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
    cache, engine = ExactCapsuleCoverCache(config), native.NativeResectionEngine(config)
    before = state(engine)
    with inject_native_capsule_cache(cache):
        for _ in range(2):
            assert encoded(engine.preview_stroke(tool.tool_id, [2, 2, 5.49])) == encoded(reference)
            assert state(engine) == before
    assert cache.stats()["hits"] > 0


def test_hard_barrier_still_runs_uncached_full_tool_check():
    config, tip = fixture()
    hard = np.zeros(config.tissue_mask.shape, bool)
    hard[4, 4, 4] = True
    config = replace(config, hard_exclusion=hard)
    expected = native.NativeResectionEngine(config).preview_stroke("native-wide-aspiration", tip)
    cache, engine = ExactCapsuleCoverCache(config), native.NativeResectionEngine(config)
    with inject_native_capsule_cache(cache):
        actual = engine.preview_stroke("native-wide-aspiration", tip)
    assert encoded(actual) == encoded(expected)
    assert not actual.feasible and actual.reason.startswith("HARD_GEOMETRY")
    assert cache.stats()["calls"] == 0 and not engine.removed_mask.any()
