"""Independent synthetic adversaries for exact coverage-only caching."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import warnings

import numpy as np
import pytest

from resectionlab import geometry, native_resection
from resectionlab.core import CaseData, SourceRef
from resectionlab.evaluation import independent_check_native_history
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache, inject_native_capsule_cache
from resectionlab.geometry import AccessWindow, GeometryScene, ToolGeometry, _immutable
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine


def fixture(*, hard=False):
    tissue = np.ones((7, 7, 8), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[:, :, 2:7] = 1
    case = CaseData("capsule-cache-review", tissue.astype(float), {"target": labels > 0}, np.eye(4),
                    (SourceRef("analytic", "synthetic:cache-review", provenance="simulated"),), brain_mask=tissue)
    blocked = np.zeros(tissue.shape, bool)
    if hard:
        blocked[3, 3, 5] = True
    config = NativeResectionConfig(tissue, labels, case.affine, AccessWindow((3, 3, -.5), (0, 0, 1), 5),
        tuple(replace(tool) for tool in NATIVE_GENERIC_TOOLS), case.semantic_hash, "explicit synthetic support",
        hard_exclusion=blocked)
    return case, config, GeometryScene(np.zeros(tissue.shape, bool), config.affine)


def test_exact_float64_bits_separate_ulp_radius_endpoint_and_order():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    start, end = np.array([2., 3., 1.]), np.array([2., 3., 4.])
    queries = [(start, end, .2), (start, end, np.nextafter(.2, np.inf)),
               (np.nextafter(start, np.inf), end, .2), (end, start, .2)]
    for one, two, radius in queries:
        np.testing.assert_array_equal(cache.query(scene, one, two, radius), geometry.capsule_voxel_indices(scene, one, two, radius))
    assert cache.stats()["misses"] == 4 and cache.stats()["hits"] == 0
    cache.query(scene, start, end, .2)
    assert cache.stats()["hits"] == 1
    assert not cache.stats()["feasibility_cached"] and not cache.stats()["cavity_cached"]


def test_output_readonly_and_forced_descriptor_tampering_fail_closed():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    result = cache.query(scene, (3, 3, 2), (3, 3, 4), .3)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    with pytest.raises(ValueError):
        result.shape = (result.size,)
    # Bypass the ndarray subclass's setter to attack the cached descriptor.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # Deliberate low-level tampering adversary.
        np.ndarray.__setattr__(result, "shape", (result.size,))
    with pytest.raises(RuntimeError, match="descriptor"):
        cache.query(scene, (3, 3, 2), (3, 3, 4), .3)


@pytest.mark.parametrize("field", ["affine", "_inverse", "_cell_radius_mm", "_orthogonal_spacing"])
def test_scene_derived_geometry_changes_reject_even_on_cache_hit(field):
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    cache.query(scene, (3, 3, 2), (3, 3, 4), .3)
    if field == "_cell_radius_mm":
        value = scene._cell_radius_mm + .1
    elif field == "_orthogonal_spacing":
        value = None
    else:
        value = getattr(scene, field).copy()
        value[0, 3] += .125
        value = _immutable(value)
    object.__setattr__(scene, field, value)
    with pytest.raises(RuntimeError):
        cache.query(scene, (3, 3, 2), (3, 3, 4), .3)


def test_source_content_replacement_and_same_shape_variants_are_distinct():
    _, config, scene = fixture()
    original = ExactCapsuleCoverCache(config)
    modified = config.target_labels.copy()
    modified[0, 0, 2] = 0
    variant = replace(config, target_labels=modified)
    assert ExactCapsuleCoverCache(variant).namespace != original.namespace
    object.__setattr__(config, "target_labels", _immutable(modified))
    with pytest.raises(RuntimeError):
        original.query(scene, (3, 3, 2), (3, 3, 4), .3)
    with pytest.raises(ValueError):
        ExactCapsuleCoverCache(config)


def test_version_namespace_changes_and_old_cache_rejects(monkeypatch):
    _, config, scene = fixture()
    original = ExactCapsuleCoverCache(config)
    original.query(scene, (3, 3, 2), (3, 3, 4), .3)
    monkeypatch.setattr(geometry, "GEOMETRY_VERSION", geometry.GEOMETRY_VERSION + ":review")
    variant = ExactCapsuleCoverCache(config)
    assert original.namespace != variant.namespace
    with pytest.raises(RuntimeError):
        original.query(scene, (3, 3, 2), (3, 3, 4), .3)


@pytest.mark.parametrize("helper", ["_segment_cell_distances", "point_segment_distances"])
def test_direct_geometry_helper_replacement_invalidates_warm_cache(monkeypatch, helper):
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    assert len(cache.query(scene, (3, 3, 2), (3, 3, 4), .3))

    def changed(*args):
        points = args[1] if helper == "_segment_cell_distances" else args[0]
        return np.full(len(points), 1e6)

    monkeypatch.setattr(geometry, helper, changed)
    assert not len(geometry.capsule_voxel_indices(scene, (3, 3, 2), (3, 3, 4), .3))
    with pytest.raises(RuntimeError):
        cache.query(scene, (3, 3, 2), (3, 3, 4), .3)


@pytest.mark.parametrize("mutation", ["source", "scene_mode"])
def test_input_conversion_mutation_is_validated_before_warm_lookup(mutation):
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    cache.query(scene, (3, 3, 2), (3, 3, 4), .3)

    class MutatingPoint:
        def __array__(self, dtype=None, copy=None):
            if mutation == "source":
                object.__setattr__(config, "source_hash", "changed-during-conversion")
            else:
                object.__setattr__(scene, "_orthogonal_spacing", None)
            return np.asarray((3, 3, 2), dtype=dtype)

    with pytest.raises(RuntimeError):
        cache.query(scene, MutatingPoint(), (3, 3, 4), .3)


@pytest.mark.parametrize("limits", [{"max_payload_bytes": 0}, {"max_entries": 0}, {"max_payload_bytes": 1}])
def test_zero_or_undersized_budget_never_retains_payload(limits):
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config, **limits)
    for _ in range(2):
        result = cache.query(scene, (3, 3, 2), (3, 3, 4), .3)
        np.testing.assert_array_equal(result, geometry.capsule_voxel_indices(scene, (3, 3, 2), (3, 3, 4), .3))
    stats = cache.stats()
    assert stats["entries"] == stats["retained_payload_bytes"] == stats["hits"] == 0
    assert stats["bypasses"] == stats["misses"] == 2


def test_empty_arrays_remain_entry_bounded_and_lru_eviction_recomputes():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config, max_entries=1, max_payload_bytes=100)
    for point in ((100, 100, 100), (101, 100, 100), (100, 100, 100)):
        assert cache.query(scene, point, point, .1).shape == (0, 3)
    assert cache.stats()["entries"] == 1 and cache.stats()["retained_payload_bytes"] == 0
    assert cache.stats()["misses"] == 3 and cache.stats()["evictions"] == 2


def test_payload_budget_evicts_lru_and_oversized_bypass_retains_existing_entries():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config, max_entries=5, max_payload_bytes=48)
    first, second, third = (1, 1, 1), (2, 2, 2), (3, 3, 3)
    for point in (first, second, first, third, second):
        assert cache.query(scene, point, point, .1).nbytes == 24
        assert cache.stats()["retained_payload_bytes"] <= 48
    stats = cache.stats()
    assert stats["entries"] == 2 and stats["retained_payload_bytes"] == 48
    assert stats["evictions"] == 2 and stats["hits"] == 1 and stats["misses"] == 4
    assert cache.query(scene, (1, 1, 1), (5, 5, 5), 1.).nbytes > 48
    assert cache.stats()["bypasses"] == 1 and cache.stats()["evictions"] == 2
    assert cache.stats()["entries"] == 2 and cache.stats()["retained_payload_bytes"] == 48


def test_reentrant_array_conversion_refuses_and_next_query_recovers():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)

    class ReentrantPoint:
        def __array__(self, dtype=None, copy=None):
            cache.query(scene, (2, 2, 2), (2, 2, 2), .1)
            return np.asarray((3, 3, 3), dtype=dtype)

    with pytest.raises(RuntimeError, match="reenter"):
        cache.query(scene, ReentrantPoint(), (3, 3, 4), .3)
    assert cache.stats()["entries"] == 0
    assert len(cache.query(scene, (3, 3, 2), (3, 3, 4), .3))


def test_invalid_query_restores_active_guard_and_never_caches_exception():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    with pytest.raises(ValueError):
        cache.query(scene, (np.nan, 3, 2), (3, 3, 4), .3)
    with pytest.raises(ValueError):
        cache.query(scene, (3, 3, 2), (3, 3, 4), np.float32(.3))
    assert cache.stats()["entries"] == 0
    assert len(cache.query(scene, (3, 3, 2), (3, 3, 4), .3))


def test_thread_crossing_nested_injection_and_body_failure_restore_patch():
    _, config, scene = fixture()
    cache = ExactCapsuleCoverCache(config)
    original = native_resection.capsule_voxel_indices
    with pytest.raises(LookupError, match="body failure"):
        with inject_native_capsule_cache(cache):
            with pytest.raises(RuntimeError, match="active"):
                with inject_native_capsule_cache(cache):
                    pass
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(native_resection.capsule_voxel_indices, scene, (3, 3, 2), (3, 3, 4), .3)
                with pytest.raises(RuntimeError, match="thread"):
                    future.result(timeout=2)
            raise LookupError("body failure")
    assert native_resection.capsule_voxel_indices is original
    with inject_native_capsule_cache(cache):
        assert len(native_resection.capsule_voxel_indices(scene, (3, 3, 2), (3, 3, 4), .3))
    assert native_resection.capsule_voxel_indices is original


def test_foreign_patch_before_or_during_scope_never_silently_survives(monkeypatch):
    _, config, _ = fixture()
    cache = ExactCapsuleCoverCache(config)
    original = native_resection.capsule_voxel_indices
    foreign = lambda *args, **kwargs: None
    monkeypatch.setattr(native_resection, "capsule_voxel_indices", foreign)
    with pytest.raises(RuntimeError, match="foreign"):
        with inject_native_capsule_cache(cache):
            pass
    assert native_resection.capsule_voxel_indices is foreign
    monkeypatch.setattr(native_resection, "capsule_voxel_indices", original)
    with pytest.raises(RuntimeError, match="changed during"):
        with inject_native_capsule_cache(cache):
            native_resection.capsule_voxel_indices = foreign
    assert native_resection.capsule_voxel_indices is original


@pytest.mark.parametrize("hard", [False, True])
def test_cached_and_uncached_acceptance_rejection_and_histories_match_exactly(hard):
    case, config, _ = fixture(hard=hard)

    def rollout(engine):
        records = []
        for tip in ((3, 3, 6), (3, 3, 6), (4, 3, 6)):
            result = engine.execute_stroke("native-fine-aspiration", tip)
            # Rejected preview diagnostics are not executed removal history.
            diagnostics = {"failure_tip_mm": result.failure_tip_mm,
                "candidate_removed_indices": result.removed_indices_native.tolist(),
                "candidate_contact_indices": result.contact_indices_native.tolist(),
                "candidate_microsteps": [step.to_dict() for step in result.microsteps],
                "source_state_hash": result.source_state_hash,
                "executed_history": result.to_history_record() if result.feasible else None}
            records.append((result.feasible, result.reason, diagnostics))
        return json.dumps(records, sort_keys=True), json.dumps(engine.history, sort_keys=True)

    baseline = NativeResectionEngine(config)
    expected = rollout(baseline)
    cached = NativeResectionEngine(config)
    cache = ExactCapsuleCoverCache(config)
    with inject_native_capsule_cache(cache):
        assert rollout(cached) == expected
        cached.reset()
        assert rollout(cached) == expected
    if hard:
        # Complete hard-constraint checking rejects these strokes before any
        # active/shaft cover queries, even with an injected cache available.
        assert cache.stats()["calls"] == 0
    else:
        assert cache.stats()["hits"] > 0
    np.testing.assert_array_equal(cached.removed_mask, baseline.removed_mask)
    np.testing.assert_array_equal(cached.contact_mask, baseline.contact_mask)
    for engine in (baseline, cached):
        audit = independent_check_native_history(case, config.tools, engine.history,
            tissue_mask=config.tissue_mask, access=config.access, hard_exclusion=config.hard_exclusion)
        assert audit.feasible and audit.unsupported_source_tissue_volume_mm3 == 0
    if hard:
        assert not cached.removed_mask.any()
    else:
        assert cached.removed_mask.any()


@pytest.mark.parametrize("step", [.5, .25, .1])
def test_cache_cannot_borrow_future_cut_for_short_tip_shaft_clearance(step):
    tissue = np.zeros((5, 5, 10), bool)
    tissue[2, 2, 5] = True
    tool = ToolGeometry("short-active-review", .9, .1, 10., tip_length_mm=.1)
    config = NativeResectionConfig(tissue, tissue.astype(np.int16), np.eye(4),
        AccessWindow((2, 2, 4.49), (0, 0, 1), 2.), (tool,), "synthetic-short-tip", "analytic single cell",
        max_tip_step_mm=step)
    original = NativeResectionEngine(config).preview_stroke(tool.tool_id, (2, 2, 5.49))
    engine = NativeResectionEngine(config)
    cache = ExactCapsuleCoverCache(config)
    with inject_native_capsule_cache(cache):
        for _ in range(2):
            candidate = engine.preview_stroke(tool.tool_id, (2, 2, 5.49))
            assert not candidate.feasible and candidate.reason == original.reason
            assert "SHAFT_BLOCKED" in candidate.reason
    assert not engine.removed_mask.any() and engine.history == []
    np.testing.assert_array_equal(engine.remaining_mask, tissue)
