"""Independent source-ID/ownership checks for the static protection fast path."""
import copy
import gc
import pickle
import warnings
import weakref

import numpy as np
import pytest
from scipy.spatial import cKDTree

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_protection import (
    GuardConfig, ProtectionAnnotation, ProtectionSet,
    load_protection, save_protection,
)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.surgical_staged_skills import (
    SkillAction, SkillLimits, checked_actions, checked_cusa_motion,
)
from resectionlab.surgical_transition_meter import TransitionMeter


def source(guard_type=ProtectionSet, *, empty=False, affine=None):
    labels = np.zeros((96, 12, 16), np.uint16)
    labels[30, 6, 10] = 1
    scene = SurgicalScene(labels, np.eye(4) if affine is None else affine,
        [{'id': 1, 'name': 'generated target', 'material': 'tumor', 'target': True}],
        {'kind': 'GENERATED independent cache control'}, {'cusaV3': default_profile()})
    mask = np.zeros(scene.shape, bool)
    if not empty:
        mask[4:20, 6, 8] = True
        mask[64:80, 6, 8] = True
    annotation = ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
        provenance='GENERATED separated clusters, not patient anatomy')
    return guard_type(scene, [annotation], clearance_mm=2.)


def full_scan(guard, start, end, radius=0., *, positions=None):
    a, b = np.asarray(start, float), np.asarray(end, float)
    p = guard._points if positions is None else np.asarray(positions, float)
    if (a.shape != (3,) or b.shape != (3,) or p.shape != guard._points.shape or
            not np.isfinite([a, b]).all() or not np.isfinite(p).all() or
            not np.isfinite(radius) or radius < 0 or
            not np.isfinite(guard.clearance_mm) or guard.clearance_mm < 0):
        raise ValueError('invalid reference input')
    d = b-a
    norm = float(d@d)
    t = np.clip((p-a)@d/norm, 0., 1.) if norm else np.zeros(len(p))
    distance = np.linalg.norm(p-a-t[:, None]*d, axis=1)
    return guard.cells[distance <= radius+guard.clearance_mm+guard._radius+1e-9]


def query(guard):
    return guard.scene.points([np.ravel_multi_index((8, 6, 8), guard.scene.shape)])[0]


def same_or_refused(guard, point, expected):
    try:
        actual = guard.capsule_hits(point, point)
    except ValueError:
        return
    assert np.array_equal(np.sort(actual), np.sort(expected)), 'A private cache edit silently lost source exclusions'


@pytest.mark.parametrize('oblique', [False, True])
def test_static_ids_match_full_scan_including_closed_boundary(oblique):
    affine = np.array([[1., .2, 0., 3.], [.1, 1.2, .1, -2.], [0., .2, .8, 1.], [0., 0., 0., 1.]]) if oblique else None
    guard = source(affine=affine)
    point = query(guard)
    for a, b, radius in [(point, point, 0.), (point-[3., 0., 7.], point+[3., 0., 7.], .5),
                          (point+[30., 0., 0.], point+[40., 0., 0.], 1.5)]:
        assert np.array_equal(np.sort(guard.capsule_hits(a, b, radius)), np.sort(full_scan(guard, a, b, radius)))
    boundary = guard._points[0]-[guard.clearance_mm+guard._radius+1e-9, 0., 0.]
    for delta in [-1e-6, 0., 1e-6]:
        a = boundary+[delta, 0., 0.]
        assert np.array_equal(np.sort(guard.capsule_hits(a, a)), np.sort(full_scan(guard, a, a)))


@pytest.mark.parametrize('field', ['cells', '_points', '_index_indices', '_index_mins', '_index_maxes'])
def test_values_and_base_aliases_are_immutable(field):
    array = getattr(source(), field)
    while isinstance(array, np.ndarray):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.setflags(write=True)
        assert memoryview(array).readonly
        array = array.base
    assert isinstance(array, bytes)


@pytest.mark.parametrize('field', ['cells', '_points', '_index_indices', '_index_mins', '_index_maxes'])
def test_same_owner_array_replacement_refuses_static_and_dynamic(field):
    guard = source(); point = query(guard)
    setattr(guard, field, getattr(guard, field).copy())
    for positions in (None, np.zeros((32, 3))):
        with pytest.raises(ValueError): guard.capsule_hits(point, point, positions=positions)


@pytest.mark.parametrize('owner,field', [('guard', '_points'), ('guard', 'cells'),
    ('index', 'data'), ('index', 'indices'), ('index', 'mins'), ('index', 'maxes')])
@pytest.mark.parametrize('edit', ['shape', 'dtype', 'strides', 'buffer'])
def test_same_array_metadata_or_buffer_edit_does_not_inherit_cache(owner, field, edit):
    guard = source(); point = query(guard)
    array = getattr(guard if owner == 'guard' else guard._tree, field)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            if edit == 'shape': array.shape = (1, array.size)
            elif edit == 'dtype': array.dtype = np.uint8
            elif edit == 'strides': array.strides = (0,)*array.ndim
            else:
                array.__setstate__((1, array.shape, array.dtype, False, array.tobytes()))
                array.setflags(write=False)
    except (ValueError, AttributeError, TypeError):
        return  # NumPy itself refused the mutation.
    with pytest.raises(ValueError): guard.capsule_hits(point, point)
    with pytest.raises(ValueError): guard.capsule_hits(point, point, positions=np.zeros((32, 3)))


def test_unregistered_copy_foreign_tree_and_same_id_forged_proof_refuse():
    guard = source(); point = query(guard)
    for unregistered in (copy.copy(guard), pickle.loads(pickle.dumps(guard))):
        with pytest.raises(ValueError): unregistered.capsule_hits(point, point)
    detached_index = copy.copy(guard._tree)
    with pytest.raises(ValueError): detached_index.query_ball_point(point, 1.)
    guard._tree = cKDTree(guard._points.copy())
    with pytest.raises(ValueError): guard.capsule_hits(point, point)
    guard = source(); point = query(guard)
    guard._points = guard._points + 100.
    guard._geometry_proof = (guard.scene, guard._points, guard.cells)
    with pytest.raises(ValueError): guard.capsule_hits(point, point)


def test_private_checker_shadow_cannot_clear_exclusions():
    guard = source(); point = query(guard); expected = guard.capsule_hits(point, point)
    assert len(expected)
    guard._points = guard._points + 100.
    guard._require_geometry = lambda: None
    same_or_refused(guard, point, expected)


def test_index_query_shadow_cannot_clear_exclusions():
    guard = source(); point = query(guard); expected = guard.capsule_hits(point, point)
    assert len(expected)
    object.__setattr__(guard._tree, 'query_ball_point', lambda *_a, **_k: [])
    same_or_refused(guard, point, expected)



def test_index_metadata_checker_shadow_cannot_admit_foreign_inspection_data():
    guard = source(); point = query(guard)
    object.__setattr__(guard._tree, 'data', guard._points + 100.)
    object.__setattr__(guard._tree, '_validate_metadata', lambda: None)
    with pytest.raises(ValueError): guard.capsule_hits(point, point)
    with pytest.raises(ValueError): guard.capsule_hits(point, point, positions=guard._points)



@pytest.mark.parametrize('field', ['n', 'm', 'leafsize'])
def test_facade_scalar_metadata_replacement_refuses(field):
    guard = source(); point = query(guard)
    object.__setattr__(guard._tree, field, getattr(guard._tree, field)+1)
    with pytest.raises(ValueError): guard.capsule_hits(point, point)
    with pytest.raises(ValueError): guard.capsule_hits(point, point, positions=guard._points)


def test_index_inspection_does_not_expose_operating_backend():
    guard = source(); index = guard._tree
    assert not isinstance(index, cKDTree)
    assert not any(isinstance(value, cKDTree) for value in vars(index).values())
    for field in ('tree', 'raw', 'backend', '__getattr__'):
        assert not hasattr(index, field)
    assert not any(isinstance(getattr(value, '__self__', None), cKDTree) for value in vars(index).values())


@pytest.mark.parametrize('empty', [False, True])
def test_dynamic_full_validation_and_empty_argument_policy(empty):
    guard = source(empty=empty); point = np.array([8., 6., 8.])
    moved = guard._points + [10., 0., 0.]
    assert np.array_equal(np.sort(guard.capsule_hits(point, point, positions=moved)), np.sort(full_scan(guard, point, point, positions=moved)))
    if len(moved):
        moved[-1, -1] = np.nan
        with pytest.raises(ValueError): guard.capsule_hits(point, point, positions=moved)
    for bad in [np.nan, np.inf, -1.]:
        with pytest.raises(ValueError): guard.capsule_hits(point, point, bad)
        guard.clearance_mm = bad
        with pytest.raises(ValueError): guard.capsule_hits(point, point)
        guard.clearance_mm = 2.
    with pytest.raises(ValueError): guard.capsule_hits([0., 0.], point)
    with pytest.raises(ValueError): guard.capsule_hits(point, point, positions=np.empty((1, 3)))


def test_scene_ownership_and_geometry_replacements_refuse():
    for edit in ['scene', 'affine', 'labels', 'radius', 'identity']:
        guard = source(); point = query(guard)
        if edit == 'scene': guard.scene = copy.copy(guard.scene)
        elif edit == 'affine': guard.scene.affine = guard.scene.affine.copy()
        elif edit == 'labels': guard.scene.labels = guard.scene.labels.copy()
        elif edit == 'identity': guard.scene.identity += '-modified'
        else: guard._radius += .01
        with pytest.raises(ValueError): guard.capsule_hits(point, point)


def test_save_open_and_registry_lifetime(tmp_path):
    guard = source(); point = query(guard)
    save_protection(guard, tmp_path/'guard.json')
    opened = load_protection(guard.scene, tmp_path/'guard.json', strict=True)
    assert np.array_equal(np.sort(opened.capsule_hits(point, point)), np.sort(guard.capsule_hits(point, point)))
    index = guard._tree; guard_ref, index_ref = weakref.ref(guard), weakref.ref(index)
    del guard, index
    gc.collect()
    assert guard_ref() is None and index_ref() is None


class FullScanReference(ProtectionSet):
    def capsule_hits(self, start, end, radius_mm=0., *, positions=None):
        return full_scan(self, start, end, radius_mm, positions=positions)


def native_control(guard_type):
    guard = source(guard_type); engine = SurgicalEngine(guard.scene, protection=guard, guard_config=GuardConfig())
    original = engine.snapshot(); meter = TransitionMeter(80); limits = SkillLimits(20, 80)
    pose = lambda z: {'position': [30., 6., float(z)], 'quaternion': [0., 0., 0., 1.]}
    prepared, setup = checked_actions(engine, [SkillAction({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1}),
        SkillAction({'slot': 'left', 'operation': 'place', 'pose': pose(3.), 'duration': .1})], limits=limits, meter=meter)
    assert setup['status'] == 'accepted'
    work, receipt = checked_cusa_motion(prepared, SkillAction({'slot': 'left', 'operation': 'apply', 'pose': pose(12.), 'duration': .05}), limits=limits, meter=meter, max_contact_replans=8)
    assert receipt['status'] == 'accepted'
    exited, end = checked_actions(work, [SkillAction({'slot': 'left', 'operation': 'withdraw', 'pose': pose(3.), 'duration': 1.}, require_endpoint=True)], limits=limits, meter=meter)
    assert end['status'] == 'accepted' and exited.metrics()['targetRemovedMm3'] == 1.
    assert engine.snapshot() == original
    assert all(not row['active'] and not row['inserted'] for row in exited.state['slots'].values())
    return exited.snapshot(), meter.trace


def test_complete_native_preview_commit_history_matches_full_scan_reference():
    optimized, optimized_trace = native_control(ProtectionSet)
    reference, reference_trace = native_control(FullScanReference)
    assert optimized == reference
    assert optimized_trace == reference_trace
