"""Opt-in brain-shift scenario on the generated layered convexity case (no patient data)."""
import copy
import math

import numpy as np
import pytest
from scipy import ndimage

from scripts.generated_glioma_walkthrough_case import build_glioma_case
from resectionlab import surgical_brain_shift as shift
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import raster
from resectionlab.surgical_scene import SurgicalScene, propagate_original_target_memberships

FACES = np.zeros((3, 3, 3), int)
FACES[1, 1, 0] = FACES[1, 1, 2] = FACES[1, 0, 1] = FACES[1, 2, 1] = FACES[0, 1, 1] = FACES[2, 1, 1] = 1


def with_properties(base, **properties):
    return propagate_original_target_memberships(base, SurgicalScene(
        base.labels, base.affine, base.regions, base.source, {**base.properties, **properties}))


@pytest.fixture(scope='module')
def case():
    _, base, landmarks = build_glioma_case()
    declared = shift.default_property(base, settlingSeconds=1.0, settlingSteps=2)
    return base, with_properties(base, brainShiftV1=declared), landmarks


def open_access(engine):
    """Reflected scalp and dura with the bone flap parked: the state the opening reaches."""
    bodies = engine.state['operativeExposureV1']['bodies']
    bone = engine.scene.properties['operativeExposureV1']['bodies']['bone']
    bodies['scalp'].update(angleRad=math.pi / 2, status='supported')
    bodies['bone'].update(status='parked', translationMm=np.subtract(bone['parkCenterMm'], bone['restCenterMm']).tolist())
    bodies['dura'].update(angleRad=math.pi / 2, status='supported')


def inspect(duration):
    return {'slot': 'left', 'operation': 'inspect', 'duration': duration}


@pytest.fixture(scope='module')
def settled(case):
    _, scene, _ = case
    engine = SurgicalEngine(scene)
    open_access(engine)
    assert engine.step(inspect(1.5))['status'] == 'accepted'
    assert engine.state['brainShiftV1']['level'] == 2
    return engine


def column_owner(engine, point):
    ijk = np.rint(engine.scene.inverse[:3, :3] @ point + engine.scene.inverse[:3, 3]).astype(int)
    return engine.occupancy()[int(np.ravel_multi_index(ijk, engine.scene.shape))]


def test_no_property_keeps_state_view_and_raster_unchanged(case):
    base, _, _ = case
    engine = SurgicalEngine(base)
    open_access(engine)
    assert engine.step(inspect(1.5))['status'] == 'accepted'
    assert 'brainShiftV1' not in engine.state and engine._brain_shift is None
    assert 'brainShiftV1' not in engine.view() and 'brainShiftPeakMm' not in engine.metrics()
    engine.state['deformation'] = {str(int(np.flatnonzero(base.labels.ravel() == 8)[100])): [0.6, 0., 0.]}
    assert raster(engine) == raster(engine, brain_shift=False) == raster(engine, brain_shift='local')


def test_shift_starts_only_at_dural_opening_and_ramps_to_the_peak(case):
    _, scene, _ = case
    engine = SurgicalEngine(scene)
    assert engine.step(inspect(0.5))['status'] == 'accepted'
    assert 'brainShiftV1' not in engine.state and engine.metrics()['brainShiftPeakMm'] == 0.
    closed = engine.occupancy()
    open_access(engine)
    result = engine.step(inspect(0.3))
    assert result['events'][1]['kind'] == 'brain_shift_onset'
    assert engine.state['brainShiftV1']['onsetSeconds'] == pytest.approx(0.6)
    assert engine.state['brainShiftV1']['directionSource'] == 'opening-inward-normal'
    assert engine.state['brainShiftV1']['gravityRasUnit'] == pytest.approx([0., 0., -1.])
    assert engine.metrics()['brainShiftPeakMm'] == 0.
    peaks = []
    for _ in range(4):
        engine.step(inspect(0.3))
        peaks.append(engine.metrics()['brainShiftPeakMm'])
    assert peaks == [2.5, 2.5, 5.0, 5.0]  # onset 0.6 s; levels at +0.5 s and +1.0 s
    assert engine.view()['brainShiftV1']['currentPeakMm'] == 5.0
    assert closed is not engine.occupancy()


def test_shift_is_zero_at_bone_attached_dura_and_rigid_neighbours(case, settled):
    _, scene, _ = case
    geo = shift.geometry(scene, settled._brain_shift)
    labels = scene.labels.ravel()
    dense = np.asarray(settled.occupancy())
    fresh = SurgicalEngine(scene)
    open_access(fresh)
    unshifted = np.asarray(fresh.occupancy())
    rigid = np.flatnonzero(geo.obstacle_flat)
    # Every rigid or non-shifting raster cell keeps exactly the owner it has without the shift.
    assert np.array_equal(dense[rigid], unshifted[rigid])
    assert set(np.unique(labels[rigid])) >= {3, 4, 5, 6}
    # Shifting cells within one cell of fixed support do not move at all.
    support = geo.support[tuple((geo.ijk - geo.box_low).T)]
    pinned = support <= geo.cell_mm + 1e-6
    assert pinned.any() and np.all(geo.amplitude[pinned] == 0.)
    moving = shift.active_map(settled).sources
    assert not np.isin(geo.cells[pinned], moving).any()


def test_target_centroid_shift_matches_the_field_at_depth(case, settled):
    _, scene, _ = case
    current = shift.active_map(settled)
    geo = current.geometry
    view = settled.view()['brainShiftV1']
    planned, moved = np.asarray(view['plannedTargetRasMm']), np.asarray(view['shiftedTargetRasMm'])
    ijk = np.column_stack(np.unravel_index(geo.target_cells, scene.shape))
    expected = (geo.evaluate(ijk) * 5.0).mean() * np.array([0., 0., -1.])
    assert moved - planned == pytest.approx(expected, abs=0.05)
    assert settled.metrics()['targetShiftMm'] == pytest.approx(np.linalg.norm(expected), abs=0.05)
    # Peak at the exposed cortex under the opening, about half at the deep target margin.
    depth, lateral = geo._frame(scene.points(geo.target_cells))
    deepest = int(np.argmax(depth))
    assert geo.evaluate(ijk[deepest:deepest + 1])[0] == pytest.approx(0.5, abs=0.06)
    surface = geo.cells[np.argmax(geo.amplitude)]
    assert geo.amplitude.max() == pytest.approx(1.0, abs=1e-9)
    assert geo._frame(scene.points([surface]))[0][0] - geo.surface < geo.cell_mm + 1e-9
    assert len(view['latticeSamples']) <= 512 and view['latticeSamples']


def test_instrument_at_preoperative_cortex_finds_space_and_meets_cortex_peak_deeper(case, settled):
    _, scene, _ = case
    labels = scene.labels
    soft = np.isin(labels[62, 60, :], [7, 8, 9, 10])
    top = int(np.max(np.flatnonzero(soft)))
    cortex = scene.points([int(np.ravel_multi_index((62, 60, top), scene.shape))])[0]
    fresh = SurgicalEngine(scene)
    open_access(fresh)
    assert fresh.touched(cortex, cortex, 0.4)
    assert not settled.touched(cortex, cortex, 0.4)
    depth = 0.
    while not settled.touched(cortex - [0, 0, depth], cortex - [0, 0, depth], 0.4):
        depth += 0.25
        assert depth < 20
    current = shift.active_map(settled)
    expected = 5.0 * current.geometry.evaluate(np.array([[62, 60, top]]))[0]
    assert expected > 4.0 and depth == pytest.approx(expected, abs=1.0)
    # Collision and removal name the original source cells, from where the image shows them.
    found = settled.touched(cortex - [0, 0, depth], cortex - [0, 0, depth], 0.4)
    origins = scene.points(found)
    assert np.all(origins[:, 2] - (cortex[2] - depth) > 3.0)
    removed = settled.clone()
    removed.state['removed'] = sorted(found)
    assert not removed.touched(cortex - [0, 0, depth], cortex - [0, 0, depth], 0.4)
    assert removed.metrics()['removedByLabelMm3']


def test_destination_map_is_a_function_without_rigid_overlap_or_one_cell_holes(case, settled):
    _, scene, _ = case
    current = shift.active_map(settled)
    geo = current.geometry
    labels = scene.labels.ravel()
    assert len(np.unique(current.sources)) == len(current.sources) and len(current.sources) > 1000
    assert not geo.obstacle_flat[current.targets].any()
    dense = np.asarray(settled.occupancy())
    owner = dense - 1
    occupied = dense != 0
    assert not (occupied & geo.obstacle_flat & geo.domain_flat[np.where(occupied, owner, 0)]).any()
    # Compressed cells keep every source; the representative follows region precedence.
    rank = {r['id']: i for i, r in enumerate(scene.regions)}
    for location, owners in list(current.members.items())[:500]:
        assert len(set(owners)) == len(owners) > 1
        assert rank[int(labels[owners[0]])] == max(rank[int(labels[c])] for c in owners)
        assert set(settled.occupancy().material_ids([location])) == set(owners)
    fresh = SurgicalEngine(scene)
    open_access(fresh)
    def isolated(values):
        empty = (values == 0).reshape(scene.shape)
        return empty & (ndimage.convolve((~empty).astype(int), FACES, mode='constant') == 6)
    new_holes = isolated(dense) & ~isolated(np.asarray(fresh.occupancy()))
    assert not new_holes.reshape(-1)[geo.domain_flat | np.isin(np.arange(labels.size), current.sources)].any()
    # The CSF gap opens on top: the vacated surface under the opening is free space.
    column = dense.reshape(scene.shape)[62, 60, :]
    soft = np.flatnonzero(np.isin(labels.reshape(scene.shape)[62, 60, :], [7, 8, 9, 10]))
    assert np.all(column[soft.max() - 3: soft.max() + 1] == 0)
    # Brain moves only into shifting tissue or free space.
    assert set(np.unique(labels[current.targets])) <= {0, *settled._brain_shift['regionIds']}


def test_full_raster_is_the_forward_map_and_the_guard_field_is_untouched(settled):
    current = shift.active_map(settled)
    pairs = raster(settled)
    assert pairs == [[int(a), int(b)] for a, b in zip(current.sources, current.targets)]
    assert raster(settled, brain_shift=False) == [] == raster(settled, brain_shift='local')
    # The protected-displacement guard reads only the local lattice field.
    assert not settled.state.get('deformation')


def test_replay_snapshot_and_restore_reproduce_shift_state_ids(case, monkeypatch):
    _, scene, _ = case
    # Open access by time so that a fresh replay reaches the same onset.
    monkeypatch.setattr(shift, 'access_open', lambda engine: engine.state['elapsedSeconds'] >= 0.45)
    actions = [inspect(0.3), inspect(0.4), inspect(0.6), inspect(0.5)]
    first = SurgicalEngine(scene)
    ids = [first.step(action)['stateId'] for action in actions]
    assert first.state['brainShiftV1']['onsetSeconds'] == pytest.approx(0.5)
    assert first.state['brainShiftV1']['level'] == 2
    occupancy = np.asarray(first.occupancy())
    scene.__dict__.pop('_brain_shift_cache', None)
    second = SurgicalEngine(scene)
    assert [second.step(action)['stateId'] for action in actions] == ids
    assert np.array_equal(np.asarray(second.occupancy()), occupancy)
    assert first.replay(first.actions).state_id == first.state_id
    restored = SurgicalEngine.restore(scene, first.snapshot(), use_cache=False, force_verify=True)
    assert restored.state_id == first.state_id
    assert np.array_equal(np.asarray(restored.occupancy()), occupancy)


def test_declaration_is_validated(case):
    base, scene, _ = case
    declared = copy.deepcopy(scene.properties['brainShiftV1'])
    for change in ({'peakSagMm': -1.}, {'onset': 'skin-incision'}, {'regionIds': [3]},
                   {'provenance': {'evidenceClass': 'measured'}}, {'gravityRasUnit': [0., 0., 2.]}):
        with pytest.raises(ValueError, match='BRAIN_SHIFT_INVALID'):
            SurgicalEngine(with_properties(base, brainShiftV1={**declared, **change}))
    assert declared['provenance']['evidenceClass'] == 'assumed-research'
    assert {s['id'] for s in declared['provenance']['sources']} == {'roberts-1998', 'nimsky-2000', 'nabavi-2001'}
