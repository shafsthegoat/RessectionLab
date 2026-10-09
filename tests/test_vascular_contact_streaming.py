"""Small generated reference grids; no patient/model payloads or full profile."""
from dataclasses import replace
import numpy as np
import pytest

from resectionlab import vascular_contact_streaming as s
from resectionlab.evaluation import segment_box_distance_sq


def fixture(angle=0., spacing=(.7, 1.1, 1.6), translation=(3., -5., 1.)):
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    affine = np.eye(4); affine[:3, :3] = rotation@np.diag(spacing); affine[:3, 3] = translation
    grid = s.Grid((7, 6, 5), affine)
    def capsule(action, part, start, end, radius):
        return s.Capsule(action, part, rotation@start+affine[:3, 3], rotation@end+affine[:3, 3], radius)
    a = np.array([.7, 1.3, -1.]); b = np.array([3., 3., 5.])
    caps = (capsule('first', 'shaft', a, b, .4), capsule('first', 'tip', a+[.2,.1,.7], b, .7),
            capsule('second', 'shaft', a, b, .4), capsule('second', 'tip', a+[.3,.2,1.], b+[.4,.4,0.], .7))
    indices = np.indices(grid.shape)
    coverage = indices[0] < 4
    positive = (indices[1] % 2 == 0) & coverage
    def sample(rows):
        return positive[tuple(rows.T)], coverage[tuple(rows.T)]
    return grid, caps, sample


def scalar_oracle(grid, capsules, sampler):
    affine = np.asarray(grid.affine_ras_mm)
    spacing = np.linalg.norm(affine[:3, :3], axis=0); rotation = affine[:3, :3]/spacing
    unions = {name:set() for name in ('shaft', 'tip', 'whole_tool')}
    per_action = {c.action_id:set() for c in capsules}; outside = {}
    for c in capsules:
        a, b = [rotation.T@(np.asarray(v)-affine[:3, 3]) for v in (c.start_ras_mm, c.end_ras_mm)]
        cells = {cell for cell in np.ndindex(grid.shape)
                 if segment_box_distance_sq(a, b, np.asarray(cell)*spacing-spacing/2,
                     np.asarray(cell)*spacing+spacing/2) <= c.radius_mm**2+1e-10}
        unions[c.part] |= cells; unions['whole_tool'] |= cells; per_action[c.action_id] |= cells
        out = bool(np.any(np.minimum(a,b)-c.radius_mm < -.5*spacing)
                   or np.any(np.maximum(a,b)+c.radius_mm > (np.asarray(grid.shape)-.5)*spacing))
        outside[c.part] = outside.get(c.part,False) or out
        outside[c.action_id] = outside.get(c.action_id,False) or out
        outside['whole_tool'] = outside.get('whole_tool',False) or out
    def record(name, cells):
        selected = np.asarray(sorted(cells), dtype=np.int64).reshape(-1,3)
        positive, known = sampler(selected)
        return s._count_record((len(cells), int(positive.sum()), int((~known).sum())),
                               outside.get(name,False), abs(np.linalg.det(affine[:3,:3])))
    return {k:record(k,v) for k,v in unions.items()}, {k:record(k,v) for k,v in per_action.items()}


@pytest.mark.parametrize('angle', [0., .37, np.pi/2])
@pytest.mark.parametrize('edge', [3, 4, 16])
def test_tiled_counts_match_full_grid_scalar_oracle(angle, edge):
    grid, caps, sample = fixture(angle)
    expected, actions = scalar_oracle(grid,caps,sample)
    result = s.evaluate_contacts(grid,caps,sample_reference=sample,budget=s.Budget(tile_edge=edge,coarse_batch=7))
    assert {key:result[key] for key in expected} == expected
    assert result['per_action'] == actions
    assert result['work']['maximum_tile_cells'] <= edge**3


def test_duplicate_capsules_preserve_strategy_and_action_unions():
    grid,caps,sample = fixture()
    first = s.evaluate_contacts(grid,caps[:2],sample_reference=sample)
    duplicate = s.evaluate_contacts(grid,caps[:2]+caps[:2],sample_reference=sample)
    repeated = s.evaluate_contacts(grid,caps[:2]+tuple(replace(c,action_id='repeat') for c in caps[:2]),sample_reference=sample)
    for key in ('shaft','tip','whole_tool'):
        assert first[key] == duplicate[key] == repeated[key]
    assert repeated['per_action']['first'] == repeated['per_action']['repeat']


@pytest.mark.parametrize('spacing', [.01, 100.])
def test_previously_admitted_spacing_and_large_translation_remain_supported(spacing):
    grid,caps,sample = fixture(spacing=(spacing,)*3,translation=(1e6,-2e6,3e6))
    expected,actions = scalar_oracle(grid,caps,sample)
    result = s.evaluate_contacts(grid,caps,sample_reference=sample)
    assert {key:result[key] for key in expected} == expected
    assert result['per_action'] == actions


@pytest.mark.parametrize('offset', [0., .5e-5, 1e-5, 1.1e-5])
def test_tolerance_contacts_cross_tile_faces(offset):
    grid = s.Grid((8,8,8),np.eye(4))
    caps = (s.Capsule('tangent','shaft',(3.5+offset,1.,1.),(3.5+offset,6.,6.),0.),)
    sample = lambda rows:(np.ones(len(rows),bool),np.ones(len(rows),bool))
    expected,_ = scalar_oracle(grid,caps,sample)
    result = s.evaluate_contacts(grid,caps,sample_reference=sample,budget=s.Budget(tile_edge=4))
    assert result['whole_tool'] == expected['whole_tool']


@pytest.mark.parametrize('kind', ['float','length','positive_outside_coverage','not_tuple'])
def test_sampler_contract_refuses_bad_labels(kind):
    grid,caps,_ = fixture()
    def sample(rows):
        a,b = np.ones(len(rows),bool),np.ones(len(rows),bool)
        if kind == 'float': return a.astype(float),b
        if kind == 'length': return np.ones(len(rows)+1,bool),b
        if kind == 'positive_outside_coverage': return a,~b
        return [a,b]
    with pytest.raises(ValueError): s.evaluate_contacts(grid,caps,sample_reference=sample)


def test_work_refusal_precedes_reference_sampler():
    grid,caps,_ = fixture()
    with pytest.raises(s.BudgetExceeded) as error:
        s.evaluate_contacts(grid,caps,sample_reference=lambda rows:pytest.fail('sampler called'),
                            budget=s.Budget(max_sampled_cells=0))
    assert error.value.outcomes is None and error.value.completed is False


def test_stop_and_unknown_outside_fov_keep_null_claims():
    grid,caps,_ = fixture()
    stop = s.evaluate_contacts(grid,(),sample_reference=lambda rows:pytest.fail('STOP sampled'))
    assert stop['whole_tool']['touched_reference_cells'] == 0
    result = s.evaluate_contacts(grid,caps,sample_reference=lambda rows:(np.zeros(len(rows),bool),np.zeros(len(rows),bool)))
    assert result['whole_tool']['annotated_positive_encounter'] is None
    assert result['whole_tool']['biological_vessel_free'] is result['whole_tool']['clinical_injury_probability'] is None
