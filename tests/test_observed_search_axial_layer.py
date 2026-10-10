"""Generated frame/retention controls only; no patient arrays or learned models."""
from dataclasses import dataclass
import os

import numpy as np
import pytest

from test_observed_search_opening_lane import Tree, clean, load

search = load(os.environ.get('OBSERVED_SEARCH_CANDIDATE', 'src/resectionlab/observed_search.py'),
              'axial_layer_candidate')
baseline = (load(os.environ['AXIAL_LAYER_BASELINE'], 'axial_layer_baseline')
            if 'AXIAL_LAYER_BASELINE' in os.environ else None)
MODE = 'return_plus_axial_layer_volume_v1'
OLD = 'return_plus_opening_depth_volume_v1'


@dataclass
class TiltedTree(Tree):
    tilt: float = 1e-10
    inward_sign: int = 1

    def observation(self):
        obs = super().observation()
        axes = self.affine[:3, :3] / np.linalg.norm(self.affine[:3, :3], axis=0)
        normal = self.inward_sign*axes[:, 2] + self.tilt*axes[:, 0]
        obs.state_features[6:9] = normal/np.linalg.norm(normal)
        if self.inward_sign < 0:
            obs.state_features[3:6] = self.affine[:3, :3] @ [0., 0., 7.] + self.affine[:3, 3]
        return obs


def projection_tree(**kwargs):
    return TiltedTree({(): {
        'cheap': (-.1, ((1, 1, 1),), False),
        'tiny': (-1., ((6, 1, 3),), False),
        'wide': (-2., ((1, 1, 3), (2, 1, 3), (3, 1, 3)), False)},
        ('cheap',): {'end': (-.1, (), True)},
        ('tiny',): {'end': (-.1, (), True)},
        ('wide',): {'reach': (5., ((1, 1, 4),), True)}}, **kwargs)


@pytest.mark.parametrize('transition', ['eager', 'lazy_planning'])
def test_transverse_projection_noise_defeats_old_volume_but_native_layer_restores_tie(transition):
    old, task = projection_tree(), projection_tree()
    options = dict(max_calls=30, beam_width=2, seconds=10, transition_mode=transition,
                   retained_prefix_diagnostics=True)
    old_path, old_report = search.observed_beam_search(old, **options, retention_mode=OLD)
    path, report = search.observed_beam_search(task, **options, retention_mode=MODE)
    assert old_path == ('STOP',) and path == ('wide', 'reach')
    old_opening = old_report['layers'][0]['opening_depth_retention'][1]
    opening = report['layers'][0]['opening_depth_retention'][1]
    assert old_opening['actions'] == ['tiny'] and opening['actions'] == ['wide']
    assert old_opening['opening_depth_mm'] > opening['opening_depth_mm']
    assert old_opening['opening_depth_mm'] - opening['opening_depth_mm'] < 1e-8
    assert opening['opening_axial_layer'] == 3 and opening['opening_removed_volume_mm3'] == 3.
    assert report['estimated_incremental_return'] == 3. and report['stop_baseline_return'] == 0.
    assert task.counters == old.counters and report['model_transition_calls'] == old_report['model_transition_calls'] == 5
    assert report['opening_depth_retention']['extra_observations_or_previews'] == 0
    assert report['opening_depth_retention']['pairwise_tolerance_or_bucket'] is False
    assert task.path == () and task.removed == ()


def test_next_integer_layer_always_beats_larger_previous_layer_volume():
    task = projection_tree()
    task.tree[()]['tiny'] = (-3., ((0, 1, 4),), False)
    _, report = search.observed_beam_search(task, max_calls=3, beam_width=2, retention_mode=MODE)
    opening = report['layers'][0]['opening_depth_retention'][1]
    assert opening['actions'] == ['tiny'] and opening['opening_axial_layer'] == 4
    assert opening['opening_removed_volume_mm3'] == 1.


def test_accumulated_layer_and_raw_depth_survive_later_shallower_widening():
    task = TiltedTree({(): {'seed': (-1., ((6, 1, 3),), False)},
        ('seed',): {'cheap': (-.1, (), False),
                   'tiny': (-1., ((5, 1, 3),), False),
                   'wide': (-2., ((1, 1, 2), (2, 1, 2), (3, 1, 2)), False)}}, max_steps=3)
    _, report = search.observed_beam_search(task, max_calls=4, beam_width=2, retention_mode=MODE)
    before = report['layers'][0]['opening_depth_retention'][0]
    after = report['layers'][1]['opening_depth_retention'][1]
    assert after['actions'] == ['seed', 'wide']
    assert after['opening_axial_layer'] == before['opening_axial_layer'] == 3
    assert after['opening_depth_mm'] == before['opening_depth_mm']
    assert after['opening_removed_volume_mm3'] == 4.


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('sign', [-1, 1])
def test_signed_indices_all_axes_physical_volume_reflection_and_zero_layer(axis, sign):
    task = projection_tree(); obs = task.observation(); outcome = task.planning_clone().step('wide')
    basis = np.diag([-2., 3., 4.]); affine = np.eye(4); affine[:3, :3] = basis; affine[:3, 3] = [10., -7., 5.]
    obs.affine_ras_mm = affine
    obs.state_features[3:6] = affine[:3, 3] + (basis[:, axis]*7 if sign < 0 else 0)
    obs.state_features[6:9] = sign*basis[:, axis]/np.linalg.norm(basis[:, axis])
    cells = np.array([[1, 1, 1], [2, 1, 1]], dtype=np.uint64)
    cells[:, axis] = [0, 2] if sign < 0 else [2, 3]
    outcome.info.update(native_affine=affine.tolist(), removed_indices_native=cells)
    depth, volume, layer = search._public_removed_opening_depth(obs, outcome, 'wide',
        with_volume=True, with_axial_layer=True)
    assert depth > 0 and volume == pytest.approx(48.)
    assert layer == (0 if sign < 0 else 3)  # Signed zero is a valid layer, never None.


def test_negative_inward_direction_plan_and_layer_order():
    task = projection_tree(inward_sign=-1)
    task.tree = {path: {action: (reward, tuple((x,y,7-z) for x,y,z in cells), terminal)
        for action,(reward,cells,terminal) in options.items()} for path,options in task.tree.items()}
    path, report = search.observed_beam_search(task, max_calls=30, beam_width=2, retention_mode=MODE)
    assert path == ('wide', 'reach')
    assert report['layers'][0]['opening_depth_retention'][1]['opening_axial_layer'] == -4


def test_translated_rotated_frame_preserves_layer_volume_and_selected_path():
    a = projection_tree()
    frame = np.array([[0.,0.,1.,30.], [1.,0.,0.,-20.], [0.,1.,0.,8.], [0.,0.,0.,1.]])
    b = projection_tree(affine=frame)
    pa, ra = search.observed_beam_search(a, max_calls=30, beam_width=2, retention_mode=MODE)
    pb, rb = search.observed_beam_search(b, max_calls=30, beam_width=2, retention_mode=MODE)
    assert pa == pb == ('wide', 'reach')
    x,y = (r['layers'][0]['opening_depth_retention'][1] for r in (ra,rb))
    assert x['opening_axial_layer'] == y['opening_axial_layer'] == 3
    assert x['opening_removed_volume_mm3'] == y['opening_removed_volume_mm3'] == 3.
    assert x['opening_depth_mm'] == pytest.approx(y['opening_depth_mm'], abs=1e-12)


@pytest.mark.parametrize('problem', ['nonaligned', 'transverse_mismatch', 'shear', 'residual_shear_extent',
                                    'normal_extent', 'anisotropic_layer_reversal', 'singular', 'nonfinite', 'bad_shape'])
def test_nonaligned_or_malformed_native_frame_refused_before_layer_credit(problem):
    task = projection_tree(); obs = task.observation(); result = task.planning_clone().step('wide')
    if problem == 'nonaligned': obs.state_features[6:9] = [0., .6, .8]
    if problem == 'transverse_mismatch': obs.state_features[6:9] = [2e-8, 0., np.sqrt(1.-4e-16)]
    if problem == 'shear': result.info['native_affine'][0][1] = 1e-4
    if problem == 'residual_shear_extent':
        result.info['native_affine'][2][1] = 5e-9
        result.info['source_shape'] = [8, 512, 8]
        obs.state_features[6:9] = [0.,0.,1.]
    if problem == 'normal_extent':
        obs.state_features[6:9] = [5e-9,0.,1.]
        result.info['source_shape'] = [512,8,8]
    if problem == 'anisotropic_layer_reversal': result.info['native_affine'][2][2] = 1e-12
    if problem == 'singular': result.info['native_affine'][0][0] = 0.
    if problem == 'nonfinite': result.info['native_affine'][0][0] = float('nan')
    if problem == 'bad_shape': result.info['source_shape'] = [8.,8.,8.]
    with pytest.raises(ValueError):
        search._public_removed_opening_depth(obs, result, 'wide', with_volume=True, with_axial_layer=True)


@pytest.mark.parametrize('condition', ['empty', 'uncovered', 'behind', 'prior_cavity'])
def test_only_eligible_actual_new_positive_depth_cells_have_layer_credit(condition):
    task = projection_tree(); obs = task.observation(); result = task.planning_clone().step('wide')
    if condition == 'empty': result.info['removed_indices_native'] = []
    if condition == 'uncovered': obs.coverage[1] = False
    if condition == 'behind': obs.state_features[3:6] = [0.,0.,7.]
    if condition == 'prior_cavity':
        obs.image_channels[3,1,1,3] = 1
        with pytest.raises(ValueError, match='previously observed cavity'):
            search._public_removed_opening_depth(obs,result,'wide',with_volume=True,with_axial_layer=True)
        return
    assert search._public_removed_opening_depth(obs,result,'wide',with_volume=True,with_axial_layer=True) == (None,0.,None)


def test_native_fixture_complete_geometry_and_final_true_objective_unchanged():
    from resectionlab.native_spatial_task import make_native_opening_task
    task = make_native_opening_task(max_steps=2)
    path, report = search.observed_beam_search(task, max_calls=32, beam_width=2, seconds=10,
        transition_mode='lazy_planning', retention_mode=MODE)
    assert len(path) == 2 and 'STOP' not in path and not report['call_cap_reached']
    for action in path: task.step(action)
    assert task.metrics()['total_reward'] == pytest.approx(report['estimated_incremental_return'])
    assert task.independent_geometry_check().feasible


@pytest.mark.parametrize('mode', ['return_only', 'return_plus_opening_depth_v1', OLD])
@pytest.mark.parametrize('transition', ['eager', 'lazy_planning'])
@pytest.mark.parametrize('cap', [0, 1, 3, 30])
@pytest.mark.skipif(baseline is None, reason='historical baseline supplied in author/root run')
def test_all_existing_modes_exact_path_full_accounting_and_callback_parity(mode,transition,cap):
    a,b = projection_tree(),projection_tree()
    options=dict(max_calls=cap,beam_width=2,seconds=10,transition_mode=transition,
                 retained_prefix_diagnostics=True,retention_mode=mode)
    pa,ra=baseline.observed_beam_search(a,**options)
    pb,rb=search.observed_beam_search(b,**options)
    assert pa==pb and clean(ra)==clean(rb) and a.counters==b.counters
