"""Generated graph arithmetic plus one native fixture; no patient/model reads."""
from dataclasses import dataclass, field, replace
import importlib.util
import os
from types import SimpleNamespace
import weakref

import numpy as np
import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


search = load(os.environ.get('OBSERVED_SEARCH_CANDIDATE', 'src/resectionlab/observed_search.py'), 'opening_search_candidate')
baseline = (load(os.environ['OBSERVED_SEARCH_BASELINE'], 'opening_search_baseline')
            if 'OBSERVED_SEARCH_BASELINE' in os.environ else None)
MODE = 'return_plus_opening_depth_v1'


@dataclass
class Tree:
    # value: (reward, newly removed cells, terminal)
    tree: dict
    max_steps: int = 2
    path: tuple = ()
    removed: tuple = ()
    planning: bool = False
    terminated: bool = False
    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    counters: dict = field(default_factory=lambda: dict(metrics=0, observations=0, clones=0, transitions=0))
    scope: str = 'permitted_nominal_model'
    absent_record: bool = False
    failure: bool = False

    @property
    def private_reference(self):
        raise AssertionError('No private reference query allowed')

    def planning_clone(self):
        return replace(self, planning=True)

    def clone(self):
        self.counters['clones'] += 1
        return replace(self)

    def metrics(self):
        self.counters['metrics'] += 1
        assert self.counters['metrics'] == 1, 'No extra metrics/history query'
        return dict(planning_estimator_only=self.planning, steps=len(self.path))

    def observation(self):
        self.counters['observations'] += 1
        ids = ('STOP', *self.tree.get(self.path, {}))
        images = np.zeros((6,8,8,8), np.float32); images[1] = 1
        for cell in self.removed:
            images[(3,*cell)] = 1
        return SimpleNamespace(action_ids=ids, action_mask=np.ones(len(ids),bool), source_id='public-source',
            assert_intact=lambda: None, image_channels=images, coverage=np.ones(images.shape,bool),
            channel_available=np.ones(6,bool), affine_ras_mm=self.affine,
            state_features=np.asarray([len(self.path),self.max_steps,0,*self.affine[:3,3],*self.affine[:3,2],3.]))

    def _advance(self, action, eager):
        self.counters['transitions'] += 1
        if self.failure: raise InterruptedError('Generated transition refusal')
        reward, removed, terminal = self.tree[self.path][action]
        self.path = (*self.path,action); self.removed = (*self.removed,*removed)
        self.terminated = terminal or len(self.path) == self.max_steps
        info = None if self.absent_record else dict(action_id=action, outcome_scope=self.scope,
            source_hash='public-source', native_affine=self.affine.tolist(), source_shape=[8,8,8],
            removed_indices_native=removed, insertion_distance_mm=1000., target_removed_mm3=0.,
            normal_removed_mm3=float(len(removed)))
        if eager: self.observation()
        return SimpleNamespace(reward=reward, terminated=self.terminated, info=info)

    def step(self, action): return self._advance(action, True)
    def advance_planning(self, action): return self._advance(action, False)


def opening_tree(**kwargs):
    return Tree({(): {'cheap1':(-.1,((1,1,1),),False), 'cheap2':(-.2,((2,1,1),),False),
                     'open':(-2.,((1,1,3),),False)},
        ('cheap1',): {'finish':(-.1,(),True)}, ('cheap2',): {'finish':(-.1,(),True)},
        ('open',): {'reach':(5.,((1,1,4),),True)}}, **kwargs)


def clean(report):
    return {k:v for k,v in report.items() if k != 'planning_seconds'}


@pytest.mark.parametrize('mode', ['eager','lazy_planning'])
@pytest.mark.parametrize('cap', [0,1,3,5,30])
@pytest.mark.skipif(baseline is None, reason='historical source supplied during staged review')
def test_default_exact_baseline_path_counters_outputs(mode,cap):
    a,b=opening_tree(),opening_tree()
    options=dict(max_calls=cap,beam_width=2,seconds=10,transition_mode=mode,retained_prefix_diagnostics=True)
    pa,ra=baseline.observed_beam_search(a,**options)
    pb,rb=search.observed_beam_search(b,**options)
    assert pa==pb and clean(ra)==clean(rb) and a.counters==b.counters
    assert 'opening_depth_retention' not in rb


@pytest.mark.parametrize('mode', ['eager','lazy_planning'])
def test_costly_opening_survives_and_terminal_return_is_unchanged(mode):
    old=opening_tree(); task=opening_tree()
    options=dict(max_calls=30,beam_width=2,seconds=10,transition_mode=mode)
    old_path,old_stats=search.observed_beam_search(old,**options)
    path,stats=search.observed_beam_search(task,**options,retention_mode=MODE,retained_prefix_diagnostics=True)
    assert old_path==('STOP',) and path==('open','reach')
    assert stats['estimated_incremental_return']==3. and stats['stop_baseline_return']==0.
    assert stats['model_transition_calls']==old_stats['model_transition_calls']==5
    assert task.counters==old.counters and task.path==() and not task.removed
    rows=stats['layers'][0]['opening_depth_retention']
    assert [(r['actions'],r['reason'],r['opening_depth_mm']) for r in rows]==[
        (['cheap1'],'best_return',1.),(['open'],'opening',3.)]
    assert stats['peak_next_layer_states']<=4
    assert stats['opening_depth_retention']['extra_observations_or_previews']==0


def test_all_negative_still_returns_stop_even_if_deepest():
    task=opening_tree(); task.tree[('open',)]={'reach':(-1.,((1,1,4),),True)}
    path,stats=search.observed_beam_search(task,max_calls=30,beam_width=2,retention_mode=MODE)
    assert path==('STOP',) and stats['estimated_incremental_return']==0


def test_no_new_removal_earns_no_depth_despite_far_tip_travel():
    task=opening_tree(); task.tree[()]['open']=(-2.,(),False)
    path,stats=search.observed_beam_search(task,max_calls=30,beam_width=2,retention_mode=MODE)
    assert path==('STOP',)
    assert [r['actions'] for r in stats['layers'][0]['opening_depth_retention']]==[['cheap1'],['cheap2']]


def test_same_return_and_opening_winner_fills_distinct_second_slot():
    task=opening_tree(); task.tree[()]['cheap1']=(-.1,((1,1,6),),False)
    _,stats=search.observed_beam_search(task,max_calls=3,beam_width=2,retention_mode=MODE)
    rows=stats['layers'][0]['opening_depth_retention']
    assert [r['actions'] for r in rows]==[['cheap1'],['cheap2']]
    assert rows[0]['reason']=='best_return_and_opening' and rows[1]['reason']=='return_fill'


def test_translated_rotated_ras_equivalent_depth_and_plan():
    transform=np.array([[0.,0.,1.,10.],[1.,0.,0.,-8.],[0.,1.,0.,5.],[0.,0.,0.,1.]])
    a,b=opening_tree(),opening_tree(affine=transform)
    pa,ra=search.observed_beam_search(a,max_calls=30,beam_width=2,retention_mode=MODE)
    pb,rb=search.observed_beam_search(b,max_calls=30,beam_width=2,retention_mode=MODE)
    assert pa==pb and clean(ra)==clean(rb)


def test_uncovered_or_outside_actor_crop_is_not_opening_progress():
    task=opening_tree(); obs=task.observation(); result=task.planning_clone().step('open')
    obs.coverage[1,1,1,3]=False
    assert search._public_removed_opening_depth(obs,result,'open') is None
    obs.coverage[:]=True; obs.affine_ras_mm=np.eye(4); obs.affine_ras_mm[0,3]=20.
    assert search._public_removed_opening_depth(obs,result,'open') is None


def test_native_anisotropic_grid_maps_into_offset_actor_crop():
    obs=opening_tree().observation()
    native=np.diag([2.,3.,4.,1.]); native[:3,3]=[10.,-7.,5.]
    crop=native.copy(); crop[:3,3]=native[:3,3]+native[:3,:3]@np.array([2.,1.,1.])
    obs.affine_ras_mm=crop
    obs.image_channels=np.zeros((6,3,3,3),np.float32); obs.image_channels[1]=1
    obs.coverage=np.ones((6,3,3,3),bool)
    obs.state_features=np.array([0.,2.,0.,10.,-7.,5.,0.,0.,1.,3.])
    record=dict(action_id='cut',source_hash=obs.source_id,outcome_scope='permitted_nominal_model',
        native_affine=native.tolist(),source_shape=[8,8,8],
        removed_indices_native=[[3,2,2],[0,0,7]])
    # Native [3,2,2] -> crop [1,1,1], RAS z13, aperture z5: depth8.
    # The deeper cell is outside the actor crop and must not outrank it.
    result=SimpleNamespace(info=record)
    assert search._public_removed_opening_depth(obs,result,'cut')==8.
    obs.coverage[3,1,1,1]=False
    assert search._public_removed_opening_depth(obs,result,'cut') is None


def test_actual_weakref_child_population_matches_reported_memory_bound():
    @dataclass
    class LiveTree(Tree):
        live: dict = field(default_factory=lambda:dict(refs=[],max_children=0))
        def __post_init__(self):
            self.live['refs'].append(weakref.ref(self))
        def _advance(self,action,eager):
            outcome=super()._advance(action,eager)
            # Observe real live cloned tasks; do not use the search's counter.
            depth=len(self.path)
            count=sum(1 for ref in self.live['refs'] if ref() is not None
                      and ref().planning and len(ref().path)==depth)
            self.live['max_children']=max(self.live['max_children'],count)
            return outcome
    root={'deep':(-5.,((1,1,6),),False),
          **{'cheap'+str(i):(-.1*i,((i+1,1,1),),False) for i in range(1,6)}}
    tree={():root}
    for action in root:
        tree[(action,) ]={'continue':(-.1,((1,2,7),),False)}
        tree[(action,'continue')]={'finish':(-.1,(),True)}
    task=LiveTree(tree,max_steps=3)
    _,stats=search.observed_beam_search(task,max_calls=50,beam_width=2,
        transition_mode='lazy_planning',retention_mode=MODE,retained_prefix_diagnostics=True)
    assert task.live['max_children']==stats['peak_next_layer_states']==4
    assert all(len(layer['opening_depth_retention'])<=2 for layer in stats['layers'])
    assert all(ref() is None or ref() is task for ref in task.live['refs'])


def test_repeated_cavity_cells_cannot_manufacture_progress():
    task=opening_tree(); obs=task.observation(); obs.image_channels[3,1,1,3]=1
    with pytest.raises(ValueError,match='previously observed'):
        search._public_removed_opening_depth(obs,task.planning_clone().step('open'),'open')


@pytest.mark.parametrize('kwargs',[dict(scope='separate_evaluator_reference'),dict(absent_record=True)])
def test_missing_or_private_progress_refuses_with_partial_accounting(kwargs):
    task=opening_tree(**kwargs)
    with pytest.raises(ValueError,match='exact public nominal') as error:
        search.observed_beam_search(task,max_calls=30,beam_width=2,retention_mode=MODE)
    assert error.value.accounting['model_transition_calls']==1
    assert error.value.accounting['completed_layers']==0


def test_reject_private_scope_before_reading_payload():
    class Private(dict):
        def get(self,key,default=None):
            assert key=='outcome_scope'
            return 'separate_evaluator_reference'
    with pytest.raises(ValueError):
        search._public_removed_opening_depth(None,SimpleNamespace(info=Private()),'x')


def test_optin_width_and_mode_refuse_before_queries():
    task=opening_tree()
    for options in (dict(beam_width=1,retention_mode=MODE),dict(beam_width=2,retention_mode='wrong')):
        with pytest.raises(ValueError): search.observed_beam_search(task,max_calls=20,**options)
    assert not any(task.counters.values())


def test_partial_cap_and_transition_failure_stay_unresolved():
    _,stats=search.observed_beam_search(opening_tree(),max_calls=1,beam_width=2,retention_mode=MODE)
    assert stats['call_cap_reached'] and stats['completed_layers']==0
    assert stats['layers'][0]['negative_prefixes_pending_at_cap']==1
    with pytest.raises(InterruptedError) as error:
        search.observed_beam_search(opening_tree(failure=True),max_calls=20,beam_width=2,retention_mode=MODE)
    assert error.value.accounting['model_transition_calls']==1
    assert error.value.accounting['evaluated_transition_prefixes']==0


def test_progress_arithmetic_counts_toward_same_wall_budget(monkeypatch):
    ticks=[0.]; original=search._public_removed_opening_depth
    monkeypatch.setattr(search,'time',SimpleNamespace(perf_counter=lambda:ticks[0]))
    def slow(*args):
        result=original(*args); ticks[0]+=2.; return result
    monkeypatch.setattr(search,'_public_removed_opening_depth',slow)
    with pytest.raises(search.ObservedSearchLimit) as error:
        search.observed_beam_search(opening_tree(),max_calls=30,beam_width=2,seconds=1.,retention_mode=MODE)
    assert error.value.accounting['time_cap_reached'] and error.value.accounting['model_transition_calls']==1


def test_real_generated_native_records_and_exact_replay():
    from resectionlab.native_spatial_task import make_native_opening_task
    task=make_native_opening_task(max_steps=2)
    path,stats=search.observed_beam_search(task,max_calls=32,beam_width=2,seconds=10.,
        transition_mode='lazy_planning',retention_mode=MODE)
    assert len(path)==2 and 'STOP' not in path and not stats['call_cap_reached']
    first=task.planning_clone(); second=task.planning_clone()
    for action in path:
        a,b=first.step(action),second.step(action)
        assert a.reward==b.reward and a.info==b.info
    assert first.terminated and second.terminated
    assert first.metrics()['target_removed_mm3']>0 and first.metrics()['total_reward']>0
    assert first.independent_geometry_check().feasible
    assert stats['opening_depth_retention']['terminal_objective_changed'] is False
