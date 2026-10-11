"""Portable generated-only fallback controls; no acquired inputs or fitted weights."""
import json
import hashlib
from dataclasses import replace
from pathlib import Path
import copy
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'tests'))
from hybrid_proposal import Costs,GeneratedSparseTask,RidgePriority,FEATURES
from fallback_policy import GeneratedFallbackTask,VERSION
import batch_contract
import patient_transfer
from batch_contract import RIDGE,template
from resectionlab.geometry import AccessWindow,ToolGeometry
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase,NativeSpatialTask


def fixture(**changes):
    support=np.zeros((13,13,10),bool);support[4:9,4:9,2:4]=True
    case=NativeSpatialCase(np.where(support,.8,0.).astype(np.float32),support,support,np.eye(4),
        AccessWindow((6.,6.,1.5),(0.,0.,1.),4.,'generated-fallback'),
        (ToolGeometry('small',.9,.5,8.,45.,.8),ToolGeometry('large',1.3,.7,8.,45.,1.5)),
        track='synthetic_scan',support_source_kind='derived_from_scan',support_derivation='invented slab',
        nominal_target=support,target_source_kind='derived_from_scan',target_derivation='invented slab',
        proposal_mode='nominal_cavity_v1',proposal_config=NominalCavityProposalConfig(
            offsets_source_voxels=tuple((a,b) for a in (-1,0,1) for b in (-1,0,1)),
            max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True))
    return replace(case,**changes) if changes else case


def ranker():return RidgePriority(np.zeros(FEATURES),np.ones(FEATURES),np.zeros(FEATURES),0.)


def order_fixture(monkeypatch,case,positive_first=False):
    # Construct an adverse ranking from an explicitly generated full catalog.
    # Production never receives this fixture oracle; it uses frozen parameters.
    full=GeneratedSparseTask(case,mode='exhaustive',max_steps=2)
    _,accounting=full.observed_one_step_search(seconds=5)
    scores={row['action_id']:row['reward'] for row in accounting['decisions'][0]['scores']}
    ids=full._shortlist_record['raw_ids']
    positive=[v for v in ids if scores.get(v,0)>0]
    nonpositive=[v for v in ids if scores.get(v,0)<=0]
    assert len(nonpositive)>=8 and positive
    ordered=(positive+nonpositive) if positive_first else (nonpositive+positive)
    priorities={v:float(len(ordered)-i) for i,v in enumerate(ordered)}
    monkeypatch.setattr(RidgePriority,'score',lambda self,rows:np.array([priorities.get(v,0.) for v in rows.ids]))
    return full,scores


def test_rescues_generated_omitted_positive_once_with_full_replay(monkeypatch):
    case=fixture();full,scores=order_fixture(monkeypatch,case)
    events=[];costs=Costs(inventory_callback=events.append)
    sparse=GeneratedSparseTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2)
    assert sparse.observed_one_step_search(seconds=5)[0]==('STOP',)
    task=GeneratedFallbackTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2,costs=costs)
    record=task.candidate_inventory()['shortlist']
    assert record['fallback_triggered'] and record['fallback_completed']
    assert len(record['initial_selected_ids'])==8 and record['queried_count']==record['raw_count']>8
    assert len(set(record['selected_ids']))==record['raw_count']
    assert costs.counts['native_previews']==record['raw_count']
    assert costs.counts['fallback_additional_previews']==record['raw_count']-8
    assert len(events)==1 and events[0]['shortlist']==record
    expected=full.observed_one_step_search(seconds=5)[0]
    actual,accounting=task.observed_one_step_search(seconds=5)
    assert actual==expected and actual[0]!='STOP' and accounting['method']==VERSION
    replay=task.fresh()
    for action in actual:replay.step(action)
    assert replay.terminated and replay.independent_geometry_check().feasible
    assert replay.metrics()['target_removed_mm3']>0


def test_positive_shortlist_does_not_expand_even_if_better_is_omitted(monkeypatch):
    case=fixture();full,scores=order_fixture(monkeypatch,case,positive_first=True)
    task=GeneratedFallbackTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2)
    baseline=GeneratedSparseTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2)
    assert not task._shortlist_record['fallback_triggered']
    assert task.costs.counts['native_previews']==8
    assert task._shortlist_record['selected_ids']==baseline._shortlist_record['selected_ids']
    assert task.observed_one_step_search(seconds=5)[0]==baseline.observed_one_step_search(seconds=5)[0]


def test_private_reference_cannot_change_fallback_catalog_mask_or_nominal_decision(monkeypatch):
    case=fixture();order_fixture(monkeypatch,case)
    a=GeneratedFallbackTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2)
    b=GeneratedFallbackTask(replace(case,reference_target=np.zeros(case.reference_target.shape)),mode='ridge_top8',ranker=ranker(),max_steps=2)
    assert a.candidate_inventory()==b.candidate_inventory()
    assert a.observation().fingerprint==b.observation().fingerprint
    aa,ac=a.observed_one_step_search(seconds=5);ba,bc=b.observed_one_step_search(seconds=5)
    assert aa==ba and ac['decisions']==bc['decisions'] and a.costs.counts==b.costs.counts


def test_cached_observation_and_clone_do_not_requery_or_relog(monkeypatch):
    case=fixture();order_fixture(monkeypatch,case)
    events=[];task=GeneratedFallbackTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2,costs=Costs(inventory_callback=events.append))
    counts=task.costs.counts.copy()
    for _ in range(3):task.observation();task.candidate_inventory();task.planning_clone().observation()
    assert task.costs.counts==counts and len(events)==1
    object.__setattr__(task._ranker,'intercept',1.)
    with pytest.raises(RuntimeError,match='changed'):task.observation()


def test_nonpositive_full_catalog_keeps_STOP_and_original_reward_ties(monkeypatch):
    case=fixture(nominal_target=np.zeros((13,13,10),bool))
    task=GeneratedFallbackTask(case,mode='ridge_top8',ranker=ranker(),max_steps=2)
    assert task._shortlist_record['fallback_triggered']
    assert task._shortlist_record['queried_count']==task._shortlist_record['raw_count']
    assert task.observed_greedy_search(seconds=5)[0]==('STOP',)
    # Emitted proposal ordering is retained even though initial score ties select top8.
    assert list(task._inventory)==[v for v in task._shortlist_record['raw_ids'] if v in task._inventory]


def test_expansion_failure_is_paid_and_never_published_or_retried(monkeypatch):
    case=fixture();order_fixture(monkeypatch,case)
    events=[];costs=Costs(inventory_callback=events.append);original=NativeResectionEngine.preview_stroke
    def capped(self,*a,**k):
        if costs.counts.get('native_previews',0)==10:raise TimeoutError('generated preview cap')
        return original(self,*a,**k)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',capped)
    task=GeneratedFallbackTask.__new__(GeneratedFallbackTask)
    with pytest.raises(TimeoutError,match='generated preview cap'):
        task.__init__(case,mode='ridge_top8',ranker=ranker(),max_steps=2,costs=costs)
    assert costs.counts['native_previews']==10 and costs.counts['fallback_additional_previews']==2
    assert not events and task._inventory is None and costs.inventory_callback is not None
    before=costs.counts.copy()
    with pytest.raises(RuntimeError,match='cannot be resumed'):task._prepare_inventory()
    assert costs.counts==before and costs.seconds['inventory_inclusive']>0


def test_unchanged_helper_bytes_and_explicit_policy_contract():
    import hashlib
    from batch_contract import HELPER_SHA
    assert hashlib.sha256((HERE/'hybrid_proposal.py').read_bytes()).hexdigest()==HELPER_SHA
    task=GeneratedFallbackTask(fixture(),mode='ridge_top8',ranker=ranker(),max_steps=2)
    original=GeneratedSparseTask(task.case,mode='ridge_top8',ranker=task._ranker,max_steps=2)
    assert task.decision_model_hash!=original.decision_model_hash
    assert task._contract_record()['conditional_catalog_expansion']['version']==VERSION


def test_actual_worker_generated_two_arms_sealed_then_uncached_replay(tmp_path,monkeypatch):
    """Actual native generated 7-cube + actual worker, no saved model or patient."""
    import initial_inventory_worker as worker
    import torch
    import test_post_exposure_learning as generated
    from resectionlab import public_patient_factory as factory
    monkeypatch.setattr(generated,'factory',factory)
    folder=tmp_path/'fixture';folder.mkdir();inputs=generated.fixture(folder,'ReMIND-050')
    manifest=json.loads(inputs['public_manifest_path'].read_bytes())
    manifest['schema']='remind-fixed-TRAIN-partial-domain-public-inputs-v1'
    manifest['source_bindings']['intended_use_review_sha256']='e'*64
    inputs['public_manifest_path'].write_text(json.dumps(manifest))
    mr={'path':str(inputs['public_manifest_path']),'sha256':factory.sha(inputs['public_manifest_path'])}
    entry={'public_manifest':mr,'intended_use_review':{'path':'synthetic-review','sha256':'e'*64}}
    model=RidgePriority(np.zeros(681),np.ones(681),np.zeros(681),0.)
    raw=json.dumps({'feature_version':model.fingerprint,'data_hash':RIDGE['data_hash'],
        'mean':[0.]*681,'scale':[1.]*681,'weights':[0.]*681,'intercept':0.}).encode()
    fake={**RIDGE,'sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':model.fingerprint}
    for module in (batch_contract,worker,patient_transfer):monkeypatch.setattr(module,'RIDGE',fake)
    monkeypatch.setattr(patient_transfer,'read_bytes',lambda path,expected,limit:raw)
    for module in (batch_contract,worker):monkeypatch.setattr(module,'OUTPUT',tmp_path/'actual-worker-output')
    # The outer generated test owner guards acquired paths. Retain, but do not
    # install, a permanent process-wide audit hook that would outlive this test.
    hooks=[];monkeypatch.setattr(worker.sys,'addaudithook',hooks.append)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    release=template('generated',{'path':'generated','sha256':'c'*64},entry)
    release['execution_released']=True
    old_small=worker.small
    def synthetic_metadata(path,expected=None):
        if str(path)==worker.BASELINE['path']:
            return {'arms':[{'method':'exhaustive','files':{'exhaustive/plan.json':'c'*64},'outcomes':{}}]}
        if str(path)==str(Path(worker.BASELINE['path']).parent/'exhaustive/plan.json'):
            return {'actions':[], 'source_hash':'generated-historical-placeholder'}
        return old_small(path,expected)
    monkeypatch.setattr(worker,'small',synthetic_metadata)
    result=worker.execute(release,'b'*64,{'blocked':0})
    batch_contract.complete_result(result,release,'b'*64)
    assert len(hooks)==1 and result['source_visits']==4 and result['all_plans_sealed_before_evaluation']
    assert result['ridge_loads']==1 and result['ridge_unchanged'] and result['cache_restored']
    assert result['fits']==result['model_calls']==result['optimizer_calls']==0
    assert all(r['complete'] and r['independent_accepted'] for r in result['arms'])
    assert result['arms'][1]['online_costs']['counts'].get('feature_rows',0)==0
    assert result['arms'][1]['online_costs']['counts'].get('ridge_calls',0)==0
    out=worker.OUTPUT
    seals=json.loads((out/'sealed-plans.json').read_text())
    assert set(seals['plans'])==set(batch_contract.ARMS)
    assert all((out/m/'replay/independent-episode.json').exists() for m in batch_contract.ARMS)
    fallback=result['arms'][0]
    assert fallback['online_costs']['counts']['fallback_gate_states']>0
    assert fallback['replay_costs']['counts']['fallback_gate_states']>0
    assert result['policy_version']==VERSION
    assert result['native_transitions']==sum(2*len(seals['plans'][m]['actions']) for m in batch_contract.ARMS)
    assert result['native_previews']==sum(r['online_previews']+r['replay_previews'] for r in result['arms'])
