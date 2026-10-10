"""One glue-only flow: fake worlds/checkpoints, real output sink, no payloads.

Canonical generated native/model/lineage semantics have separate saved controls.
This checks only four-arm routing, exact factory kwargs, actual greedy-record
forwarding, zero-update guards, cleanup and bounded result/output wiring.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import comparison_contract as contract
import select_worker as worker


@pytest.mark.parametrize('beam_capped',[False,True])
def test_fixed_four_arm_glue_without_payloads(tmp_path,monkeypatch,beam_capped):
    import torch
    from resectionlab import patient_select_inference as inference
    from resectionlab import public_patient_factory as factory
    from resectionlab import patient_planning_admission as admission
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab import observed_search
    from resectionlab import planning_budget
    config={'max_candidates':120,'intermediate_opening_mm':1.,'tool_footprint_opening':True,'obstruction_opening':False}
    protocols={name:{'updates_per_method':64 if name=='IL64' else 8,
        'public_target_context_variant':contract.TARGET_CONTEXT,
        'cohort_execution':{'proposal_config':config,'occupancy_condition':contract.OCCUPANCY}}
        for name in contract.ENDPOINTS}
    data={name:{'protocol':protocols[name],'checkpoint':{'path':name+'.unused','sha256':'a'*64},
        'context_hashes':{},'method':'IL' if name=='IL64' else 'RL','updates':64 if name=='IL64' else 8,
        'parameter_hash':'sha256:'+'b'*64,'training_release_sha256':'c'*64,'training_evidence':{'generated':True}} for name in contract.ENDPOINTS}
    loaded=[];factory_calls=[];collected=[];replayed=[];greedy_raw=[];live={}
    class Budget:
        def __init__(self,*args,**kwargs):self.count=0;live['budget']=self
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def check(self):pass
        def complete(self,*,history_complete):
            assert history_complete;live['history_attested']=True
        def snapshot(self):return {'native_preview_entries':self.count}
    class Costs:
        def __init__(self,*args):self.rows=[];self.forwards=0;live['costs']=self
        def __enter__(self):return self
        def __exit__(self,*args):return False
        @contextmanager
        def scope(self,name):
            yield
            self.rows.append({'phase':name,'wall_seconds':0.})
    class Source:
        source_hash='generated-source'
        def __init__(self,config):
            self.proposal_config=config;self.post_exposure=SimpleNamespace(record={'version':contract.EXPOSURE})
            self._domain_record={'source_support_unknown_voxels':0}
    class Context:
        fingerprint='generated-context'
        def record(self):return {'occupancy_condition':contract.OCCUPANCY}
    class Task:
        max_steps=24;decision_model_hash='generated-model'
        def __init__(self,case):self.case=case
        def observation(self):return SimpleNamespace(fingerprint='generated-observation',action_ids=('STOP',),
            action_mask=SimpleNamespace(tolist=lambda:[True]))
        def observed_greedy_search(self,*,seconds):
            assert seconds==60
            raw={'method':'observed_greedy','complete':True,'generated_glue_only':True}
            greedy_raw.append(raw);return ('STOP',),raw
    class Checkpoint:
        def __init__(self,method):self.method=method
        def lineage_record(self):return {'method':self.method,'completed_updates':64 if self.method=='IL' else 8,'parameter_hash':'sha256:'+'b'*64}
        def require_task(self,*args):return self
        def require(self,*args):return self
    def load(path,**kwargs):
        loaded.append(kwargs['expected_method']);return Checkpoint(kwargs['expected_method'])
    def prepare(out,release,sha,expected,progress,**kwargs):
        assert len(loaded)==2
        assert kwargs['expected_role']=='SELECT' and kwargs['occupancy_condition']==contract.OCCUPANCY
        method=kwargs['checkpoint_lineage']['method'];name='IL64' if method=='IL' else 'RL8'
        assert kwargs['occupancy_inference_protocol']==protocols[name] and kwargs['post_exposure_condition']==contract.EXPOSURE
        assert kwargs['partial_domain_inputs'] is True
        assert '037' not in str(kwargs['public_manifest_path'])
        factory_calls.append(method);live['budget'].count+=1
        return Source(kwargs['proposal_config']),{}, {},{}
    def collect(base,context,checkpoint,*,output,guard,**kwargs):
        guard();arm=output.name;collected.append(arm)
        if kwargs:
            assert kwargs['actions']==('STOP',)
            if arm=='observed_greedy':assert kwargs['accounting'] is greedy_raw[0] and 'call_cap_reached' not in kwargs['accounting']
        else:live['costs'].forwards+=1
        return SimpleNamespace(transitions=[SimpleNamespace(action_id='STOP',reward=0.)])
    def replay(base,context,trace,checkpoint,*,output,guard):
        guard();replayed.append(output.name)
        return {'replayed_history':[{'target_removed_mm3':0.,'normal_removed_mm3':0.}], 'plan_seal':'generated-seal'}
    monkeypatch.setattr(worker,'authenticate_comparison',lambda release:data)
    monkeypatch.setattr(worker,'read_pinned',lambda *args:{})
    monkeypatch.setattr(worker,'select_public',lambda *args:{'path':'generated013.unused','sha256':'d'*64})
    monkeypatch.setattr(planning_budget,'PlanningBudget',Budget)
    monkeypatch.setattr(preflight,'_CallCosts',Costs)
    monkeypatch.setattr(factory,'prepare_public_source',prepare)
    monkeypatch.setattr(admission,'make_patient_planning_task',lambda source,**kwargs:(Task(source),Context()))
    monkeypatch.setattr(inference,'load_select_checkpoint',load)
    monkeypatch.setattr(inference,'collect_select_greedy',collect)
    monkeypatch.setattr(inference,'collect_select_search',collect)
    monkeypatch.setattr(inference,'seal_and_replay_select',replay)
    monkeypatch.setattr(observed_search,'observed_beam_search',lambda *args,**kwargs:(('STOP',),{'call_cap_reached':beam_capped,'time_cap_reached':False}))
    release={'public_index':{},'greedy_seconds':60,'public_preparation':{'generated':True}}
    if beam_capped:
        with pytest.raises(ValueError,match='Search unresolved'):
            worker.execute(release,'f'*64,tmp_path/'out',lambda *a,**k:None)
        result=json.loads((tmp_path/'out/result.json').read_text())
    else:
        result=worker.execute(release,'f'*64,tmp_path/'out',lambda *a,**k:None)
    assert contract.complete_result(result) is (not beam_capped)
    assert collected==replayed==list(contract.ARMS[:-1] if beam_capped else contract.ARMS)
    assert loaded==['IL','RL'] and factory_calls==['IL','IL','RL','IL']
    assert result['native_budget']['native_preview_entries']==4
    assert result['total_policy_forward_calls']==2
    assert result['checkpoint_loads']==2 and result['optimizer_attempts']==result['gradient_attempts']==0
    assert result['held_cases']==['ReMIND-037'] and result['EVAL_opened'] is False
    assert result['partial_domain_inputs'] is True and result['public_preparation']==release['public_preparation']
    assert json.loads((tmp_path/'out/result.json').read_text())==result
    if beam_capped:
        assert result['status']=='failed_or_unresolved'
        assert live.get('history_attested') is not True
        row=result['arms']['observed_beam']
        assert row['status']=='search_unresolved' and row['partial_actions']==['STOP']
        assert row['accounting']['call_cap_reached'] is True
        assert not (tmp_path/'out/observed_beam/selection.json').exists()
        assert 'plan_seal' not in row
        return
    assert live['history_attested'] is True
    for arm in ('observed_greedy','observed_beam'):
        assert result['arms'][arm]['checkpoint_author'] is None
    wrong=copy.deepcopy(result);wrong['checkpoint_loads']=1;assert not contract.complete_result(wrong)
    wrong=copy.deepcopy(result);wrong['arms']['observed_beam']['status']='search_unresolved';assert not contract.complete_result(wrong)


@pytest.mark.parametrize('pin',[None,'','pending','G'*64])
def test_terminal_RL_pin_required_before_any_read(monkeypatch,pin):
    def forbidden(*args,**kwargs):raise AssertionError('No metadata may open before terminal pin')
    monkeypatch.setattr(contract,'sha',forbidden)
    monkeypatch.setattr(contract,'read_pinned',forbidden)
    with pytest.raises(ValueError,match='Root-reviewed terminal RL'):
        contract.endpoint_refs('RL8',rl_result_sha256=pin)


@pytest.mark.parametrize('changed',[None,'held037','missing_Ds'])
def test_expanded_saved_metadata_join_without_payloads(monkeypatch,changed):
    original=contract.read_pinned
    def metadata_only(ref,**kwargs):
        assert Path(ref['path']).suffix=='.json'
        row=original(ref,**kwargs)
        if changed=='held037' and ref['path']==contract.PUBLIC_INDEX:
            row['cases'][1]['quarantined_conversion_manifest_sha256']='0'*64
        if changed=='missing_Ds' and row.get('schema')=='remind-fixed-SELECT-partial-domain-public-inputs-v1':
            del row['input_files']['supplied_support_domain']
        return row
    monkeypatch.setattr(contract,'read_pinned',metadata_only)
    ref={'path':contract.PUBLIC_INDEX,'sha256':contract.PUBLIC_SHA}
    if changed:
        with pytest.raises(ValueError,match='held037|five-array'):
            contract.qualified_public(ref)
        return
    active,manifest,refs,record=contract.qualified_public(ref)
    assert active['sha256']=='841261035a7d7be5c55984e717b8d65b712065a708e9260231ebcd49cc4d700e'
    assert manifest['public_support_domain_fully_covered'] is False
    assert record['partial_domain_inputs'] is True
    assert record['source_domain_counts']['target_in_unknown_support_domain']==0
    assert record['original_failed_attempt']['failure']['message']=='POST_EXPOSURE_EMPTY_FIXED_K'
    assert record['original_failed_attempt']['policy_forwards']==0
    assert all(Path(ref['path']).suffix=='.json' for ref in refs)
