"""One glue-only flow: fake worlds/checkpoints, real output sink, no payloads.

Canonical generated native/model/lineage semantics are covered by128 controls.
This checks only four-arm routing, exact factory kwargs, actual greedy-record
forwarding, zero-update guards, cleanup and bounded result/output wiring.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import comparison_contract as contract
import select_worker as worker


def test_fixed_four_arm_glue_without_payloads(tmp_path,monkeypatch):
    import torch
    from resectionlab import patient_select_inference as inference
    from resectionlab import public_patient_factory as factory
    from resectionlab import patient_planning_admission as admission
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab import observed_search
    from resectionlab import planning_budget
    config={'max_candidates':120,'intermediate_opening_mm':1.,'tool_footprint_opening':True,'obstruction_opening':True}
    protocols={name:{'updates_per_method':64 if name=='IL64' else 8,
        'public_target_context_variant':contract.TARGET_CONTEXT,
        'cohort_execution':{'proposal_config':config,'occupancy_condition':contract.OCCUPANCY}}
        for name in contract.ENDPOINTS}
    data={name:{'protocol':protocols[name],'checkpoint':{'path':name+'.unused','sha256':'a'*64},
        'context_hashes':{},'method':'IL' if name=='IL64' else 'RL','updates':64 if name=='IL64' else 8,
        'parameter_hash':'sha256:'+'b'*64,'training_release_sha256':'c'*64} for name in contract.ENDPOINTS}
    loaded=[];factory_calls=[];collected=[];replayed=[];greedy_raw=[];live={}
    class Budget:
        def __init__(self,*args,**kwargs):self.count=0;live['budget']=self
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def check(self):pass
        def complete(self,*,history_complete):assert history_complete
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
        def __init__(self,config):self.proposal_config=config
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
        assert kwargs['occupancy_inference_protocol']==protocols[name]
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
    monkeypatch.setattr(observed_search,'observed_beam_search',lambda *args,**kwargs:(('STOP',),{'call_cap_reached':False,'time_cap_reached':False}))
    result=worker.execute({'public_index':{},'greedy_seconds':60},'f'*64,tmp_path/'out',lambda *a,**k:None)
    assert contract.complete_result(result)
    assert collected==replayed==list(contract.ARMS)
    assert loaded==['IL','RL'] and factory_calls==['IL','IL','RL','IL']
    assert result['native_budget']['native_preview_entries']==4
    assert result['total_policy_forward_calls']==2
    assert result['checkpoint_loads']==2 and result['optimizer_attempts']==result['gradient_attempts']==0
    assert result['held_cases']==['ReMIND-037'] and result['EVAL_opened'] is False
    assert json.loads((tmp_path/'out/result.json').read_text())==result
    for arm in ('observed_greedy','observed_beam'):
        assert result['arms'][arm]['checkpoint_author'] is None
    wrong=copy.deepcopy(result);wrong['checkpoint_loads']=1;assert not contract.complete_result(wrong)
    wrong=copy.deepcopy(result);wrong['arms']['observed_beam']['status']='search_unresolved';assert not contract.complete_result(wrong)
