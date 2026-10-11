"""Tiny metadata/contract controls; no patient, checkpoint, model or task access."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import TRAIN,semantic,useful_gate,join_planning_replay


def cases():
    outcome={'target_removed_mm3':1.,'normal_removed_mm3':0.,'cumulative_contacted_tissue_upper_bound_mm3':2.,
             'total_reward':.8,'complete_tool_path_length_mm':2.,'nonstop_actions':1}
    row={'complete':True,'audit':{'accepted':True,'outcomes':outcome},'world':{'source':'same'},
         'planning_costs':{'native_preview_entries':10,'native_transition_calls':2}}
    return {s:{m:copy.deepcopy(row) for m in ('GREEDY','IL64','RL8','Q_GUIDED')} for s in TRAIN}


def test_tie_is_negative():
    result=useful_gate(cases());assert result['comparison_admissible'] and not result['useful_gain']


def test_strict_gain_needs_all_four_outcomes():
    rows=cases();rows[TRAIN[0]]['Q_GUIDED']['planning_costs']['native_preview_entries']=9
    assert useful_gate(rows)['useful_gain']
    rows[TRAIN[3]]['Q_GUIDED']['audit']['outcomes']['target_removed_mm3']=.9
    assert not useful_gate(rows)['useful_gain']
    rows[TRAIN[3]]['Q_GUIDED']['complete']=False
    assert not useful_gate(rows)['comparison_admissible']


@pytest.mark.parametrize('field',['normal_removed_mm3','cumulative_contacted_tissue_upper_bound_mm3'])
def test_cost_improvement_cannot_cover_harm(field):
    rows=cases();rows[TRAIN[0]]['Q_GUIDED']['planning_costs']['native_preview_entries']=1
    rows[TRAIN[0]]['Q_GUIDED']['audit']['outcomes'][field]+=1
    assert not useful_gate(rows)['useful_gain']


def test_original_negative_current_cost_examples():
    # Independent stdlib recurrence oracle, not an alternate fitting implementation.
    def oracle(rewards):
        out=[];future=0.
        for reward in reversed(rewards):
            q=reward+max(0.,future);out.append(q);future=q
        return list(reversed(out))
    assert oracle([-2,-3])==[-2,-3]
    assert oracle([-2,5])==[3,5]
    cfg=json.loads((HERE/'config.json').read_text())
    assert 'current negative action cost preserved' in cfg['target']
    assert cfg['head_updates']==64 and cfg['traces']==36 and cfg['states']==497


def test_runner_source_compiles_and_retains_single_replay_admission():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    source=(HERE/'worker.py').read_text()
    assert 'audit=audit_actual(actual)' in source
    assert source.index('audit=audit_actual(actual)') < source.index('nominal=actual.planning_clone()')
    assert 'trace=admit_collected_trace(context,transitions,nominal' in source
    assert 'builder.add_trace(row[\'key\'],trace)' in source
    assert 'torch.load' not in source and 'PatientTrainSession' not in source


def test_missing_method_or_failed_certificate_refuses():
    rows=cases();del rows[TRAIN[0]]['RL8'];assert not useful_gate(rows)['comparison_admissible']
    rows=cases();rows[TRAIN[1]]['Q_GUIDED']['audit']['accepted']=False
    assert not useful_gate(rows)['comparison_admissible']


def test_q_guard_to_replay_join_is_not_action_ids_only():
    row={'action_id':'a','reward':1.,'terminated':True,'source_state_hash':'x','result_state_hash':'y','observation_hash':'z'}
    planned={'selected_action_id':'a','scores':[{'action_id':'a','reward':1.}],**{k:v for k,v in row.items() if k not in ('action_id','reward')}}
    a={'decisions':[planned],'estimated_incremental_return':1.,'nominal_history_hash':'h','source_hash':'s','decision_model_hash':'m'}
    b={'decisions':[row],'nominal_history_hash':'h','world':{'source_hash':'s','decision_model_hash':'m'}}
    assert join_planning_replay(a,b,method='Q_GUIDED')
    for field,value in (('reward',2.),('result_state_hash','changed')):
        changed=copy.deepcopy(b);changed['decisions'][0][field]=value
        with pytest.raises(ValueError):join_planning_replay(a,changed,method='Q_GUIDED')


def test_real_saved_metadata_joins_without_array_or_checkpoint_reads(monkeypatch):
    import os
    from common import ROOT
    from inputs import metadata_pins,manifest
    original=os.open
    def no_payload(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).suffix in ('.npy','.psckpt'):
            raise AssertionError('Actual payload forbidden in metadata control')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'open',no_payload)
    pins,arrays=metadata_pins();rows=manifest({'input_metadata':pins})
    assert len(rows)==36 and sum(len(r['decisions']) for r in rows)==497 and len(arrays)==20


def load_entry(name):
    spec=importlib.util.spec_from_file_location('q_runner_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_owned_parent_check_only_and_false_release_never_spawn(tmp_path,monkeypatch):
    from common import sha
    parent=load_entry('run_owned');stage=tmp_path/'runner';stage.mkdir()
    (stage/'source-index.json').write_text(json.dumps({'files':{},'execution_released':False}))
    (stage/'config.json').write_text((HERE/'config.json').read_text())
    monkeypatch.setattr(parent,'HERE',stage);monkeypatch.setattr(parent,'ROOT',tmp_path)
    monkeypatch.setattr(parent.subprocess,'Popen',lambda *a,**k:pytest.fail('No child in source-only controls'))
    args=['run_owned','--source-index-sha256',sha(stage/'source-index.json')]
    monkeypatch.setattr(sys,'argv',args);assert parent.main()==0
    monkeypatch.setattr(sys,'argv',args+['--execute'])
    with pytest.raises(ValueError,match='not released'):parent.main()


def test_worker_false_release_refuses_before_torch_or_source(tmp_path,monkeypatch):
    from common import sha
    worker=load_entry('worker');stage=tmp_path/'runner';stage.mkdir()
    (stage/'source-index.json').write_text(json.dumps({'files':{},'execution_released':False}))
    (stage/'config.json').write_text((HERE/'config.json').read_text())
    monkeypatch.setattr(worker,'HERE',stage);monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'execute',lambda *a,**k:pytest.fail('No science in false-release control'))
    monkeypatch.setattr(sys,'argv',['worker','--source-index-sha256',sha(stage/'source-index.json'),'--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='root-released'):worker.main()


def test_actual_isolated_worker_bootstrap_under_boundary(tmp_path):
    import subprocess
    import os
    output=tmp_path/'bootstrap';cache=tmp_path/'absent-source-cache'
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','DYLD_','LD_'))}
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['TMPDIR']=str(tmp_path)
    result=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+str(cache),str(HERE/'bootstrap_smoke.py'),'--output',str(output)],
        capture_output=True,text=True,timeout=28,env=env)
    assert result.returncode==0,result.stderr
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['status']=='passed' and receipt['payload_open_attempts']==[]
    assert receipt['native_calls']==receipt['policy_forwards']==receipt['optimizer_steps']==0
