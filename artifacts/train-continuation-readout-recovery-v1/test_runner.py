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
    return {s:{m:copy.deepcopy(row) for m in ('GREEDY','Q64','CALIBRATED')} for s in TRAIN}


def test_tie_is_negative():
    result=useful_gate(cases());assert result['comparison_admissible'] and not result['useful_gain']


def test_strict_gain_needs_all_four_outcomes():
    rows=cases();rows[TRAIN[0]]['CALIBRATED']['planning_costs']['native_preview_entries']=9
    assert useful_gate(rows)['useful_gain']
    rows[TRAIN[3]]['CALIBRATED']['audit']['outcomes']['target_removed_mm3']=.9
    assert not useful_gate(rows)['useful_gain']
    rows[TRAIN[3]]['CALIBRATED']['complete']=False
    assert not useful_gate(rows)['comparison_admissible']


@pytest.mark.parametrize('field',['normal_removed_mm3','cumulative_contacted_tissue_upper_bound_mm3'])
def test_cost_improvement_cannot_cover_harm(field):
    rows=cases();rows[TRAIN[0]]['CALIBRATED']['planning_costs']['native_preview_entries']=1
    rows[TRAIN[0]]['CALIBRATED']['audit']['outcomes'][field]+=1
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


def test_runner_source_compiles_and_reuses_cache_without_reconstruction():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    source=(HERE/'worker.py').read_text()
    assert 'load_continuation_cache(ROOT/CACHE_ATTEMPT' in source
    assert 'ContinuationCorpusBuilder' not in source and 'admit_collected_trace' not in source
    assert 'save_continuation_cache' not in source
    assert 'torch.load' not in source and 'PatientTrainSession' not in source
    assert source.count('write_logged_targets(sink,output,config')==1


def test_missing_method_or_failed_certificate_refuses():
    rows=cases();del rows[TRAIN[0]]['Q64'];assert not useful_gate(rows)['comparison_admissible']
    rows=cases();rows[TRAIN[1]]['CALIBRATED']['audit']['accepted']=False
    assert not useful_gate(rows)['comparison_admissible']


def test_q_guard_to_replay_join_is_not_action_ids_only():
    row={'action_id':'a','reward':1.,'terminated':True,'source_state_hash':'x','result_state_hash':'y','observation_hash':'z'}
    planned={'selected_action_id':'a','scores':[{'action_id':'a','reward':1.}],**{k:v for k,v in row.items() if k not in ('action_id','reward')}}
    a={'decisions':[planned],'estimated_incremental_return':1.,'nominal_history_hash':'h','source_hash':'s','decision_model_hash':'m'}
    b={'decisions':[row],'nominal_history_hash':'h','world':{'source_hash':'s','decision_model_hash':'m'}}
    assert join_planning_replay(a,b,method='CALIBRATED')
    for field,value in (('reward',2.),('result_state_hash','changed')):
        changed=copy.deepcopy(b);changed['decisions'][0][field]=value
        with pytest.raises(ValueError):join_planning_replay(a,changed,method='CALIBRATED')


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


def test_beating_poor_Q64_does_not_count_as_beating_greedy():
    rows=cases();rows[TRAIN[0]]['Q64']['audit']['outcomes']['target_removed_mm3']=0.
    assert not useful_gate(rows)['useful_gain']


def test_baseline_reproduction_allows_only_provenance_and_timing_changes():
    from common import require_baseline_fit,require_baseline_route
    keys=('base_parameter_hash','architecture_hash','initial_head_hash','fitted_head_hash',
          'config','updates','final_weighted_mse','target','weighting','initialization')
    old={k:1 for k in keys};new=dict(old,corpus_seal='new',elapsed_seconds=4)
    assert require_baseline_fit(new,old)
    for key in keys:
        changed=dict(new);changed[key]=2
        with pytest.raises(ValueError):require_baseline_fit(changed,old)
    route={'actions':['STOP'],'world':{'source':'same'},'nominal_history_hash':'a','audit':{'outcomes':{'target':0}}}
    assert require_baseline_route(dict(route,elapsed_seconds=2),route)
    changed=copy.deepcopy(route);changed['audit']['outcomes']['target']=1
    with pytest.raises(ValueError):require_baseline_route(changed,route)


def test_cache_reused_before_fit_with_external_original_binding():
    source=(HERE/'worker.py').read_text()
    assert source.index("with at('authenticated_existing_cache_load')") < source.index("with at('fixed_64_head_updates')")
    assert "expected_cache_sha256=CACHE_PINS['authenticated-features.zip'],expected_binding=binding" in source
    assert "'feature_array_not_saved':False" in source
    assert "for method,base in (('IL64',IL),):" in source
    assert "for method in config['methods']" in source
    assert 'continuation_calibration_accounting' in source
    config=json.loads((HERE/'config.json').read_text())
    assert config['methods']==['GREEDY','Q64','CALIBRATED']
    assert config['caps']['output_bytes']==12*1024**2
    assert config['caps']['whole_build_bytes']==32*1024**2
    assert config['logged_prediction_readouts']==2


def test_real_exclusive_output_budget_publication_and_prior_failure(tmp_path):
    from common import ROOT
    sys.path.insert(0,str(ROOT/'src'))
    from resectionlab.patient_planning_cohort_io import OutputBudget
    worker=load_entry('worker')
    prior=tmp_path/'prior';prior.mkdir();sink=OutputBudget(prior,128*1024)
    sink.write(prior/'logged-targets.json',{'feature_array_not_saved':True})
    with pytest.raises(FileExistsError):
        sink.write(prior/'logged-targets.json',{'feature_array_not_saved':False})
    assert json.loads((prior/'logged-targets.json').read_text())=={'feature_array_not_saved':True}
    fresh=tmp_path/'recovery';fresh.mkdir();sink=OutputBudget(fresh,128*1024)
    worker.write_logged_targets(sink,fresh,'fixed','sha256:manifest',[1.,0.],[.5,.5],[False,True],'prior/cache.zip','a'*64)
    record=json.loads((fresh/'logged-targets.json').read_text())
    assert record['feature_array_not_saved'] is False and record['feature_cache']=='prior/cache.zip'
    assert len(list(fresh.iterdir()))==1
    with pytest.raises(FileExistsError):
        worker.write_logged_targets(sink,fresh,'fixed','sha256:manifest',[1.,0.],[.5,.5],[False,True],'prior/cache.zip','a'*64)


def test_exact_saved_cache_origin_metadata_and_substitute_refusals():
    from common import ROOT,CACHE_ATTEMPT,CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA,CACHE_PINS,read_json,require_cache_origin
    names=('report.json','launcher.json','cache-binding.json','cache-receipt.json','reconstruction-evidence.json')
    rows=[read_json(ROOT/CACHE_ATTEMPT/name,CACHE_PINS[name]) for name in names]
    rows.append(read_json(ROOT/CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA))
    assert require_cache_origin(*rows)
    for index,key,value in ((0,'head_optimizer_updates',1),(1,'worker_exit_code',0),
            (2,'corpus_seal','changed'),(3,'sha256','0'*64),(4,'source_index_sha256','0'*64)):
        changed=copy.deepcopy(rows);changed[index][key]=value
        with pytest.raises(ValueError):require_cache_origin(*changed)
