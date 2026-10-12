"""Root-run generated representation/metadata controls; no native or payload work."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT,canonical,semantic,sha,read_json,TAIL_RUN,TAIL_PINS
from recovery_inputs import FAILED,PINS,OLD_INDEX,OLD_INDEX_SHA,require_saved_plans,exact_json_equal


def load(name):
    spec=importlib.util.spec_from_file_location('recovery_'+name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def original_compactor():
    """Execute exactly the old pure production function, not a mirrored implementation."""
    old=read_json(ROOT/OLD_INDEX,OLD_INDEX_SHA)
    path='build/continuation-endgame-pair-v1/runner/worker.py'
    from common import read_bytes
    tree=ast.parse(read_bytes(ROOT/path,old['files'][path]))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compact_history')
    namespace={'semantic':semantic}
    exec(compile(ast.Module(body=[function],type_ignores=[]),path,'exec'),namespace)
    return namespace['compact_history']


def test_actual_old_compact_history_tuple_json_mismatch_and_exact_repair():
    row={'action_id':'fixed','reward':1.25,'target_removed_mm3':2.,'normal_removed_mm3':0.,
        'tool_id':'tool','entry_mm':(1.,2.,3.),'tip_mm':(4.,5.,6.),'physical_start_mm':(1.,2.,3.),
        'complete_tool_path_length_mm':5.196,'interaction_mode':'aspirate','outcome_scope':'source',
        'microsteps':({'step':0,'point':(1.,2.,3.)},),'removed_indices_native':((1,2,3),),
        'contact_indices_native':((0,0,1),),'source_state_hash':'state-before','result_state_hash':'state-after'}
    actual=original_compactor()([row]);stored=json.loads(canonical(actual))
    assert actual!=stored and isinstance(actual[0]['tip_mm'],tuple) and isinstance(stored[0]['tip_mm'],list)
    assert exact_json_equal(actual,stored,'all compact fields')
    assert load('worker').compact_history([row])==actual
    for field,value in [('reward',1.2500000000000002),('tip_mm',[4.,5.,6.000000000000001]),
                        ('history_row_hash','changed-state-bound-digest')]:
        changed=copy.deepcopy(stored);changed[0][field]=value
        with pytest.raises(ValueError):exact_json_equal(actual,changed,'numeric/state mutation')
    changed=copy.deepcopy(stored);del changed[0]['entry_mm']
    with pytest.raises(ValueError):exact_json_equal(actual,changed,'omitted field')


def saved_inputs():
    return [read_json(ROOT/FAILED/'report.json',PINS['report.json']),
        read_json(ROOT/FAILED/'launcher.json',PINS['launcher.json']),
        read_json(ROOT/FAILED/'plans.json',PINS['plans.json']),
        read_json(ROOT/TAIL_RUN/'report.json',TAIL_PINS['report.json']),read_json(ROOT/OLD_INDEX,OLD_INDEX_SHA)]


def test_real_saved_plans_admit_without_payload_open(monkeypatch):
    oldopen=os.open
    def forbidden(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).suffix in ('.npy','.zip','.psckpt','.qckpt'):
            pytest.fail('No scientific payload permitted in metadata controls')
        return oldopen(path,*args,**kwargs)
    monkeypatch.setattr(os,'open',forbidden)
    result=require_saved_plans(*saved_inputs())
    assert result['accepted'] and result['new_model_cache_checkpoint_loads']==result['new_planning_calls']==0
    from recovery_inputs import metadata_pins
    pins,arrays=metadata_pins()
    assert len(arrays)==5 and all(not p.endswith(('.zip','.psckpt','.qckpt')) for p in pins)


@pytest.mark.parametrize('changed',['head','state','extra_intervention'])
def test_resealed_metadata_mutations_still_refuse(changed):
    failed,parent,plans,retained,index=saved_inputs();plan=plans['IMMEDIATE_BEST']
    if changed=='head':plan['planning']['fitted_head_hash']='sha256:wrong'
    if changed=='state':plan['planning']['decisions'][8]['source_state_hash']='sha256:wrong'
    if changed=='extra_intervention':plan['planning']['decisions'][9]['intervention']=True
    plan['plan_seal']=semantic({k:v for k,v in plan.items() if k!='plan_seal'})
    failed['plans']=copy.deepcopy(plans);failed['plans_sealed_before_replay']['seals']['IMMEDIATE_BEST']=plan['plan_seal']
    with pytest.raises(ValueError):require_saved_plans(failed,parent,plans,retained,index)


def test_source_evidence_precedes_parity_and_no_model_decode():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    source=(HERE/'worker.py').read_text()
    assert source.index("sink.write(dest/(method+'-evidence.json')")<source.index("join_planning_replay(plan['planning']")
    assert source.index("sink.write(dest/(method+'-evidence.json')")<source.index("require_baseline_route(result,retained_arm)")
    assert all(name not in source for name in ('load_cohort_checkpoint','load_continuation_cache','fit_continuation_',
        'calibrate_continuation_','score_tails(','endgame_tail_search'))
    config=json.loads((HERE/'config.json').read_text())
    assert config['caps']['frozen_policy_forwards']==0 and config['caps']['native_previews']==12000


def test_false_release_never_runs(tmp_path,monkeypatch):
    stage=tmp_path/'runner';stage.mkdir()
    (stage/'source-index.json').write_text(json.dumps({'files':{},'execution_released':False}))
    (stage/'config.json').write_text((HERE/'config.json').read_text())
    parent=load('run_owned');monkeypatch.setattr(parent,'HERE',stage);monkeypatch.setattr(parent,'ROOT',tmp_path)
    monkeypatch.setattr(parent.subprocess,'Popen',lambda *a,**k:pytest.fail('No child'))
    args=['parent','--source-index-sha256',sha(stage/'source-index.json')]
    monkeypatch.setattr(sys,'argv',args);assert parent.main()==0
    monkeypatch.setattr(sys,'argv',args+['--execute'])
    with pytest.raises(ValueError,match='not released'):parent.main()


def test_actual_isolated_bootstrap_no_checkpoint_cache_or_patient(tmp_path):
    output=tmp_path/'bootstrap';cache=tmp_path/'absent-cache'
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','DYLD_','LD_'))}
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['TMPDIR']=str(tmp_path)
    result=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+str(cache),str(HERE/'bootstrap_smoke.py'),
        '--output',str(output)],capture_output=True,text=True,timeout=28,env=env)
    assert result.returncode==0,result.stderr
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['status']=='passed' and receipt['payload_open_attempts']==[]
    assert receipt['native_calls']==receipt['policy_forwards']==receipt['optimizer_steps']==0
