"""Preparation and mocked orchestration only; never invoke a solver."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('accelerate_controls_owner',ROOT/'scripts/mechanics_accelerate_controls.py')
v=importlib.util.module_from_spec(s);s.loader.exec_module(v)


def test_all_eight_decks_only_replace_solver(tmp_path):
 manifest=v.prepare(tmp_path/'decks')
 assert manifest['case_order']==list(v.CASES) and len(manifest['cases'])==8
 for name,row in manifest['cases'].items():
  raw=(ROOT/row['original_path']).read_bytes();changed=(tmp_path/'decks/decks'/(name+'.feb')).read_bytes()
  assert changed.replace(v.backend.ACCELERATE_XML.encode(),v.backend.SKYLINE_XML.encode())==raw
  assert row['adapted_sha256']==v.sha(tmp_path/'decks/decks'/(name+'.feb'))
 with pytest.raises(FileExistsError):v.prepare(tmp_path/'decks')


def test_missing_runtime_identity_fails_closed(tmp_path):
 with pytest.raises((ValueError,FileNotFoundError)):
  v.runtime_binding(tmp_path,{'path':'missing-identity.json','sha256':'a'*64})


@pytest.mark.parametrize('console',['Default linear solver: accelerate','* Selecting linear solver skyline *',
 '* Selecting linear solver accelerate *\n* Selecting linear solver skyline *'])
def test_backend_selection_is_actual_and_unambiguous(console):
 original=(ROOT/v.ORIGINAL['zero']).read_bytes();changed=v.backend.transform_deck(original).encode()
 with pytest.raises(ValueError,match='selection evidence'):v.backend_evidence(console,original,changed,'a'*64)


def test_backend_evidence_binds_original_adapted_and_identity():
 original=(ROOT/v.ORIGINAL['zero']).read_bytes();changed=v.backend.transform_deck(original).encode()
 evidence=v.backend_evidence('Default linear solver: skyline\n* Selecting linear solver accelerate *',original,changed,'a'*64)
 assert evidence['runtime_identity_sha256']=='a'*64 and evidence['solver_only_change_verified']
 with pytest.raises(ValueError):v.backend_evidence('* Selecting linear solver accelerate *',original,changed.replace(b'<k>',b'<different_k>'),'a'*64)


def success_result(inputs):
 result=v.initial_result();result.update(status='completed',solver_invocations=8,runtime_identity_sha256='a'*64,
  stiffness_scaling={'passed':True},inputs_after=v.unchanged(inputs))
 for row in result['cases']:
  row.update(status='passed',checker_passed=True,solver_exit_code=0,deck_sha256='d'*64,command=['mock'])
  row['backend_evidence']={'runtime_identity_sha256':'a'*64,'solver_backend':'accelerate',
    'solver_only_change_verified':True,'solver_xml':v.backend.ACCELERATE_XML,
    'executed_deck_sha256':'d'*64,'actual_selection_lines':['Selecting linear solver accelerate']}
 return result


@pytest.mark.parametrize('fault',['cap','identity','xml','backend','scaling','count','input','partial'])
def test_parent_acceptance_requires_complete_matching_evidence(tmp_path,fault):
 p=tmp_path/'input';p.write_bytes(b'fixed');inputs={str(p):v.sha(p)}
 result=success_result(inputs);supervision={'status':'completed','exit_code':0};after=v.unchanged(inputs)
 assert v.accept(supervision,result,after)
 if fault=='cap':supervision['status']='failed_or_incomplete'
 elif fault=='identity':result['cases'][0]['backend_evidence']['runtime_identity_sha256']='b'*64
 elif fault=='xml':result['cases'][0]['backend_evidence']['solver_xml']='<linear_solver type="skyline" />'
 elif fault=='backend':result['cases'][0]['backend_evidence']['actual_selection_lines']=['Selecting linear solver skyline']
 elif fault=='scaling':result['stiffness_scaling']['passed']=False
 elif fault=='count':result['solver_invocations']=7
 elif fault=='input':after[str(p)]['unchanged']=False
 elif fault=='partial':result['cases'][0]['status']='not_executed'
 assert not v.accept(supervision,result,after)


def mock_worker(tmp_path,monkeypatch,*,fail_case=None,wrong_backend=None,scaling=True):
 output=tmp_path/'attempt';output.mkdir();p=tmp_path/'bound';p.write_bytes(b'fixed')
 base={'release_path':'mock','attempt_directory':str(output),'input_hashes':{str(p):v.sha(p)},
       'executable':'never-executed','runtime_identity_sha256':'a'*64}
 v.write(output/'results.json',v.initial_result());v.write(output/'execution-baseline.json',base)
 monkeypatch.setattr(v,'baseline',lambda _:base)
 calls=[]
 def run(command,**kwargs):
  calls.append(command);name=Path(command[-1]).stem;directory=kwargs['cwd']
  assert command[1:3]==['-noconfig','-no_title'] and kwargs['timeout']<=60
  kwargs['stdout'].write('Selecting linear solver '+('skyline' if name==wrong_backend else 'accelerate'))
  for suffix in ('nodes.log','elements.log','log'):(directory/(name+'.'+suffix)).write_text('MOCK OUTPUT ONLY')
  return SimpleNamespace(returncode=1 if name==fail_case else 0)
 monkeypatch.setattr(v.subprocess,'run',run)
 fake=SimpleNamespace(__file__='mock-checker',np=SimpleNamespace(__version__='mock'),
    read_bounded_text=lambda path:Path(path).read_text(),check_outputs=lambda *args:{'case':args[0],'passed':True},
    check_stiffness_scaling=lambda *args:{'passed':scaling})
 monkeypatch.setattr(v,'load',lambda *args:fake)
 return output,calls,base


@pytest.mark.parametrize('fault',['exit','backend','scaling'])
def test_stop_at_first_failure_without_retry(tmp_path,monkeypatch,fault):
 kwargs={'fail_case':'zero'} if fault=='exit' else {'wrong_backend':'zero'} if fault=='backend' else {'scaling':False}
 output,calls,_=mock_worker(tmp_path,monkeypatch,**kwargs)
 assert v.worker(output)==1
 result=json.loads((output/'results.json').read_text())
 count=5 if fault=='scaling' else 1
 assert len(calls)==count and result['solver_invocations']==count
 assert all(row['status']=='not_executed' for row in result['cases'][count:])
 before=(output/'results.json').read_bytes()
 with pytest.raises(ValueError,match='repeat'):v.worker(output)
 assert (output/'results.json').read_bytes()==before


def test_successful_mock_sequence_records_eight_bound_calls(tmp_path,monkeypatch):
 output,calls,base=mock_worker(tmp_path,monkeypatch)
 assert v.worker(output)==0 and len(calls)==8
 result=json.loads((output/'results.json').read_text())
 assert v.accept({'status':'completed','exit_code':0},result,v.unchanged(base['input_hashes']))
 for row in result['cases']:
  assert row['backend_evidence']['executed_deck_sha256']==row['deck_sha256']
  assert row['backend_evidence']['runtime_identity_sha256']=='a'*64


def test_parent_fixed_caps_private_environment_and_missing_release(tmp_path,monkeypatch):
 output=tmp_path/'attempt';p=tmp_path/'bound';p.write_bytes(b'fixed')
 base={'attempt_directory':str(output),'input_hashes':{str(p):v.sha(p)},'runtime_identity_sha256':'a'*64,
       'source_commit':'a'*40,'release':{'repository_directory':str(tmp_path)}}
 monkeypatch.setattr(v,'baseline',lambda _:base)
 runtime=v.load(ROOT/'scripts/febio_runtime.py','mock_supervisor_source')
 def supervise(command,directory,**kwargs):
  assert kwargs['seconds']==60 and kwargs['rss_bytes']==3*1024**3
  assert kwargs['environment']['OMP_NUM_THREADS']=='1' and kwargs['environment']['OMP_DYNAMIC']=='FALSE'
  v.write(output/'results.json',success_result(base['input_hashes']))
  return {'status':'failed_or_incomplete','exit_code':0}
 runtime.supervise=supervise;monkeypatch.setattr(v,'load',lambda *args:runtime)
 assert v.launch('mock',output)==1
 assert json.loads((output/'execution.json').read_text())['status']=='failed_or_incomplete'
 assert json.loads((output/'hex8-summary.json').read_text())['status']=='failed_or_incomplete'
 missing=tmp_path/'missing-release'
 monkeypatch.setattr(v,'baseline',lambda _: (_ for _ in ()).throw(FileNotFoundError('missing identity')))
 assert v.launch('mock',missing)==1
 receipt=json.loads((missing/'execution.json').read_text());assert receipt['worker_started'] is False
 assert len(json.loads((missing/'results.json').read_text())['cases'])==8


def test_summaries_bind_same_parent_and_each_original_primitive(tmp_path,monkeypatch):
 output,calls,base=mock_worker(tmp_path,monkeypatch)
 assert v.worker(output)==0
 result=json.loads((output/'results.json').read_text())
 archive=tmp_path/'archive'
 for relative in [*v.ORIGINAL.values(),*v.CHECKER_PINS]:
  target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
 monkeypatch.setattr(v,'ROOT',archive)
 base.update(source_commit='a'*40,release={'repository_directory':str(tmp_path)})
 v.write(output/'execution-baseline.json',base);v.write(output/'execution.json',{'status':'completed','test_only':True})
 v.summaries(output,base,result,True)
 hx=json.loads((output/'hex8-summary.json').read_text());tet=json.loads((output/'tet10_mpc-summary.json').read_text())
 assert hx['status']=='passed_all_five_fixed_patch_controls' and hx['stiffness_scaling']['passed']
 assert tet['status']=='three_actual_fixed_software_controls_passed'
 assert hx['runtime_identity_sha256']==tet['runtime_identity_sha256']=='a'*64
 assert hx['evidence']==tet['evidence']
 assert hx['solver_invocations']==5 and tet['solver_invocations']==3
 for row in hx['rows']+tet['case_rows']:
  assert set(row['evidence'])=={'original_deck','executed_deck','console','nodes','elements','solver_log','checked','checker_source'}
  for record in row['evidence'].values():assert v.sha(tmp_path/record['path'])==record['sha256']
 for record in hx['evidence'].values():assert v.sha(tmp_path/record['path'])==record['sha256']
