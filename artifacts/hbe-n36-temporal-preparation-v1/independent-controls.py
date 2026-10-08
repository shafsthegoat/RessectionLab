"""Targeted metadata/orchestration controls; no native, geometry or response execution."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,math
import numpy as np
import pytest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from scripts import mechanics_hbe_halfheight_global_n36_temporal as core
from scripts import mechanics_hbe_halfheight_global_n36_temporal_readout as rd
from scripts import mechanics_hbe_halfheight_global_n36_temporal_experiment as run

@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
 def no(*a,**kw):raise AssertionError('Native/geometry/raw access forbidden in independent controls')
 for m,n in [(run.subprocess,'Popen'),(run.old,'solve'),(run.runtime,'supervise'),(core,'baseline_geometry'),(core,'case_contents'),(rd,'_records'),(rd.spatial_reader,'replay_prior_runs')]:monkeypatch.setattr(m,n,no)

def test_exact_source_and_case_specific_contracts():
 expected={'scripts/mechanics_hbe_halfheight_global_n36_temporal.py':'5af7a1d9beac06412c9a20c91fb6cf8d3834c2b8eb91c3053541f8d0256a9c93','scripts/mechanics_hbe_halfheight_global_n36_temporal_readout.py':'550d8300ebcdef18778026d1150bcd32feca4991bd94a57406a360234800ede1','scripts/mechanics_hbe_halfheight_global_n36_temporal_experiment.py':'7fbd7dc4fdf773c5db2f0bbaa0a9a20891b7828ba57731817e62456f928faa8e','manifests/experiments/hbe-01-03-halfheight-global-n36-temporal-v1.json':'c726bc9299ed0cbb394a4889586df9487aa26cf6b932bae9ddc5b1db5914991f','tests/test_mechanics_hbe_halfheight_global_n36_temporal.py':'3fa8d8675c741a33732b63e65245db1ed0e88eaaa6f65c06f688156a1bd46cb4'}
 for p,h in expected.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
 study=json.loads((R/core.DECLARATION_PATH).read_text());core.require_study(study)
 assert study['budgets']['preparation_nested_in_aggregate'] is True
 assert rd.primitive_limit('nodes')==768*1024**2 and rd.primitive_limit('elements')==512*1024**2
 assert rd.spatial_reader.primitive_limit('nodes')==384*1024**2 and rd.spatial_reader.primitive_limit('elements')==256*1024**2
 assert np.array_equal(core.TIMES[::2],rd.inherited.TIMES)
 for k,v in study['baseline']['source_bindings'].items():
  assert hashlib.sha256((R/'scripts'/Path(v['path']).name).read_bytes()).hexdigest()==study['inherited_source_sha256'][k]

# Explicit software-only durable-child receipt, not a scientific case/release.
def arrange(tmp_path,monkeypatch,after_publication,child_error=False):
 d=tmp_path/'experiment';d.mkdir();clock=[100.];events=[]
 study={'budgets':{'pure_preparation_seconds':60},'baseline':{'primitive_bindings':{'mesh':{'path':'retained-mesh','sha256':'0'*64}},'reconstruction':{'path':'retained-map','sha256':'1'*64}}}
 plan={'directory':str(d),'study':study,'study_binding':{'path':'software-only','sha256':'2'*64}}
 class Fake:
  pid=101
  killed=False
  def wait(self,timeout=None):
   events.append(('wait',timeout))
   if child_error and not self.killed:raise run.subprocess.TimeoutExpired(['software-mock'],timeout)
   return 0
  def poll(self):return None if child_error and not self.killed else 0
  def kill(self):self.killed=True;events.append(('kill',None))
 def child(command,**kwargs):
  events.append(('child',command));prep=d/'preparation';case={}
  for name,file in [('deck','specimen.feb'),('loading','loading.json'),('backend_source_deck','skyline.feb')]:
   p=prep/file;p.write_text('software-only non-scientific placeholder');case[name]=run.old.binding(tmp_path,p)
  case.update(mesh=study['baseline']['primitive_bindings']['mesh'],reconstruction=study['baseline']['reconstruction'])
  run.previous_runner.durable_json(tmp_path,prep/'prepared.json',{'schema':'hbe-n36-temporal-pure-deck-v1','study':plan['study_binding'],'run_id':core.RUN_ID,'case':case,'solver_calls':0,'gmsh_generation_calls':0,'baseline_geometry_reused':True,'measured_data_accessed':False})
  return Fake()
 original=run.previous_runner.durable_json
 def delayed(root,path,value):
  b=original(root,path,value)
  if Path(path).name=='execution.json':clock[0]=after_publication
  return b
 monkeypatch.setattr(run.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(run.subprocess,'Popen',child);monkeypatch.setattr(run.previous_runner,'durable_json',delayed)
 return plan,{},SimpleNamespace(active=None),events

@pytest.mark.parametrize('aggregate_deadline,post_time',[(2500.,160.),(150.,150.)])
def test_nested_publication_timeout_never_returns_case(tmp_path,monkeypatch,aggregate_deadline,post_time):
 plan,state,watch,events=arrange(tmp_path,monkeypatch,post_time)
 with pytest.raises(TimeoutError):run.prepare_with_deadline(tmp_path,plan,{'path':'mock-baseline','sha256':'3'*64},aggregate_deadline,state,watch)
 assert 'preparation_elapsed_seconds_including_publication' not in state
 assert state['preparation'] and watch.active is None
 assert events[1][1]==min(60,aggregate_deadline-100)

def test_successful_nested_preparation_includes_publication(tmp_path,monkeypatch):
 plan,state,watch,events=arrange(tmp_path,monkeypatch,130.)
 case=run.prepare_with_deadline(tmp_path,plan,{'path':'mock-baseline','sha256':'3'*64},2500.,state,watch)
 assert case==state['case'] and state['preparation_elapsed_seconds_including_publication']==30.
 assert watch.active is None and events[1][1]==60.

def test_nested_timeout_kills_and_reaps_child(tmp_path,monkeypatch):
 plan,state,watch,events=arrange(tmp_path,monkeypatch,130.,True)
 with pytest.raises(run.subprocess.TimeoutExpired):run.prepare_with_deadline(tmp_path,plan,{'path':'mock-baseline','sha256':'3'*64},2500.,state,watch)
 assert ('kill',None) in events and events[-1]==('wait',None)
 assert 'case' not in state and 'preparation_elapsed_seconds_including_publication' not in state
 p=tmp_path/state['preparation']['path'];assert json.loads(p.read_text())['status']=='failed_or_incomplete'
 assert watch.active is None

@pytest.mark.parametrize('delta,inside',[(0.,True),(-8.317139479872837e-7,False),(1.3257825473393658e-6,False)])
def test_directional_margin_does_not_become_force_gate(delta,inside):
 study=json.loads((R/core.DECLARATION_PATH).read_text())
 report=rd.signed_margin_report(delta,study['classification_fragility']['observed_S60_endpoint_margin'])
 assert report['inside_fixed_other_state_scalar_interval'] is inside
 assert report['force_error_bound'] is False and report['used_as_acceptance_gate'] is False
