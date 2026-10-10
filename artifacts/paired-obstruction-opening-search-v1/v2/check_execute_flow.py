"""Run the exact worker execute/counter bodies with tiny generated runtime doubles.

This is an orchestration and exclusive-output regression, not a physics test.
The prior separately saved actual-native capture tests cover geometry integration.
"""
import ast,copy,gc,hashlib,json,signal,sys,tempfile,time,types,weakref
from contextlib import contextmanager
from dataclasses import asdict,dataclass
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));import pilot_contract as c

def timeout(*unused):raise TimeoutError('Generated execute-flow control30s cap')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30)
start=time.perf_counter()
import numpy as np

def digest(value):
 a=np.ascontiguousarray(value);return 'sha256:'+hashlib.sha256(a.tobytes()+str(a.shape).encode()+a.dtype.str.encode()).hexdigest()
def thaw(value):return json.loads(json.dumps(value))
@dataclass
class Tool:tool_id:str='generated_tool'
@dataclass
class Reward:target_per_mm3:float=1.;normal_per_mm3:float=.2
class Config:
 def __init__(self,**kwargs):self.record=kwargs;self.obstruction_opening=kwargs.get('obstruction_opening',False);self.fingerprint=c.semantic(self.to_record())
 def to_record(self):return {k:v for k,v in self.record.items() if k!='obstruction_opening' or v}
class Source:
 def __init__(self,subject,config):
  self.source_hash=c.semantic({'subject':subject,'config':config.to_record()});self.config=config
  self.structural_intensity=np.zeros((3,3,3),dtype=np.float32);self.observed_support=np.ones((3,3,3),dtype=bool)
  self.nominal_target=np.zeros((3,3,3),dtype=np.float32);self.nominal_target[1,1,1]=1
  self.reference_target=self.nominal_target;self.affine_ras_mm=np.eye(4);self._native_affine_ras_mm=self.affine_ras_mm
  self.public_target_domain=np.ones((3,3,3),dtype=bool);self.tools=(Tool(),)
  self.access=types.SimpleNamespace(center_mm=np.zeros(3),normal_inward=np.array([0.,0.,1.]),radius_mm=6.,window_id='generated')
  self.support_provenance={'target_domain_hash':digest(self.public_target_domain)}
  self._native_config=types.SimpleNamespace(fingerprint=c.semantic({'source':self.source_hash}))
  self._occupancy_derivation={'source_support_positive_voxels':26,'added_region_positive_voxels':1}
  self._supplied_goal_extent={'full_region_positive_voxels':1,'full_region_membership_mm3':1.}
  self._normalization_record={'source_image_hash':digest(self.structural_intensity),'method':'generated'}
class Engine:pass
class Budget:
 def __init__(self,*args,**kwargs):self.count=0;self.complete_flag=False
 def __enter__(self):return self
 def __exit__(self,kind,*unused):
  if kind is None:assert self.complete_flag
 def check(self):pass
 def snapshot(self):return {'native_preview_entries':self.count}
 def complete(self,*,history_complete):assert history_complete;self.complete_flag=True
class BudgetViolation(RuntimeError):pass
class SearchLimit(RuntimeError):pass
class Policy:
 def forward(self,*a,**k):raise AssertionError('Model prohibited in generated flow')
class Adam:
 def step(self,*a,**k):raise AssertionError('Optimizer prohibited in generated flow')
class Context:
 def __init__(self,task):self.fingerprint=c.semantic({'context':task._source_hash})
 def require_task(self,task):assert task.case.source_hash==task._source_hash
 def record(self):return {'context_hash':self.fingerprint,'generated':True}
class Task:
 def __init__(self,source):
  self.case=source;self._source_hash=source.source_hash;self.decision_model_hash=c.semantic({'model':source.source_hash})
  self._config=source._native_config;self._planning=False;self.terminated=False;self.history=[];self._steps=0;self.reward_spec=Reward()
  self._engine=types.SimpleNamespace(state_hash=c.semantic({'source':source.source_hash,'steps':0}))
 def observation(self):return types.SimpleNamespace(fingerprint=c.semantic({'observation':self._source_hash}))
 def _prepare_inventory(self):return {r['action_id']:r for r in self.candidate_rows() if r['feasible']}
 def candidate_rows(self):
  base={'family':'exposed_opening','action_id':'base-'+self._source_hash,'proposal_id':'base-'+self._source_hash,
    'proposal_reason':'PROPOSED_UNCERTIFIED','reason':'FEASIBLE','feasible':True,'tool_id':'generated_tool',
    'entry_mm':[0.,0.,0.],'tip_mm':[0.,0.,1.]}
  rows=[base]
  if self.case.config.obstruction_opening and self._steps==0:
   rows.append({**base,'family':'obstruction_opening','action_id':'added-'+self._source_hash,'proposal_id':'added-'+self._source_hash,'tip_mm':[1.,1.,1.]})
  return rows
 def candidate_inventory(self):
  self._prepare_inventory();rows=self.candidate_rows()
  return {'source_hash':self._source_hash,'decision_model_hash':self.decision_model_hash,'cavity_state_hash':self._engine.state_hash,
   'provider_version':'generated','declared_slots':len(rows),'emitted_count':len(rows),'accepted_count':len(rows),'rejected_count':0,
   'omitted_count':0,'duplicate_count':0,'unavailable_count':0,'complete':True,'ledger_complete':True,'ledger':rows}
 def planning_clone(self):
  result=Task(copy.copy(self.case));result._planning=True;return result
 def observed_greedy_search(self,**unused):
  model=self.planning_clone();actions=[];decisions=[]
  while not model.terminated:
   model._prepare_inventory();rows=model.candidate_rows()
   scores=[{'action_id':'STOP','reward':0.,'target_removed_mm3':0.,'normal_removed_mm3':0.}]
   scores.extend({'action_id':r['action_id'],'reward':1. if r['family']=='obstruction_opening' else -.2,
      'target_removed_mm3':1. if r['family']=='obstruction_opening' else 0.,'normal_removed_mm3':0.} for r in rows)
   picked=max(scores,key=lambda r:r['reward'])['action_id']
   decisions.append({'step':model._steps,'source_state_hash':model._engine.state_hash,'scores':scores,'selected_action_id':picked,
       'legal_nonstop_actions':len(rows),'scored_nonstop_actions':len(rows)})
   model._transition(picked);actions.append(picked)
  return tuple(actions),{'complete':True,'decisions':decisions,'model_transition_calls':len(actions)}
 def _transition(self,action):
  stopped=action=='STOP';record={'action_id':action,'reward':0. if stopped else 1.,'target_removed_mm3':0. if stopped else 1.,
      'normal_removed_mm3':0.,'removed_indices_native':[] if stopped else [[1,1,1]]}
  self.history.append(record);self._steps+=1;self.terminated=stopped
  self._engine.state_hash=c.semantic({'source':self._source_hash,'steps':self._steps})
  return types.SimpleNamespace(info=record)
 def step(self,action):return self._transition(action)
 def fresh(self):return Task(self.case)
 def metrics(self):return {'history':copy.deepcopy(self.history),'terminated':self.terminated,'total_reward':sum(r['reward'] for r in self.history)}
def audit(task,**unused):return {'accepted':True,'complete_episode':True,'committed_history_hash':c.semantic(task.history),
 'source_hash':task._source_hash,'decision_model_hash':task.decision_model_hash,'outcomes':{'total_reward':task.metrics()['total_reward']},'target_access_success':task.metrics()['total_reward']>0}

# Use the real OutputBudget class body (exclusive creation and byte accounting),
# avoiding imports of its unrelated checkpoint/ML dependencies.
io=ast.parse((ROOT/'src/resectionlab/patient_planning_cohort_io.py').read_text())
io_class=next(n for n in io.body if isinstance(n,ast.ClassDef) and n.name=='OutputBudget')
ioenv={'Path':Path,'json':json,'thaw_json':lambda v:v,'freeze_json':lambda v:v}
exec(compile(ast.fix_missing_locations(ast.Module(body=[io_class],type_ignores=[])),'OutputBudget','exec'),ioenv)
factory=types.ModuleType('resectionlab.public_patient_factory');factory.write=lambda *a,**k:None
modules={
 'torch':{'optim':types.SimpleNamespace(Adam=Adam),'load':lambda *a,**k:None},
 'resectionlab.core':{'thaw_json':lambda v:v,'freeze_json':lambda v:v,'array_digest':digest},
 'resectionlab.native_resection':{'NativeResectionEngine':Engine},
 'resectionlab.native_proposals':{'NominalCavityProposalConfig':Config},
 'resectionlab.native_spatial_task':{'NativeSpatialTask':Task},
 'resectionlab.patient_planning_admission':{'make_patient_planning_task':lambda source,**kw:(Task(source),Context(Task(source)))},
 'resectionlab.native_spatial_evaluation':{'evaluate_native_spatial_episode':audit},
 'resectionlab.observed_search':{'ObservedSearchLimit':SearchLimit},
 'resectionlab.patient_planning_cohort_io':{'OutputBudget':ioenv['OutputBudget']},
 'resectionlab.planning_budget':{'PlanningBudget':Budget,'PlanningBudgetViolation':BudgetViolation},
 'resectionlab.spatial_policy':{'SpatialPolicy':Policy}}
saved={k:sys.modules.get(k) for k in (*modules,'resectionlab','resectionlab.public_patient_factory')}
for name,fields in modules.items():
 mod=types.ModuleType(name);mod.__dict__.update(fields);sys.modules[name]=mod
pkg=types.ModuleType('resectionlab');pkg.public_patient_factory=factory;sys.modules['resectionlab']=pkg;sys.modules['resectionlab.public_patient_factory']=factory
try:
 with tempfile.TemporaryDirectory(prefix='generated-worker-flow-') as directory:
  temp=Path(directory);manifestrows=[];records={'release':{'learning_protocol':{'cohort_execution':{'proposal_config':{'max_candidates':120}},'public_target_context_variant':'generated'}},'public_index':{'cases':manifestrows}}
  (temp/'cohort.json').write_text('{}');refs={'cohort':{'path':str(temp/'cohort.json')}};visits=[]
  for subject in c.TRAIN:
   path=temp/(subject+'.json');path.write_text(json.dumps({'patient_id':subject,'input_files':{'image':{'sha256':'generated'}}}))
   manifestrows.append({'patient_id':subject,'path':str(path),'sha256':c.sha(path)})
   source=Source(subject,Config(max_candidates=120,obstruction_opening=False));task=Task(source)
   records['world_'+subject]={'source_hash':source.source_hash,'decision_model_hash':task.decision_model_hash,'initial_observation_hash':task.observation().fingerprint}
   records['initial-inventory_'+subject]=task.candidate_inventory()
  del source,task
  def prepare(dest,*args,public_manifest_path,proposal_config,occupancy_condition,**kwargs):
   assert occupancy_condition==c.OCCUPANCY
   subject=json.loads(public_manifest_path.read_text())['patient_id'];visits.append((subject,proposal_config.obstruction_opening))
   return Source(subject,proposal_config),{},{},{}
  factory.prepare_public_source=prepare
  release={'condition_admission':{v:{'limits':{},'learning_protocol_hash':'generated'} for v in c.CONDITIONS}}
  outcomes=[]
  for version in ('v1','v2'):
   path=HERE.with_name('paired-obstruction-opening-search-'+version)/'search_worker.py'
   tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('execution_counters','execute')]
   env={'ROOT':temp,'TRAIN':c.TRAIN,'CONDITIONS':c.CONDITIONS,'OCCUPANCY':c.OCCUPANCY,'ARMS':c.ARMS,'EXECUTION':c.EXECUTION,
    'WORKER_SECONDS':30,'OUTPUT_BYTES':2*1024**2,'inputs':lambda:(records,refs),'canonical':c.canonical,'semantic':c.semantic,
    'sha':c.sha,'without_ids':c.without_ids,'inventory_summary':c.inventory_summary,'contextmanager':contextmanager,'asdict':asdict,
    'gc':gc,'json':json,'Path':Path,'time':time,'weakref':weakref}
   exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
   output=temp/version;before=len(visits)
   try:result=env['execute'](release,'generated',output,lambda *a,**kw:None)
   except RuntimeError:
    result=json.loads((output/'result.json').read_text())
    if version!='v1':raise
   if version=='v1':
    assert len(visits)==before and all(r['failure']['type']=='UnboundLocalError' for r in result['arms'])
    assert result['native_counts']=={'search':0,'rollout':0,'replay':0}
   else:
    assert c.complete_result(result) and len(visits)-before==8
    assert visits[before:]==[(s,flag) for s in c.TRAIN for flag in (False,True)]
    assert result['native_counts']=={'search':12,'rollout':12,'replay':12}
    for i,row in enumerate(result['arms']):
     assert row['actions'][-1]=='STOP' and row['source_released']
     assert row['decision_summary'][0]['selected_family']==('obstruction_opening' if i%2 else 'STOP')
     assert row['decision_summary'][0]['added_target_removing_count']==i%2
     plan=json.loads((output/f'arm-{i:02d}/plan.json').read_text());replay=json.loads((output/f'arm-{i:02d}/replay.json').read_text())
     assert c.semantic(plan['plan'])==plan['plan_seal'] and plan['plan']['history']==replay['metrics']['history']
    assert factory.write is not None and Task._prepare_inventory.__name__=='_prepare_inventory'
   outcomes.append({'version':version,'status':result['status'],'source_visits':len(visits)-before,'native_counts':result['native_counts']})
finally:
 for key,value in saved.items():
  if value is None:sys.modules.pop(key,None)
  else:sys.modules[key]=value
receipt={'status':'pass','elapsed_seconds':time.perf_counter()-start,'controls':outcomes,'worker_sha256':c.sha(HERE/'search_worker.py'),
 'scope':'Exact execute/run_arm/counters with generated runtime doubles and real exclusive OutputBudget; no patient/native physics/model execution',
 'patient_reads':0,'real_native_steps':0,'model_forwards':0,'optimizer_updates':0}
(HERE/'execute-flow-controls.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
signal.setitimer(signal.ITIMER_REAL,0)
