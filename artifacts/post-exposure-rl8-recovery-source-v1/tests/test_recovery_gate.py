"""Generated metadata only: no numerical imports, checkpoint decoder or patient reads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rl8_recovery_contract',HERE/'pilot_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class RecoveryGate(unittest.TestCase):
 def setUp(self):
  self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.addCleanup(patch.stopall);patch.object(c,'ROOT',self.root).start()
  self.rows={};self.generated={};self.param='sha256:'+'1'*64;self.initial='sha256:'+'2'*64
  def put(path,value):
   data=(json.dumps(value,sort_keys=True)+'\n').encode();p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
   return {'path':path,'sha256':hashlib.sha256(data).hexdigest()}
  self.put=put
  cp={'path':'original/IL-final.psckpt','parameter_hash':self.param}; raw=b'generated checkpoint bytes; not a model'
  (self.root/'original').mkdir();(self.root/cp['path']).write_bytes(raw);cp.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  patch.object(c,'ORIGINAL_IL_CHECKPOINT',cp).start()
  def add(key,value):
   self.rows[key]=value; self.generated[key]=put('original/'+key+'.json',value)
  def plan(subject,n):
   history=[{'action_id':'MOVE'+str(i),'reward':1.,'outcome_scope':'planning'} for i in range(n)]
   if n==1:history=[{'action_id':'STOP','reward':0.,'outcome_scope':'planning'}]
   value={'source_hash':'source-'+subject,'decision_model_hash':'model-'+subject,'context_hash':'context-'+subject,
    'initial_observation_hash':'obs-'+subject,'max_steps':24,'actions':[x['action_id'] for x in history],
    'history':history,'terminal_reason':'HORIZON' if n==24 else 'STOP','parameter_hash':self.param,'learning_updates':64}
   return {'plan':value,'plan_seal':c.semantic(value)}
  accepted={}
  for subject in c.TRAIN[:-1]:
   p=plan(subject,1);add('plan_'+subject,p)
   history=[{**h,'outcome_scope':'authoritative'} for h in p['plan']['history']]
   audit={'accepted':True,'complete_episode':True,'committed_history_hash':c.semantic(history),
          'geometry':{'feasible':True,'complete_tool_checked':True}}
   add('replay_'+subject,{'metrics':{'history':history},'independent_geometry':audit})
   accepted[subject]={'complete':True,'parameter_hash':self.param,'plan_seal':p['plan_seal']}
  p=plan('ReMIND-045',24);add('plan045',p)
  add('release',{'source_index':{'path':'original/source_index.json','sha256':'3'*64}})
  add('source_index',{'generated':True});add('configuration',{'generated':True})
  add('reload',{'completed_updates':64,'exact_parameter_match':True,'parameter_hash':self.param,'sha256':cp['sha256']})
  add('update64',{'completed_updates':64,'after_parameter_hash':self.param})
  add('dynamics',{'updates':[{'after_parameter_hash':self.param} for _ in range(64)]})
  add('result',{'status':'failed_or_unresolved','method':'IL','TRAIN':list(c.TRAIN),'optimizer_updates':{'IL':64,'RL':0},
   'failure':{'type':'ValueError','message':'Independent native geometry rejected replay','committed':False},
   'fresh_common_initialization_verified':True,'initial_parameter_hash':self.initial,
   'teacher_decisions':29,'teacher_steps':c.TEACHER_STEPS,'teacher_statuses':{s:'complete_replayed' for s in c.TRAIN},
   'loss_forward_calls':1856,'teacher_logit_forwards':29,'checkpoint_loads':1,'search_calls':0,'private_reference_reads':0,
   'SELECT_EVAL_opened':False,'TRAIN_greedy':accepted,
   'checkpoints':{'IL':{'path':'IL-final.psckpt',**{k:cp[k] for k in ('sha256','bytes','parameter_hash')}}}})
  add('worker_final',{'canonical_result_sha256':self.generated['result']['sha256']})
  add('receipt',{'status':'failed_or_unresolved','exit_code':1,'cleanup_errors':[],'final_owned_pids':[],
   'worker_termination_confirmed':True,'result_sha256':self.generated['result']['sha256'],
   'release_sha256':self.generated['release']['sha256'],'source_index':self.rows['release']['source_index']})
  patch.object(c,'ORIGINAL_IL',self.generated).start()
  for name in ('evaluation.py','patient_planning_preflight.py'):
   path=self.root/'src/resectionlab'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# generated source\n')
  self.plan=p['plan'];hist=[{**h,'outcome_scope':'authoritative'} for h in self.plan['history']]
  self.audit={'accepted':True,'complete_episode':True,'source_hash':self.plan['source_hash'],
   'decision_model_hash':self.plan['decision_model_hash'],'committed_history_hash':c.semantic(hist),
   'geometry':{'feasible':True,'complete_tool_checked':True,'action_count':24,'failures':[]}}
  self.replay={'independent_geometry':self.audit,'metrics':{'history':hist,'steps':24,'terminated':True,
   'source_hash':self.plan['source_hash'],'decision_model_hash':self.plan['decision_model_hash']}}
  self.index={'head':'4'*40,'files':{f'src/resectionlab/{n}':c.sha(self.root/f'src/resectionlab/{n}') for n in ('evaluation.py','patient_planning_preflight.py')}}
  self.rindex=put(c.RECOVERY_ROOT+'/source-index.json',self.index)
  joins={'failed_result':'result','failed_receipt':'receipt','original_release':'release','original_source_index':'source_index','configuration':'configuration','plan':'plan045'}
  self.release={'execution_released':True,'subject':'ReMIND-045','parameter_hash':self.param,'expected_head':self.index['head'],
   'source_index':self.rindex,'inputs':{key:self.generated[v] for key,v in joins.items()}}
  self.release['inputs']['checkpoint']={k:cp[k] for k in ('path','sha256')}
  self.rrelease=put(c.RECOVERY_ROOT+'/root-release.json',self.release)
  self.result={'status':'complete_saved_plan_diagnostic','subject':'ReMIND-045','independent_accepted':True,
   'full_history_equal':True,'committed_actions':24,'source_visits':1,'checkpoint_loads':0,'optimizer_updates':0,
   'policy_forwards':0,'teacher_search_calls':0,'training_admitted':False,'release_sha256':self.rrelease['sha256'],
   'checkpoint_sha256':cp['sha256'],'saved_plan_sha256':self.generated['plan045']['sha256'],
   'original_IL_result_sha256':self.generated['result']['sha256'],'original_IL_parent_receipt_sha256':self.generated['receipt']['sha256'],
   'saved_plan_seal':p['plan_seal'],'parameter_hash':self.param,'prior_learning_updates':64,
   'source_hash':self.plan['source_hash'],'context_hash':self.plan['context_hash'],'decision_model_hash':self.plan['decision_model_hash'],
   'saved_history_hash':c.semantic(self.plan['history']),'independent_geometry':self.audit['geometry']}
  self.rpins={};patch.object(c,'RECOVERY_INPUTS',self.rpins).start()
  self.receipt={'status':'complete','exit_code':0,'cleanup_errors':[],'remaining_owned_pids':[],
   'worker_termination_confirmed':True,'source_index':self.rindex,'release_sha256':self.rrelease['sha256']}
  self.refresh()
 def refresh(self):
  ref=self.put(c.RECOVERY_ROOT+'/attempt-01/native-replay.json',self.replay);self.result['native_replay_sha256']=ref['sha256']
  ref=self.put(c.RECOVERY_ROOT+'/attempt-01/result.json',self.result);self.receipt['result_sha256']=ref['sha256']
  self.ref=self.put(c.RECOVERY_RECEIPT,self.receipt)
  self.rpins.update(receipt=self.ref,result={'path':c.RECOVERY_ROOT+'/attempt-01/result.json','sha256':self.receipt['result_sha256']},
   replay={'path':c.RECOVERY_ROOT+'/attempt-01/native-replay.json','sha256':self.result['native_replay_sha256']},release=self.rrelease,source_index=self.rindex)
 def test_exact_generated_gate_accepts_failed_original_only_with_separate_recovery(self):
  evidence=c.recovered_il_evidence(self.ref)
  self.assertEqual(evidence['original_status_preserved'],'failed_or_unresolved')
  self.assertTrue(evidence['reevaluation_accepted'])
 def test_placeholder_and_old_diagnostic_refuse_before_original_reads(self):
  for ref in (c.PENDING_RECOVERY,{'path':c.RECOVERY_RECEIPT,'sha256':'x'*64},
              {'path':'build/post-exposure-IL045-evaluation-diagnostic-v1/attempt-01.supervision/receipt.json','sha256':'a'*64}):
   with self.subTest(ref=ref),patch.object(c,'original_il_evidence',side_effect=AssertionError('must not read')):
    with self.assertRaises(ValueError):c.recovered_il_evidence(ref)
 def test_recovery_failure_partial_updates_or_changed_identity_refused(self):
  for field,value in [('independent_accepted',False),('full_history_equal',False),('committed_actions',23),
   ('policy_forwards',1),('optimizer_updates',1),('checkpoint_loads',1),('teacher_search_calls',1),
   ('prior_learning_updates',63),('parameter_hash','wrong'),('source_hash','wrong'),('context_hash','wrong'),
   ('decision_model_hash','wrong'),('saved_history_hash','wrong'),('original_IL_result_sha256','wrong')]:
   old=self.result[field];self.result[field]=value;self.refresh()
   with self.subTest(field=field),self.assertRaises(ValueError):c.recovered_il_evidence(self.ref)
   self.result[field]=old
  self.refresh()
 def test_clean_parent_required(self):
  for field,value in [('status','failed_or_unresolved'),('exit_code',1),('cleanup_errors',['bad']),
   ('remaining_owned_pids',[123]),('worker_termination_confirmed',False)]:
   old=self.receipt[field];self.receipt[field]=value;self.refresh()
   with self.subTest(field=field),self.assertRaises(ValueError):c.recovered_il_evidence(self.ref)
   self.receipt[field]=old
 def test_geometry_and_replayed_history_authenticate_even_if_result_says_accepted(self):
  cases=[('accepted',False),('complete_episode',False),('committed_history_hash','wrong')]
  for field,value in cases:
   old=self.audit[field];self.audit[field]=value;self.refresh()
   with self.subTest(field=field),self.assertRaises(ValueError):c.recovered_il_evidence(self.ref)
   self.audit[field]=old
  self.replay['metrics']['history'][0]['reward']=2.;self.audit['committed_history_hash']=c.semantic(self.replay['metrics']['history']);self.refresh()
  with self.assertRaises(ValueError):c.recovered_il_evidence(self.ref)
 def test_original_failure_cannot_be_rewritten_as_clean_or_update63(self):
  with patch.object(c,'small',side_effect=lambda ref:copy.deepcopy(next(self.rows[k] for k,v in self.generated.items() if v==ref))):
   for field,value in [('status','complete_matched_TRAIN_endpoint'),('optimizer_updates',{'IL':63,'RL':0})]:
    old=self.rows['result'][field];self.rows['result'][field]=value
    with self.subTest(field=field),self.assertRaises(ValueError):c.original_il_evidence()
    self.rows['result'][field]=old
 def test_old_accepted_three_routes_still_required(self):
  with patch.object(c,'small',side_effect=lambda ref:copy.deepcopy(next(self.rows[k] for k,v in self.generated.items() if v==ref))):
   self.rows['replay_ReMIND-015']['independent_geometry']['accepted']=False
   with self.assertRaises(ValueError):c.original_il_evidence()
 def test_actual_file_digest_and_checkpoint_bytes_are_enforced(self):
  p=self.root/c.ORIGINAL_IL_CHECKPOINT['path'];p.write_bytes(b'changed')
  with self.assertRaises(ValueError):c.recovered_il_evidence(self.ref)
 def test_checker_binding_enforced(self):
  (self.root/'src/resectionlab/evaluation.py').write_text('# different checker\n')
  with self.assertRaises(ValueError):c.recovered_il_evidence(self.ref)
 def test_worker_supervisor_and_protocol_generators_unchanged(self):
  old=HERE.parent/'post-exposure-learning-v1'
  for name in ('cohort_worker.py','run_owned.py'):
   self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes())
  import ast
  before=ast.parse((old/'pilot_contract.py').read_text());after=ast.parse((HERE/'pilot_contract.py').read_text())
  for name in ('expected_configuration','canonical_configuration','complete_result','inputs'):
   one=next(x for x in before.body if isinstance(x,ast.FunctionDef) and x.name==name)
   two=next(x for x in after.body if isinstance(x,ast.FunctionDef) and x.name==name)
   self.assertEqual(ast.dump(one),ast.dump(two),name)
  self.assertEqual(c.METHODS['RL'],{'updates':8,'worker_seconds':3540,'parent_seconds':3600,'source_visits':44,
   'loss_forward_cap':768,'policy_forward_cap':1661,'native_preview_cap':385440,'teacher_trace_reuses':0,'teacher_cache_payload_bytes':0})

if __name__=='__main__':unittest.main(verbosity=2)
