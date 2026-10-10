"""Generated absent-option fingerprints; invoked only in released test slot."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
BASE=Path(__file__).resolve().parents[1]
branch=sys.argv[1];assert branch in ('baseline','stage')
sys.path.insert(0,str(ROOT/'src'))
import resectionlab
resectionlab.__path__.insert(0,str(BASE/branch/'resectionlab'))
from dataclasses import replace
import numpy as np
from resectionlab.core import semantic_digest,array_digest
from resectionlab.native_spatial_task import make_native_opening_task,NativeSpatialTask,DERIVED_OCCUPANCY_SOURCE_KIND,OPENING_TOOLS
from resectionlab.native_proposals import SUPPLIED_GOAL_REGION
from resectionlab.public_target_context import VERSION
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode

def obs(o):
 return {'source':o.source_id,'images':array_digest(o.image_channels),'coverage':array_digest(o.coverage),
  'available':array_digest(o.channel_available),'state':array_digest(o.state_features),
  'affine':array_digest(o.affine_ras_mm),'context':None if o.public_target_context is None else o.public_target_context.fingerprint}

def run(context):
 task=make_native_opening_task()
 if context:
  c=task.case
  task=NativeSpatialTask(replace(c,track='annotation_assisted',support_source_kind=DERIVED_OCCUPANCY_SOURCE_KIND,
   support_derivation='generated explicit union',occupancy_source_support=c.observed_support,
   target_source_kind='supplied_annotation',target_derivation='generated supplied whole region',
   target_semantics=SUPPLIED_GOAL_REGION,public_target_context_variant=VERSION,
   public_target_domain=np.ones(c.observed_support.shape,bool)),max_steps=3)
 states=[]
 states.append({'observation':obs(task.observation()),'inventory':semantic_digest(task.candidate_inventory())})
 row=next(r for r in task.candidate_inventory()['ledger'] if r['feasible'] and r['tool_id']==OPENING_TOOLS[0].tool_id and r['voxel']==[4,4,1])
 task.step(row['action_id'])
 states.append({'observation':obs(task.observation()),'inventory':semantic_digest(task.candidate_inventory())})
 task.step('STOP')
 evaluation=evaluate_native_spatial_episode(task)
 evaluation.pop('evaluation_seconds',None)
 return {'source':task.case.source_hash,'model':task.decision_model_hash,'states':states,
  'history':semantic_digest(task.metrics()['history']),'metrics':semantic_digest(task.metrics()),'evaluation':semantic_digest(evaluation)}
print(json.dumps({'historical_generated':run(False),'union_global_default':run(True)},sort_keys=True))
