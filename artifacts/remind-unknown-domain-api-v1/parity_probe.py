"""Generated historical default records; no source data or model access."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import resectionlab
resectionlab.__path__.insert(0,str(HERE/sys.argv[1]/'resectionlab'))
from dataclasses import replace
import numpy as np
from resectionlab.core import array_digest,semantic_digest
from resectionlab.native_spatial_task import (make_native_opening_task,NativeSpatialTask,
    OPENING_TOOLS,DERIVED_OCCUPANCY_SOURCE_KIND)
from resectionlab.native_proposals import SUPPLIED_GOAL_REGION
from resectionlab.public_target_context import VERSION
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode

def record(task):
    obs=task.observation()
    return {'source_hash':task.case.source_hash,'reference_hash':task.case.reference_hash,
        'model_hash':task.decision_model_hash,'config_hash':task._config.fingerprint,
        'observation_hash':obs.fingerprint,'inventory_hash':semantic_digest(task.candidate_inventory()['ledger']),
        'array_hashes':{name:array_digest(getattr(obs,name)) for name in ('image_channels','coverage','action_geometry','action_mask','state_features')},
        'target_context':None if obs.public_target_context is None else obs.public_target_context.fingerprint}

out={}
for condition in ('historical_generated','historical_union_context'):
    task=make_native_opening_task()
    if condition=='historical_union_context':
        raw=task.case.observed_support.copy();raw[4,4,5]=False
        task=NativeSpatialTask(replace(task.case,track='annotation_assisted',support_source_kind=DERIVED_OCCUPANCY_SOURCE_KIND,
            target_source_kind='supplied_annotation',target_derivation='generated full supplied region',
            target_semantics=SUPPLIED_GOAL_REGION,occupancy_source_support=raw,
            public_target_context_variant=VERSION,public_target_domain=np.ones(raw.shape,bool)),max_steps=2)
    rows=[record(task)]
    for tool,voxel in ((OPENING_TOOLS[0].tool_id,(4,4,1)),(OPENING_TOOLS[1].tool_id,(4,4,5))):
        key=next(row['action_id'] for row in task.candidate_inventory()['ledger'] if row['feasible'] and row['tool_id']==tool and row['voxel']==list(voxel))
        task.step(key)
        if not task.terminated:rows.append(record(task))
    evaluation=evaluate_native_spatial_episode(task);evaluation.pop('evaluation_seconds',None)
    out[condition]={'states':rows,'history_hash':semantic_digest(task.metrics()['history']),
        'metrics_hash':semantic_digest(task.metrics()),'evaluation_hash':semantic_digest(evaluation)}
print(json.dumps(out,sort_keys=True,indent=2))
