"""Small generated source-review reproductions; no model/patient input."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'build/healthy-corridor-bridge-v1';OUT=Path(__file__).resolve().parent
source_sha=hashlib.sha256((C/'healthy_corridor.py').read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('corridor_review_fixture',C/'test_corridor.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
# Constructor on the smallest possible two-interior-cell support.
task,prepared,arrays=t.fixture();support=np.zeros(task.support.shape,bool);support[6:9,6:9,10:14]=True
actor=t.thaw_json(task.actor_contract);actor['support']['output_sha256']=t.array_digest(support);actor['task']['support_output_sha256']=t.array_digest(support)
small=t.c.CorridorTask(actor,task.t1,support,task.coverage,task.definition)
findings={'source_sha256':source_sha,'two_interior_cells_waypoints':[list(p) for p in small.waypoints],'waypoints_distinct':len(set(small.waypoints))==len(small.waypoints)}
# Recompute valid seal for a suboptimal SEARCH STOP and ask public-only preflight.
accounting={'policy_callback_calls':0,'nominal_candidate_comparisons':len(task.candidates)+1,'actor_forward_calls':0,'optimizer_updates':0}
record=t.c._strategy(task,'SEARCH','STOP',accounting)
try:t.c._preflight(task,record,'SEARCH');findings['suboptimal_search_STOP_accepted']=True
except ValueError as error:findings['suboptimal_search_STOP_accepted']=False;findings['STOP_refusal']=str(error)
# Public metadata altered after admission should not weaken actor-only prerequisites.
actor=t.thaw_json(task.actor_contract);actor['support']['project_fit_roles']=['SELECT']
try:t.c.CorridorTask(actor,task.t1,task.support,task.coverage,task.definition);findings['protected_fit_role_accepted']=True
except ValueError as error:findings['protected_fit_role_accepted']=False;findings['fit_refusal']=str(error)
(OUT/'initial-findings.json').write_text(json.dumps(findings,indent=2)+'\n');print(json.dumps(findings))
