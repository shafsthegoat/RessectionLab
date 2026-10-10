"""Stdlib metadata packaging only; no project, checkpoint decode or model."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
FIT=ROOT/'build/goal-conditioned-policy-v1/full-teacher-refit-run-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
prior=json.loads((ROOT/'build/goal-conditioned-policy-v1/full-teacher-refit-v1/input-index.json').read_text())
data={k:v for k,v in prior['data'].items() if k.startswith('teacher-')}
for key,name in [('refit-experiment.json','experiment-freeze.json'),('refit-result.json','result.json')]:
    path=FIT/name;data[key]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
fit=json.loads((FIT/'result.json').read_text());checkpoint=fit['final_checkpoint']
assert checkpoint['sha256']=='5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7'
assert fit['experiment_hash']=='sha256:fd211e00dbe6dd1c9ed50a58d6c3b9f9f8896a0acd3a7815b2c52e4a8dc1014f'
# Artifact hash comes from the root-supervised fit receipt. Parent and decoder
# both check actual checkpoint bytes before the released run; packaging does not load it.
caps={'hard_total_wall_seconds':45,'sampled_owned_tree_rss_bytes':1073741824,'geometry_previews':2048,'actor_forwards':48,'planning_transitions':48,'authoritative_execution_transitions':48,'threads':1,'attempts':1}
inputs={'version':'fixed24-TRAIN-greedy-rollout-inputs-v1','data':data,'experiment_hash':fit['experiment_hash'],'checkpoint':{'path':str(Path(checkpoint['path']).relative_to(ROOT)),'sha256':checkpoint['sha256'],'parameter_hash':checkpoint['parameter_hash']},'caps':caps}
dump(OUT/'input-index.json',inputs)
old=json.loads((ROOT/'build/goal-conditioned-policy-v1/full-teacher-refit-v1/source-index.json').read_text())
files={k:v for k,v in old['source_files'].items() if 'full-teacher-refit-v1/' not in k and not k.endswith('train_diagnostic_readout.py')}
for relative,expected in files.items():assert sha(ROOT/relative)==expected,relative
for name in ('rollout_worker.py','run_rollout_owned.py','rollout_summary.py'):
    path=OUT/name;ast.parse(path.read_text());files[str(path.relative_to(ROOT))]=sha(path)
source={'version':'fixed24-TRAIN-greedy-rollout-source-v1','source_files':files,'input_index':{'path':str((OUT/'input-index.json').relative_to(ROOT)),'sha256':sha(OUT/'input-index.json')},'experiment_hash':fit['experiment_hash'],'caps':caps,'frozen_counts':{'tasks':24,'checkpoint_loads':1,'optimizer_updates':0,'new_search_calls':0,'new_teacher_calls':0,'SELECT_or_MEASUREMENT_tasks':0,'patient_reads':0}}
dump(OUT/'source-index.json',source)
recipe={'status':'STAGED_NOT_RELEASED_NOT_EXECUTED','command':'.venv/bin/python build/goal-conditioned-policy-v1/full-teacher-rollout-v1/run_rollout_owned.py --expected-head <RELEASED_COMMIT> --source-index build/goal-conditioned-policy-v1/full-teacher-rollout-v1/source-index.json --source-index-sha256 '+sha(OUT/'source-index.json'),'source_index_sha256':sha(OUT/'source-index.json'),'input_index_sha256':sha(OUT/'input-index.json'),'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'caps':caps,'output':'build/goal-conditioned-policy-v1/full-teacher-rollout-run-v1','checkpoint':inputs['checkpoint']}
dump(OUT/'run-recipe.json',recipe)
print(json.dumps({'source_index_sha256':sha(OUT/'source-index.json'),'input_index_sha256':sha(OUT/'input-index.json'),'run_recipe_sha256':sha(OUT/'run-recipe.json'),'data_files':len(data),'execution':'none'},indent=2))
