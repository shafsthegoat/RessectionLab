"""Static source hashes/AST and JSON identities only; no candidate imports/tests."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/goal-conditioned-policy-v1/goal-relation-policy-v1'
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
digest=lambda x:'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
promotion=read(BASE/'promotion-index.json');source=read(BASE/'source-index.json');inputs=read(BASE/'input-index.json');recipe=read(BASE/'run-recipe.json')
assert sha(BASE/'promotion-index.json')=='d9aedf35f5819bb350951ce11f0172e893e70a7e7553614fad3c2d5b034ea512'
assert sha(BASE/'source-index.json')=='c2feb9288dcd15f4feeb85cb999d86fa4c190600c80069ac11cde3352b39274c'
assert sha(BASE/'input-index.json')==source['input_index']['sha256']
assert sha(BASE/'promotion.patch')==promotion['patch_sha256']
staged={r['destination']:r for r in promotion['files']}
for name,row in staged.items():
    assert sha(ROOT/row['source'])==row['source_sha256']
    previous=ROOT/name
    assert (not previous.exists()) if row['previous_sha256'] is None else sha(previous)==row['previous_sha256']
    ast.parse((ROOT/row['source']).read_text())
for name,value in source['source_files'].items():
    assert sha(ROOT/(staged[name]['source'] if name in staged else name))==value,name
assert set(inputs['data'])=={*(f'teacher-{i:02d}.json' for i in range(24)),*(f'IL-update-{i:02d}.json' for i in range(1,33)),*(f'baseline-before-state-{i:02d}.json' for i in range(40)),'experiment-freeze.json'}
for name,row in inputs['data'].items():
    assert row['path'].endswith('.json') and sha(ROOT/row['path'])==row['sha256']
    if name.startswith('baseline-before-state-'):
        d=read(ROOT/row['path']);assert d['parameter_hash']=='sha256:e7215950e221045b8e9612e4482837f9b1ff2c52278454b26efc746598cf1487'
new=read(BASE/'prospective-experiment.json');old=read(ROOT/'build/goal-conditioned-policy-v1/full-teacher-refit-v1/prospective-experiment.json')
arch=read(BASE/'prospective-architecture.json')
assert digest(new)==promotion['experiment_hash']==source['refit_experiment_hash']==inputs['refit_experiment_hash']==recipe['experiment_hash']
assert digest(arch)==promotion['architecture_hash']==new['architecture_hash']
assert new['teacher_states']==old['teacher_states'] and new['family_manifest']==old['family_manifest']
assert new['protocol']['updates']==old['protocol']['updates']==32 and new['protocol']['batch_size']==40
assert new['protocol']['loss_forward_calls']==1280 and new['protocol']['fixed_before_after_TRAIN_forwards']==80
assert source['caps']==recipe['caps']=={'attempts':1,'geometry_previews':2048,'sampled_owned_tree_rss_bytes':1073741824,'threads':1,'wall_seconds':180}
assert recipe['source_index_sha256']==sha(BASE/'source-index.json') and recipe['promotion_index_sha256']==sha(BASE/'promotion-index.json')
# Existing body remains the old exact loss; only opt-in type/initialization checks change.
for function in ('contact_imitation_loss','contact_gradient_step','contact_reinforce_loss'):
    trees=[ast.parse(p.read_text()) for p in (ROOT/'src/resectionlab/contact_learning.py',ROOT/staged['src/resectionlab/contact_learning.py']['source'])]
    bodies=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==function) for t in trees]
    assert ast.dump(bodies[0],include_attributes=False)==ast.dump(bodies[1],include_attributes=False)
summary={'status':'STATIC_SOURCE_INVENTORY_VERIFIED','promotion_index_sha256':sha(BASE/'promotion-index.json'),'source_index_sha256':sha(BASE/'source-index.json'),'input_index_sha256':sha(BASE/'input-index.json'),'recipe_sha256':sha(BASE/'run-recipe.json'),'patch_sha256':sha(BASE/'promotion.patch'),'source_promotion_files':len(staged),'input_metadata_files':len(inputs['data']),'teacher_corpus_and_family_unchanged':True,'existing_loss_and_gradient_bodies_unchanged':True,'source_tests_executed':0,'candidate_imports':0,'model_constructions':0,'native_tasks':0,'checkpoint_reads':0,'forwards':0,'optimizer_steps':0}
(OUT/'source-inventory.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2))
