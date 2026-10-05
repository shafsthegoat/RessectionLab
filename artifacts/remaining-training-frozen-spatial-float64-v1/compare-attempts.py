"""Compare completed saved attempts without task/model imports or new episodes."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'remaining-training-frozen-spatial-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
indices={str(p.name):load(p/'output-sha256.json') for p in (OLD,ROOT)}
for folder in (OLD,ROOT):
    for name,digest in indices[folder.name].items():
        assert sha(folder/name)==digest,(folder.name,name)
a=load(OLD/'declaration-input.json');b=load(ROOT/'declaration-input.json')
new_fields={'changed_numerical_sources','revision_of','revision_created_at','revision_reason'}
assert set(b)-set(a)==new_fields
assert {k:v for k,v in a.items() if k!='source_sha256'}=={k:v for k,v in b.items() if k not in new_fields|{'source_sha256'}}
changed_sources=sorted(k for k in set(a['source_sha256'])|set(b['source_sha256']) if a['source_sha256'].get(k)!=b['source_sha256'].get(k))
assert changed_sources==['src/resectionlab/native_spatial_evaluation.py','src/resectionlab/native_spatial_task.py']
assert b['revision_of']['sha256']==sha(OLD/'declaration-input.json')
assert sha(OLD/'frozen-checkpoint.pt')==sha(ROOT/'frozen-checkpoint.pt')==a['checkpoint']['sha256']
arithmetic={'target_removed_mm3','normal_removed_mm3','reward'}
decision_arithmetic={'target_removed_mm3','normal_removed_mm3','reward'}
def without_arithmetic(value):
    if isinstance(value,dict):return {k:without_arithmetic(v) for k,v in value.items() if k not in arithmetic}
    if isinstance(value,list):return [without_arithmetic(v) for v in value]
    return value
rows=[]
for subject in a['subjects']:
    p=load(OLD/subject/'preparation.json');q=load(ROOT/subject/'preparation.json')
    assert {k:v for k,v in p.items() if k!='preparation_seconds'}=={k:v for k,v in q.items() if k!='preparation_seconds'}
    row={'subject':subject,'preparation_equal_except_time':True,'status':q['status'],'episodes':[]}
    if q['status']=='prepared':
        for method in a['methods']:
            x=load(OLD/subject/f'{method}.json');y=load(ROOT/subject/f'{method}.json')
            assert len(x['decisions'])==len(y['decisions'])
            assert len(x['metrics']['history'])==len(y['metrics']['history'])
            changes=[]
            for i,(hx,hy) in enumerate(zip(x['metrics']['history'],y['metrics']['history'])):
                keys={k for k in set(hx)|set(hy) if hx.get(k)!=hy.get(k)}
                assert keys<=arithmetic,(subject,method,i,keys)
                changes.append({k:{'old':hx[k],'new':hy[k],'difference':hy[k]-hx[k]} for k in sorted(keys)})
            assert without_arithmetic(x['metrics']['history'])==without_arithmetic(y['metrics']['history'])
            for dx,dy in zip(x['decisions'],y['decisions']):
                keys={k for k in set(dx)|set(dy) if dx.get(k)!=dy.get(k)}
                assert keys<={'committed_info','reward','decision_seconds','transition_seconds'},(subject,method,keys)
                assert without_arithmetic(dx['committed_info'])==without_arithmetic(dy['committed_info'])
            assert y['status']=='complete' and y['independent_evaluation']['accepted'] is True
            row['episodes'].append({'method':method,'old_status':x['status'],'new_status':y['status'],
                'actions':[d['action_id'] for d in y['decisions']],
                'actions_ids_masks_observations_logits_probabilities_values_equal':True,
                'removed_contact_cells_microsteps_geometry_provenance_equal':True,
                'history_arithmetic_changes':changes,
                'old_unaccepted_simulator_return_if_failed':x.get('simulated_return') if x['status']!='complete' else None,
                'old_accepted_return':x.get('simulated_return') if x['status']=='complete' else None,
                'new_accepted_return':y['independent_evaluation']['outcomes']['total_reward'],
                'return_arithmetic_difference':y['simulated_return']-x['simulated_return'],
                'old_episode_sha256':sha(OLD/subject/f'{method}.json'),'new_episode_sha256':sha(ROOT/subject/f'{method}.json')})
    rows.append(row)
report={'status':'passed','source_sha256':sha(Path(__file__)),'original_attempt_unchanged':True,
        'comparison_scope':'All saved preparations, nine decision sequences and complete committed physical histories; no patient arrays, policy forward, or native replay.',
        'checkpoint_bytes_equal':True,'declared_support_timestamp_equal':a['declared_at']==b['declared_at'],
        'declaration_settings_roles_order_seed_equal':True,'changed_numerical_sources':changed_sources,
        'original_execution_index_sha256':sha(OLD/'output-sha256.json'),'repeat_execution_index_sha256':sha(ROOT/'output-sha256.json'),
        'original_declaration_sha256':sha(OLD/'declaration-input.json'),'repeat_declaration_sha256':sha(ROOT/'declaration-input.json'),
        'history_changed_fields_allowed':sorted(arithmetic),'new_episodes':0,'new_policy_forwards':0,
        'old_failed_outcomes_reclassified':False,'patients':rows}
(ROOT/'attempt-comparison.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('PASS: five preparations unchanged except timing; nine behavior/physical histories unchanged except target/normal/reward arithmetic.')
