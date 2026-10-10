"""Metadata-only completion of a prior native audit. No project imports/replay."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'build/learned-transfer-root-integration-v1';ATTEMPT=BASE/'actual-attempt'
def h(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return 'sha256:'+h(encoded(v))
def read(p):return json.loads(p.read_bytes())
def exclusive(p,v):
    with p.open('x') as f:f.write(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
# Preserve the first attempt, including its reviewer bookkeeping failure.
first={}
for name in ('audit_saved.py','run-output.txt','run-receipt.json'):
    original=(OUT/name).read_bytes();copy=OUT/('first-'+name)
    if not copy.exists():copy.write_bytes(original)
    assert copy.read_bytes()==original
    first[name]={'sha256':h(original),'bytes':len(original)}
trace=(OUT/'run-output.txt').read_text();run=read(OUT/'run-receipt.json')
assert 'line 129' in trace and "if k!='sessionHash'" in trace and trace.endswith('AssertionError\n')
assert run['exit_code']==1 and run['numeric_threads']==1 and run['elapsed_seconds']<30
# Rehash only saved result/source bytes; do not import any runtime/replay module.
index=read(ATTEMPT/'copy-index.json');inputs={}
for row in index['files']:
    p=ATTEMPT/row['path'];b=p.read_bytes();assert len(b)==row['bytes'] and h(b)==row['sha256'];inputs[str(p.relative_to(ROOT))]=h(b)
assert index['original_attempt_directory_count']==1
r=read(ATTEMPT/'result.json');e=r['episode'];pair=r['pair'];actor=pair['actor']['strategy'];search=pair['search']['strategy']
assert inputs[str((ATTEMPT/'result.json').relative_to(ROOT))]=='9dbd781099dbf1bf6e44bd9cef9bfb4cff7ef9ebc2139906ea979c1de66ab295'
release=read(BASE/'live-release.json');sources={}
for rel,expected in release['controller_sources'].items():
    p=ROOT/'src/resectionlab'/rel.removeprefix('local:') if rel.startswith('local:') else ROOT/rel
    assert h(p.read_bytes())==expected;sources[str(p.relative_to(ROOT))]=expected
for row in read(BASE/'backend-promotion.json')['copy']:
    assert h((ROOT/row['to']).read_bytes())==row['sha256'];sources[row['to']]=row['sha256']
workspace=ROOT/'build/integrated-workspace-persistence-root/trained-transfer-live.ressectionlab'
published=read(ROOT/'artifacts/learned-aspiration-desktop-live-v1/source-index.json')['saved_generated_workspace']
assert workspace.stat().st_size==published['bytes'] and h(workspace.read_bytes())==published['sha256']
with zipfile.ZipFile(workspace) as z:
    assert set(z.namelist())=={'manifest.json','arrays.npz','workspace.json','workspace/series-0.ressectionlab'}
    raw=z.read('workspace.json');saved=json.loads(raw);manifest=json.loads(z.read('manifest.json'))
assert manifest['workspace_sha256']==h(raw)
# Correct established bundle recipe: logical image identities, not ZIP storage metadata.
logical={k:v for k,v in saved.items() if k not in ('sessionHash','images')}
logical['images']=[row['source'] for row in saved['images']]
assert digest(logical)==saved['sessionHash']
assert saved['referenceCaseHash']==manifest['case_semantic_hash']==r['caseHash']==e['caseHash']
assert saved['episodeReplay']['envelope']['episode']==e
assert saved['episodeReplay']['envelope']['episodeCanonicalJson']==encoded({k:v for k,v in e.items() if k!='episodeId'}).decode()
assert saved['episodeReplay']['selection']=={'episodeId':e['episodeId'],'frameIndex':71,'visible':True}
assert len(saved['images'])==1 and saved['images'][0]['source']['scope']=='display-only-native-grid'
assert saved['images'][0]['source']['annotationKind']=='estimated'
assert digest({k:v for k,v in e.items() if k!='episodeId'})==e['episodeId']
assert digest({k:v for k,v in pair.items() if k!='seal'})==pair['seal']
assert actor['action_ids']==search['action_ids'] and actor['decisions']==search['decisions']
assert pair['actor']['projection_trace']==pair['search']['projection_trace']
ws_source=(ROOT/'src/resectionlab/workspace_bundle.py').read_text()
assert "'imported_computational_provenance_unverified'" in ws_source
summary={
 'status':'PASS_saved_result_native_replay_and_corrected_metadata',
 'source_commit':release['source_commit'],'result_sha256':inputs[str((ATTEMPT/'result.json').relative_to(ROOT))],
 'episode_id':e['episodeId'],'pair_seal':pair['seal'],'strategy_seal':e['planning']['strategySeal'],
 'first_audit':{'files':first,'receipt':run,'disposition':'Native and numerical assertions through line 128 passed; line 129 failed solely because reviewer used whole workspace image storage rows instead of established logical source identities. First attempt retained unchanged.','native_replay_seconds':None},
 'metadata_repair':{'correct_logical_session_digest':saved['sessionHash'],'workspace_metadata_sha256':h(raw),'second_native_replay':False,'first_metadata_finalizer_disposition':'An additional metadata-only assertion assumed episodeCanonicalJson included episodeId; established canonical_episode excludes that digest field. Initial finalizer source preserved; corrected without project imports or replay.'},
 'equal_permitted_inputs':{'same_source':True,'same_decision_model':True,'same_full_and_projected_inventory_trace':True,'same_recorded_actions':True,'same_native_decisions_and_terminal_state':True,'private_reference_scored_during_planning':False},
 'action_ids':actor['action_ids'],'actor_forwards_recorded':2,'new_updates_recorded':0,
 'search_accounting':search['search_accounting'],
 'outcomes':{'removed_mm3':7.0,'nominal_target_removed_mm3':4.0,'other_removed_mm3':3.0,'contact_cells':63,'probe_contact_cells':0,'modeled_reward':e['metrics']['total_reward'],'clinical_probability':None},
 'replay_frames':len(e['replayFrames']),'microsteps':sum(len(x['microsteps']) for x in e['history']),
 'all_replay_prefix_mask_hashes_match':True,'full_saved_episode_matches_native_replay':True,
 'workspace':{'sha256':published['sha256'],'saved_same_episode':True,'selection':saved['episodeReplay']['selection'],'images':len(saved['images']),'image_arrays_decoded':0,'primaryPlanningHash_saved_value':saved['primaryPlanningHash'],'primary_planning_hash_recomputed':False,'reopen_authorship':'unverified_imported','computational_provenance':'imported_computational_provenance_unverified'},
 'supervision':read(ATTEMPT/'supervision.json'),'source_hashes':sources,'saved_input_hashes':inputs,
 'review_activity':{'checkpoint_opens':0,'actor_calls':0,'search_runs':0,'private_reference_loads':0,'patient_reads':0,'native_replay_operations':1,'metadata_finalizer_project_imports':0},
 'limitations':['Actor-only latency was not recorded; worker time includes load, actor, search, checks and serialization.','Search counters are saved execution accounting checked for internal consistency; search was not rerun.','Checkpoint provenance follows the bound original execution receipt; reviewer did not reopen weights.','UI vascular encounter readout is root-observed; private-reference scoring was not rerun or independently rescored here.','Primary planning hash is preserved in saved JSON but not independently recomputed after the bookkeeping assertion.','Generated geometry/software transfer only; no physical tissue or clinical validation.']}
exclusive(OUT/'audit.json',summary)
print(json.dumps({'status':summary['status'],'audit_sha256':h((OUT/'audit.json').read_bytes()),'metadata_only':True,'new_replay_operations':0}))
