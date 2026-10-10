"""Saved generated-result audit plus bounded recorded-action replay; no policy run."""
from pathlib import Path
import hashlib,json,math,sys,time,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'build/learned-transfer-root-integration-v1'
ATTEMPT=BASE/'actual-attempt'
EXPECTED_RESULT='9dbd781099dbf1bf6e44bd9cef9bfb4cff7ef9ebc2139906ea979c1de66ab295'
blocked=[]
def access_guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
        if path.suffix in ('.pt','.pth') or (str(path).startswith(str(ROOT/'data')+'/')):
            blocked.append(str(path));raise AssertionError('Checkpoint/data access forbidden in saved audit')
    if event in ('subprocess.Popen','os.system','socket.connect'):
        blocked.append(event);raise AssertionError('New process/network forbidden in replay audit')
sys.addaudithook(access_guard)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def digest(v):return 'sha256:'+sha_bytes(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def read_json(path,maxbytes=3*1024**2):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<=maxbytes
    return json.loads(path.read_bytes())
index=read_json(ATTEMPT/'copy-index.json');input_hashes={}
for row in index['files']:
    assert Path(row['path']).name==row['path']
    path=ATTEMPT/row['path'];data=path.read_bytes()
    assert len(data)==row['bytes'] and sha_bytes(data)==row['sha256']
    input_hashes[str(path.relative_to(ROOT))]=sha_bytes(data)
assert index['original_attempt_directory_count']==1
result=read_json(ATTEMPT/'result.json');assert input_hashes[str((ATTEMPT/'result.json').relative_to(ROOT))]==EXPECTED_RESULT
assert (ATTEMPT/'result.json').stat().st_size==184977
completion=read_json(ATTEMPT/'completion.json');supervision=read_json(ATTEMPT/'supervision.json')
assert completion['status']=='complete' and completion['resultSha256']==EXPECTED_RESULT
assert completion['supervisionSha256']==input_hashes[str((ATTEMPT/'supervision.json').relative_to(ROOT))]
assert supervision['status']=='complete' and supervision['workerReturnCode']==0 and supervision['workerTerminationConfirmed']
assert supervision['unresolvedWorkerPid'] is None and supervision['cleanupErrors']==[] and supervision['stopReason'] is None
assert 0<supervision['elapsedSeconds']<20 and 0<supervision['sampledPeakRssBytes']<=1024**3 and supervision['samples']==24
release=read_json(BASE/'live-release.json');assert release['source_commit']=='ed87de4b8931819aac583ecd6f11efdbcf0c5361'
source_hashes={}
for relative,expected in release['controller_sources'].items():
    path=ROOT/'src/resectionlab'/relative.removeprefix('local:') if relative.startswith('local:') else ROOT/relative
    actual=sha_bytes(path.read_bytes());assert actual==expected
    source_hashes[str(path.relative_to(ROOT))]=actual
assert sha_bytes((ROOT/'src/resectionlab/legacy_transfer_supervisor.py').read_bytes())==release['controller_sha256']
for row in read_json(BASE/'backend-promotion.json')['copy']:
    assert sha_bytes((ROOT/row['to']).read_bytes())==row['sha256']
    source_hashes[row['to']]=row['sha256']
# Ban actual compute before importing replay helpers. These modules never deserialize weights on import.
import torch
torch.set_num_threads(1)
def forbidden(*a,**k):raise AssertionError('Actor/search/weights/private scoring forbidden in saved replay')
torch.load=forbidden
from resectionlab.spatial_policy import SpatialPolicy
SpatialPolicy.forward=forbidden;SpatialPolicy.act=forbidden
import resectionlab.shared_episode as shared
shared.execute_policy_episode=forbidden;shared.execute_search_episode=forbidden;shared.observed_beam_search=forbidden
import resectionlab.observed_search as search_module
search_module.observed_beam_search=forbidden
import resectionlab.shared_vascular_evaluation as vascular
vascular._load_generated_reference=forbidden
from resectionlab.legacy_transfer_episode import execute_transfer_episode_from_pair,transfer_authorship
from resectionlab.core import array_digest
pair=result['pair'];episode=result['episode']
assert result['status']=='complete' and result['newOptimizerUpdates']==0 and result['clinicalValidation'] is False
assert digest({k:v for k,v in pair.items() if k!='seal'})==pair['seal']
assert digest({k:v for k,v in episode.items() if k!='episodeId'})==episode['episodeId']
assert digest(episode['planning']['strategy'])==episode['planning']['strategySeal']
assert digest(result['originalLoaderReceipt'])==pair['original_checkpoint_validation_receipt_sha256']
assert digest(pair['projection_receipt'])==pair['projection_receipt_hash']
assert pair['projection_receipt']['checkpoint_validation_receipt_sha256']==digest(result['originalLoaderReceipt'])
assert result['checkpointSha256']==result['originalLoaderReceipt']['checkpoint_file_sha256']==pair['projection_receipt']['checkpoint_sha256']
assert pair['checkpoint_training_domain_matches_target'] is False and pair['private_reference_scored'] is False
actor=pair['actor']['strategy'];search=pair['search']['strategy']
assert actor['action_ids']==search['action_ids'] and len(actor['decisions'])==len(search['decisions'])==2
assert pair['actor']['projection_trace']==pair['search']['projection_trace']
for arm in (actor,search):
    assert arm['source_hash']==episode['sourceHash'] and arm['environment_contract_hash']==episode['decisionModelHash']
    assert digest([row['physical_transition'] for row in arm['decisions']])==arm['physical_history_hash']
    assert [row['action_id'] for row in arm['decisions']]==arm['action_ids']
assert actor['decisions']==search['decisions']
assert actor['method']=='LEARNED_POLICY' and search['method']=='SEARCH'
assert actor['policy_identity']['checkpoint_validation_receipt_sha256']==pair['projection_receipt_hash']
assert actor['policy_identity']['architecture_hash']==digest({'base_architecture_hash':pair['projection_receipt']['base_architecture_hash'],'projection_hash':pair['projection_hash'],'wrapper':'bound-legacy-aspiration-transfer-policy-v1'})
assert episode['planning']['actorForwardCalls']==episode['planning']['actor_forward_calls']==2
assert episode['planning']['optimizerUpdates']==episode['planning']['optimizer_updates']==0
account=search['search_accounting'];layers=account['layers']
assert account['model_transition_calls']==account['eager_transition_calls']==account['evaluated_transition_prefixes']==sum(x['model_transition_calls'] for x in layers)==14
assert account['max_calls']==24 and account['beam_width']==4 and account['actor_forward_calls']==0
assert account['expanded_nodes']==account['observation_requests']==sum(x['expanded_nodes'] for x in layers)==10
assert account['negative_prefixes_evaluated']==sum(x['negative_prefixes_evaluated'] for x in layers)==6
assert account['negative_prefixes_pruned_by_beam']==sum(x['negative_prefixes_pruned_by_beam'] for x in layers)==4
assert account['completed_layers']==sum(x['completed'] for x in layers)==4
assert not account['time_cap_reached'] and not account['call_cap_reached']
assert account['guidance']=='none' and account['objective_source']=='observed_scan_estimator_only'
# One bounded saved-action replay operation; internal validators replay both fixed arms and native episode.
began=time.perf_counter()
case,replayed=execute_transfer_episode_from_pair(pair)
replay_seconds=time.perf_counter()-began
assert replayed==episode and case.semantic_hash==result['caseHash']
assert result['episodeAuthorship']==transfer_authorship(episode,live_backend_run=True)
frames=episode['replayFrames'];shape=tuple(episode['shape'])
removed=np.zeros(shape,bool);contact=np.zeros(shape,bool);probe=np.zeros(shape,bool)
for frame in frames:
    for key,mask in (('removedIndicesNative',removed),('contactIndicesNative',contact),('probeContactIndicesNative',probe)):
        for point in frame[key]:mask[tuple(point)]=True
    assert array_digest(removed)==frame['cavityHash']
    assert array_digest(contact)==frame['contactHash']
    assert array_digest(probe)==frame['probeContactHash']
assert len(frames)==72 and np.array_equal(np.argwhere(removed),episode['finalRemovedIndicesNative'])
voxel_mm3=abs(float(np.linalg.det(case.affine[:3,:3])))
nominal=case.compartments['generated_nominal_target']
outcomes={'removed_mm3':float(removed.sum()*voxel_mm3),'nominal_target_removed_mm3':float((removed&nominal).sum()*voxel_mm3),'other_removed_mm3':float((removed&~nominal).sum()*voxel_mm3),'contact_cells':int(contact.sum()),'probe_contact_cells':int(probe.sum())}
metrics=episode['metrics']
assert outcomes['removed_mm3']==metrics['simulated_removed_volume_mm3']==7
assert outcomes['nominal_target_removed_mm3']==metrics['target_removed_mm3']==4
assert outcomes['other_removed_mm3']==metrics['normal_removed_mm3']==3
assert metrics['total_reward']==account['estimated_incremental_return']==3.353
assert metrics['clinical_deficit_probability'] is None
history=episode['history'];assert [r['interaction_mode'] for r in history]==['aspirate','stop']
assert len(history[-1]['microsteps'])==0
micro_count=sum(len(r['microsteps']) for r in history)
# Read only the JSON metadata from the generated saved ZIP, never primary/auxiliary image arrays.
workspace=ROOT/'build/integrated-workspace-persistence-root/trained-transfer-live.ressectionlab'
with zipfile.ZipFile(workspace) as archive:
    infos=archive.infolist();assert len(infos)==4 and len({i.filename for i in infos})==4
    info=archive.getinfo('workspace.json');assert info.file_size<2*1024**2
    saved=json.loads(archive.read(info))
assert digest({k:v for k,v in saved.items() if k!='sessionHash'})==saved['sessionHash']
assert saved['episodeReplay']['envelope']['episode']==episode
assert saved['referenceCaseHash']==case.semantic_hash and saved['primaryPlanningHash']==case.planning_hash
assert saved['episodeReplay']['selection']['episodeId']==episode['episodeId']
from resectionlab.workspace_bundle import canonical_episode
assert saved['episodeReplay']['envelope']['episodeCanonicalJson']==canonical_episode(episode)
assert transfer_authorship(episode,live_backend_run=False)['status']=='unverified_imported'
# Qualifier behavior is pinned in reviewed workspace module; saved file itself carries no live authority.
assert "'imported_computational_provenance_unverified'" in (ROOT/'src/resectionlab/workspace_bundle.py').read_text()
for path,digest_value in source_hashes.items():assert sha_bytes((ROOT/path).read_bytes())==digest_value
for path,digest_value in input_hashes.items():assert sha_bytes((ROOT/path).read_bytes())==digest_value
assert not blocked
summary={'status':'PASS_saved_result_and_native_replay','source_commit':release['source_commit'],'result_sha256':EXPECTED_RESULT,'episode_id':episode['episodeId'],'pair_seal':pair['seal'],'strategy_seal':episode['planning']['strategySeal'],'equal_permitted_inputs':{'same_source':True,'same_decision_model':True,'same_full_and_projected_inventory_trace':True,'same_recorded_actions':True,'same_native_decisions_and_terminal_state':True,'private_reference_scored_during_planning':False},'action_ids':actor['action_ids'],'actor_forwards_recorded':2,'new_updates_recorded':0,'search_accounting':account,'outcomes':{**outcomes,'modeled_reward':metrics['total_reward'],'clinical_probability':None},'native_replay_seconds':replay_seconds,'replay_frames':len(frames),'microsteps':micro_count,'all_replay_prefix_mask_hashes_match':True,'full_saved_episode_matches_native_replay':True,'workspace':{'sha256':sha_bytes(workspace.read_bytes()),'saved_same_episode':True,'selection':saved['episodeReplay']['selection'],'images':len(saved['images']),'image_arrays_decoded':0,'reopen_authorship':'unverified_imported','computational_provenance':'imported_computational_provenance_unverified'},'supervision':supervision,'source_hashes':source_hashes,'saved_input_hashes':input_hashes,'review_activity':{'checkpoint_opens':0,'actor_calls':0,'search_runs':0,'private_reference_loads':0,'patient_reads':0,'native_replay_operations':1,'blocked_calls':blocked},'limitations':['Actor-only latency was not recorded; worker time includes load, actor, search, checks and serialization.','Search counters are saved execution accounting checked for internal consistency; search was not rerun.','Checkpoint provenance follows the bound original execution receipt; reviewer did not reopen weights.','UI vascular encounter readout was reported by root; transient private-reference scoring was not rerun or independently rescored here.','Generated geometry/software transfer only; no physical tissue or clinical validation.']}
(OUT/'audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':summary['status'],'native_replay_seconds':replay_seconds,'actor_forwards_recorded':2,'search_calls_recorded':14,'same_actions':True,'frames':len(frames),'microsteps':micro_count,'outcomes':summary['outcomes'],'workspace_selection':summary['workspace']['selection']}))
