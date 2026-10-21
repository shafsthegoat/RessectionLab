"""Read-only source-record milestone audit. Never constructs an engine."""
from pathlib import Path
import sys, json, importlib.util, subprocess
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from scipy import ndimage
from resectionlab.core import semantic_digest, array_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment

def no_engine(frame, event, arg):
    if event == 'call' and frame.f_globals.get('__name__') == 'resectionlab.surgical_engine' and frame.f_code.co_name not in ('<module>', 'SurgicalEngine', '_OperationCheckFailed'):
        raise AssertionError('Patient engine execution forbidden in source-record audit: '+frame.f_code.co_name)
sys.setprofile(no_engine)
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('accept_native.py'))
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020'
P=B/'primary-v2'; cp=P/'checkpoints/primary-over-eighty-percent-target.ressectionlab'
prior=P/'checkpoints/primary-before-superior-resume-1918.ressectionlab'
provenance=json.loads(cp.with_suffix('.provenance.json').read_text())
proof_path=P/'superior-pass-resume-execution-proof.json'; proof=json.loads(proof_path.read_text())
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit == proof['codeCommit'] == '7cbecb11a11a925bdb25065b414546532cb61da8'
subprocess.run(['git','diff','--quiet',commit,'--','src/resectionlab'],cwd=ROOT,check=True)
assert a.file_hash(a.SOURCE)==a.SOURCE_SHA256
assert a.file_hash(cp)==provenance['checkpointSha256']
assert a.file_hash(prior)==proof['checkpointSha256']
source,_,source_ws=load_workspace(a.SOURCE)
case,artifacts,ws=load_workspace(cp)
pre_case,pre_artifacts,pre_ws=load_workspace(prior)
source_scene=load_environment(a.SOURCE); scene=load_environment(cp)
source_scene.require_cache_source(); scene.require_cache_source()
assert scene.identity == source_scene.identity == a.SCENE_ID
assert array_digest(scene.labels)==array_digest(source_scene.labels)==provenance['labelArrayHash']
source_arrays=a.array_hashes(source)
assert source_arrays==a.array_hashes(case)==a.array_hashes(pre_case)
assert source.source_refs==case.source_refs==pre_case.source_refs
assert source.semantic_hash==case.semantic_hash==pre_case.semantic_hash==provenance['caseSemanticHash']
assert a.originals(source_ws)==a.originals(ws)==a.originals(pre_ws)
r=artifacts['surgical_toolbox']; before=pre_artifacts['surgical_toolbox']
s=r['branches']['full-primary']; pre=before['branches']['full-primary']
m=r['fullCase']['full-primary']; pm=before['fullCase']['full-primary']
assert r['branchId']=='full-primary' and r['position']==len(s['actions'])==1954
assert len(pre['actions'])==1918 and s['actions'][:1918]==pre['actions']
assert r['mapping']==before['mapping'] and r['physicsProfile']==before['physicsProfile']
protection=a.protection_audit(scene,s)
assert s['protectionContract']==pre['protectionContract']
for snapshot,n in ((s,1954),(pre,1918)):
    assert snapshot['sceneId']==scene.identity and snapshot['checkpoint']['position']==n
    state_id=semantic_digest({'scene':scene.identity,'state':snapshot['checkpoint']['state'],
        'protectionContract':snapshot['protectionContract']['contentHash']})
    assert snapshot['stateId']==snapshot['checkpoint']['stateId']==state_id
assert s['stateId']==m['lastStateId']==provenance['stateId']
assert pre['stateId']==proof['startStateId']
state=s['checkpoint']['state']; removed=state['removed']
assert all(type(v) is int for v in removed) and len(removed)==len(set(removed))
ids=np.array(removed,dtype=np.int64); labels=scene.labels.ravel()
assert ids.min()>=0 and ids.max()<labels.size
assert np.all(labels[ids]>0)
target_ids=[v['id'] for v in scene.regions if v.get('target')]
owned=np.isin(scene.labels,target_ids); residual=owned.copy(); residual.ravel()[ids]=False
components,number=ndimage.label(residual,structure=ndimage.generate_binary_structure(3,1))
sizes=np.bincount(components.ravel())[1:]; sizes=np.sort(sizes)[::-1]
by_material={}
for region in scene.regions:
    by_material.setdefault(region['material'],0)
    by_material[region['material']]+=int(np.count_nonzero(labels[ids]==region['id']))
original_target=int(np.count_nonzero(owned)); retained=int(np.count_nonzero(residual))
removed_target=original_target-retained
assert (original_target,removed_target,retained,number)==(43573,34905,8668,157)
assert by_material=={'soft_tissue':0,'tumor':34905,'bone':15238,'vessel':0}
assert provenance['removedByOwnedMaterialMm3']=={k:v*scene.voxel_mm3 for k,v in by_material.items()}
assert provenance['removedOwnedTargetMm3']==removed_target*scene.voxel_mm3
assert provenance['remainingOwnedTargetMm3']==retained*scene.voxel_mm3
assert provenance['remainingSixConnectedRegions']==number
memberships=a.annotation_memberships(source,scene,removed)
brain=source.compartments['brain'].ravel(); tumor=source.compartments['tumor'].ravel()
cross={}
for material in by_material:
    mids=np.array([v['id'] for v in scene.regions if v['material']==material])
    selected=ids[np.isin(labels[ids],mids)]
    b=brain[selected];t=tumor[selected]
    cross[material]={'bothRawBrainAndTumor':int(np.count_nonzero(b&t)),
      'rawBrainOnly':int(np.count_nonzero(b&~t)), 'rawTumorOnly':int(np.count_nonzero(~b&t)),
      'neither':int(np.count_nonzero(~b&~t))}
raw_removed={k:int(np.count_nonzero(v.ravel()[ids])) for k,v in source.compartments.items()}
assert raw_removed==provenance['removedRawAnnotationMembershipCells']
assert raw_removed['brain']==35766 and raw_removed['tumor']==35227
pending=m['deferredAccessPass']; old=pm['deferredAccessPass']; frozen=pending['pass']
assert old['status']=='pending' and pending['status']=='resumed'
for key,value in old.items():
    if key!='status': assert pending[key]==value
assert semantic_digest(frozen)==pending['passHash']==proof['passHash']
archive={k:pending[k] for k in ('sourceSceneId','requestBoundary','deferBoundary','targetAtDeferMm3','pass')}
assert semantic_digest(archive)==pending['archiveHash']==proof['archiveHash']
assert frozen['accessAxis']==2 and frozen['accessIndex']==22 and len(frozen['accesses'])==99
assert m['accessAxis']==2 and m['accessIndex']==22 and m['passIndex']==frozen['passIndex']==0
assert m['accesses'][:22]==frozen['accesses'][:22]
assert len(m['accesses'])==99
rank_fields={'remainingGeometricCapacityCells','capacityStateId'}
tail={v['id']:v for v in frozen['accesses'][22:]}
assert set(tail)=={v['id'] for v in m['accesses'][22:]}
for row in m['accesses'][22:]:
    assert {k:v for k,v in row.items() if k not in rank_fields}=={k:v for k,v in tail[row['id']].items() if k not in rank_fields}
positions={}
for key in ('requestBoundary','deferBoundary','stoppedBoundary','resumedBoundary'):
    boundary=pending[key]; p=boundary['position'];positions[key]=p
    assert boundary['actionPrefixHash']==semantic_digest(s['actions'][:p])
assert positions==dict(proof['checkedHistoricalBoundaryPositions'],resumedBoundary=1918)
assert pending['resumedBoundary']['stateId']==pre['stateId']
for trial in pending['trials']:
    b=trial['startBoundary']; assert b['actionPrefixHash']==semantic_digest(s['actions'][:b['position']])
assert pending['targetAtResumeMm3']==proof['targetAtResumeMm3']==34695
assert pending['targetAtDeferMm3']==proof['targetAtDeferMm3']==34262
baseline=frozen['passStartTargetMm3']+pending['targetAtResumeMm3']-pending['targetAtDeferMm3']
assert baseline==m['passStartTargetMm3']==pending['restoredPassStartTargetMm3']==proof['restoredPassBaselineMm3']==433
assert m['directionalPasses']==pm['directionalPasses']
assert len(m['interveningAccessTrials'])==len(pm.get('interveningAccessTrials',[]))+1
assert m['interveningAccessTrials'][:-1]==pm.get('interveningAccessTrials',[])
trial=m['interveningAccessTrials'][-1]
fields=('accesses','accessIndex','accessAxis','passIndex','passStartTargetMm3','drillAttempts','cycleStartTargetMm3',
        'cycleBoneRemovedCells','coreSearchExhausted','coreAttempts','corePositioning')
for k in fields:
    assert (k in trial)==(k in pm)
    if k in pm: assert trial[k]==pm[k]
assert trial['directionalTransition']==pm['directionalTransition'] and trial['stoppedStage']==pm['stage']=='endpoint'
assert trial['stopReason']==pm['stopReason'] and trial['endBoundary']==pending['resumedBoundary']
assert state['holds']=={} and all(not v['bleedingActive'] for v in state['sources'])
receipt={'utc':datetime.now(timezone.utc).isoformat(),'result':'pass: source-record milestone only','codeCommit':commit,
 'checkpoint':str(cp),'checkpointSha256':a.file_hash(cp),'source':str(a.SOURCE),'originalSha256':a.SOURCE_SHA256,
 'identity':{'caseSemanticHash':case.semantic_hash,'sceneId':scene.identity,'labelArrayHash':array_digest(scene.labels),
   'sourceArrayHashes':source_arrays,'acquisitionSourceRefsEqual':True,'embeddedOriginalFilesEqual':True,
   'embeddedOriginalFileCount':len(a.originals(ws)),'snapshotSemanticHash':semantic_digest(s),
   'checkpointSemanticHash':semantic_digest(s['checkpoint']),'stateId':s['stateId']},
 'protection':protection,'actionPrefix':{'preservedActions':1918,'totalActions':1954,'exactFullPrefixEqual':True,
   'priorCheckpointSha256':a.file_hash(prior),'prefixSemanticHash':semantic_digest(pre['actions']),
   'fullActionSemanticHash':semantic_digest(s['actions'])},
 'sourceAccounting':{'voxelMm3':scene.voxel_mm3,'originalOwnedTargetCells':original_target,
   'removedOwnedTargetCells':removed_target,'remainingOwnedTargetCells':retained,
   'removedOwnedTargetPercent':100*removed_target/original_target,'sixConnectedResidualRegions':int(number),
   'residualRegionSizesDescendingCells':sizes.tolist(),'removedByOwnedMaterialCells':by_material,
   'rawAnnotationRemovedMembershipCells':raw_removed,'removedOwnedMaterialByRawMembershipCells':cross,
   'rawAnnotationMemberships':memberships,
   'rawMembershipInterpretation': {'rawBrainMinusRawTumorCells':raw_removed['brain']-raw_removed['tumor'],
      'removedRawBrainOnlyCells':sum(v['rawBrainOnly'] for v in cross.values()),
      'removedRawTumorOnlyCells':sum(v['rawTumorOnly'] for v in cross.values()),
      'removedBoneOwnedCellsInsideRawTumor':cross['bone']['bothRawBrainAndTumor']+cross['bone']['rawTumorOnly'],
      'meaning':'The raw-brain excess is 666 brain-only minus 127 tumor-only memberships. All 666 brain-only cells are bone-owned; 322 removed bone-owned cells also overlap raw tumor. Neither mask overlap nor ownership constitutes reviewed clinical healthy-tissue truth.'},
   'directProtectionTargetMembership':a.protected_target_membership(scene,state,s['protectionContract'])},
 'resumeRecords':{'proof':str(proof_path),'proofSha256':a.file_hash(proof_path),'passHash':pending['passHash'],
   'archiveHash':pending['archiveHash'],'historicalActionPrefixHashesMatch':True,'boundaryPositions':positions,
   'frozenSuperiorAccessCount':99,'restoredAndCurrentCursor':22,'finished22AndTailRouteDescriptorsPreserved':True,
   'leftTrialActionsAndPassMetadataPreserved':True,'currentPassBaselineMm3':baseline,
   'baselineCorrectionOnceInRecords':True,'currentStage':m['stage'],'currentStatus':m['status']},
 'limits':['Read-only saved source-record integrity and counts; no engine construction, restore, preview, step or replay.',
   'Snapshot/checkpoint IDs and historical action-prefix hashes are recomputed; historical intermediate physical states and runtime proof assertions were not independently replayed.',
   'Raw annotation memberships overlap. Zero owned soft_tissue removal does not establish zero clinical healthy-brain injury; bone ownership outside raw tumor is not reviewed clinical normal tissue.',
   'The denominator is 43573 original OWNED target cells, distinct from 45115 full raw tumor annotation cells. Residual connectivity and direct protection membership do not establish reachability or exhausted legal approaches.',
   'Final forced canonical acceptance, endpoint cleanup, full frame coverage and final Electron GUI playback remain pending.']}
out=Path(__file__).with_name('primary-1954-source-record-review.json')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(out),'removedOwnedTarget':removed_target,'remainingOwnedTarget':retained,
 'sixConnectedRegions':int(number),'removedByOwnedMaterial':by_material,'rawMemberships':raw_removed,'rawCrossTab':cross}))
