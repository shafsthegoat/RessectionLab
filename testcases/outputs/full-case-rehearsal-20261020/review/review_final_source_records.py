"""Final saved-source/record audit only. No patient engine execution."""
from pathlib import Path
import sys,json,importlib.util,math,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from resectionlab.core import semantic_digest,array_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment

def no_engine(frame,event,arg):
 if event=='call' and frame.f_globals.get('__name__')=='resectionlab.surgical_engine' and frame.f_code.co_name not in ('<module>','SurgicalEngine','_OperationCheckFailed'):
  raise AssertionError('Forbidden patient engine call: '+frame.f_code.co_name)
sys.setprofile(no_engine)
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('accept_native.py'));a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020';native=B/'final/rehearsal.ressectionlab';raw=B/'final/rehearsal-raw.ressectionlab'
provenance=json.loads(native.with_suffix(native.suffix+'.display-provenance.json').read_text())
assert a.file_hash(a.SOURCE)==a.SOURCE_SHA256
assert a.file_hash(native)==provenance['outputSha256'] and a.file_hash(raw)==provenance['nativeSha256']
source,_,source_ws=load_workspace(a.SOURCE);case,artifacts,ws=load_workspace(native)
assert a.array_hashes(source)==a.array_hashes(case)
assert source.semantic_hash==case.semantic_hash and source.source_refs==case.source_refs
assert a.originals(source_ws)==a.originals(ws)
with zipfile.ZipFile(native) as final,zipfile.ZipFile(raw) as before:
 assert final.namelist()==before.namelist()
 for name in before.namelist():
  if name not in ('manifest.json','workspace.json'):assert final.read(name)==before.read(name)
 assert json.loads(final.read('manifest.json'))['artifacts']['surgical_toolbox']==json.loads(before.read('manifest.json'))['artifacts']['surgical_toolbox']
scene=load_environment(native);original_scene=load_environment(a.SOURCE)
assert scene.identity==original_scene.identity==a.SCENE_ID
assert array_digest(scene.labels)==array_digest(original_scene.labels)
r=artifacts['surgical_toolbox'];branches=r['branches'];assert {k:len(v['actions']) for k,v in branches.items()}=={'exposure-prefix':42,'full-primary':2148,'full-vascular':73,'tapered-frontier-prefix':19}
checks={}
for name,s in branches.items():
 assert s['sceneId']==scene.identity and len(s['actions'])==s['checkpoint']['position']
 sid=semantic_digest({'scene':scene.identity,'state':s['checkpoint']['state'],'protectionContract':s['protectionContract']['contentHash']})
 assert sid==s['stateId']==s['checkpoint']['stateId']
 checks[name]={'actions':len(s['actions']),'stateId':sid,'snapshotHash':semantic_digest(s),'protection':a.protection_audit(scene,s)}
for name in ('full-primary','full-vascular'):assert branches[name]['actions'][:42]==branches['exposure-prefix']['actions']
assert branches['full-primary']['actions'][:19]==branches['tapered-frontier-prefix']['actions']
assert r['parents']['full-vascular']=={'branchId':'exposure-prefix','position':42}
rows={}
for name in ('full-primary','full-vascular'):
 s=branches[name];state=s['checkpoint']['state'];ids=np.asarray(state['removed'],dtype=np.int64)
 accounting=a.source_accounting(scene,state); raw_removed={k:int(np.count_nonzero(v.ravel()[ids])) for k,v in source.compartments.items()}
 brain=source.compartments['brain'].ravel();tumor=source.compartments['tumor'].ravel();labels=scene.labels.ravel();cross={}
 for material in accounting['materials']:
  mids=[x['id'] for x in scene.regions if x['material']==material];selected=ids[np.isin(labels[ids],mids)];bb=brain[selected];tt=tumor[selected]
  cross[material]={'bothRawBrainAndTumor':int(np.count_nonzero(bb&tt)),'rawBrainOnly':int(np.count_nonzero(bb&~tt)),'rawTumorOnly':int(np.count_nonzero(~bb&tt)),'neither':int(np.count_nonzero(~bb&~tt))}
 present=math.fsum(state['debris'].values());removed=state.get('debrisRemovedMm3',0.);escaped=state.get('debrisEscapedMm3',0.)
 assert math.isclose(present,state['debrisMm3'],abs_tol=1e-8)
 assert not state['holds'] and all(not v['inserted'] and not v['active'] and not v['grip'] for v in state['slots'].values())
 assert all(not x['bleedingActive'] for x in state['sources'])
 rows[name]={'accounting':accounting,'rawAnnotationRemovedMembershipCells':raw_removed,'removedOwnedMaterialByRawMembershipCells':cross,'rawAnnotationMemberships':a.annotation_memberships(source,scene,state['removed']),'directProtectionTargetMembership':a.protected_target_membership(scene,state,s['protectionContract']),'fluid':a.fluid_audit(state),'wounds':a.wound_audit(state,scene),'debris':{'spatialPresentMm3':present,'removedMm3':removed,'escapedMm3':escaped,'sumPresentRemovedEscapedMm3':math.fsum([present,removed,escaped]),'drillingV2RemovedVolumeMm3':state.get('drillingV2',{}).get('removedVolumeMm3'),'limit':'Endpoint spatial/published ledger agreement; per-action generation provenance/conservation is pending final canonical audit.'},'elapsedSimulationSeconds':state['elapsedSeconds'],'bothHandsOutInactiveReleased':True,'metadataStatus':r['fullCase'][name].get('status'),'metadataStopReason':r['fullCase'][name].get('stopReason')}
primary=rows['full-primary'];target=primary['accounting']['target'];assert (target['removedOwnedCells'],target['retainedOwnedCells'])==(35444,8129)
assert primary['accounting']['ownedHealthyOrUnknownNonTargetRemoval']['cells']==0
meta=r['fullCase']['full-primary'];assert meta['status']=='stopped' and meta['stage']=='endpoint' and 'budget' in meta['stopReason'].lower()
assert branches['full-primary']['stateId'].startswith('sha256:c97de565')
receipt={'status':'passed: final source records only','utc':datetime.now(timezone.utc).isoformat(),'native':str(native),'nativeSha256':a.file_hash(native),'originalSha256':a.SOURCE_SHA256,'sourceCaseHash':case.semantic_hash,'sceneId':scene.identity,'labelArrayHash':array_digest(scene.labels),'sourceArrayHashes':a.array_hashes(case),'sourceReferencesAndEmbeddedOriginalsPreserved':True,'displayMergeToolboxAndNonMetadataPayloadBytesUnchanged':True,'branchRecords':checks,'exactCommonExposurePrefixActions':42,'exactPrimaryTaperedPrefixActions':19,'branches':rows,'limits':['Budget-limited substantial removal, not maximal feasible removal or evidence residual is unresectable.','43573 original owned target denominator differs from45115 raw tumor annotation.','Zero owned non-target soft-tissue removal is not zero clinical healthy-brain injury; overlapping raw annotation memberships/ownership are not reviewed clinical truth.','Residual six-connectivity/direct mask membership do not prove reachability or clinical clearance.','Read-only records: no Engine construction/restore/preview/step/replay; final forced canonical, observed frame coverage, GUI/video/persistence acceptance remain separate pending gates.']}
out=B/'final/independent-source-record-review.json';out.open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'report':str(out),'nativeSha256':receipt['nativeSha256'],'primaryTarget':{k:v for k,v in target.items() if k!='residualConnectedRegions'},'primaryResidualRegions':len(target['residualConnectedRegions']),'primaryRawMemberships':primary['rawAnnotationRemovedMembershipCells'],'primaryMaterials':primary['accounting']['materials'],'primaryDebris':primary['debris'],'vascularFluid':rows['full-vascular']['fluid']}))
