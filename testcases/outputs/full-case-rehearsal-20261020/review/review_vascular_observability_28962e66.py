"""Saved-record/source review only; no engine construction or physical replay."""
from pathlib import Path
import sys,json,hashlib,math,importlib.util,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy import ndimage
from resectionlab.core import semantic_digest,array_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment

def no_engine(frame,event,arg):
 if event=='call' and frame.f_globals.get('__name__')=='resectionlab.surgical_engine' and frame.f_code.co_name not in ('<module>','SurgicalEngine','_OperationCheckFailed'):
  raise AssertionError('Forbidden engine execution: '+frame.f_code.co_name)
sys.setprofile(no_engine)
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('accept_native.py'));a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020';native=B/'gui/vascular-observability-28962e66.ressectionlab';old=B/'vascular-v2/vascular.ressectionlab'
report_path=B/'review/vascular-observability-28962e66-canonical.json';report=json.loads(report_path.read_text())
provenance_path=native.with_suffix(native.suffix+'.display-provenance.json');provenance=json.loads(provenance_path.read_text())
assert a.file_hash(native)==report['nativeSha256']==provenance['outputSha256']
assert a.file_hash(old)==provenance['nativeSha256']=='7348f05c91c2ac3cd6689807e7c00da99d21d23dac2be8722fca74eddafd6fcb'
assert a.file_hash(a.SOURCE)==a.SOURCE_SHA256==report['originalSha256']
assert a.file_hash(Path(provenance['reference']))==provenance['referenceSha256']
assert a.file_hash(B/'review/apply_rehearsal_display.py')==provenance['helperSha256']
with zipfile.ZipFile(native) as z,zipfile.ZipFile(old) as before:
 assert set(z.namelist())==set(before.namelist())
 for name in z.namelist():
  if name not in ('manifest.json','workspace.json'):assert z.read(name)==before.read(name)
 new_manifest=json.loads(z.read('manifest.json'));old_manifest=json.loads(before.read('manifest.json'))
 assert new_manifest['artifacts']['surgical_toolbox']==old_manifest['artifacts']['surgical_toolbox']
 for key,value in old_manifest['artifacts'].items():
  if key!='workspace':assert new_manifest['artifacts'][key]==value
source,_,source_ws=load_workspace(a.SOURCE);case,artifacts,ws=load_workspace(native);oldcase,oldartifacts,oldws=load_workspace(old)
assert artifacts['surgical_toolbox']==oldartifacts['surgical_toolbox']
assert a.array_hashes(source)==a.array_hashes(case)==a.array_hashes(oldcase)==report['sourceArrays']
assert source.semantic_hash==case.semantic_hash==oldcase.semantic_hash==report['caseHash']
assert source.source_refs==case.source_refs==oldcase.source_refs
assert a.originals(source_ws)==a.originals(ws)==a.originals(oldws)
scene=load_environment(native);original_scene=load_environment(a.SOURCE)
assert scene.identity==original_scene.identity==report['sceneId']==a.SCENE_ID
assert array_digest(scene.labels)==array_digest(original_scene.labels)==report['labelArrayHash']
r=artifacts['surgical_toolbox'];s=r['branches']['full-vascular'];state=s['checkpoint']['state'];actions=s['actions']
assert len(actions)==s['checkpoint']['position']==73
assert s['stateId']==s['checkpoint']['stateId']==report['stateId']==semantic_digest({'scene':scene.identity,'state':state,'protectionContract':s['protectionContract']['contentHash']})
assert a.protection_audit(scene,s)==report['sourceProtectionEvidence']
assert r['parents']['full-vascular']==report['parent']=={'branchId':'exposure-prefix','position':42}
prefix=r['branches']['exposure-prefix'];assert prefix['actions']==actions[:42] and prefix['stateId']=='sha256:6817a85eae927401b66d32eff4627d66c73a37da9fb19b481085c12496c38bd4'
removed=np.asarray(state['removed'],dtype=np.int64);assert len(set(removed.tolist()))==len(removed)==3765
counts={}
for region in scene.regions:counts[region['material']]=counts.get(region['material'],0)+int(np.count_nonzero(scene.labels.ravel()[removed]==region['id']))
assert counts=={'soft_tissue':0,'tumor':312,'bone':3453,'vessel':0}
for key,n in counts.items():assert report['accounting']['materials'][key]['removedCells']==n
memberships=a.annotation_memberships(source,scene,state['removed']);assert memberships==report['rawAnnotationMemberships']
fluid=a.fluid_audit(state);assert fluid==report['fluid']
faces=state['vesselsV2']['faces'];dose=scene.properties['tissuePhysicsV2']['vessel']['coagulationDose']
assert len(faces)==5 and len(state['vesselsV2']['wounds'])==1
assert all(f['patent'] is False and f['clot']==1. and f['coagulationDose']>=dose for f in faces.values())
assert all(not src['bleedingActive'] and not src['patent'] for src in state['sources'])
assert not state['holds'] and all(not v['inserted'] and not v['active'] and not v['grip'] for v in state['slots'].values())
trace_path=Path(report['canonicalReplay']['trace']);rows=[json.loads(t) for t in trace_path.read_text().splitlines()]
assert len(rows)==73 and [v['action'] for v in rows]==actions
seen=set();previous=None;previous_time=0.;max_mass=0.
for pos,row in enumerate(rows,1):
 assert row['position']==row['vascularState']['position']==pos and row['status'] in ('accepted','partial')
 if previous is not None:assert row['stateBefore']==previous
 previous=row['stateAfter'];assert not seen.intersection(row['removedCells']);seen.update(row['removedCells'])
 v=row['vascularState'];assert math.isclose(v['elapsedSeconds']-previous_time,row['elapsedSeconds'],abs_tol=1e-8)
 previous_time=v['elapsedSeconds'];f=row['fluid']
 residual=f['addedMm3']-f['presentMm3']-f['removedMm3']-f['escapedMm3']-f['trappedOutletMm3']
 assert abs(residual)<1e-9 and math.isclose(residual,f['residualMm3'],abs_tol=1e-12)
 max_mass=max(max_mass,abs(residual))
 assert v['presentMm3']==f['presentMm3'] and v['generatedMm3']==f['vascular']['generatedMm3']
assert seen==set(removed.tolist()) and previous==s['stateId']
assert rows[-1]['vascularState']['faces']==faces and rows[-1]['vascularState']['slots']==state['slots']
assert math.isclose(previous_time,state['elapsedSeconds'],abs_tol=1e-8)
causal=a.causal_vascular_audit([v['vascularState'] for v in rows],[v['action'] for v in rows[1:]],dose,required=True)
assert causal==report['causalVascularChain']
for p in (58,72,73):
 before,after=rows[p-2]['vascularState'],rows[p-1]['vascularState']
 assert after['activeWounds']==0 and before['generatedMm3']==after['generatedMm3']
points=[]
for pos in (42,45,47,48,49,50,51,54,55,56,58,62,63,67,68,69,72,73):
 row=rows[pos-1];v=row['vascularState'];before=rows[pos-2]['vascularState']
 points.append({'position':pos,'simulationSeconds':v['elapsedSeconds'],'operation':row['action']['operation'],
 'instrument':v['slots'][row['action']['slot']]['instrument'],'activeWounds':v['activeWounds'],
 'damagedFaces':len(v['faces']),'actualFaceDoseTotal':sum(f['coagulationDose'] for f in v['faces'].values()),
 'minimumFaceClot':min((f['clot'] for f in v['faces'].values()),default=None),
 'generatedBloodMm3':v['generatedMm3'],'spatialFluidMm3':v['presentMm3'],
 'actualFluidRemovedThisActionMm3':v['removedMm3']-before['removedMm3']})
output={'utc':datetime.now(timezone.utc).isoformat(),'result':'pass: saved vascular observability records',
 'native':str(native),'nativeSha256':a.file_hash(native),'originalSha256':a.SOURCE_SHA256,
 'canonicalReceiptSha256':a.file_hash(report_path),'traceSha256':a.file_hash(trace_path),'displayProvenanceSha256':a.file_hash(provenance_path),
 'sceneId':scene.identity,'stateId':s['stateId'],'surgicalArtifactEncodedAndDecodedUnchanged':True,
 'sourceArraysLabelsAcquisitionRefsAndOriginalsUnchanged':True,'displayOnlyRequestedBoundsNotActualMotion':True,
 'matchingStrictProtectionMasks':11,'clearanceMm':2.,'criticalDisplacementMm':0.,
 'savedSharedPrefixActions':42,'recordedActions':73,'recordedStatuses':{k:sum(v['status']==k for v in rows) for k in ('accepted','partial')},
 'recordedStateIdChainAndRemovalUnionMatchEndpoint':True,'ownedMaterialRemovalCells':counts,
 'rawAnnotationRemovalMembershipCells':{k:v['removedSourceCellsWithinAnnotation'] for k,v in memberships['groups']['compartments'].items()},
 'fluid':fluid,'maximumRecordedFluidResidualMm3':max_mass,'oneWoundFiveSeveredNonpatentClottedFacesActualDoseEachAtLeast2':True,
 'causalVascularChainRecomputedFromRecordedBoundaryStates':causal,'guiReviewPositions':points,
 'limits':['Read-only source/native/trace review. No patient engine construction, restore, preview, step, physics replay, live-session access or GUI control.',
 'Root supplied the current forced-canonical receipt/trace. Their recorded causality and chain were recomputed, but intermediate full physical states were not independently replayed in this review.',
 'Source memberships are overlapping unreviewed estimates; zero owned soft removal is not clinical normal-tissue safety proof. Vascular branch retains43261/43573 owned target cells and is not the primary-resection endpoint.',
 'Display provenance proves artifact/source preservation and conservative camera computation, not actual complete playback visibility. Telemetry/frame coverage and actual secondary-display GUI evidence are separate gates.',
 'This review concerns the completed vascular observability branch only, not final primary or combined-native acceptance.']}
out=Path(__file__).with_name('vascular-observability-28962e66-independent-record-review.json');out.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'receipt':str(out),'counts':counts,'fluid':fluid,'guiPositions':points}))
