import assert from 'node:assert/strict';
import {test} from 'node:test';
import {modeledVolume,referenceModeledVolume,modeledVolumeWork,modeledVolumeCacheStatistics,registerSourceInspection} from '../src/modeled-volume.ts';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {hydrateCase} from '../src/case-data.ts';

const affine=[[1,.2,0,12],[0,2,.1,-7],[0,0,1.5,3],[0,0,0,1]];
const regions=[{id:9,name:'brain',material:'soft_tissue',color:'#ccc',target:false},
  {id:2,name:'tumor',material:'tumor',color:'#a33',target:true},
  {id:17,name:'vessel',material:'vessel',color:'#f33',target:false}];
const randomSource=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
const source=(count=105)=>({caseHash:'generated-source',shape:[count/15,5,3],affine,frame:'RAS+',mri:new Float32Array(count),
  compartments:regions.map(r=>({name:r.name,kind:r.material,color:r.color,mask:new Uint8Array(count),sourceIdentity:`original-${r.name}`}))});
const view=(s,extra={},coupled=false)=>({stateId:'generated-state',scene:{shape:s.shape,affine,regions},removed:[],moved:[],
  ...(coupled?{physics:{profile:{schema:'resectionlab.tissue-physics.v2'}}}:{}),...extra});
function compare(actual,expected){
  assert.equal(actual.mri,expected.mri);assert.deepEqual(actual.affine,expected.affine);assert.deepEqual(actual.shape,expected.shape);
  assert.equal(actual.compartments.length,expected.compartments.length);
  actual.compartments.forEach((layer,i)=>{const reference=expected.compartments[i];
    assert.equal(layer.name,reference.name);assert.equal(layer.kind,reference.kind);assert.equal(layer.color,reference.color);
    assert.deepEqual(layer.mask,reference.mask);assert.equal(layer.surface,reference.surface);assert.equal(layer.sourceIdentity,reference.sourceIdentity);
  });
}

for(const coupled of [false,true])for(const categorical of [false,true])test(`sparse modeled states match independent full reconstruction across random overlaps/moves/backseeks (coupled=${coupled},labels=${categorical})`,()=>{
  const random=randomSource(0x91e5+(coupled?1:0)+(categorical?2:0)),s=source();
  for(const layer of s.compartments)for(let i=0;i<s.mri.length;i++)layer.mask[i]=Number(random()<.42);
  s.compartments.push({name:'unmapped',kind:'unknown',color:'#555',mask:Uint8Array.from(s.mri,()=>Number(random()<.1))});
  if(categorical){const data=Uint16Array.from(s.mri,()=>Math.floor(random()*4));s.labelSets=[{id:'categorical',data}];
    s.compartments[0]={...s.compartments[0],labelSet:'categorical',labelValue:1};s.compartments[1]={...s.compartments[1],labelSet:'categorical',labelValue:2};}
  const original=s.compartments.map(layer=>layer.mask.slice()),states=[];
  for(let j=0;j<60;j++)states.push(view(s,{stateId:`state-${j}`,removed:Array.from({length:j%11},()=>Math.floor(random()*s.mri.length)),
    moved:Array.from({length:j%9},()=>[Math.floor(random()*s.mri.length),Math.floor(random()*s.mri.length)])},coupled));
  let previous=null,previousMasks=null;
  for(const requested of [...states,...states.slice().reverse(),...states.slice(10,30)]){
    const actual=modeledVolume(s,requested),expected=referenceModeledVolume(s,requested);compare(actual,expected);
    if(previous)previous.compartments.forEach((layer,i)=>assert.deepEqual(layer.mask,previousMasks[i],'published earlier masks must stay immutable'));
    previous=actual;previousMasks=actual.compartments.map(layer=>layer.mask.slice());
    s.compartments.forEach((layer,i)=>assert.deepEqual(layer.mask,original[i]));
    assert.ok(modeledVolumeCacheStatistics(s).frames<=3);assert.equal(modeledVolumeCacheStatistics(s).baseOwnerBytes,s.mri.length*2);
  }
});

test('localized changes copy only changed layers, preserve source surfaces, and restore exact source ownership after eviction',()=>{
  const s=source(131070);s.compartments[0].mask.fill(1);s.compartments[0].mask[7]=0;s.compartments[1].mask[7]=1;
  const surface={source:new Float32Array(0),positions:new Float32Array(0),normals:new Float32Array(0),sourceNormals:new Float32Array(0),indices:new Uint32Array(0)};
  s.compartments.forEach(layer=>layer.surface=surface);
  const initial=modeledVolume(s,view(s));assert.equal(initial.compartments[0].mask,s.compartments[0].mask);
  const changed=modeledVolume(s,view(s,{removed:[7]})),measurement=modeledVolumeWork(changed);
  assert.equal(measurement.fullBuild,false);assert.equal(measurement.sourceCellsScanned,0);assert.equal(measurement.patchedCells,1);
  assert.equal(measurement.copiedMaskBytes,s.mri.length);assert.equal(changed.compartments[0],initial.compartments[0]);
  assert.equal(changed.compartments[2],initial.compartments[2]);assert.equal(changed.compartments[1].surface,undefined);
  for(const cell of [8,9,10,11])modeledVolume(s,view(s,{removed:[7,cell]}));
  const restored=modeledVolume(s,view(s));compare(restored,referenceModeledVolume(s,view(s)));
  assert.equal(restored.compartments[1].mask,s.compartments[1].mask);assert.equal(restored.compartments[1].surface,surface);
  const same=modeledVolume(s,view(s,{stateId:'pose-only',elapsedSeconds:999}));assert.equal(same,restored);
  assert.deepEqual(modeledVolumeWork(same),{fullBuild:false,sourceCellsScanned:0,copiedMaskBytes:0,patchedCells:0,baseOwnerBytes:s.mri.length*2});
});

test('source and mapping changes use a complete independent fallback; invalid native indices still fail',()=>{
  const a=source(),b=source();a.compartments[0].mask.fill(1);b.compartments[2].mask.fill(1);
  const previous=modeledVolume(a,view(a));compare(modeledVolume(b,view(b)),referenceModeledVolume(b,view(b)));
  const reordered=[regions[2],regions[0],{...regions[1],material:'unknown'}],request=view(a,{scene:{shape:a.shape,affine,regions:reordered},moved:[[0,1],[1,2]]},true);
  const changed=modeledVolume(a,request);compare(changed,referenceModeledVolume(a,request));assert.equal(modeledVolumeWork(changed).fullBuild,true);
  const restored=modeledVolume(a,view(a));compare(restored,referenceModeledVolume(a,view(a)));assert.equal(modeledVolumeWork(restored).fullBuild,true);
  assert.notEqual(restored,previous,'mapping replacement releases the previous three-frame owner');
  assert.throws(()=>modeledVolume(a,view(a,{removed:[a.mri.length]})),/Invalid removed/);
  assert.throws(()=>modeledVolume(a,view(a,{moved:[[0,-1]]})),/Invalid displacement/);
});

test('whole membrane patch rendering survives release and clears only after actual extraction',()=>{
  const count=125,mask=new Uint8Array(count);mask[62]=mask[63]=1;
  const region={id:31,name:'source membrane',material:'membrane',color:'#82c8d8',target:false};
  const s={caseHash:'generated-membrane-source',shape:[5,5,5],affine,frame:'RAS+',mri:new Float32Array(count),compartments:[{name:region.name,kind:'membrane',color:region.color,mask}]};
  const frame={stateId:'held',scene:{shape:s.shape,affine,regions:[region]},removed:[],moved:[[62,37],[63,38]],holds:{left:{kind:'membrane-patch-v1',cells:[62,63]}}};
  const held=modeledVolume(s,frame);assert.equal(held.compartments[0].mask[62],0);assert.equal(held.compartments[0].mask[37],1);assert.equal(held.compartments[0].mask[38],1);
  const released=modeledVolume(s,{...frame,stateId:'released',holds:{}});assert.equal(released,held,'native moved source IDs retain the piece without a hold');
  const extracted=modeledVolume(s,{...frame,stateId:'extracted',holds:{},moved:[],removed:[62,63]});assert.equal(extracted.compartments[0].mask.reduce((a,b)=>a+b,0),0);
  assert.equal(mask[62],1);assert.equal(mask[63],1);assert.equal(mask[37],0);assert.equal(mask[38],0);
});

function inspectionFixture(){
  const s=source();s.caseId='generated-inspection';
  s.compartments.forEach((layer,i)=>{layer.kind=i===1?'tumor':'unknown';layer.mask.fill(1);layer.surface={source:new Float32Array(9),positions:new Float32Array(9),normals:new Float32Array(9),sourceNormals:new Float32Array(9),indices:new Uint32Array([0,1,2])};});
  s.packedLabels={data:new Uint8Array(s.mri.length*4),codes:[]};
  const hashes=Object.fromEntries(s.compartments.map((l,i)=>[l.name,'sha256:'+String(i+1).repeat(64)]));
  const marker={schema:'resectionlab.cranial-planning-candidate.v1',status:'inspection-only',sourcePatientGroup:'generated-person',sourceCollectionHash:'generated-collection',targetLayerIds:['tumor'],patientTaskAdmission:false,officialTrainClaim:false,annotationAuthority:false,physicalProtectionIndexConstructed:false};
  const v={...view(s),scene:{shape:s.shape,affine:s.affine,regions:regions.map(r=>({...r,material:r.target?'tumor':'unknown'})),source:{caseHash:s.caseHash,caseId:s.caseId,annotationHashes:hashes,adapter:{type:'native-patient-source-inspection',sourceRole:'DEVELOPMENT_CANDIDATE',sourcePatient:marker.sourcePatientGroup,sourceCollectionHash:marker.sourceCollectionHash,physicalProtectionIndexConstructed:false}}},history:[],cuts:[],contact:[],thermal:{},fluid:{},slots:{},metrics:{elapsedSeconds:0,targetRemovedMm3:0,nonTargetRemovedMm3:0},protection:{configured:false}};
  const annotations=Object.fromEntries(Object.entries(hashes).map(([name,maskHash])=>[name,{maskHash}]));
  registerSourceInspection(s,marker,annotations);return {s,v,marker,annotations};
}
test('unacted inspection reuses exact source overlaps, masks, packed labels and surfaces with no owner allocation',()=>{
  const {s,v}=inspectionFixture(),masks=s.compartments.map(l=>l.mask),surfaces=s.compartments.map(l=>l.surface),packed=s.packedLabels;
  assert.equal(modeledVolume(s,v),s);assert.equal(modeledVolume({...s,planningHash:'viewer-wrapper'},v).mri,s.mri);
  assert.equal(modeledVolume(s,v).packedLabels,packed);assert.deepEqual(s.compartments.map(l=>l.mask),masks);assert.deepEqual(s.compartments.map(l=>l.surface),surfaces);
  assert(s.compartments.every(l=>l.mask.every(x=>x===1)),'independent overlaps cannot become exclusive owner masks');
  assert.deepEqual(modeledVolumeWork(s),{fullBuild:false,sourceCellsScanned:0,copiedMaskBytes:0,patchedCells:0,baseOwnerBytes:0});assert.equal(modeledVolumeCacheStatistics(s).baseOwnerBytes,0);
});
test('inspection requires source marker/image/grid/group/collection/annotation identity, not an adapter string',()=>{
  const {s,v}=inspectionFixture();const unregistered={...s,mri:s.mri.slice()};assert.throws(()=>modeledVolume(unregistered,v),/inspection marker/);
  for(const mutate of [x=>x.scene.source.caseHash='wrong',x=>x.scene.source.adapter.sourcePatient='other',x=>x.scene.source.adapter.sourceCollectionHash='other',x=>x.scene.source.annotationHashes.brain='wrong',x=>x.scene.regions[0].material='bone']){const bad=structuredClone(v);mutate(bad);assert.throws(()=>modeledVolume(s,bad),/binding changed/);}
  assert.throws(()=>modeledVolume({...s,compartments:s.compartments.map(l=>({...l,mask:l.mask.slice()}))},v),/binding changed/);
  assert.throws(()=>modeledVolume({...s,affine:s.affine.map((r,i)=>r.map((x,j)=>x+(i===0&&j===3?1:0)))},v),/affine mismatch/);
});
test('inspection never conceals native history, tissue effects, tool activation or physical permission',()=>{
  const {s,v}=inspectionFixture();
  for(const extra of [{history:[{action:{operation:'configure'}}]},{removed:[1]},{moved:[[1,2]]},{thermal:{1:1}},{deformation:{1:[0,0,0]}},{drillingV2:{inputEnergyJ:1}},{metrics:{elapsedSeconds:1}},{slots:{left:{active:true,inserted:false,grip:[]}}},{protection:{configured:true}}])assert.throws(()=>modeledVolume(s,{...v,...extra}),/unacted source/);
});
test('real generated standard-physics BridgeSession create DTO satisfies the pristine inspection predicate through the actual loader',async()=>{
  const repo=fileURLToPath(new URL('../../',import.meta.url));
  const script=`import base64,json,tempfile\nfrom pathlib import Path\nimport numpy as np\nfrom resectionlab.core import CaseData,SourceRef,array_digest,thaw_json\nfrom resectionlab.desktop_bridge import BridgeSession,_Request\nshape=(4,5,6);a=np.eye(4);mri=np.arange(120,dtype=np.float32).reshape(shape);t=np.zeros(shape,bool);t[1:3,2:4,2:4]=1;g=np.ones(shape,bool);masks={'model-target':t,'critical':g}\nmarker={'schema':'resectionlab.cranial-planning-candidate.v1','status':'inspection-only','sourcePatientGroup':'generated-person','sourceCollectionHash':'generated-collection','targetLayerIds':['model-target'],'patientTaskAdmission':False,'officialTrainClaim':False,'annotationAuthority':False,'physicalProtectionIndexConstructed':False}\nrows={k:{'kind':'tumor'if k=='model-target'else'unknown','rehearsal_target':k=='model-target'}for k in masks}\nmeta={'generated':True,'is_synthetic':True,'data_case':{'family_id':'GENERATED:inspection','role':'DEVELOPMENT_CANDIDATE'},'anatomy_layers':rows,'surgical_properties':{'cranialPlanningCandidateV1':marker},'source_inspection_annotations':{k:{'maskHash':array_digest(v)}for k,v in masks.items()},'surgical_adapter':{'type':'native-patient-source-inspection','sourceRole':'DEVELOPMENT_CANDIDATE','sourcePatient':'generated-person','sourceCollectionHash':'generated-collection','physicalProtectionIndexConstructed':False}}\ncase=CaseData('generated-inspection',mri,masks,a,(SourceRef('generated','synthetic://inspection',provenance='simulated'),),metadata=meta)\nwith tempfile.TemporaryDirectory()as d:\n owner=BridgeSession(Path(d)/'transfers')\n try:\n  payload=owner._install_case(case,{},_Request('generated-install'));result=owner.execute('surgicalRehearsal',{'caseHash':case.semantic_hash,'command':'create'},_Request('generated-create'),lambda*a:None)\n  assets={}\n  def transport(row):\n   row['assetId']=row.pop('path');assets[row['assetId']]=base64.b64encode(Path(row['assetId']).read_bytes()).decode()\n  transport(payload['mri'])\n  for row in payload['compartments']:\n   transport(row['array']);transport(row['sourceArray'])\n  print(json.dumps({'payload':payload,'assets':assets,'view':result['view']}))\n finally:owner.close()\n`;
  const run=spawnSync(`${repo}/.venv/bin/python`,['-B','-c',script],{cwd:repo,encoding:'utf8',timeout:20000,maxBuffer:4*1024*1024,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});assert.equal(run.status,0,run.stderr);
  const actual=JSON.parse(run.stdout),loaded=await hydrateCase(actual.payload,{readAsset:async id=>new Uint8Array(Buffer.from(actual.assets[id],'base64'))});
  assert.equal(actual.view.physics.profile.schema,'resectionlab.tissue-physics.v2');assert.equal(modeledVolume(loaded,actual.view),loaded);assert.equal(modeledVolumeWork(loaded).baseOwnerBytes,0);assert.equal(actual.view.history.length,0);
  const bad=structuredClone(actual.view);bad.drillingV2.inputEnergyJ=1;assert.throws(()=>modeledVolume(loaded,bad),/unacted source/);
});
