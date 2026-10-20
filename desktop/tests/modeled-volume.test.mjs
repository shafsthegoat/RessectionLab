import assert from 'node:assert/strict';
import {test} from 'node:test';
import {modeledVolume,referenceModeledVolume,modeledVolumeWork,modeledVolumeCacheStatistics} from '../src/modeled-volume.ts';

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
