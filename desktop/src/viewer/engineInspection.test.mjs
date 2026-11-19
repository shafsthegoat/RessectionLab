import assert from 'node:assert/strict';
import test from 'node:test';
import {modeledVolume} from '../modeled-volume.ts';
import {inspectionSection,inspectionSource,sectionClasses} from './engineInspection.ts';
import {resultMembership,resultMaskCrop,resultMaskHash} from './resultInspectionData.ts';
import {transformPoint} from './coordinates.ts';
import {modeledAnatomy} from './modeledAnatomy.ts';

const shape=[7,8,9],affine=[[0,-1,0,12],[2,0,0,-8],[0,0,1,4],[0,0,0,1]];
const index=(i,j,k)=>(i*shape[1]+j)*shape[2]+k;
const cells=[index(3,4,5),index(4,4,5),index(4,5,5)],vessel=index(4,5,5);
const volumes=shape.reduce((n,s)=>n*s,1);
const mask=a=>Uint8Array.from({length:volumes},(_,i)=>Number(a.includes(i)));
const source={caseId:'grid',caseHash:'grid',shape,affine,frame:'RAS+',mri:new Float32Array(volumes),compartments:[
  {name:'tissue',mask:new Uint8Array(volumes).fill(1)},
  {name:'target',mask:mask(cells)},{name:'vessel',mask:mask([vessel])},
]};
const original={scene:{shape,affine,resolutionMm:[2,1,1],regions:[
  {id:1,material:'soft_tissue',name:'tissue',target:false},{id:2,material:'tumor',name:'target',target:true},
  {id:3,material:'vessel',name:'vessel',target:false}]},
  removed:[],moved:[],slots:{right:{inserted:true,instrument:'cusa',position:[8,-2,9]}}};
const at=(section,i,j)=>section.membership[(section.height-1-j+section.origin[1])*section.width+i-section.origin[0]];

test('operated section follows exact source ownership, accepted poses and non-monotonic replay',()=>{
  const intact=modeledVolume(source,original);
  const base=inspectionSource(modeledAnatomy(intact).surfaceReference.owner,original);
  const inspect=v=>inspectionSection(modeledVolume(source,v),v,base,2,null);
  const before=inspect(original);
  assert.equal(before.depth,5);
  assert.equal(before.targetPixels,2);assert.equal(before.cavityPixels,0);
  assert.equal(at(before,4,5),sectionClasses.vessel);
  assert.deepEqual(before.labels,['P','A','L','R']);
  assert.equal(before.tips[0].inPlane,true);
  assert.equal(before.tips[0].x,3-before.origin[0]+.5);
  assert.equal(before.tips[0].y,before.origin[1]+before.height-4-.5);
  const removed={...original,removed:[cells[0]],slots:{}};
  const after=inspect(removed);
  assert.equal(after.targetPixels,1);assert.equal(after.cavityPixels,1);
  assert.equal(at(after,3,4),sectionClasses.cavity);assert.equal(after.tips.length,0);
  assert.deepEqual(inspect(original),before);assert.deepEqual(inspect(removed),after);
  const outside=inspectionSection(intact,original,base,2,2);
  assert.equal(outside.depth,2);assert.equal(outside.tips[0].inPlane,false);
  assert.equal(outside.cavityPixels,0);assert.equal(outside.targetPixels,0);
});

test('moved tissue that occupies an excavated cell is retained in the current cut',()=>{
  const view={...original,removed:[cells[0]],moved:[[cells[1],cells[0]]]};
  const volume=modeledVolume(source,view);
  const section=inspectionSection(volume,view,inspectionSource(modeledAnatomy(volume).surfaceReference.owner,view),2,null);
  assert.equal(at(section,3,4),sectionClasses.target);
  assert.equal(at(section,4,4),sectionClasses.tissue);
  assert.equal(section.cavityPixels,0);
});

test('3D cavity cast, displaced residual ownership and cropped affine follow forward and backward seeks',()=>{
  const inspect=view=>resultMembership(modeledVolume(source,view),view);
  const before=inspect(original);
  assert.equal(before.cellMm3,2);assert.equal(before.overlapOriginal,1);assert.equal(before.overlapRemaining,1);
  const overlappingRemoval=inspect({...original,removed:[vessel]});
  assert.equal(overlappingRemoval.overlapRemaining,0);
  assert.deepEqual([...overlappingRemoval.residual],cells.slice(0,2));
  assert.deepEqual([...before.residual],cells.slice(0,2));assert.equal(before.cavity.length,0);
  const destination=index(1,3,7);
  const state={...original,removed:[cells[0],index(2,1,1)],moved:[[cells[1],destination]]};
  const next=inspect(state);assert.deepEqual([...next.residual],[destination]);
  assert.deepEqual([...next.cavity],state.removed.toSorted((a,b)=>a-b));
  assert.deepEqual(inspect(original),before);assert.deepEqual(inspect(state),next);
  assert.notEqual(resultMaskHash(before.residual),resultMaskHash(next.residual));
  const cropped=resultMaskCrop(next.cavity,state);assert.equal(cropped.mask.reduce((a,b)=>a+b,0),2);
  assert.deepEqual(transformPoint(cropped.affine,[1,3,4]),transformPoint(affine,[3,4,5]));
  const filled=inspect({...original,removed:[cells[0]],moved:[[cells[1],cells[0]]]});
  assert.equal(filled.cavity.length,0);assert.deepEqual([...filled.residual],[cells[0]]);
});
