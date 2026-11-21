import test from 'node:test';
import assert from 'node:assert/strict';
import {membraneTransport} from './membraneTransport.ts';

const stage={schema:'resectionlab.cranial-stages.v1',contractHash:'patch-A',sceneId:'a',stateId:'s',
  assessmentCurrent:true,bonePrefixVerified:true,patchCells:2,patchHeld:true,requiredBoundaryEdges:6};
const view={stateId:'s',scene:{sceneId:'a',shape:[4,5,6],affine:[[1,1,0,80],[0,2,0,-75],[0,0,-2,100],[0,0,0,1]]},
  cranialStages:stage,slots:{right:{instrument:'membrane_forceps'}},moved:[[46,51],[76,81]],removed:[],
  holds:{right:{kind:'membrane-patch-v1',contractHash:'patch-A',cells:[46,76],offset:[0,1,-1]}}};
test('patch readout follows accepted affine translation and exact whole-patch destinations',()=>{
  assert.equal(membraneTransport(view),'Membrane held · 3 mm from source · right hand');
  assert.equal(membraneTransport({...view,moved:[],holds:{right:{...view.holds.right,offset:[0,0,0]}}}),'Membrane held at source · right hand');
  assert.equal(membraneTransport({...view,moved:[[46,51]]}),'Membrane held');
  assert.equal(membraneTransport({...view,removed:[76]}),'Membrane held');
  assert.equal(membraneTransport({...view,holds:{right:{...view.holds.right,contractHash:'other'}}}),'Membrane held');
});
test('only the bound assessed stage distinguishes cutting, release and verified extraction',()=>{
  const caption=patch=>membraneTransport({...view,cranialStages:{...stage,patchHeld:false,...patch}});
  assert.equal(caption({stateId:'unpublished'}),null);
  assert.equal(caption({sceneId:'other'}),null);
  assert.equal(caption({assessmentCurrent:false}),null);
  assert.equal(caption({cutBoundaryEdges:3}),'Membrane cut · 3 of 6 edges');
  assert.equal(caption({boundaryCutComplete:true}),'Membrane boundary cut · patch retained');
  assert.equal(caption({patchDetached:true}),'Membrane released in field');
  assert.equal(caption({patchExtracted:true,duraCleared:true,membrane:{actualNativeExtractionEvents:1,removedMembraneMm3:49}}),'Membrane extracted · 49 mm³');
});
