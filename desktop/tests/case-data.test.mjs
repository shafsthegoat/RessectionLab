import assert from 'node:assert/strict';
import {test} from 'node:test';
import {hydrateCase, initialCursor, validateCaseDescriptor, anatomyColor} from '../src/case-data.ts';

const hash = 'a'.repeat(64);
const descriptor = (dtype, assetId) => ({assetId,dtype,shape:[2,2,2],byteOrder:'little',order:'C',sha256:hash,byteLength:dtype==='float32'?32:8});
function fixture() {
  return {caseId:'fixture',caseHash:`sha256:${hash}`,frame:'RAS+',affine:[[2,0,0,10],[0,3,0,-5],[0,0,4,7],[0,0,0,1]],shape:[2,2,2],spacingMm:[2,3,4],mri:descriptor('float32','mri'),compartments:[{name:'enhancing_target',volumeMm3:24,array:descriptor('uint8','mask')}],unknowns:['motor_unknown'],metadata:{is_synthetic:true}};
}
function api(overrides={}) {
  const assets={mri:new Uint8Array(new Float32Array([0,1,2,3,4,5,6,7]).buffer),mask:new Uint8Array([0,0,0,0,0,0,0,1]),...overrides};
  return {readAsset:async id=>assets[id]};
}

test('valid anisotropic affine preserves C-order arrays and physical cursor',async()=>{
  const data=await hydrateCase(fixture(),api());
  assert.deepEqual([...data.mri],[0,1,2,3,4,5,6,7]);
  assert.deepEqual(initialCursor(data),[12,-2,11]);
  assert.deepEqual(data.affine,fixture().affine);
});
test('LPS is converted once without resampling voxel data',async()=>{
  const input=fixture();input.frame='LPS+';
  const data=await hydrateCase(input,api());
  assert.equal(data.frame,'RAS+');assert.deepEqual(initialCursor(data),[-12,2,11]);
  assert.deepEqual([...data.compartments[0].mask],[0,0,0,0,0,0,0,1]);
  assert.equal(input.affine[0][0],2);
});
test('MRI and mask encodings are rejected before transfer',async()=>{
  const mutations=[p=>p.mri.dtype='float64',p=>p.compartments[0].array.dtype='int8',p=>p.mri.byteOrder='big',p=>p.mri.order='F',p=>p.mri.shape=[2,4,1],p=>p.compartments[0].array.shape=[8],p=>p.mri.byteLength=64,p=>p.compartments[0].array.byteLength=7];
  for(const mutate of mutations){const p=fixture();mutate(p);let calls=0;await assert.rejects(hydrateCase(p,{readAsset:async()=>{calls++;return new Uint8Array();}}));assert.equal(calls,0);}
});
test('invalid identities, grids and frames fail closed',()=>{
  const mutations=[p=>p.caseHash='unknown',p=>p.mri.sha256='bad',p=>p.frame='RAS',p=>p.frame='LPS-danger',p=>p.shape=[2,0,2],p=>p.shape=[2,2,2.5],p=>p.shape=[4096,4096,4096],p=>p.affine[0][0]=NaN,p=>p.affine[3][0]=1,p=>p.affine[2]=[0,0,0,0],p=>p.spacingMm[0]=1];
  for(const mutate of mutations){const p=fixture();mutate(p);assert.throws(()=>validateCaseDescriptor(p));}
});
test('descriptor-only source masks and brain mask must match source grid',()=>{
  const p=fixture();p.compartments[0].sourceArray={...descriptor('uint8','source'),shape:[1,2,4]};assert.throws(()=>validateCaseDescriptor(p));
  delete p.compartments[0].sourceArray;p.brainMask=descriptor('float32','brain');assert.throws(()=>validateCaseDescriptor(p));
});
test('truncated buffers and non-finite voxels fail during hydration',async()=>{
  await assert.rejects(hydrateCase(fixture(),api({mri:new Uint8Array(31)})),/incomplete/);
  await assert.rejects(hydrateCase(fixture(),api({mask:new Uint8Array(7)})),/Incomplete/);
  await assert.rejects(hydrateCase(fixture(),api({mri:new Uint8Array(new Float32Array([NaN,0,0,0,0,0,0,0]).buffer)})),/non-finite/);
});
test('binary source mask and displayed physical volume reconcile',async()=>{
  await assert.rejects(hydrateCase(fixture(),api({mask:new Uint8Array([0,0,0,0,0,0,0,2])})),/binary/);
  const p=fixture();p.compartments[0].volumeMm3=1e9;await assert.rejects(hydrateCase(p,api()),/volume disagrees/);
});
test('semantic colors do not depend on compartment order',()=>{
  assert.equal(anatomyColor('enhancing_target'),'#e5a269');
  assert.equal(anatomyColor('nonenhancing_or_necrotic_core'),'#b894d9');
  assert.equal(anatomyColor('FLAIR_abnormality'),'#55bace');
});
