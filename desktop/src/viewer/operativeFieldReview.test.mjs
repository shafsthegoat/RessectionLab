import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {operativeVoxelSurface} from './operativeVoxelSurface.ts';
import {operativeFieldContext} from './operativeFieldContext.ts';
import {operativeSectionVisible,operativeSectionContext} from './operativeSectionContext.ts';
import {fieldTargetOnOverview} from './operativeFieldSlice.ts';

const field={shape:[2,2,2],origin:[0,0,0],spacingMm:2,voxels:Uint8Array.of(8,0,0,0,8,0,0,0),
  snapshotId:'rev',targetRasMm:[12,20,30],basisRas:[[0,1,0],[-1,0,0],[0,0,1]],materialBits:{tumor:8},
  layers:[{id:'skull',label:'Supplied skull',sourceSeries:'bone-ct',role:'bone',sampledVoxels:1,sampledDataSha256:'supported'}]};

test('raw adjacent cells share no hidden faces and preserve exact cell and section boundaries',()=>{
  const mask=Uint8Array.of(1,0,1,0,0,0,0,0),bounds={low:[0,0,0],high:[0,1,0]};
  for(const [row,faces,minY] of [[0,10,-1],[1,6,1]]){
    const geometry=operativeVoxelSurface(field,mask,bounds,row);
    assert.equal(geometry.index.count,faces*6);
    geometry.computeBoundingBox();
    assert.deepEqual(geometry.boundingBox.min.toArray(),[-1,minY,-1]);
    assert.deepEqual(geometry.boundingBox.max.toArray(),[1,3,1]);
    geometry.dispose();
  }
  assert.deepEqual([...mask],[1,0,1,0,0,0,0,0]);
});

test('source outline is sampled in patient orientation around the target pixel',()=>{
  const view={shape:[3,3],spacingMm:2,targetPx:[1,1],rightRas:[0,1,0],upRas:[-1,0,0]};
  assert.deepEqual([...fieldTargetOnOverview(field,view)],[0,0,0,0,1,1,0,0,0]);
});

test('surrounding source geometry maps the patient target to the local origin and checks identity and support',()=>{
  const volume={displayIdentity:JSON.stringify(['collection','series','image','rev',false,2]),frame:'RAS+',
    affine:[[1,0,0,12],[0,1,0,20],[0,0,1,30],[0,0,0,1]],compartments:[{name:'skull',
      sourceSupport:{sampledDataSha256:'supported'},
      surface:{indices:Uint32Array.of(0,1,2),positions:new Float32Array(9),normals:new Float32Array(9)}}]};
  const group=operativeFieldContext(volume,field,'collection');
  assert.equal(group.children.length,1);
  group.updateWorldMatrix(false,true);
  assert.deepEqual(new THREE.Vector3(0,0,0).applyMatrix4(group.children[0].matrixWorld).toArray(),[0,0,0]);
  assert.deepEqual(new THREE.Vector3(0,1,0).applyMatrix4(group.children[0].matrixWorld).toArray(),[1,0,0]);
  assert.equal(operativeFieldContext(volume,{...field,snapshotId:'other'},'collection').children.length,0);
  assert.equal(operativeFieldContext(volume,field,'other').children.length,0);
  // The inspected MRI origin maps to voxel [1,0,0] in the contributing CT.
  const ctFrame=[[1,0,0,11],[0,1,0,20],[0,0,1,30],[0,0,0,1]];
  const ct={seriesId:'bone-ct',modality:'CT',sourceSha256:'ct-original',signalSha256:'ct-signal',
    data:Float32Array.of(2000,2000,2000,2000,500,0,0,0),shape:[2,2,2],affine:ctFrame};
  volume.compartments[0].appearanceSource=ct;
  volume.patientContext={sources:[{seriesId:'bone-ct',sourceSha256:'ct-original',displayWindow:[0,1000]}]};
  const scanField={...field,intensity:{sourceSeries:'series',sourceSha256:'mr-original',window:[20,60]}};
  const mesh=operativeFieldContext(volume,scanField,'collection').children[0];
  assert.equal(mesh.geometry.getAttribute('scanSignal').array[0],128);
  assert.equal(mesh.userData.sourceLayer.appearance.seriesId,'bone-ct');
  volume.compartments[0].appearanceSource={...ct,seriesId:'other'};
  assert.equal(operativeFieldContext(volume,scanField,'collection').children[0].geometry.hasAttribute('scanSignal'),false);
  volume.compartments[0].sourceSupport.sampledDataSha256='raw';
  assert.equal(operativeFieldContext(volume,field,'collection').children.length,0);
});

test('depth following retains the exterior for shallow work and respects explicit view choices',()=>{
  const route={stations:{bone:28,dura:22,brain:16,tumor:12},spacingMm:.5,routeRadiusMm:3};
  assert(!operativeSectionVisible(route,'dura','follow'));
  assert(operativeSectionVisible(route,'brain','follow'));
  assert(operativeSectionVisible(route,'tumor','follow'));
  assert(!operativeSectionVisible({...route,stations:{bone:null,dura:null,brain:14,tumor:12}},'tumor','follow'));
  assert(!operativeSectionVisible(route,'tumor','exterior'));
  assert(operativeSectionVisible(route,'overview','section'));
});

test('intact MRI continuation samples the matching source outside the operated footprint only',()=>{
  const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const source={seriesId:'mri',sourceSha256:'original',affine,shape:[8,8,8],data:new Float32Array(512).fill(20)};
  const volume={displayIdentity:JSON.stringify(['study','mri','signal','rev',false,1]),compartments:[{appearanceSource:source,alignment:'reference'}]};
  const local={...field,origin:[0,0,0],targetRasMm:[2,2,2],basisRas:[[1,0,0],[0,1,0],[0,0,1]],spacingMm:1,
    intensity:{sourceSeries:'mri',sourceSha256:'original',window:[0,40]}};
  const plane=operativeSectionContext(volume,'study',local,0);
  assert(plane);
  const {data,width}=plane.material.map.image;
  // Native source extends from RAS 0 to 7, sampled every 0.5 mm.
  assert.deepEqual([...data.slice(0,4)],[128,128,128,255]);
  assert.equal(data[(4*width+4)*4+3],0);
  assert.equal(operativeSectionContext(volume,'other',local,0),null);
  assert.equal(operativeSectionContext(volume,'study',{...local,snapshotId:'old'},0),null);
  source.sourceSha256='another-acquisition';
  assert.equal(operativeSectionContext(volume,'study',local,0),null);
  plane.geometry.dispose();plane.material.map.dispose();plane.material.dispose();
});
