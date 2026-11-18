import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {detailBits,operativeRouteIssue,playableSteps,readOperativeField,timelinePose} from '../src/operative-field.ts';

const route={id:'A',sourceSnapshotId:'patient-A-revision-8',targetLayerId:'lesion',parameterHash:'geometry',assessmentStatus:'current'};
const voxels=Uint8Array.of(1,2,4,8,16,24,8,31);
const bytes={dtype:'uint8',order:'C',byteOrder:'little',shape:[2,2,2],byteLength:8,assetId:'a',
  sha256:createHash('sha256').update(voxels).digest('hex')};
const packet={snapshotId:route.sourceSnapshotId,approachId:'A',targetLayerId:'lesion',parameterHash:'geometry',
  units:'mm',basisRas:[[1,0,0],[0,1,0],[0,0,1]],targetRasMm:[0,0,0],materialBits:{bone:1,dura:2,brain:4,tumor:8,vessels:16},
  shape:[2,2,2],origin:[-1,-1,0],spacingMm:1,radiusMm:1,routeRadiusMm:1,entryDepthMm:1,
  stations:{bone:null,dura:null,brain:1,tumor:0},stationEvidence:{bone:null,dura:null,brain:'source',tumor:'source'},
  layers:[],excluded:[],encounters:[],blockers:[],unknowns:[],stationIssues:[],simulationExecutionAuthorized:false,labels:bytes};

test('read a bound overlapping-label field and reject changed source, approach, layout and bytes',async()=>{
  const read=async()=>voxels;
  const field=await readOperativeField(packet,route,read);
  assert.deepEqual(field.voxels,voxels);
  for(const mismatch of [{parameterHash:'other'},{targetLayerId:'other'},{snapshotId:'old'},{approachId:'B'},
    {simulationExecutionAuthorized:true},{spacingMm:0},{stations:{}},{shape:[2,2,8]},{units:'cm'},
    {materialBits:{...packet.materialBits,dura:8}},{stations:{...packet.stations,tumor:32}},
    {stationEvidence:{...packet.stationEvidence,tumor:'proxy'}},{stationEvidence:{}},
    {labels:{...bytes,order:'F'}},{labels:{...bytes,shape:[8,1,1]}}]){
    await assert.rejects(()=>readOperativeField({...packet,...mismatch},route,read));
  }
  await assert.rejects(()=>readOperativeField(packet,route,async()=>new Uint8Array(8)),/verify/);
});
test('stale, missing and wrong-target plans cannot open an earlier field',()=>{
  assert.equal(operativeRouteIssue(route,'lesion',route.sourceSnapshotId),null);
  for(const args of [[null,'lesion',route.sourceSnapshotId],[route,'other',route.sourceSnapshotId],[route,'lesion','old'],
    [{...route,assessmentStatus:'stale'},'lesion',route.sourceSnapshotId]]){
    assert.ok(operativeRouteIssue(...args));
  }
});
test('timeline skips missing layers, optionally omits retraction, and keeps the last stage fully applied',()=>{
  const field={...packet,voxels},steps=playableSteps(field,true);
  assert.deepEqual(steps.map(s=>s.id),['brain','tumor']);
  assert.deepEqual(timelinePose(steps,0),{phase:'overview',progress:0});
  assert.deepEqual(timelinePose(steps,1),{phase:'brain',progress:1});
  assert.deepEqual(timelinePose(steps,1.25),{phase:'tumor',progress:.25});
  assert.deepEqual(timelinePose(steps,2),{phase:'tumor',progress:1});
  assert.deepEqual(playableSteps(field,false).map(s=>s.id),['tumor']);
  assert.deepEqual(playableSteps({...field,voxels:voxels.map(v=>v|16)},true).map(s=>s.id),['brain']);
  assert.equal(timelinePose([],2).phase,'overview');
});

test('operative scan texture verifies its local frame, source identity, coverage, and transferred bytes',async()=>{
  const brightness=Uint8Array.of(0,25,72,102,140,211,222,255);
  const descriptor={...bytes,assetId:'brightness',sha256:createHash('sha256').update(brightness).digest('hex')};
  const intensity={signal:descriptor,shape:[2,2,2],origin:[-1,-1,0],spacingMm:[1,1,1],unknownValue:0,
    sourceSeries:'t1',sourceSha256:'c'.repeat(64),dataSha256:'b'.repeat(64),modality:'T1',
    window:[3,225],windowUnits:'source intensity',windowMethod:'Local measured percentiles',coveredVoxels:7,
    sourceShape:[9,9,9],sourceVoxelToPatientRasMm:[[0,-1,0,2],[1,0,0,-3],[0,0,1,4],[0,0,0,1]]};
  const read=async id=>id==='a'?voxels:brightness;
  const field=await readOperativeField({...packet,intensity},route,read);
  assert.deepEqual(field.intensity.voxels,brightness);
  assert.deepEqual(field.voxels,voxels);
  assert.equal(field.intensity.signal,undefined);
  assert.equal((await readOperativeField(packet,route,read)).intensity,undefined);
  const fine=Uint8Array.from({length:18},(_,i)=>i+1);
  const fineShape=[3,2,3];
  const fineDescriptor={...descriptor,shape:fineShape,byteLength:fine.length,sha256:createHash('sha256').update(fine).digest('hex')};
  const finer=await readOperativeField({...packet,intensity:{...intensity,shape:fineShape,spacingMm:[.5,1,.5],
    signal:fineDescriptor,coveredVoxels:fine.length}},route,async id=>id==='a'?voxels:fine);
  assert.deepEqual(finer.intensity.voxels,fine);
  for(const shape of [[257,2,2],[126,126,126]]){
    await assert.rejects(()=>readOperativeField({...packet,intensity:{...intensity,shape,
      spacingMm:shape.map(n=>1/(n-1))}},route,read),/intensity grid/);
  }
  for(const bad of [{origin:[1,-1,0]},{spacingMm:[1,1,.5]},{shape:[2,2,3]},
    {unknownValue:255},{coveredVoxels:8},{sourceSha256:'unknown'},{modality:'MIP'},
    {sourceVoxelToPatientRasMm:[[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},{window:[34,4]},{windowUnits:'HU'},{signal:{...descriptor,order:'F'}},{signal:{...descriptor,sha256:'a'.repeat(64)}}]){
    await assert.rejects(()=>readOperativeField({...packet,intensity:{...intensity,...bad}},route,read),/intensity|scan/i);
  }
});

test('extended source labels survive transfer and require the matching class map',async()=>{
  const extended=Uint16Array.of(4|32,4|64,128,256,16|512,16|1024,16|2048,8|4096);
  const raw=new Uint8Array(extended.buffer);
  const labels={...bytes,dtype:'uint16',byteLength:raw.byteLength,sha256:createHash('sha256').update(raw).digest('hex')};
  const result={...packet,labels,materialBits:{...packet.materialBits,...detailBits}};
  const field=await readOperativeField(result,route,async()=>raw);
  assert.deepEqual(field.voxels,extended);
  await assert.rejects(()=>readOperativeField({...result,materialBits:{...result.materialBits,artery:1024}},route,async()=>raw),/classes/);
  await assert.rejects(()=>readOperativeField(result,route,async()=>raw.subarray(1)),/verify/);
});
