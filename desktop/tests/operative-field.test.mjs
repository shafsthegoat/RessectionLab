import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {operativeRouteIssue,playableSteps,readOperativeField,timelinePose} from '../src/operative-field.ts';

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
