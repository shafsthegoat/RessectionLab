import test from 'node:test';
import assert from 'node:assert/strict';
import {missingOperativeSkull,loadOperativeSkull} from './operativeResidentContext.ts';

const skull={id:'skull',role:'bone',sampledVoxels:12,sourceSeries:'mri',sampledDataSha256:'reviewed-label'};
const field={layers:[skull],snapshotId:'revision'};
const base={displayIdentity:JSON.stringify(['collection','mri','signal','revision',false,1]),compartments:[]};

test('supplement only a missing unambiguous source skull on the exact displayed revision',()=>{
  assert.equal(missingOperativeSkull(base,field,'collection'),skull);
  for(const displayIdentity of ['surgical:case','null','{}','[1,2]'])assert.equal(missingOperativeSkull({...base,displayIdentity},field,'collection'),null);
  assert.equal(missingOperativeSkull({...base,compartments:[{name:'skull'}]},field,'collection'),null);
  assert.equal(missingOperativeSkull(base,field,'other'),null);
  assert.equal(missingOperativeSkull(base,{...field,snapshotId:'other'},'collection'),null);
  assert.equal(missingOperativeSkull(base,{...field,layers:[skull,skull]},'collection'),null);
  assert.equal(missingOperativeSkull(base,{...field,layers:[{...skull,adjudication:{}}]},'collection'),null);
  assert.equal(missingOperativeSkull({...base,sourceFrameIssue:'unknown frame'},field,'collection'),null);
});

test('reject changed scan, unsupported registration or support and cancelled hydration',async()=>{
  const controller=new AbortController();let response;
  const api={async anatomyView(request){
    assert.deepEqual(request.layers,['skull']);assert.equal(request.seriesId,'mri');assert.equal(request.snapshotId,'revision');
    return response;
  },readAsset(){throw Error('Unexpected transfer');}};
  response={mri:{sha256:'other'}};
  await assert.rejects(loadOperativeSkull(api,base,skull,controller.signal),/different source/);
  const source={name:'skull',surface:{},sourceSeries:'mri',alignment:'estimated'};
  response={mri:{sha256:'signal'},compartments:[source]};
  await assert.rejects(loadOperativeSkull(api,base,skull,controller.signal),/no matching/);
  source.alignment='reviewed';source.sourceSupport={sampledDataSha256:'different-label'};
  await assert.rejects(loadOperativeSkull(api,base,skull,controller.signal),/no matching/);
  controller.abort();
  await assert.rejects(loadOperativeSkull(api,base,skull,controller.signal),e=>e.name==='AbortError');
});
