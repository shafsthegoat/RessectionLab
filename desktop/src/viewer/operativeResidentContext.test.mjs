import test from 'node:test';
import assert from 'node:assert/strict';
import {missingOperativeSurfaces,loadOperativeSurfaces} from './operativeResidentContext.ts';

const skull={id:'skull',role:'bone',sampledVoxels:12,sourceSeries:'mri',sampledDataSha256:'reviewed-label'};
const field={layers:[skull],snapshotId:'revision'};
const base={displayIdentity:JSON.stringify(['collection','mri','signal','revision',false,1]),compartments:[]};

test('supplement only a missing unambiguous source skull on the exact displayed revision',()=>{
  assert.deepEqual(missingOperativeSurfaces(base,field,'collection'),[skull]);
  const gm={...skull,id:'cortex',role:'brain',detail:'grayMatter'};
  assert.deepEqual(missingOperativeSurfaces(base,{...field,layers:[skull,gm]},'collection'),[skull,gm]);
  assert.deepEqual(missingOperativeSurfaces(base,{...field,layers:[gm,{...gm,id:'second'}]},'collection'),[]);
  for(const displayIdentity of ['surgical:case','null','{}','[1,2]'])assert.deepEqual(missingOperativeSurfaces({...base,displayIdentity},field,'collection'),[]);
  assert.deepEqual(missingOperativeSurfaces({...base,compartments:[{name:'skull'}]},field,'collection'),[]);
  assert.deepEqual(missingOperativeSurfaces(base,field,'other'),[]);
  assert.deepEqual(missingOperativeSurfaces(base,{...field,snapshotId:'other'},'collection'),[]);
  assert.deepEqual(missingOperativeSurfaces(base,{...field,layers:[skull,skull]},'collection'),[]);
  assert.deepEqual(missingOperativeSurfaces(base,{...field,layers:[{...skull,adjudication:{}}]},'collection'),[]);
  assert.deepEqual(missingOperativeSurfaces({...base,sourceFrameIssue:'unknown frame'},field,'collection'),[]);
});

test('reject changed scan, unsupported registration or support and cancelled hydration',async()=>{
  const controller=new AbortController();let response;
  const api={async anatomyView(request){
    assert.deepEqual(request.layers,['skull']);assert.equal(request.seriesId,'mri');assert.equal(request.snapshotId,'revision');
    return response;
  },readAsset(){throw Error('Unexpected transfer');}};
  response={mri:{sha256:'other'}};
  await assert.rejects(loadOperativeSurfaces(api,base,[skull],controller.signal),/different source/);
  const source={name:'skull',surface:{},sourceSeries:'mri',alignment:'estimated'};
  response={mri:{sha256:'signal'},compartments:[source]};
  await assert.rejects(loadOperativeSurfaces(api,base,[skull],controller.signal),/No matching/);
  source.alignment='reviewed';source.sourceSupport={sampledDataSha256:'different-label'};
  await assert.rejects(loadOperativeSurfaces(api,base,[skull],controller.signal),/No matching/);
  controller.abort();
  await assert.rejects(loadOperativeSurfaces(api,base,[skull],controller.signal),e=>e.name==='AbortError');
});
