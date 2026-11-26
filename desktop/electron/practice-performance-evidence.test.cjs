'use strict';

const assert=require('node:assert/strict');
const test=require('node:test');
const {alignRendererClock,findCompletion}=require('./practice-performance-evidence.cjs');
const {distribution,validatePair}=require('../../scripts/summarize_practice_desktop.cjs');

const sample=()=>({
  frames:[{at:11,serial:3,stateId:'accepted',source:'source',modelGeneration:7,displayIdentity:'case:2'}],
  fences:[{at:12,serial:3,stateId:'accepted',source:'source',modelGeneration:7,displayIdentity:'case:2',completed:true}],
  publications:[{at:13,serial:3,stateId:'accepted',source:'source',modelGeneration:7,displayIdentity:'case:2'}],
});

test('only the GPU fence for the submitted accepted state can end the action interval',()=>{
  const completion=trace=>findCompletion(trace,{at:10},'source','accepted');
  const trace=sample();
  assert.equal(completion(trace).fence.at,12);
  for(const change of [
    row=>{row.fences[0].serial=2;},
    row=>{row.fences[0].modelGeneration=6;},
    row=>{row.frames[0].preview=true;},
    row=>{row.fences[0].source='old';},
    row=>{row.fences[0].at=9;},
    row=>{row.fences[0].completed=false;},
    row=>{row.publications[0].displayIdentity='case:1';},
  ]){
    const stale=sample();
    change(stale);
    assert.equal(completion(stale),null);
  }
});

test('clock alignment bounds quantization and withholds incompatible offsets',async()=>{
  const aligned=await alignRendererClock(async()=>performance.now());
  assert(aligned.valid && aligned.offset>=aligned.low && aligned.offset<=aligned.high);
  let calls=0;
  const incompatible=await alignRendererClock(async()=>performance.now()+100000*(++calls%2));
  assert.equal(incompatible.valid,false);
  assert.equal(incompatible.offset,undefined);
});

test('summary keeps the largest sample and requires enough observations for p99',()=>{
  assert.deepEqual(distribution([1,3,7,1000,null]),{n:4,p50:5,p95:1000,max:1000});
  assert.equal(distribution(Array.from({length:300},(_,i)=>i)).p99,296);
});

test('paired actions retain exact physics and poses with separate practice episode IDs',()=>{
  const report={originalSha256:'source-file',actionStreamSha256:'source-actions',
    performance:{viewport:{width:1440,height:900}},
    parent:{caseHash:'case',stateId:'parent',position:1,branchId:'source-branch'},
    machine:{cpu:'test'},savedActionsVerified:1,actions:[{
      before:'parent',label:'Place tool',display:{camera:[0,0,1]},
      result:{requested:{operation:'place'},status:'accepted',completedFraction:1,
        witness:{branchId:'episode-a',stateId:'committed',physicalDigest:'exact'}},
    }]};
  const independent=structuredClone(report);
  independent.actions[0].result.witness.branchId='episode-b';
  assert.match(validatePair(report,independent),/^[0-9a-f]{64}$/);
  independent.actions[0].result.witness.physicalDigest='changed';
  assert.throws(()=>validatePair(report,independent),/Accepted physical states/);
});
