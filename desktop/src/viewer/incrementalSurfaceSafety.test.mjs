import test from 'node:test';
import assert from 'node:assert/strict';
import {IncrementalSurfaceIntersectionAudit,intersectionVertices} from './surfaceSafety.ts';

const budget=16*1024**2;
const members=set=>[...set].sort((a,b)=>a-b);
function fixture(){
  const source=new Float64Array(30*9);
  for(let f=0;f<30;f++)source.set([f*10,0,0,f*10+2,0,0,f*10,2,0],f*9);
  const indices=Uint32Array.from({length:90},(_,i)=>i),data=indices.slice();
  const layout={vertexIds:indices.slice(),vertexCapacity:90,rows:[{buffer:data.buffer,offset:0,begin:0,end:30}]};
  const owner=new IncrementalSurfaceIntersectionAudit(),observer=owner.begin(layout,indices,budget),physical=source.slice();
  const result=observer.run(source,physical,indices);assert.equal(result.size,0);
  observer.complete(source,physical,indices,result);assert(owner.retainedBytes>0);
  return {source,physical,indices,layout,owner};
}
function check(owner,layout,source,candidate,indices,maximum=budget){
  const observer=owner.begin(layout,indices,maximum),actual=observer.run(source,candidate,indices);
  assert.deepEqual(members(actual),members(intersectionVertices(source,candidate,indices)));
  observer.complete(source,candidate,indices,actual);return actual;
}
test('dirty/clean contacts are found on either side of current canonical face order',()=>{
  const {source,indices,layout,owner}=fixture();
  for(const [moving,fixed] of [[0,20],[25,3]]){
    const candidate=source.slice();candidate.set(source.subarray(fixed*9,fixed*9+9),moving*9);
    assert.equal(check(owner,layout,source,candidate,indices).size,6);
    assert.equal(owner.statistics.partialAudits,1);
  }
  assert.equal(check(owner,layout,source,source.slice(),indices).size,0);
  assert.equal(owner.statistics.dirtyFaces,0);
});
test('removed clean faces and newly appended faces preserve exact audit membership',()=>{
  const {source,indices,layout,owner}=fixture();
  const removed=indices.slice(3),rows=[{...layout.rows[0],offset:1,end:29}];
  assert.equal(check(owner,{...layout,rows},source,source,removed).size,0);
  assert.equal(owner.statistics.partialAudits,1);
  const added=Float64Array.from([...source,400,0,0,401,0,0,400,1,0]),allIds=Uint32Array.from({length:93},(_,i)=>i);
  const extra=Uint32Array.of(90,91,92),next={vertexIds:allIds.slice(),vertexCapacity:93,
    rows:[...layout.rows,{buffer:extra.buffer,offset:0,begin:30,end:31}]};
  check(owner,next,added,added,allIds);assert.equal(owner.statistics.partialAudits,1);
});
test('reordered provenance, duplicate stable IDs, large changes and budget exhaustion use full audits',()=>{
  for(const mode of ['reorder','duplicate','large','budget']){
    const {source,indices,layout,owner}=fixture();let ids=indices,metadata=layout,candidate=source.slice();
    if(mode==='reorder'){
      const order=[29,...Array.from({length:29},(_,i)=>i)];
      ids=Uint32Array.from(order.flatMap(f=>Array.from(indices.subarray(f*3,f*3+3))));
      metadata={...layout,rows:order.map((offset,begin)=>({buffer:layout.rows[0].buffer,offset,begin,end:begin+1}))};
    }else if(mode==='duplicate'){
      metadata={...layout,vertexIds:layout.vertexIds.slice()};metadata.vertexIds[1]=metadata.vertexIds[0];
    }else if(mode==='large')for(let i=0;i<candidate.length;i++)candidate[i]+=.1;
    check(owner,metadata,source,candidate,ids,mode==='budget'?1:budget);
    assert.equal(owner.statistics.fullAudits,1);
    if(mode==='budget')assert.equal(owner.retainedBytes,0);
  }
});
test('a nonempty audit cannot publish a replacement certificate',()=>{
  const {source,indices,layout,owner}=fixture(),before=owner.retainedBytes,candidate=source.slice();
  candidate.set(source.subarray(90,99),0);
  assert(check(owner,layout,source,candidate,indices).size>0);
  assert.equal(owner.retainedBytes,before);
  check(owner,layout,source,source,indices);assert.equal(owner.statistics.dirtyFaces,0);
});
test('certificate ownership survives caller mutation and transferred input buffers',()=>{
  const {source,physical,indices,layout,owner}=fixture(),nextIds=indices.slice();
  physical.set(source.subarray(72,81),108);
  assert.equal(check(owner,layout,source,physical,indices).size,6);
  structuredClone({physical,indices},{transfer:[physical.buffer,indices.buffer]});
  assert.equal(physical.byteLength,0);assert.equal(indices.byteLength,0);
  const next=source.slice();next.set(source.subarray(27,36),225);
  assert.equal(check(owner,layout,source,next,nextIds).size,6);
  assert.equal(owner.statistics.partialAudits,1);
});
