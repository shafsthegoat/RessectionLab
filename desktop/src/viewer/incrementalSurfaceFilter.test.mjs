import test from 'node:test';
import assert from 'node:assert/strict';
import {IncrementalSurfaceFilter} from './incrementalSurfaceFilter.ts';
import {sincCoefficients} from './surfacePolicy.ts';

// Retained full recurrence: independent of halo selection, local IDs and cache.
function reference(original,offsets,neighbors,pinned,c){
  const result=Float64Array.from(original,v=>c[0]*v);
  let previous=original.slice(),current=new Float64Array(original.length),next=new Float64Array(original.length);
  for(let j=1;j<c.length;j++){
    for(let v=0;v<pinned.length;v++)for(let a=0;a<3;a++){
      const i=v*3+a,degree=offsets[v+1]-offsets[v];let sum=0;
      for(let e=offsets[v];e<offsets[v+1];e++)sum+=previous[neighbors[e]*3+a];
      next[i]=pinned[v]||!degree?original[i]:j===1?sum/degree:2*sum/degree-current[i];result[i]+=c[j]*next[i];
    }[current,previous,next]=[previous,next,current];
  }return result;
}
function chain(n=301,cut=-1){
  const offsets=new Uint32Array(n+1),neighbors=[];
  for(let v=0;v<n;v++){
    // Duplicate entries are intentional: preserve their exact addition order.
    if(v&&v!==cut)neighbors.push(v-1,v-1);
    if(v+1<n&&v+1!==cut)neighbors.push(v+1,v+1);
    offsets[v+1]=neighbors.length;
  }
  const pinned=new Uint8Array(n);pinned[0]=pinned[n-1]=1;
  return [offsets,new Uint32Array(neighbors),pinned];
}
function fixture(c=sincCoefficients('context')){
  const n=301,source=Float64Array.from({length:n*3},(_,i)=>i%11===0?-0:Math.sin(i*.071)*2),ids=Uint32Array.from({length:n},(_,i)=>i);
  const layout={vertexIds:ids,vertexCapacity:n,registry:{},generation:0,generationValid:true,dirtyVertices:new Uint32Array(),rows:[]};
  const owner=new IncrementalSurfaceFilter(),args=chain(n),frame=owner.begin(layout,1e7),raw=reference(source,...args,c);
  assert.equal(frame.filter.compute(source,...args,c),undefined);frame.filter.capture(source,raw);frame.filter.complete();
  return {owner,layout,source,args,c};
}
const exact=(a,b)=>assert.deepEqual(new Uint8Array(a.buffer,a.byteOffset,a.byteLength),new Uint8Array(b.buffer,b.byteOffset,b.byteLength));
test('K output/2K dependency halo matches full bits with cuts, coordinates, pins and duplicate sums',()=>{
  for(const kind of ['context','tumor']){
    const {owner,layout,source,c}=fixture(sincCoefficients(kind)),next=source.slice(),args=chain(301,150);
    next[151*3+1]+=.013;args[2][149]=1;
    const frame=owner.begin({...layout,generation:1,dirtyVertices:Uint32Array.of(149,150)},1e7);
    const result=frame.filter.compute(next,...args,c);
    assert(result);exact(result,reference(next,...args,c));assert(owner.statistics.local);
    assert(owner.statistics.affectedVertices<owner.statistics.haloVertices);
    assert(owner.statistics.haloVertices<source.length/3);
  }
});
test('failed/skipped generations, invalid updates, changed coefficients and registries take full path',()=>{
  for(const mode of ['gap','invalid','coefficients','registry']){
    const {owner,layout,source,args,c}=fixture(),changed={...layout,generation:mode==='gap'?2:1};
    if(mode==='invalid')changed.generationValid=false;
    if(mode==='registry')changed.registry={};
    const frame=owner.begin(changed,1e7),coefficients=mode==='coefficients'?sincCoefficients('tumor'):c;
    assert.equal(frame.filter?.compute(source,...args,coefficients),undefined);
    assert.equal(owner.statistics.local,false);
  }
});
test('raw state stages before cap mutation/transfer and commits only on successful completion',()=>{
  const {owner,layout,source,args,c}=fixture(),changed=source.slice();changed[450]+=.017;
  const next={...layout,generation:1,dirtyVertices:Uint32Array.of(150)},first=owner.begin(next,1e7);
  const raw=first.filter.compute(changed,...args,c);first.filter.capture(changed,raw);
  raw.fill(1e6);structuredClone(raw,{transfer:[raw.buffer]});
  // Retry retains the exact transition seeds, with the old committed cache.
  const retry=owner.begin(next,1e7),replayed=retry.filter.compute(changed,...args,c);
  exact(replayed,reference(changed,...args,c));retry.filter.capture(changed,replayed);retry.filter.complete();
  const gap=owner.begin({...layout,generation:3},1e7);
  assert.equal(gap.filter.compute(changed,...args,c),undefined);
});
test('identity/budget guards include duplicate IDs and simultaneous old/staged fields',()=>{
  const {owner,layout}=fixture(),bytes=owner.retainedBytes;
  const limited=owner.begin({...layout,generation:1},bytes);assert.equal(limited.filter,undefined);assert.equal(owner.retainedBytes,0);
  const other=fixture(),ids=other.layout.vertexIds.slice();ids[1]=ids[0];
  assert.equal(other.owner.begin({...other.layout,vertexIds:ids},1e7).filter,undefined);
});
test('coefficient identity preserves signed zero',()=>{
  const {owner,layout,source,args}=fixture(Float64Array.of(-0,-0));
  const frame=owner.begin({...layout,generation:1},1e7);
  assert.equal(frame.filter.compute(source,...args,Float64Array.of(0,-0)),undefined);
  assert.equal(owner.statistics.local,false);
});
