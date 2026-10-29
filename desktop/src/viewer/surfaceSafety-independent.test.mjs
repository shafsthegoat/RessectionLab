import test from 'node:test';
import assert from 'node:assert/strict';
import {intersectionVertices,trianglesIntersect} from './surfaceSafety.ts';

// Exhaustive nonadjacent pairs independently check BVH candidate rejection.
// The SAT primitive is separately exercised by surfaceQuality.test.mjs.
function exhaustive(source,candidate,ids){
  const bad=new Set();
  for(let a=0;a<ids.length;a+=3)for(let b=a+3;b<ids.length;b+=3){
    const left=Array.from(ids.slice(a,a+3)),right=Array.from(ids.slice(b,b+3));
    if(left.some(id=>right.includes(id)))continue;
    if(trianglesIntersect(candidate,candidate,ids,a,b)&&!trianglesIntersect(source,source,ids,a,b))
      for(const id of [...left,...right])bad.add(id);
  }
  return [...bad].sort((a,b)=>a-b);
}
test('BVH agrees with exhaustive pairs across contacts, shared vertices and near boxes',()=>{
  let seed=8191;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  for(const scale of [1e-8,1,1e5])for(const shared of [false,true]){
    const source=Float64Array.from({length:90},()=>scale*(random()*12-6));
    const candidate=Float64Array.from(source,(v,i)=>i%3===2?v*.03:v*.4);
    const ids=Uint32Array.from({length:90},(_,i)=>shared?i%30:Math.floor(i/3)%10*3+i%3);
    for(const [s,c] of [[source,candidate],[source,source],[candidate,source]])
      assert.deepEqual([...intersectionVertices(s,c,ids)].sort((a,b)=>a-b),exhaustive(s,c,ids));
  }
  assert.deepEqual([...intersectionVertices(new Float64Array(),new Float64Array(),new Uint32Array())],[]);
});
