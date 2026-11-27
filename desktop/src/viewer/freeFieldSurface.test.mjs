import assert from 'node:assert/strict';
import test from 'node:test';
import {joinedFreeField} from './freeFieldSurface.ts';

test('adjacent deposits form a closed continuous skin confined to wet cells',()=>{
  const samples=[{cell:[1,1,1],amount:1},{cell:[2,1,1],amount:.4},{cell:[2,2,1],amount:.012}];
  const result=joinedFreeField(samples,[5,5,5],1);
  assert.equal(result.joined.size,3);
  const position=result.geometry.getAttribute('position'), normals=result.geometry.getAttribute('normal');
  const indices=result.geometry.index.array, edges=new Map();
  for(let i=0;i<indices.length;i+=3){
    for(let k=0;k<3;k++){
      const a=indices[i+k], b=indices[i+(k+1)%3], key=[Math.min(a,b),Math.max(a,b)].join(',');
      edges.set(key,(edges.get(key)??0)+1);
    }
  }
  assert(edges.size>36);assert([...edges.values()].every(n=>n===2));
  for(let i=0;i<position.count;i++){
    const p=[position.getX(i),position.getY(i),position.getZ(i)];
    assert(samples.some(s=>s.cell.every((v,j)=>Math.abs(v-p[j])<.50001)));
    assert(Math.hypot(normals.getX(i),normals.getY(i),normals.getZ(i))>.99);
  }
  const smaller=joinedFreeField(samples.map(s=>({...s,amount:s.amount/4})),[5,5,5],1);
  result.geometry.computeBoundingBox();smaller.geometry.computeBoundingBox();
  assert(smaller.geometry.boundingBox.max.z<result.geometry.boundingBox.max.z);
});

test('diagonal, row-wrap and remote cells have no shared fluid surface',()=>{
  for(const cell of [[2,2,4],[1,2,0],[4,4,4]]){
    assert.equal(joinedFreeField([{cell:[1,1,4],amount:1},{cell,amount:1}],[5,5,5],1),null);
  }
});
