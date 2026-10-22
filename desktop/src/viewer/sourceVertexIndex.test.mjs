import assert from 'node:assert/strict';
import {test} from 'node:test';
import {SourceVertexIndex,SurfaceIndexBudgetExceeded} from './sourceVertexIndex.ts';

test('typed source registry preserves exact stable IDs through collisions, growth and unsigned lattice keys',()=>{
  const budgets=[],registry=new SourceVertexIndex([1000,1000,500],bytes=>budgets.push(bytes)),points=[];
  for(let i=0;i<4000;i++){
    const point=i===0?[-.5,-.5,-.5]:i===3999?[999.5,999.5,499.5]:[Math.floor(i/1001)/2-.5,20.5,(i%1001)/2-.5];
    points.push(point);assert.equal(registry.intern(...point),i);
  }
  const output=new Float32Array(3);
  for(let i=points.length-1;i>=0;i--){assert.equal(registry.intern(...points[i]),i);registry.copy(i,output,0);assert.deepEqual([...output],points[i]);}
  assert.equal(registry.vertexCount,points.length);assert(budgets.length>4);assert.equal(budgets.at(-1),registry.retainedBytes);
  for(const point of [[NaN,0,0],[.1,0,0],[-1,0,0],[1000,0,0]])assert.throws(()=>registry.intern(...point),/half-grid/);
  assert.equal(registry.vertexCount,points.length);
});
test('registry refuses unsafe lattice keys and capacity growth before retaining storage outside its budget',()=>{
  assert.throws(()=>new SourceVertexIndex([2**20,2**20,2],()=>{}),SurfaceIndexBudgetExceeded);
  const registry=new SourceVertexIndex([100,100,100],bytes=>{if(bytes>40000)throw new SurfaceIndexBudgetExceeded('test budget');});
  let added=0;
  assert.throws(()=>{for(let x=0;x<100;x++)for(let y=0;y<100;y++){registry.intern(x,y,0);added++;}},SurfaceIndexBudgetExceeded);
  assert(added>1000);assert(registry.retainedBytes<=40000);
  const output=new Float32Array(3);registry.copy(0,output,0);assert.deepEqual([...output],[0,0,0]);assert.equal(registry.intern(0,0,0),0);
});

test('rounded Float32 half-grid axes fall back before registry admission',()=>{
  const registry=new SourceVertexIndex([2**23,1,1],()=>{}),output=new Float32Array(3);
  const id=registry.intern(2**23-.5,0,0);registry.copy(id,output,0);assert.equal(output[0],2**23-.5);
  assert.throws(()=>new SourceVertexIndex([2**23+1,1,1],()=>{}),SurfaceIndexBudgetExceeded);
  assert.throws(()=>new SourceVertexIndex([1,2**23+2,1],()=>{}),SurfaceIndexBudgetExceeded);
});
