import assert from 'node:assert/strict';
import test from 'node:test';
import {indexedSurfaceNormals} from './surfaceNormals.ts';

// Retained independent array/reduction implementation, including its original
// arithmetic order. Optimizing normals must not change displayed source shape.
function reference(points,indices,affine){
  const sums=new Float64Array(points.length),linear=affine.slice(0,3).map(r=>r.slice(0,3));
  const determinant=linear[0][0]*(linear[1][1]*linear[2][2]-linear[1][2]*linear[2][1])-linear[0][1]*(linear[1][0]*linear[2][2]-linear[1][2]*linear[2][0])+linear[0][2]*(linear[1][0]*linear[2][1]-linear[1][1]*linear[2][0]);
  for(let f=0;f<indices.length;f+=3){const ids=[indices[f]*3,indices[f+1]*3,indices[f+2]*3];
    const delta=id=>linear.map(r=>r.reduce((s,v,k)=>s+v*(points[id+k]-points[ids[0]+k]),0));
    const a=delta(ids[1]),b=delta(ids[2]),normal=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    for(const id of ids)for(let axis=0;axis<3;axis++)sums[id+axis]+=normal[axis]*Math.sign(determinant);
  }
  const output=new Float32Array(points.length);
  for(let i=0;i<points.length;i+=3){const covector=[0,1,2].map(a=>linear.reduce((s,r,k)=>s+r[a]*sums[i+k],0)),length=Math.hypot(...covector);
    for(let a=0;a<3;a++)output[i+a]=length?covector[a]/length:0;}
  return output;
}

test('specialized physical normals are byte-identical across mirrored, sheared, anisotropic and degenerate surfaces',()=>{
  let seed=126391;const random=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)/2**32);
  for(const linear of [[[1,0,0],[0,1,0],[0,0,1]],[[-1,0,0],[0,2.7,0],[0,0,.31]],[[1.2,-.27,.03],[.12,.9,-.21],[.03,-.09,-2.1]],[[0,0,0],[0,0,0],[0,0,0]]]){
    const affine=[...linear.map(row=>[...row,27]),[0,0,0,1]],points=Float32Array.from({length:501*3},()=>random()*400-200),indices=Uint32Array.from({length:9000},()=>Math.floor(random()*501));
    const original=points.slice(),faces=indices.slice(),expected=reference(points,indices,affine),actual=indexedSurfaceNormals(points,indices,affine);
    assert.deepEqual(new Uint8Array(actual.buffer),new Uint8Array(expected.buffer));
    assert.deepEqual(points,original);assert.deepEqual(indices,faces);
  }
  assert.deepEqual(indexedSurfaceNormals(new Float32Array(),new Uint32Array(),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),new Float32Array());
});
