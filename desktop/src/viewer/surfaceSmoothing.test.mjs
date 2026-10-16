import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {maskSurface} from './surface.ts';
import {smoothSurface} from './surfaceSmoothing.ts';
import {surfaceNormals} from './surfaceNormals.ts';
import {inverseAffine} from './coordinates.ts';

const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const digest = data => createHash('sha256').update(new Uint8Array(data.buffer)).digest('hex');
function fixture(shape, inside) {
  const mask = new Uint8Array(shape.reduce((a,b)=>a*b));
  for(let x=0;x<shape[0];x++) for(let y=0;y<shape[1];y++) for(let z=0;z<shape[2];z++)
    mask[(x*shape[1]+y)*shape[2]+z] = inside(x,y,z) ? 1 : 0;
  return mask;
}
function topology(positions) {
  const ids = new Map(), triangles=[], edges=new Map(), parents=[];
  for(let i=0;i<positions.length;i+=3){
    const key=positions.subarray(i,i+3).join(',');
    if(!ids.has(key)){ids.set(key,ids.size);parents.push(parents.length);}
    triangles.push(ids.get(key));
  }
  const find = n => parents[n]===n?n:(parents[n]=find(parents[n]));
  for(let i=0;i<triangles.length;i+=3) for(let a=0;a<3;a++){
    const v=triangles[i+a],w=triangles[i+(a+1)%3],key=[v,w].sort((a,b)=>a-b).join(',');
    edges.set(key,(edges.get(key)??0)+1); parents[find(v)]=find(w);
  }
  assert.ok([...edges.values()].every(n=>n===2),'Every source edge remains closed');
  return {triangles,vertices:ids.size,edges:edges.size,components:new Set(parents.map((_,i)=>find(i))).size};
}
function faceNormal(p,i) {
  const ab=[0,1,2].map(a=>p[i+3+a]-p[i+a]),ac=[0,1,2].map(a=>p[i+6+a]-p[i+a]);
  return [ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];
}
test('brain/tumor-like surface is measurably smoother with immutable source data',()=>{
  const shape=[25,25,25],mask=fixture(shape,(x,y,z)=>(x-12)**2+(y-12)**2+(z-12)**2<100);
  const source=maskSurface(mask,shape),before=[digest(mask),digest(source)],smooth=smoothSurface(source,identity);
  assert.deepEqual([digest(mask),digest(source)],before);
  assert.deepEqual(topology(smooth),topology(source));
  // Analytic sphere: normals should agree more closely with the radial normal.
  const error=p=>{
    const normals=surfaceNormals(p);let sum=0;
    for(let i=0;i<p.length;i+=3){const radial=[p[i]-12,p[i+1]-12,p[i+2]-12],length=Math.hypot(...radial);
      sum+=1-radial.reduce((s,v,a)=>s+v*normals[i+a],0)/length;}
    return sum/(p.length/3);
  };
  assert.ok(error(smooth)<error(source)*.65,'Reduces stair-step normal error by at least 35%');
});
test('thin branching vessels, nearby components and skull-like cavity retain their triangle connectivity',()=>{
  for(const [shape,inside] of [
    [[15,15,15],(x,y,z)=>(x===7&&y===7)||(y===7&&z===7)||(x===7&&z===7)||(x===10&&y===10&&z>3&&z<12)],
    [[17,17,17],(x,y,z)=>{const r=(x-8)**2+(y-8)**2+(z-8)**2;return r>=25&&r<49;}],
  ]){
    const source=maskSurface(fixture(shape,inside),shape),smooth=smoothSurface(source,identity);
    assert.deepEqual(topology(smooth),topology(source));
    for(let i=0;i<source.length;i+=9){
      const a=faceNormal(source,i),b=faceNormal(smooth,i);
      assert.ok(a.reduce((sum,v,a)=>sum+v*b[a],0)>0,'No reversed triangle');
      assert.ok(Math.hypot(...b)>=.1*Math.hypot(...a),'No collapsed triangle');
    }
  }
});
test('displacement is physically bounded on anisotropic, oblique, mirrored and sheared grids',()=>{
  const shape=[12,12,12],source=maskSurface(fixture(shape,(x,y,z)=>(x-5)**2+(y-5)**2+(z-5)**2<16),shape);
  for(const affine of [identity,[[-.8,-.6,0,100],[-.6,.8,0,-80],[0,0,4,50],[0,0,0,1]],[[.7,.4,.2,9000],[0,2,.3,-3000],[0,0,3,-500],[0,0,0,1]]]){
    const smooth=smoothSurface(source,affine),inverse=inverseAffine(affine);
    const limit=.5/Math.max(...inverse.slice(0,3).map(row=>Math.hypot(...row.slice(0,3))));
    for(let i=0;i<source.length;i+=3){
      const delta=[0,1,2].map(a=>smooth[i+a]-source[i+a]);
      const world=affine.slice(0,3).map(row=>delta.reduce((s,v,a)=>s+v*row[a],0));
      assert.ok(Math.hypot(...world)<=limit+1e-5,'Maximum half-voxel physical displacement');
    }
    assert.ok(surfaceNormals(smooth).every(Number.isFinite));
    const translated=affine.map((row,i)=>row.map((v,j)=>j===3&&i<3?v+12345:v));
    assert.deepEqual(smoothSurface(source,translated),smooth,'Translation never affects smoothing');
  }
});
test('open boundaries remain fixed and malformed display geometry is rejected',()=>{
  const source=new Float32Array([0,0,0,1,0,0,0,1,0]);
  assert.deepEqual(smoothSurface(source,identity),source);
  assert.equal(smoothSurface(new Float32Array(),identity).length,0);
  assert.throws(()=>smoothSurface(new Float32Array(8),identity),/triangles/);
  assert.throws(()=>smoothSurface(new Float32Array([NaN,0,0,1,0,0,0,1,0]),identity),/finite/);
  assert.throws(()=>smoothSurface(source,[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]),/singular/);
});
