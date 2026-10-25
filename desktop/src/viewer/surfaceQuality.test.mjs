import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {maskSurface} from './surface.ts';
import {prepareSurface} from './prepareSurface.ts';
import {fairIndexedSurface,indexSurface} from './surfaceSmoothing.ts';
import {intersectionVertices,trianglesIntersect} from './surfaceSafety.ts';
import {SURFACE_POLICY,sincCoefficients} from './surfacePolicy.ts';
import {cachedSurface,retainSurface} from './surfaceCache.ts';
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const hash=a=>createHash('sha256').update(new Uint8Array(a.buffer)).digest('hex');
function mask(shape,inside){return Uint8Array.from({length:shape.reduce((a,b)=>a*b)},(_,i)=>inside(Math.floor(i/(shape[1]*shape[2])),Math.floor(i/shape[2])%shape[1],i%shape[2])?1:0);}
test('tumor profile removes acquisition steps more strongly while retaining source and indexed topology',()=>{
  const shape=[25,25,25],m=mask(shape,(x,y,z)=>(x-12)**2+(y-12)**2+(z-12)**2<90),before=hash(m);
  const generic=prepareSurface(m,shape,identity,'brain'),tumor=prepareSurface(m,shape,identity,'tumor');
  const error=p=>{let sum=0;for(let i=0;i<p.smoothPositions.length;i+=3){const radial=[0,1,2].map(a=>p.smoothPositions[i+a]-12),length=Math.hypot(...radial);sum+=1-radial.reduce((s,v,a)=>s+v*p.smoothNormals[i+a],0)/length;}return sum/(p.positions.length/3);};
  assert.ok(error(tumor)<error(generic)*.9);assert.ok(error(tumor)<.003);assert.equal(hash(m),before);assert.deepEqual(tumor.indices,generic.indices);
  assert.ok(tumor.diagnostics.maxDisplacementMm<=1.00001);assert.equal(tumor.diagnostics.policy,SURFACE_POLICY.version);
});

test('complex context uses an explicit source fallback while tissue profiles remain eligible for smoothing',()=>{
  const budget=SURFACE_POLICY.maxContextTriangles;
  try{SURFACE_POLICY.maxContextTriangles=0;const p=Float32Array.from([0,0,0,1,0,0,0,1,0]),ids=Uint32Array.from([0,1,2]);
    const context=fairIndexedSurface(p,ids,identity,'skull');assert.deepEqual(context.positions,p);assert.match(context.diagnostics.fallbackReason,/context/);
    assert.equal(fairIndexedSurface(p,ids,identity,'tumor').diagnostics.fallbackReason,undefined);
  }finally{SURFACE_POLICY.maxContextTriangles=budget;}
});
test('thin and nearby vessels retain every source triangle and physical radius cap across grids',()=>{
  const shape=[15,15,15],m=mask(shape,(x,y,z)=>(x===5&&y===7)||(x===7&&y===7)||(z===7&&x===5));
  for(const affine of [identity,[[-.5,0,0,70],[0,1,0,-20],[0,0,3,90],[0,0,0,1]],[[1,.2,0,0],[0,2,.3,0],[0,0,3,0],[0,0,0,1]]]){
    const s=prepareSurface(m,shape,affine,'vessels');assert.equal(s.indices.length,maskSurface(m,shape).length/3);
    assert.ok(s.diagnostics.maxDisplacementMm<=.25001);assert.ok(s.smoothNormals.every(Number.isFinite));
  }
});
test('SAT detects crossing/coplanar triangles, excludes disjoint triangles and preserves existing source contacts',()=>{
  const ids=Uint32Array.from([0,1,2,3,4,5]),p=Float64Array.from([0,0,0,2,0,0,0,2,0, .5,.5,-1,.5,.5,1,1.5,.5,0]);
  assert.ok(trianglesIntersect(p,p,ids,0,3));const moved=p.slice();moved[9]+=10;moved[12]+=10;moved[15]+=10;
  assert.ok(!trianglesIntersect(moved,moved,ids,0,3));assert.equal(intersectionVertices(p,p,ids).size,0);
  assert.equal(intersectionVertices(moved,p,ids).size,6);
  const coplanar=Float64Array.from([0,0,0,2,0,0,0,2,0, .2,.2,0,1,.2,0,.2,1,0]);assert.ok(trianglesIntersect(coplanar,coplanar,ids,0,3));
});
test('source cache is tissue/grid specific and polynomial preserves DC exactly',()=>{
  const m=new Uint8Array(8),prepared=prepareSurface(m,[2,2,2],identity,'tumor');retainSurface(m,identity,'tumor',prepared);
  assert.equal(cachedSurface(m,identity,'tumor'),prepared);assert.equal(cachedSurface(m,identity,'vessels'),undefined);
  for(const kind of ['context','tumor','vessel'])assert.ok(Math.abs(sincCoefficients(kind).reduce((a,b)=>a+b)-1)<1e-12);
  const p=Float32Array.from([0,0,0,1,0,0,0,1,0]),s=fairIndexedSurface(p,Uint32Array.from([0,1,2]),identity,'tumor');assert.deepEqual(s.positions,p);
  assert.deepEqual(indexSurface(p).indices,Uint32Array.from([0,1,2]));
});
test('raw stable-ID snapshots retain active bounds and cannot satisfy smoothed cache requests',()=>{
  const m=new Uint8Array(8),surface={positions:Float32Array.of(100,100,100,0,0,0,1,0,0,0,1,0),normals:new Float32Array(12),indices:Uint32Array.of(1,2,3),smoothingPrepared:false,representation:'stable-source-ids',activeBounds:[[0,0,0],[1,1,0]]};
  retainSurface(m,identity,'bone',surface,[2,2,2]);const cached=cachedSurface(m,identity,'bone',[2,2,2],false);
  assert.equal(cached,surface);assert.equal(cached.activeBounds,surface.activeBounds);assert.equal(cached.representation,'stable-source-ids');assert.equal(cachedSurface(m,identity,'bone',[2,2,2],true),undefined);
});
