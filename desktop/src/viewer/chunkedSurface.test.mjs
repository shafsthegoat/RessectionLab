import assert from 'node:assert/strict';
import {test} from 'node:test';
import {ChunkedMaskSurface} from './chunkedSurface.ts';
import {maskSurface} from './surface.ts';
import {prepareSurface} from './prepareSurface.ts';
import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import {indexSurface} from './surfaceSmoothing.ts';
const shape=[9,8,7],affine=[[1,.2,0,10],[0,2,.1,20],[0,0,-1.5,30],[0,0,0,1]],at=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
const source=()=>Uint8Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>{const x=Math.floor(i/(shape[1]*shape[2])),y=Math.floor(i/shape[2])%shape[1],z=i%shape[2];return Number((x>1&&x<7&&y>0&&y<6&&z>1&&z<6)||(x===8&&y===7&&z===6));});
test('chunk cores stitch the exact full reference triangle order across boundaries, new surfaces and deletion',()=>{
  const mask=source(),chunks=new ChunkedMaskSurface(mask,shape,4);assert.deepEqual(chunks.positions(),maskSurface(mask,shape));
  for(const indices of [[at(3,3,3),at(4,4,4)],[at(0,0,0),at(8,7,6)],[at(7,7,6),at(8,7,6)]]){
    for(const i of indices)mask[i]=1-mask[i];chunks.update(indices);assert.deepEqual(chunks.positions(),maskSurface(mask,shape));assert(chunks.lastRemeshedChunks<=16);
  }
  const removed=[];for(let i=0;i<mask.length;i++)if(mask[i]){mask[i]=0;removed.push(i);}chunks.update(removed);assert.equal(chunks.positions().length,0);
  mask[at(4,4,4)]=1;chunks.update([at(4,4,4)]);assert.deepEqual(chunks.positions(),maskSurface(mask,shape));
});
test('source chunks preserve exact global indexing, full fairing and physical normals under anisotropic sheared mirrored affine',()=>{
  const mask=source(),engine=new IncrementalSurfaceEngine(),base={layerId:'vessel',shape,affine,kind:'vessel'};
  let result=engine.prepare({...base,jobId:1,previousIdentity:'',identity:'a',mask:mask.slice()});
  const equal=()=>{const reference=prepareSurface(mask,shape,affine,'vessel',true);for(const key of ['positions','indices','normals','smoothPositions','smoothNormals'])assert.deepEqual(result[key],reference[key],key);};equal();
  const indices=Uint32Array.from([at(3,3,3),at(4,4,4)]),values=Uint8Array.from([0,0]);indices.forEach((id,i)=>mask[id]=values[i]);
  result=engine.prepare({...base,jobId:2,previousIdentity:'a',identity:'b',indices,values});equal();assert.equal(result.work.incremental,true);assert(result.work.remeshedChunks<=8);
  assert.equal(result.identity,'b');assert.equal(result.resident[0].identity,'b');assert(mask.every(n=>n===0||n===1));
});
test('source/geometry identity mismatch requires full admission; bounded caches and separate engines never borrow masks',()=>{
  const mask=source(),base={jobId:1,layerId:'brain',shape,affine,kind:'brain',previousIdentity:'',identity:'a'};
  const tiny=new IncrementalSurfaceEngine(1),result=tiny.prepare({...base,mask:mask.slice()});assert.deepEqual(result.resident,[]);assert.equal(result.work.retainedBytes,0);
  assert.throws(()=>tiny.prepare({...base,jobId:2,previousIdentity:'a',identity:'b',indices:new Uint32Array(),values:new Uint8Array()}),/admission/);
  const engine=new IncrementalSurfaceEngine();engine.prepare({...base,mask:mask.slice()});
  assert.throws(()=>engine.prepare({...base,previousIdentity:'wrong',identity:'b',indices:new Uint32Array(),values:new Uint8Array()}),/admission/);
  assert.throws(()=>new IncrementalSurfaceEngine().prepare({...base,previousIdentity:'a',identity:'b',indices:new Uint32Array(),values:new Uint8Array()}),/admission/);
});
test('lazy zero-effect admission matches all reference buffers and starts resident chunks only on a real cut',()=>{
  const mask=source(),engine=new IncrementalSurfaceEngine(),base={layerId:'brain',shape,affine,kind:'brain'};
  let result=engine.prepare({...base,jobId:1,previousIdentity:'',identity:'mapped',mask:mask.slice(),retainChunks:false});
  const equal=()=>{const reference=prepareSurface(mask,shape,affine,'brain',true);for(const key of ['positions','indices','normals','smoothPositions','smoothNormals'])assert.deepEqual(result[key],reference[key],key);};
  equal();assert.equal(result.work.fullReference,true);assert.equal(result.work.retainedBytes,0);assert.equal(result.work.remeshedChunks,0);assert.deepEqual(result.resident,[]);
  assert.throws(()=>engine.prepare({...base,jobId:2,previousIdentity:'mapped',identity:'cut',indices:Uint32Array.of(at(3,3,3)),values:Uint8Array.of(0)}),/admission/);
  mask[at(3,3,3)]=0;
  result=engine.prepare({...base,jobId:3,previousIdentity:'mapped',identity:'cut',mask:mask.slice()});
  equal();assert.equal(result.work.incremental,false);assert(result.work.retainedBytes>0);assert.deepEqual(result.resident,[{layerId:'brain',identity:'cut'}]);
  mask[at(4,4,4)]=0;
  result=engine.prepare({...base,jobId:4,previousIdentity:'cut',identity:'delta',indices:Uint32Array.of(at(4,4,4)),values:Uint8Array.of(0)});
  equal();assert.equal(result.work.incremental,true);assert(result.work.remeshedChunks<=8);assert.deepEqual(result.resident,[{layerId:'brain',identity:'delta'}]);
});
test('half-grid numeric indexing is byte-identical to generic exact keys, with arbitrary geometry fallback',()=>{
  const soup=maskSurface(source(),shape);assert.deepEqual(indexSurface(soup,shape),indexSurface(soup));
  const arbitrary=new Float32Array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.1,.2,.3,.7,.8,.9,1.1,1.2,1.3]);
  assert.deepEqual(indexSurface(arbitrary,shape),indexSurface(arbitrary));
  assert.deepEqual(indexSurface(soup,[2**40,2**40,2**40]),indexSurface(soup),'unsafe numeric spans retain exact generic indexing');
});
