import assert from 'node:assert/strict';
import {test} from 'node:test';
import {ChunkedMaskSurface} from './chunkedSurface.ts';
import {maskSurface} from './surface.ts';
import {prepareSurface} from './prepareSurface.ts';
import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import {indexSurface,indexSurfaceLookupStatistics} from './surfaceSmoothing.ts';
const shape=[9,8,7],affine=[[1,.2,0,10],[0,2,.1,20],[0,0,-1.5,30],[0,0,0,1]],at=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
const source=()=>Uint8Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>{const x=Math.floor(i/(shape[1]*shape[2])),y=Math.floor(i/shape[2])%shape[1],z=i%shape[2];return Number((x>1&&x<7&&y>0&&y<6&&z>1&&z<6)||(x===8&&y===7&&z===6));});
test('bounded lattice indexing preserves first appearance and exact arbitrary-order fallback',()=>{
  const soup=new Float32Array(100000*9);
  for(let i=0;i<soup.length;i+=3){const v=Math.floor(i/3)%70000;soup[i]=(v%90)/2;soup[i+1]=(Math.floor(v/90)%90)/2;soup[i+2]=Math.floor(v/8100)/2;}
  for(const i of [0,6,300,303,306])soup[i]=.1234567;
  const expected=indexSurface(soup),actual=indexSurface(soup,[100,100,100]);
  assert.deepEqual(new Uint8Array(actual.positions.buffer),new Uint8Array(expected.positions.buffer));
  assert.deepEqual(actual.indices,expected.indices);
  assert.throws(()=>indexSurface(Float32Array.of(NaN,0,0,0,1,0,0,0,1),[2,2,2]),/finite/);
});
test('rolling source lookup retains exact corner IDs across plane reuse and separate requests',()=>{
  const soup=new Float32Array(100000*9);
  for(let f=0;f<100000;f++){
    const x=Math.floor(f/10000),y=f%97,z=Math.floor(f/97)%97;
    soup.set([x,y,z,x+.5,y+.5,z,x+1,y,z+.5],f*9);
  }
  const expected=indexSurface(soup),first=indexSurface(soup,[100,100,100]);
  assert.deepEqual(first,expected);
  const reversed=new Float32Array(soup.length);for(let i=0;i<soup.length;i+=9)reversed.set(soup.subarray(soup.length-i-9,soup.length-i),i);
  assert.deepEqual(indexSurface(reversed,[100,100,100]),indexSurface(reversed));
  assert.deepEqual(indexSurface(soup,[100,100,100]),expected,'lookup contents cannot cross requests');
  const stats=indexSurfaceLookupStatistics();assert(stats.retainedBytes>0);assert(stats.retainedBytes<=stats.maximumBytes);
});
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
test('raw preparation retains exact source buffers and omits conditioning, including initial mapping',()=>{
  const mask=source(),before=mask.slice(),reference=prepareSurface(mask,shape,affine,'vessel',true);
  for(const retainChunks of [true,false]){
    const result=new IncrementalSurfaceEngine().prepare({jobId:1,layerId:'vessel',previousIdentity:'',identity:'raw',shape,affine,kind:'vessel',mask:mask.slice(),smooth:false,retainChunks});
    for(const key of ['positions','indices','normals'])assert.deepEqual(result[key],reference[key],key);
    assert.equal(result.smoothingPrepared,false);assert.equal(result.work.smoothingPrepared,false);
    assert.equal(result.smoothPositions,undefined);assert.equal(result.smoothNormals,undefined);assert.equal(result.diagnostics,undefined);
  }
  assert.deepEqual(mask,before);
});
test('resident cell identity survives raw/full preference changes and later incremental cuts',()=>{
  const mask=source(),engine=new IncrementalSurfaceEngine(),base={layerId:'vessel',shape,affine,kind:'vessel'};
  engine.prepare({...base,jobId:1,previousIdentity:'',identity:'a',mask:mask.slice(),smooth:false});
  let result=engine.prepare({...base,jobId:2,previousIdentity:'a',identity:'a',indices:new Uint32Array(),values:new Uint8Array(),smooth:true});
  const full=prepareSurface(mask,shape,affine,'vessel',true);
  for(const key of ['positions','indices','normals','smoothPositions','smoothNormals'])assert.deepEqual(result[key],full[key],key);
  assert.equal(result.work.incremental,true);assert.equal(result.work.remeshedChunks,0);
  const index=at(3,3,3);mask[index]=0;
  result=engine.prepare({...base,jobId:3,previousIdentity:'a',identity:'b',indices:Uint32Array.of(index),values:Uint8Array.of(0),smooth:false});
  const cut=prepareSurface(mask,shape,affine,'vessel',true);
  for(const key of ['positions','indices','normals'])assert.deepEqual(result[key],cut[key],key);
  assert.equal(result.work.incremental,true);assert.equal(result.smoothPositions,undefined);
});
test('indexed chunk rows preserve canonical bytes through seams, removed components, undo and new isolated surfaces',()=>{
  const mask=source(),chunks=new ChunkedMaskSurface(mask,shape,4,2**20);
  const equal=()=>{
    const reference=indexSurface(maskSurface(mask,shape));
    for(const key of ['positions','indices'])assert.deepEqual(new Uint8Array(chunks.indexed()[key].buffer),new Uint8Array(reference[key].buffer),key);
    assert.deepEqual(chunks.positions(),maskSurface(mask,shape));
    assert(chunks.retainedBytes<=2**20);
  };
  equal();
  const original=mask.slice();
  for(const ids of [[at(3,3,3),at(4,4,4)],[at(0,0,0),at(8,7,6)],[at(7,7,6),at(8,7,6)]]){for(const id of ids)mask[id]=1-mask[id];chunks.update(ids);equal();}
  const changed=[];for(let i=0;i<mask.length;i++)if(mask[i]!==original[i]){mask[i]=original[i];changed.push(i);}chunks.update(changed);equal();
  const deleted=[];for(let i=0;i<mask.length;i++)if(mask[i]){mask[i]=0;deleted.push(i);}chunks.update(deleted);equal();
  mask[at(4,4,4)]=1;chunks.update([at(4,4,4)]);equal();
});
test('indexed worker buffers remain exact across 32-cell seams and survive transfer without detaching residency',()=>{
  const shape=[35,8,7],at=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
  const mask=Uint8Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>{const x=Math.floor(i/56),y=Math.floor(i/7)%8,z=i%7;return Number(x>0&&x<34&&y>0&&y<7&&z>0&&z<6);});
  const engine=new IncrementalSurfaceEngine(),base={layerId:'brain',shape,affine,kind:'brain'};
  let result=engine.prepare({...base,jobId:1,previousIdentity:'',identity:'a',mask:mask.slice()});
  const equal=()=>{const reference=prepareSurface(mask,shape,affine,'brain',true);for(const key of ['positions','indices','normals','smoothPositions','smoothNormals'])assert.deepEqual(new Uint8Array(result[key].buffer),new Uint8Array(reference[key].buffer),key);assert.equal(result.work.indexedChunks,true);};
  equal();
  const transfers=['positions','indices','normals','smoothPositions','smoothNormals'].map(key=>result[key].buffer);
  structuredClone(result,{transfer:transfers});assert.equal(result.positions.byteLength,0);
  const ids=[at(31,4,4),at(32,4,4),at(0,0,0)],values=[0,0,1];ids.forEach((id,i)=>mask[id]=values[i]);
  result=engine.prepare({...base,jobId:2,previousIdentity:'a',identity:'b',indices:Uint32Array.from(ids),values:Uint8Array.from(values)});equal();
  result=engine.prepare({...base,jobId:3,previousIdentity:'b',identity:'b',indices:new Uint32Array(),values:new Uint8Array()});equal();assert.equal(result.work.remeshedChunks,0);assert.equal(result.work.indexedCorners,0);
});
test('indexed budget exhaustion discards partial rows and returns the complete exact updated surface',()=>{
  const shape=[40,12,12],mask=new Uint8Array(40*12*12);mask[0]=1;
  const first=new ChunkedMaskSurface(mask.slice(),shape,32,2**20),budget=first.retainedBytes+100;
  const engine=new IncrementalSurfaceEngine(budget),base={layerId:'brain',shape,affine,kind:'brain',smooth:false};
  let result=engine.prepare({...base,jobId:1,previousIdentity:'',identity:'a',mask:mask.slice()});assert.equal(result.work.indexedChunks,true);
  const ids=[];for(let x=1;x<40;x+=2)for(let y=1;y<12;y+=2)for(let z=1;z<12;z+=2){const id=(x*12+y)*12+z;mask[id]=1;ids.push(id);}
  result=engine.prepare({...base,jobId:2,previousIdentity:'a',identity:'b',indices:Uint32Array.from(ids),values:new Uint8Array(ids.length).fill(1)});
  const expected=prepareSurface(mask,shape,affine,'brain',false);for(const key of ['positions','indices','normals'])assert.deepEqual(result[key],expected[key],key);
  assert.equal(result.work.indexedChunks,false);assert(result.work.retainedBytes<=budget);assert.deepEqual(result.resident,[]);
  mask.fill(0);mask[0]=1;
  result=engine.prepare({...base,jobId:3,previousIdentity:'b',identity:'c',mask:mask.slice()});assert.equal(result.work.indexedChunks,true);assert.deepEqual(result.resident,[{layerId:'brain',identity:'c'}]);
});
