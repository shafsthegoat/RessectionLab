import assert from 'node:assert/strict';
import {test} from 'node:test';
import * as THREE from 'three';
import {LabelTextureUpdates} from './labelTextureUpdates.ts';
function fixture(shape){
  const data=Uint8Array.from({length:shape.reduce((a,b)=>a*b,4)},(_,i)=>(i*17+3)%256),texture=new THREE.Data3DTexture(data,shape[2],shape[1],shape[0]);
  texture.format=THREE.RGBAFormat;texture.type=THREE.UnsignedByteType;texture.unpackAlignment=1;const copies=[];let gpu,failAt=0;
  const renderer={initTexture(t){gpu=Uint8Array.from(t.image.data);},copyTextureToTexture(source,destination,box,position){
    assert.equal(source.version,0,'CPU wrapper must never be initialized on the GPU');assert.notEqual(source,destination);assert.equal(source.image.data,destination.image.data);assert.equal(source.unpackAlignment,1);
    if(copies.length+1===failAt)throw Error('injected copy failure');copies.push({low:box.min.toArray(),high:box.max.toArray(),destination:position.toArray()});
    const w=source.image.width,h=source.image.height;
    for(let z=box.min.z;z<box.max.z;z++)for(let y=box.min.y;y<box.max.y;y++)for(let x=box.min.x;x<box.max.x;x++){
      const input=((z*h+y)*w+x)*4,output=(((position.z+z-box.min.z)*h+position.y+y-box.min.y)*w+position.x+x-box.min.x)*4;
      gpu.set(source.image.data.subarray(input,input+4),output);
    }
  }};
  return {texture,data,renderer,copies,gpu:()=>gpu,failOn(n){failAt=n;}};
}
function patch(data,indices){const values=Uint8Array.from({length:indices.length*4},(_,i)=>255-i);indices.forEach((index,i)=>data.set(values.subarray(i*4,i*4+4),index*4));return {indices:Uint32Array.from(indices),values};}
test('partial labels preserve exact RGBA and z-fastest source order at every axis boundary',()=>{
  const shape=[4,3,5],f=fixture(shape),updates=new LabelTextureUpdates();updates.sync(f.renderer,f.texture);const initial=f.data.slice();
  for(const index of [0,(1*3+1)*5+2,shape.reduce((a,b)=>a*b,1)-1]){
    const p=patch(f.data,[index]);assert.equal(updates.update(f.renderer,f.texture,shape,p,false),'partial');updates.sync(f.renderer,f.texture);assert.deepEqual(f.gpu(),f.data);
  }
  assert.deepEqual(f.copies[1],{low:[2,1,1],high:[3,2,2],destination:[2,1,1]});
  assert.deepEqual(f.copies[2],{low:[4,2,3],high:[5,3,4],destination:[4,2,3]});
  assert.equal(updates.work.fullUploads,1);assert.equal(updates.work.partialUploadBytes,12);assert.equal(updates.work.partialRegions,3);assert.equal(initial.byteLength,f.data.byteLength);
});
test('first admission/layout replacement and unavailable API use complete exact upload fallback',()=>{
  const shape=[4,3,5],f=fixture(shape),updates=new LabelTextureUpdates(),p=patch(f.data,[2]);
  assert.equal(updates.update(f.renderer,f.texture,shape,p,false),'full');updates.sync(f.renderer,f.texture);assert.deepEqual(f.gpu(),f.data);assert.equal(f.copies.length,0);
  f.texture.image.data=f.data.slice().fill(43);assert.equal(updates.update(f.renderer,f.texture,shape,undefined,true),'full');updates.sync(f.renderer,f.texture);assert.deepEqual(f.gpu(),f.texture.image.data);
  const noCopy={initTexture:f.renderer.initTexture};const q=patch(f.texture.image.data,[5]);assert.equal(updates.update(noCopy,f.texture,shape,q,false),'full');updates.sync(noCopy,f.texture);assert.deepEqual(f.gpu(),f.texture.image.data);
});
test('copy failure after a successful region falls back to a complete final image and records known bytes',()=>{
  const shape=[65,2,3],f=fixture(shape),updates=new LabelTextureUpdates();updates.sync(f.renderer,f.texture);f.failOn(2);
  const p=patch(f.data,[0,f.data.length/4-1]);assert.equal(updates.update(f.renderer,f.texture,shape,p,false),'full');updates.sync(f.renderer,f.texture);
  assert.deepEqual(f.gpu(),f.data);assert.equal(updates.work.fallbacks,1);assert.equal(updates.work.partialUploadBytes,4);assert.equal(updates.work.fullUploadBytes,f.data.length*2);
});
test('dirty command budget stays bounded and empty patches allocate/upload nothing',()=>{
  const shape=[257,257,1],f=fixture(shape),updates=new LabelTextureUpdates();updates.sync(f.renderer,f.texture);
  const indices=[];for(let x=0;x<257;x+=32)for(let y=0;y<257;y+=32)indices.push(x*257+y);
  assert.equal(updates.update(f.renderer,f.texture,shape,patch(f.data,indices),false),'full');assert.equal(f.copies.length,0);updates.sync(f.renderer,f.texture);assert.deepEqual(f.gpu(),f.data);
  const version=f.texture.version;assert.equal(updates.update(f.renderer,f.texture,shape,{indices:new Uint32Array(),values:new Uint8Array()},false),'none');assert.equal(f.texture.version,version);
});
