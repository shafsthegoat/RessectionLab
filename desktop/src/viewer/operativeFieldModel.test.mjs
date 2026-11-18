import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {buildOperativeModel,hasAspirationSeed} from './operativeFieldModel.ts';
import {operativeSurface} from './operativeSurface.ts';
import {operativeInstruments} from './operativeInstruments.ts';

function field() {
  const shape=[25,25,21], voxels=new Uint8Array(25*25*21), index=(x,y,z)=>(x*25+y)*21+z;
  for(let x=0;x<25;x++)for(let y=0;y<25;y++)for(let z=0;z<21;z++){
    let bits=z===19?1:z===17?2:z<17?4:0;
    if((x-12)**2+(y-12)**2+(z-9)**2<=25)bits|=8;
    if(x===13&&y===12&&z>=4&&z<=16)bits|=16;
    voxels[index(x,y,z)]=bits;
  }
  return {shape,origin:[-12,-12,-9],spacingMm:1,radiusMm:12,routeRadiusMm:2,entryDepthMm:9,stations:{bone:10,dura:8,brain:7,tumor:5},voxels};
}
const count=mask=>mask.reduce((total,n)=>total+n,0);

test('cut progression, lifted source patches, aspiration and scrubbing preserve the source',()=>{
  const f=field(), original=f.voxels.slice(), m=buildOperativeModel(f);
  for(const kind of ['bone','dura']){
    const layer=m.layers[kind],flap=m.layers[kind+'Flap'];
    const initial=count(layer.mask);assert.ok(count(flap.mask)>0);
    m.applyOpening(kind,64,false);const halfway=count(layer.mask);
    m.applyOpening(kind,128,false);const opened=count(layer.mask);
    assert.ok(initial>halfway&&halfway>opened);
    m.applyOpening(kind,128,true);assert.equal(count(layer.mask)+count(flap.mask),opened);
    m.applyOpening(kind,0,false);assert.equal(count(layer.mask),initial);
  }
  const tumor=m.layers.tumor.mask.slice(), vessel=m.layers.vessel.mask.slice(), brain=m.layers.brain.mask.slice();
  let previous=null;
  for(let step=1;step<=m.stepCount;step++){
    const tip=m.tipAt(step), [x,y,z]=tip.map((v,i)=>v-f.origin[i]), sample=f.voxels[(x*25+y)*21+z];
    assert.ok(sample&8);assert.equal(sample&16,0);
    if(previous)assert.ok(new THREE.Vector3(...previous).distanceTo(new THREE.Vector3(...tip))<=1.00001);
    previous=tip;
  }
  m.applyAspiration(Math.ceil(m.stepCount*.35));const mid=count(m.layers.tumor.mask);
  const removed=m.applyAspiration(m.stepCount),done=count(m.layers.tumor.mask);
  assert.equal(removed,count(tumor)-done);
  assert.ok(count(tumor)>mid&&mid>done&&done>0);
  assert.deepEqual(m.layers.vessel.mask,vessel);assert.deepEqual(m.layers.brain.mask,brain);
  for(let i=0;i<tumor.length;i++)if(tumor[i]!==m.layers.tumor.mask[i]){assert.ok(original[i]&8);assert.equal(original[i]&16,0);}
  m.applyAspiration(0);assert.deepEqual(m.layers.tumor.mask,tumor);assert.deepEqual(f.voxels,original);
});

test('missing stations leave visible sampled sources intact and omit corresponding poses',()=>{
  const f=field();f.stations={bone:null,dura:null,brain:null,tumor:null};const m=buildOperativeModel(f);
  assert.equal(count(m.layers.boneFlap.mask)+count(m.layers.duraFlap.mask),0);
  assert.ok(count(m.layers.bone.mask)>0 && count(m.layers.tumor.mask)>0);
  assert.equal(m.stepCount,0);assert.equal(m.tipAt(0),null);assert.equal(m.surfaceHeight('dura',0,0),null);
});

test('section surface has valid indexed triangles, outward normals and closed edges',()=>{
  const f=field(),m=buildOperativeModel(f),{mask,bounds}=m.layers.tumor;
  const mesh=operativeSurface(f,mask,bounds,m.firstSectionRow),pos=mesh.getAttribute('position'),indices=mesh.index.array;
  assert.ok(pos.count>20);assert.ok(indices.length>3*pos.count);const edges=new Map();
  let signedVolume=0;
  for(let i=0;i<indices.length;i+=3){
    const ids=Array.from(indices.slice(i,i+3));assert.ok(ids.every(v=>v>=0&&v<pos.count));assert.equal(new Set(ids).size,3);
    const [a,b,c]=ids.map(j=>new THREE.Vector3().fromBufferAttribute(pos,j));signedVolume+=a.dot(b.cross(c))/6;
    for(let k=0;k<3;k++){const a=ids[k],b=ids[(k+1)%3],key=[Math.min(a,b),Math.max(a,b)].join(',');edges.set(key,(edges.get(key)??0)+1);}
  }
  assert.ok(signedVolume>0);assert.ok([...edges.values()].every(n=>n===2));mesh.dispose();
});

test('aspirator annulus has an open lumen and a separate long shaft',()=>{
  const m=operativeInstruments(1.6,9,48),[tip,shaft]=m.cusa.children;
  const ray=new THREE.Raycaster(new THREE.Vector3(0,0,-2),new THREE.Vector3(0,0,1));
  m.cusa.updateMatrixWorld(true);assert.equal(ray.intersectObject(tip).length,0);
  ray.ray.origin.x=1.25;assert.ok(ray.intersectObject(tip).length>0);
  assert.ok(new THREE.Box3().setFromObject(shaft).getSize(new THREE.Vector3()).z>40);
  for(const tool of Object.values(m))tool.traverse(obj=>{if(obj instanceof THREE.Mesh){obj.geometry.dispose();obj.material.dispose();}});
});

const index=(x,y,z)=>(x*25+y)*21+z;

test('protected station or endpoint never redirects aspiration to another eligible nearby focus',()=>{
  for(const protectedZ of [9,14]){
    const f=field();f.voxels.fill(0);
    for(let z=9;z<=14;z++)f.voxels[index(12,12,z)]=8;
    f.voxels[index(12,12,protectedZ)]|=16;
    f.voxels[index(14,12,14)]=8;
    assert.equal(hasAspirationSeed(f),false);
    const m=buildOperativeModel(f),initial=count(m.layers.tumor.mask);
    assert.equal(m.stepCount,0);assert.equal(m.tipAt(1),null);
    assert.equal(m.applyAspiration(100),0);assert.equal(count(m.layers.tumor.mask),initial);
    assert.equal(count(m.layers.vessel.mask),1);
  }
});

test('aspiration brush cannot cross a one-cell vessel or empty wall to another tumor island',()=>{
  for(const wall of [0,8|16]){
    const f=field();f.voxels.fill(0);f.routeRadiusMm=6;
    for(let z=9;z<=14;z++){
      f.voxels[index(12,12,z)]=8;
      f.voxels[index(13,12,z)]=wall;
      f.voxels[index(14,12,z)]=8;
    }
    assert.equal(hasAspirationSeed(f),true);
    const m=buildOperativeModel(f);assert.ok(m.tipRadius>=2);assert.ok(m.stepCount>0);
    assert.equal(m.applyAspiration(m.stepCount),6);
    for(let z=9;z<=14;z++){
      assert.equal(m.layers.tumor.mask[index(12,12,z)],0);
      assert.equal(m.layers.tumor.mask[index(14,12,z)],1);
    }
    assert.equal(count(m.layers.vessel.mask),wall?6:0);
    assert.equal(m.applyAspiration(0),0);assert.equal(count(m.layers.tumor.mask),12);
  }
});
