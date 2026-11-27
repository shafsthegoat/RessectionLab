import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {craniotomyOutline,presentParkedTool,installCropSection} from './hostSceneDisplay.ts';
import {sampleSection,depthRuns,paintSection,sectionWindow,exteriorAbove,UNLABELLED_INSIDE} from './operativeSection.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const n=40,shape=[n,n,n],at=(x,y,z)=>(x*n+y)*n+z;
// Layered slab along +z: bone (1) z 20-24, skin (2) z 25-27, air above; a 6-cell bone flap at the centre.
function head(){
  const owner=new Uint16Array(n**3),flap=[];
  for(let x=0;x<n;x++)for(let y=0;y<n;y++){
    for(let z=20;z<25;z++){owner[at(x,y,z)]=1;if(Math.hypot(x-20,y-20)<=6)flap.push(at(x,y,z));}
    for(let z=25;z<28;z++)owner[at(x,y,z)]=2;
  }
  return {owner,flap};
}

test('the craniotomy outline follows the bone flap rim and lies on the skin',()=>{
  const {owner,flap}=head();
  const loop=craniotomyOutline(flap,owner,shape,identity,[0,0,1]);
  assert(loop&&loop.length>=77);
  for(const p of loop){
    const r=Math.hypot(p[0]-20,p[1]-20);
    assert(r>5&&r<7.5,`radius ${r}`);
    // Just outside the last skin cell (z=27, face at 27.5).
    assert(p[2]>27&&p[2]<28.5,`height ${p[2]}`);
  }
  assert.equal(craniotomyOutline(flap.slice(0,3),owner,shape,identity,[0,0,1]),undefined);
});

test('a parked tool is hidden in the microscope, faint outside it and restored when inserted',()=>{
  const material=new THREE.MeshStandardMaterial({opacity:1});
  const group=new THREE.Group();group.add(new THREE.Mesh(new THREE.BoxGeometry(),material));
  presentParkedTool(group,true,true);
  assert.equal(group.visible,false);
  presentParkedTool(group,true,false);
  assert.equal(group.visible,true);assert(material.transparent&&material.opacity<.2);
  presentParkedTool(group,false,false);
  assert.equal(material.opacity,1);assert.equal(material.transparent,false);assert.equal(material.depthWrite,true);
});

test('crop-section shading is installed once and keys its own program',()=>{
  const material=new THREE.MeshStandardMaterial();
  assert(installCropSection(material,[10,10,10]));
  assert(!installCropSection(material,[10,10,10]));
  assert.match(material.customProgramCacheKey(),/crop-section/);
});

test('a depth section lists the layers met along its axis from the skin, with unlabelled space kept unlabelled',()=>{
  // Down the image is -z: air, skin (3 cells), bone (5), unlabelled (2), cortex.
  const owner=new Uint16Array(n**3);
  for(let x=0;x<n;x++)for(let y=0;y<n;y++){
    for(let z=0;z<18;z++)owner[at(x,y,z)]=3;
    for(let z=20;z<25;z++)owner[at(x,y,z)]=1;
    for(let z=25;z<28;z++)owner[at(x,y,z)]=2;
  }
  const mri=Float32Array.from(owner,v=>v*10+5);
  const layers=[{name:'Skull',displayKind:'skull',color:'#fff',mask:Uint8Array.from(owner,v=>Number(v===1))},
    {name:'Scalp skin',displayKind:'scalp',color:'#fff',mask:Uint8Array.from(owner,v=>Number(v===2))},
    {name:'Cerebral cortex',displayKind:'grayMatter',color:'#fff',mask:Uint8Array.from(owner,v=>Number(v===3))}];
  const sample=sampleSection(layers,mri,shape,identity,{origin:[10,20,34],across:[1,0,0],down:[0,0,-1],width:20,height:30,stepMm:1});
  const {runs}=depthRuns(sample,layers,10);
  assert.deepEqual(runs.map(r=>r.name),['Scalp','Bone (cortical)',UNLABELLED_INSIDE,'Cortex']);
  // Boundaries follow the half-cell contour: within a millimetre of the cell layers (3, 5 and 2 mm).
  const expected=[[0,3],[3,8],[8,10],[10,runs[3].toMm]];
  runs.forEach((run,i)=>{assert(Math.abs(run.fromMm-expected[i][0])<=1&&Math.abs(run.toMm-expected[i][1])<=1,JSON.stringify(runs));});
  const exterior=exteriorAbove(sample),pixels=paintSection(sample,layers,sectionWindow(sample),exterior);
  assert.equal(pixels[3],0,'air above the skin is transparent');
  assert.equal(pixels[4*(10*20+10)+3],255,'tissue is opaque');
});
