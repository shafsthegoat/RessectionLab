import assert from 'node:assert/strict';
import test from 'node:test';
import {prepareSurface} from './prepareSurface.ts';
import {HOST_FAIRING_POLICY,hostFairingMobility} from './hostSurfaceFairing.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const shape=[40,40,24],[nx,ny,nz]=shape;
const at=(x,y,z)=>(x*ny+y)*nz+z;
// A gently sloped terrace (one cell rise every 6 cells along x) of "cortex"
// over "white matter", under free space: the kind of 1 mm staircase a host head has.
function terraceOwner(){
  const owner=new Uint16Array(nx*ny*nz);
  for(let x=0;x<nx;x++)for(let y=0;y<ny;y++){
    const top=6+Math.floor(x/6);
    for(let z=0;z<top;z++)owner[at(x,y,z)]=z<top-3?2:1;
  }
  return owner;
}
function prepare(owner,region,hostFairing){
  const mask=Uint8Array.from(owner,v=>Number(v===region));
  return prepareSurface(mask,shape,identity,'soft_tissue',true,{owner,protectedRegions:[],...(hostFairing?{hostFairing:true}:{})});
}
// Deviation of free-surface heights from the ideal ramp through the riser midpoints.
function terraceError(surface){
  let worst=0,sum=0,n=0;
  for(let i=0;i<surface.positions.length;i+=3){
    const [x,y,z]=surface.positions.subarray(i,i+3),shown=surface.smoothPositions.subarray(i,i+3);
    if(x<8||x>nx-9||y<3||y>ny-4||z<6)continue;
    // Only outward (free-space) faces: the top of the column at this x.
    if(Math.abs(z-(5.5+Math.floor(Math.round(x)/6)))>.01||!Number.isInteger(x))continue;
    const ideal=5.5+(shown[0]-2.5)/6,error=Math.abs(shown[2]-ideal);
    worst=Math.max(worst,error);sum+=error;n++;
  }
  return {worst,mean:sum/n};
}

test('host fairing turns 1 mm terraces into a ramp within half a cell of the source',()=>{
  const owner=terraceOwner();
  const plain=prepare(owner,1,false),host=prepare(owner,1,true);
  assert.equal(host.diagnostics.policy,HOST_FAIRING_POLICY);
  assert(host.diagnostics.maxDisplacementMm<.5,'bounded by half a cell');
  const before=terraceError(plain),after=terraceError(host);
  assert(after.mean<.5*before.mean,`terrace error ${before.mean.toFixed(3)} -> ${after.mean.toFixed(3)}`);
  for(let i=0;i<host.positions.length;i+=3){
    const d=Math.hypot(...[0,1,2].map(a=>host.smoothPositions[i+a]-host.positions[i+a]));
    assert(d<=.5+1e-6);
  }
});

test('faces shared with another material keep the shared implicit position',()=>{
  const owner=terraceOwner();
  const cortex=prepare(owner,1,true),white=prepare(owner,2,true);
  const shown=new Map();
  for(let i=0;i<white.positions.length;i+=3)shown.set(white.positions.subarray(i,i+3).join(','),white.smoothPositions.subarray(i,i+3).join(','));
  let shared=0;
  for(let i=0;i<cortex.positions.length;i+=3){
    const key=cortex.positions.subarray(i,i+3).join(',');
    if(!shown.has(key))continue;
    shared++;assert.equal(cortex.smoothPositions.subarray(i,i+3).join(','),shown.get(key));
  }
  assert(shared>1000,'the cortex/white interface is present in both layers');
});

test('a generated phantom without the host flag keeps its source-edge display',()=>{
  const owner=terraceOwner(),plain=prepare(owner,1,false);
  assert.notEqual(plain.diagnostics.policy,HOST_FAIRING_POLICY);
});

test('brain faces against the head wall may move; faces inside one group stay shared',()=>{
  // Column: white matter (2) under cortex (1) under bone (3), no free space.
  const big=[5,5,8],o=new Uint16Array(200);
  for(let x=0;x<5;x++)for(let y=0;y<5;y++)for(let z=0;z<8;z++)o[(x*5+y)*8+z]=z<3?2:z<5?1:3;
  const m=Uint8Array.from(o,v=>Number(v===1)),p=Float32Array.from([2,2,2.5, 2,2,4.5]);
  const grouped=hostFairingMobility(p,p,o,m,big,new Set(),[[1,2],[3]]).mobile;
  assert.deepEqual([...grouped],[0,1]);
  const ungrouped=hostFairingMobility(p,p,o,m,big,new Set()).mobile;
  assert.deepEqual([...ungrouped],[0,0]);
});

test('a host crop through the brain keeps subarachnoid space out of exterior air',async()=>{
  const {exteriorAirFlags,EXTERIOR_AIR}=await import('./exteriorSkinSurface.ts');
  // Skin slab on top (z 8-9), cortex below (z<5) and a CSF gap reaching every crop face.
  const n=12,shape=[n,n,n],owner=new Uint16Array(n**3),id=(x,y,z)=>(x*n+y)*n+z;
  for(let x=0;x<n;x++)for(let y=0;y<n;y++){for(let z=0;z<5;z++)owner[id(x,y,z)]=3;for(let z=8;z<10;z++)owner[id(x,y,z)]=2;}
  const plain=exteriorAirFlags(owner,shape,2),host=exteriorAirFlags(owner,shape,2,new Set([3]));
  // CSF touching the cortex is exterior air only without the host rule.
  assert(plain[id(6,6,5)]&EXTERIOR_AIR);
  assert(!(host[id(6,6,5)]&EXTERIOR_AIR));
  // Air above the skin is unchanged.
  assert(host[id(6,6,11)]&EXTERIOR_AIR);
});

test('brain pinholes close for display; an open sulcus and the engine owner stay as they are',async()=>{
  const {hostDisplayOwner,hostDisplayMask}=await import('./hostSurfaceFairing.ts');
  const n=12,shape=[n,n,n],at=(x,y,z)=>(x*n+y)*n+z,owner=new Uint16Array(n**3);
  for(let x=1;x<n-1;x++)for(let y=1;y<n-1;y++)for(let z=1;z<8;z++)owner[at(x,y,z)]=1;
  owner[at(5,5,4)]=0;                       // enclosed pinhole
  for(let y=1;y<n-1;y++)for(let z=3;z<8;z++)owner[at(8,y,z)]=0; // one-cell sulcus open along y and to the surface
  const before=owner.slice(),closed=hostDisplayOwner(owner,shape,[1]);
  assert.equal(closed.owner[at(5,5,4)],1);
  assert.equal(closed.owner[at(8,6,6)],0);
  assert.deepEqual(owner,before);
  const mask=Uint8Array.from(owner,v=>Number(v===1)),display=hostDisplayMask(mask,owner,closed.owner,shape);
  assert.equal(display[at(5,5,4)],1);assert.equal(mask[at(5,5,4)],0);
});
