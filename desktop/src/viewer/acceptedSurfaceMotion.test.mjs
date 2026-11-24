import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {AcceptedSurfaceMotion,applyAcceptedSurfaceMotion} from './acceptedSurfaceMotion.ts';
import {installRetractionMorph} from './retractionMorph.ts';
import {retractionOffset} from './operativeFieldModel.ts';

const shape=[12,12,12],affine=[[2,.25,0,5],[0,1.5,.2,-3],[0,0,-1,9],[0,0,0,1]];
const linear=new THREE.Matrix3().set(...affine.slice(0,3).flatMap(r=>r.slice(0,3)));
const id=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
const view=(deformation={},moved=[])=>({scene:{shape,affine,regions:[{id:1,material:'brain'},{id:2,material:'vessel'}]},deformation,moved,removed:[]});
const storage=type=>new type(shape.reduce((a,b)=>a*b));
const vec=(attribute,i)=>new THREE.Vector3().fromBufferAttribute(attribute,i);
const close=(a,b,tol=1e-6)=>assert(a.distanceTo(b)<tol,a.toArray()+' vs '+b.toArray());
function geometry(points){
  const result=new THREE.BufferGeometry();
  result.setAttribute('position',new THREE.Float32BufferAttribute(points.flat(),3));
  result.setAttribute('normal',new THREE.Float32BufferAttribute(points.flatMap(()=>[0,0,1]),3));
  return result;
}

test('unique integer transport moves MRI lookup to the originating cell on raw and conditioned surfaces',()=>{
  const from=id(2,3,4),to=id(4,3,4),owner=storage(Uint16Array),mask=storage(Uint8Array);
  owner[from]=owner[id(10,10,10)]=1;mask[to]=mask[id(10,10,10)]=1;
  const field=new AcceptedSurfaceMotion(view({[from]:new THREE.Vector3(2,0,.23).applyMatrix3(linear).toArray()},[[from,to]]),owner);
  const base=[[4,3,4],[10,10,10]],expectedRest=new THREE.Vector3(2,3,4);
  for(const smooth of [true,false]){
    const geo=geometry(base),original=geo.getAttribute('position').array;
    applyAcceptedSurfaceMotion(geo,field,mask,smooth,true);
    const position=geo.getAttribute('position'),offset=geo.getAttribute('corticalRestOffset');
    close(vec(position,0),new THREE.Vector3(4,3,smooth?4.23:4));
    close(vec(position,0).add(vec(offset,0)),expectedRest);
    assert.equal(geo.getAttribute('corticalUnknown').getX(0),0);
    close(vec(position,1).add(vec(offset,1)),new THREE.Vector3(...base[1]));
    assert.deepEqual([...original],base.flat());
    applyAcceptedSurfaceMotion(geo,null,mask,smooth,true);
    assert.deepEqual([...geo.getAttribute('position').array],base.flat());
    assert(geo.getAttribute('corticalRestOffset').array.every(x=>x===0));
  }
});

test('many-to-one or mixed raster origins withhold only local MRI and fixed structures pin adjacent interpolation',()=>{
  const a=id(3,4,4),b=id(4,4,4),owner=storage(Uint16Array),mask=storage(Uint8Array);
  owner[a]=owner[b]=owner[id(9,9,9)]=1;mask[b]=mask[id(9,9,9)]=1;
  const collision=new AcceptedSurfaceMotion(view({},[[a,b]]),owner);
  assert(collision.sample([4,4,4],mask).ambiguous);
  assert(!collision.sample([9,9,9],mask).ambiguous);
  const mixed=new AcceptedSurfaceMotion(view({},[[a,id(5,4,4)]]),owner);mask[id(5,4,4)]=1;
  assert(mixed.sample([4.5,4,4],mask).ambiguous);
  owner[id(3,3,5)]=2;
  const pinned=new AcceptedSurfaceMotion(view({[a]:[.2,0,0]}),owner);
  assert.deepEqual(pinned.sample([3,4,4],mask).shift,[0,0,0]);
});

test('generated vessel motion follows only its declared, intact source attachment and matching accepted tissue shift',()=>{
  const site=id(4,4,4),tissue=id(4,4,3),fixed=id(5,4,4);
  const owner=storage(Uint16Array),mask=storage(Uint8Array);
  owner[site]=owner[fixed]=2;owner[tissue]=1;mask[site]=mask[fixed]=1;
  const delta=new THREE.Vector3(.25,0,0).applyMatrix3(linear).toArray();
  const native=view({[site]:delta,[tissue]:delta,[fixed]:delta});
  const schema='resectionlab.generated-glioma-materials.v1';
  native.scene.source={procedurePolicySourceClass:'generated-engineering'};
  native.physics={brain:{model:schema,assignment:{schema,kind:'assigned-generated-research',vesselAttachments:[[site,tissue]]}}};
  const field=new AcceptedSurfaceMotion(native,owner),pts=[[4,4,4],[5,4,4],[4,4,3]];
  assert.deepEqual([...field.movingVesselRegions],[2]);
  const original=geometry(pts),smooth=original.clone();
  applyAcceptedSurfaceMotion(smooth,field,mask,true,true,2);
  close(vec(smooth.getAttribute('position'),0),new THREE.Vector3(4.25,4,4));
  // An attached cell cannot animate an untethered vessel center or extend the
  // vessel mesh onto the donor cell. Source registration and masks stay put.
  assert.deepEqual([...smooth.getAttribute('position').array.slice(3)],pts.slice(1).flat());
  applyAcceptedSurfaceMotion(original,field,mask,false,true);
  assert.deepEqual([...original.getAttribute('position').array],pts.flat());
  assert.equal(mask[site],1);assert.equal(owner[site],2);
  for(const changed of [
    {...native,physics:undefined},
    {...native,scene:{...native.scene,source:{procedurePolicySourceClass:'acquired'}}},
    {...native,deformation:{[site]:delta,[tissue]:[0,0,0]}},
    {...native,removed:[tissue]},
    {...native,cutEdges:[[tissue,site]]},
    {...native,moved:[[fixed,site]]},
  ]){
    const inactive=new AcceptedSurfaceMotion(changed,owner);
    assert.equal(inactive.movingVesselRegions.size,0);
    applyAcceptedSurfaceMotion(smooth,inactive,mask,false,true);
    assert.deepEqual([...smooth.getAttribute('position').array],pts.flat());
  }
});

test('the retained vessel profile renders the same accepted source position after its attachment is lost',()=>{
  const site=id(4,4,4),support=id(4,4,3),to=id(5,4,4),unmapped=id(8,4,4);
  const owner=storage(Uint16Array),mask=storage(Uint8Array);
  owner[site]=owner[unmapped]=2;owner[support]=1;mask[to]=mask[unmapped]=1;
  const delta=new THREE.Vector3(.76,0,0).applyMatrix3(linear).toArray();
  const native=view({[site]:delta,[support]:delta,[unmapped]:delta},[[site,to]]);
  const schema='resectionlab.generated-glioma-materials.v1';
  native.scene.source={procedurePolicySourceClass:'generated-engineering'};
  native.physics={brain:{model:schema,assignment:{schema,kind:'assigned-generated-research',
    vesselAttachments:[[site,support]],vesselMotionModel:'retained-source-segments-v1'}}};
  const geo=geometry([[5,4,4],[8,4,4]]),expected=new THREE.Vector3(4.76,4,4);
  for(const sample of [native,{...native,deformation:{[site]:delta,[unmapped]:delta},removed:[support]},{
    ...native,deformation:{[site]:delta},transectedEdges:[[site,support]]
  }]){
    const field=new AcceptedSurfaceMotion(sample,owner);
    applyAcceptedSurfaceMotion(geo,field,mask,true,true,2);
    close(vec(geo.getAttribute('position'),0),expected);
    close(vec(geo.getAttribute('position'),1),new THREE.Vector3(8,4,4));
    close(vec(geo.getAttribute('position'),0).add(vec(geo.getAttribute('corticalRestOffset'),0)),new THREE.Vector3(4,4,4));
  }
});

test('conditioned surface and lighting follow the local accepted gradient and raw/generated sources stay untouched',()=>{
  const pos=[4.25,4.33,4],cell=id(4,4,4),owner=storage(Uint16Array),mask=storage(Uint8Array);
  owner[cell]=mask[cell]=1;
  const move=new AcceptedSurfaceMotion(view({[cell]:[.04,.02,.3]}),owner),geo=geometry([pos]);
  const deform=p=>new THREE.Vector3(...p).add(new THREE.Vector3(...move.sample(p,mask).shift));
  const tangent=(axis)=>{
    const a=[...pos],b=[...pos];a[axis]+=.0001;b[axis]-=.0001;
    return deform(a).sub(deform(b)).normalize();
  };
  applyAcceptedSurfaceMotion(geo,move,mask,true,false);
  const physical=vec(geo.getAttribute('normal'),0);
  assert(Math.abs(physical.dot(tangent(0)))<1e-6&&Math.abs(physical.dot(tangent(1)))<1e-6);
  close(vec(geo.getAttribute('position'),0),deform(pos));
  assert(!geo.hasAttribute('corticalRestOffset'));
  const still=geometry([pos]);applyAcceptedSurfaceMotion(still,move,mask,false,false);
  close(vec(still.getAttribute('position'),0),new THREE.Vector3(...pos));
});

test('route morph normals stay perpendicular to deformed tangents across intermediate influences on an oblique grid',()=>{
  const p=new THREE.Vector3(1,.7,5),g=geometry([p.toArray()]),toRoute=new THREE.Matrix4().fromArray(affine.flat()).transpose();
  const opening=6,station=4;
  installRetractionMorph(g,opening,station,toRoute);
  const deform=(q,t)=>{
    const world=q.clone().applyMatrix4(toRoute);
    world.addScaledVector(new THREE.Vector3(...retractionOffset(...world.toArray(),opening,station)),t);
    return world.applyMatrix4(toRoute.clone().invert());
  };
  for(const t of [0,.2,.65,1]){
    const n=vec(g.getAttribute('normal'),0).addScaledVector(vec(g.morphAttributes.normal[0],0),t).normalize();
    for(const axis of [0,1]){
      const d=new THREE.Vector3();d.setComponent(axis,.001);
      const tangent=deform(p.clone().add(d),t).sub(deform(p.clone().sub(d),t)).normalize();
      assert(Math.abs(n.dot(tangent))<1e-7);
    }
    close(p.clone().addScaledVector(vec(g.morphAttributes.position[0],0),t),deform(p,t));
  }
});
