import * as THREE from 'three';
import { surgicalToolMesh } from './surgicalGeometry.ts';

/** Tip-local positive Z points out of the field. The illustrative tools use
 * the saved route scale independently of the acquired tissue labels. */
export function operativeInstruments(tipRadius:number, opening:number, fieldDepth:number) {
  const shaftLength=Math.max(opening*1.4,Math.min(fieldDepth+opening*.3,90));
  const metal=new THREE.MeshStandardMaterial({color:'#c5d4da',metalness:.55,roughness:.25});
  const grip=new THREE.MeshStandardMaterial({color:'#386a83',metalness:.32,roughness:.38});
  const accent=new THREE.MeshStandardMaterial({color:'#e4b576',metalness:.5,roughness:.32});
  function axial(group:THREE.Group, points:number[][], material:THREE.Material) {
    const profile=points.map(([r,z])=>new THREE.Vector2(r,z));
    const mesh=new THREE.Mesh(new THREE.LatheGeometry(profile,28),material);
    mesh.rotation.x=Math.PI/2; group.add(mesh);
  }
  const cusa=new THREE.Group(), r=tipRadius;
  cusa.name='hollow-aspirator';
  // The annular cross section includes the inner lumen wall and bevel.
  axial(cusa,[[r*.63,r*3],[r*.63,r*.14],[r*.7,0],[r*.91,0],[r,r*.15],[r,r*3],[r*.63,r*3]],metal);
  axial(cusa,[[r,r*2.9],[r*1.13,r*4],[r*1.13,shaftLength],[r,shaftLength]],metal);
  axial(cusa,[[r*1.12,shaftLength*.68],[r*1.53,shaftLength*.7],[r*1.53,shaftLength*.85],[r*1.12,shaftLength*.88]],grip);
  axial(cusa,[[r*1.12,shaftLength*.91],[r*2.1,shaftLength],[r*2.7,shaftLength+r*2],[r*2.7,shaftLength+r*6],[0,shaftLength+r*6]],grip);
  axial(cusa,[[r*2.55,shaftLength+r*1.8],[r*2.76,shaftLength+r*2],[r*2.76,shaftLength+r*2.45],[r*2.65,shaftLength+r*2.6]],accent);
  const generic=(id:string,radius:number,length:number)=>{
    const mesh=surgicalToolMesh({id,name:id,category:'field',geometry:id,materials:[],effect:'illustrative',radiusMm:radius,shaftRadiusMm:radius*.69,workingLengthMm:length},
      {instrument:id,position:[0,0,0],quaternion:[1,0,0,0],active:true,inserted:true,grip:[],parameters:{}});
    const group=new THREE.Group();group.add(mesh);return group;
  };
  const drill=generic('cranial_drill',Math.max(.85,opening*.09),opening*1.18);
  const scalpel=generic('scalpel',Math.max(.7,opening*.075),opening*.95);
  const blade=new THREE.Group();
  const width=Math.max(2,opening*.38), length=Math.max(4,opening*.58), t=Math.max(.35,width*.08);
  const outline=new THREE.Shape();outline.moveTo(-width*.42,length);outline.lineTo(-width*.5,width*.28);
  outline.quadraticCurveTo(-width*.5,0,0,0);outline.quadraticCurveTo(width*.5,0,width*.5,width*.28);
  outline.lineTo(width*.42,length);outline.closePath();
  const pad=new THREE.Mesh(new THREE.ExtrudeGeometry(outline,{depth:t,bevelEnabled:true,bevelSegments:2,bevelSize:t*.25,bevelThickness:t*.25,steps:1}),metal);
  pad.rotation.set(Math.PI/2,0,Math.PI/2,'ZXY');pad.position.x=t*.5;blade.add(pad);
  axial(blade,[[0,length*.88],[t*1.8,length],[t*1.8,length+opening*.8],[width*.45,length+opening],[width*.45,length+opening*1.42],[0,length+opening*1.42]],grip);
  return {drill,scalpel,blade,cusa};
}
