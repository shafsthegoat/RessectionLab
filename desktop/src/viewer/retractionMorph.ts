import * as THREE from 'three';
import {retractionField} from './retractionField.ts';

/** Relative position and normal targets for the existing route-space geometric
 * retraction. Its offset is one fixed direction times a Gaussian. The cofactor
 * of I + t a grad(w)^T is linear in t, so Three's blended relative normals
 * represent every intermediate influence without rebuilding this surface. */
export function installRetractionMorph(geometry:THREE.BufferGeometry,opening:number,station:number,
  toRoute=new THREE.Matrix4()){
  const positions=geometry.getAttribute('position'),normals=geometry.getAttribute('normal');
  if(!positions||!positions.count||opening<=0)return;
  const shift=new Float32Array(positions.count*3),normalChange=new Float32Array(shift.length);
  const basis=new THREE.Matrix3().setFromMatrix4(toRoute);
  const inverse=basis.clone().invert(),field=retractionField(opening,station);
  const direction=new THREE.Vector3(...field(0,0,0).direction).applyMatrix3(inverse);
  const gradientToLocal=basis.transpose(),point=new THREE.Vector3(),gradient=new THREE.Vector3(),n=new THREE.Vector3();
  for(let i=0;i<positions.count;i++){
    point.fromBufferAttribute(positions,i).applyMatrix4(toRoute);
    if(Math.abs(point.x)>5*opening||Math.abs(point.y)>5*opening||Math.abs(point.z-station)>5*opening)continue;
    const sample=field(point.x,point.y,point.z);
    point.set(...sample.shift).applyMatrix3(inverse).toArray(shift,i*3);
    if(!normals)continue;
    gradient.set(...sample.gradient).applyMatrix3(gradientToLocal);
    n.fromBufferAttribute(normals,i);
    const normalAlongShift=n.dot(direction);
    n.multiplyScalar(gradient.dot(direction)).addScaledVector(gradient,-normalAlongShift).toArray(normalChange,i*3);
  }
  geometry.morphAttributes.position=[new THREE.BufferAttribute(shift,3)];
  if(normals)geometry.morphAttributes.normal=[new THREE.BufferAttribute(normalChange,3)];
  geometry.morphTargetsRelative=true;
}
