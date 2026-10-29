import * as THREE from 'three';
import type {Point3} from './coordinates';
import type {MicroscopeSettings,ViewerCameraState} from './workspaceSettings';

/** Optical controls use RAS mm only; they never enter a surgical request. */
export function microscopeCamera(s:MicroscopeSettings):ViewerCameraState {
  const outward=new THREE.Vector3(...s.outward).normalize();
  if(!Number.isFinite(outward.length())||outward.length()<.5)throw Error('Invalid microscope direction');
  const reference=Math.abs(outward.z)>.95?new THREE.Vector3(0,1,0):new THREE.Vector3(0,0,1);
  const right=new THREE.Vector3().crossVectors(reference,outward).normalize();
  const tilt=right.clone().applyAxisAngle(outward,THREE.MathUtils.degToRad(s.azimuthDeg));
  const direction=outward.clone().applyAxisAngle(tilt,THREE.MathUtils.degToRad(s.angleDeg));
  const target=new THREE.Vector3(...s.target).addScaledVector(outward,s.focusOffsetMm);
  const up=reference.clone().addScaledVector(direction,-reference.dot(direction)).normalize();
  return {target:target.toArray() as Point3,position:target.clone().addScaledVector(direction,s.workingDistanceMm).toArray() as Point3,up:up.toArray() as Point3};
}
export function microscopeFov(s:MicroscopeSettings):number {
  return THREE.MathUtils.radToDeg(2*Math.atan(s.fieldMm/(2*s.workingDistanceMm)));
}
