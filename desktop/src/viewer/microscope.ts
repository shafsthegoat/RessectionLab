import * as THREE from 'three';
import {transformPoint,type Point3} from './coordinates.ts';
import {sourceCenter} from './exposureAnatomy.ts';
import type {SurgicalView} from '../surgical-types';
import type {MicroscopeSettings,ViewerCameraState,ViewerWorkspaceSettings} from './workspaceSettings';

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
export function microscopeFov(s:Pick<MicroscopeSettings,'workingDistanceMm'|'fieldMm'>):number {
  return THREE.MathUtils.radToDeg(2*Math.atan(s.fieldMm/(2*s.workingDistanceMm)));
}


/** Frame the declared aperture from outside the retained skin. This uses
 * its source cells so parked flaps and an exchanged tool never move the lens. */
export function exposureMicroscope(view:SurgicalView):MicroscopeSettings|undefined{
  const bodies=view.operativeExposureV1?.bodies;
  if(!bodies?.scalp.cells.length||!bodies.dura.cells.length)return;
  const points=(cells:number[])=>cells.map(id=>new THREE.Vector3(...sourceCenter(id,view.scene.shape,view.scene.affine)));
  const mean=(v:THREE.Vector3[])=>v.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).divideScalar(v.length);
  const skin=points(bodies.scalp.cells),top=mean(skin),dura=mean(points(bodies.dura.cells));
  const normal=view.operativeExposureV1?.exteriorNormalRas;
  const outward=normal?new THREE.Vector3(...normal).normalize():top.clone().sub(dura).normalize();
  const radius=skin.reduce((r,p)=>Math.max(r,p.distanceTo(top)),0);
  return {target:top.lerp(dura,.70).toArray(),outward:outward.toArray(),angleDeg:18,azimuthDeg:15,
    workingDistanceMm:160,fieldMm:Math.min(120,Math.max(22,Math.round(2.85*radius))),focusOffsetMm:0,light:2.1};
}

/** External view of an operative scene from outside its declared opening.
 * The camera looks down the scene's outward normal, tilted toward the patient's
 * superior side, and fits the working grid (34 degree external field). */
export function exposureOverviewCamera(view:SurgicalView,fovDeg=34):ViewerCameraState|undefined{
  const normal=view.operativeExposureV1?.exteriorNormalRas;
  if(!normal)return;
  const outward=new THREE.Vector3(...normal).normalize();
  if(!(outward.length()>.5))return;
  const [nx,ny,nz]=view.scene.shape,corners:THREE.Vector3[]=[];
  for(const i of [0,nx-1])for(const j of [0,ny-1])for(const k of [0,nz-1])
    corners.push(new THREE.Vector3(...transformPoint(view.scene.affine,[i,j,k])));
  const centre=corners.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).divideScalar(corners.length);
  const radius=corners.reduce((r,p)=>Math.max(r,p.distanceTo(centre)),0);
  const reference=Math.abs(outward.z)>.95?new THREE.Vector3(0,1,0):new THREE.Vector3(0,0,1);
  const right=new THREE.Vector3().crossVectors(reference,outward).normalize();
  // Rotate toward the reference so the head below the opening stays in view.
  let direction=outward.clone().applyAxisAngle(right,-THREE.MathUtils.degToRad(28)).normalize();
  // An inserted instrument's handpiece leaves the opening outward; the camera turns
  // about the normal away from it so the handpiece does not fill the foreground.
  const handles=Object.values(view.slots??{}).filter(slot=>slot?.inserted&&slot.quaternion?.length===4)
    .map(slot=>new THREE.Vector3(0,0,-1).applyQuaternion(new THREE.Quaternion(...slot.quaternion).normalize()));
  if(handles.length){
    const clearance=(d:THREE.Vector3)=>Math.min(...handles.map(h=>d.angleTo(h)));
    const minimum=THREE.MathUtils.degToRad(55);
    if(clearance(direction)<minimum){
      let best=direction,score=-Infinity;
      for(let k=0;k<16;k++)for(const tilt of [28,36,44]){
        const candidate=direction.clone().applyAxisAngle(outward,k*Math.PI/8);
        const turned=outward.clone().applyAxisAngle(new THREE.Vector3().crossVectors(outward,candidate).normalize(),THREE.MathUtils.degToRad(tilt));
        // Prefer clearing the handpiece, then staying near the default view.
        const value=Math.min(clearance(turned),minimum)*4-turned.angleTo(direction)*.5;
        if(value>score){score=value;best=turned;}
      }
      direction=best.normalize();
    }
  }
  const distance=.86*radius/Math.sin(THREE.MathUtils.degToRad(fovDeg/2));
  const up=reference.clone().addScaledVector(direction,-reference.dot(direction)).normalize();
  return {target:centre.toArray() as Point3,position:centre.clone().addScaledVector(direction,distance).toArray() as Point3,up:up.toArray() as Point3};
}

/** Establish the microscope once per scene. An explicit saved view takes
 * precedence, including an external camera or a pinned inspection section. */
export function defaultSurgeonView(view:SurgicalView,settings:ViewerWorkspaceSettings,annotationCenter:Point3|null=null):ViewerWorkspaceSettings {
  if(settings.operativeView)return settings;
  const slots=Object.values(view.slots);
  const slot=slots.find(item=>item.inserted&&item.active)??slots.find(item=>item.inserted)??slots[0];
  if(!slot)return settings;
  const blade=view.planarBladeManeuver?.opticalFrame;
  // Tool +Z points into the working field. A small off-axis angle keeps the
  // tip visible beside the shaft without moving it or exposing hidden tissue.
  const outward=new THREE.Vector3(0,0,-1).applyQuaternion(new THREE.Quaternion(...slot.quaternion).normalize());
  const center=blade?.targetMm??(slot.inserted?slot.position:annotationCenter)
    ??transformPoint(view.scene.affine,view.scene.shape.map(n=>(n-1)/2) as Point3);
  const microscope:MicroscopeSettings={target:[...center],outward:blade?[...blade.outward]:outward.toArray(),
    angleDeg:blade?30:12,azimuthDeg:0,workingDistanceMm:160,fieldMm:blade?12:42,focusOffsetMm:0,light:2.5,
    ...exposureMicroscope(view),...settings.microscope,inspectionTransparency:false,inspectionCutaway:false};
  return {...settings,operativeView:'surgeon',microscope,externalCamera:settings.camera,camera:microscopeCamera(microscope),
    layout:'3d-focus',showPlane:false,contextStyle:'uniform',clipping:undefined,clipBox:undefined,obliqueClip:undefined,
    volumeRendering:{...settings.volumeRendering,enabled:false}};
}
