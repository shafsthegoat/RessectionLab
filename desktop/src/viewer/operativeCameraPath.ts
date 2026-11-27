import * as THREE from 'three';
import type {OperativeField,OperativePhase} from '../operative-field';
import type {OperativeViewChoice} from './operativeSectionContext';

const up=new THREE.Vector3(0,0,1);
const sourceDirection=new THREE.Vector3(.20,-.37,1).normalize();
// View slightly across the shaft while retaining an open line along the planned entry.
const entryDirection=new THREE.Vector3(.22,-.17,1).normalize();
export const entryApproachEnd=.55;
const ease=(value:number)=>THREE.MathUtils.smootherstep(value,0,1);
type Fit={center:THREE.Vector3;size:number};

/** The bounding sphere of a rotated volume leaves much of the entry view unused.
 * Fit the actual registered surfaces along the initial camera axes. */
export function operativeSourceFit(group:THREE.Group,plannedEntry?:THREE.Vector3):Fit|null {
  const right=new THREE.Vector3().crossVectors(up,sourceDirection).normalize();
  const vertical=new THREE.Vector3().crossVectors(sourceDirection,right);
  const bounds=new THREE.Box3(),vertex=new THREE.Vector3(),projected=new THREE.Vector3();
  group.updateMatrixWorld(true);
  group.traverse(object=>{
    if(!(object instanceof THREE.Mesh))return;
    const points=object.geometry.getAttribute('position');
    for(let i=0;i<points.count;i++){
      vertex.fromBufferAttribute(points,i).applyMatrix4(object.matrixWorld);
      bounds.expandByPoint(projected.set(vertex.dot(right),vertex.dot(vertical),vertex.dot(sourceDirection)));
    }
  });
  if(bounds.isEmpty())return null;
  if(plannedEntry)bounds.expandByPoint(projected.set(plannedEntry.dot(right),plannedEntry.dot(vertical),plannedEntry.dot(sourceDirection)));
  const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());
  return {center:right.clone().multiplyScalar(center.x).addScaledVector(vertical,center.y).addScaledVector(sourceDirection,center.z),
    size:1.19*Math.max(size.x,size.y,1)};
}

export function operativeCameraPose(field:OperativeField,opening:number,source:Fit|null,phase:OperativePhase,
  progress:number,retract:boolean,view:OperativeViewChoice) {
  const first=(['bone','dura','brain','tumor'] as const).find(id=>field.stations[id]!==null&&(id!=='brain'||retract));
  const entry=field.stations.bone??field.stations.dura??field.stations.brain??field.stations.tumor??0;
  // Frame the actual aperture with room for the tool and intact rim. The
  // display sampling crop may be much wider than the operative opening.
  const localSize=Math.max(3.5*opening,8*field.spacingMm);
  const target=new THREE.Vector3(0,0,entry),bearing=entryDirection.clone();
  let size=localSize,marker=0;
  if(view==='exterior')return {...(source??{center:target,size}),bearing:sourceDirection,marker:1};
  if(view==='section'){
    const station=(phase==='overview'?null:field.stations[phase])??entry;
    return {center:new THREE.Vector3(0,0,station),size:Math.max(opening*4,field.radiusMm*1.82),
      bearing:new THREE.Vector3(.36,-.91,.90).normalize(),marker};
  }
  // Bring the entry toward center in screen space while magnification changes.
  const approach=phase==='overview'?0:phase===first?ease(progress/entryApproachEnd):1;
  if(source&&approach<1){
    size=Math.exp(THREE.MathUtils.lerp(Math.log(source.size),Math.log(localSize),approach));
    target.addScaledVector(source.center.clone().sub(target),(1-approach)*size/source.size);
    bearing.copy(sourceDirection).lerp(entryDirection,approach).normalize();
    marker=1-ease((approach-.30)/.52);
  }
  return {center:target,size,bearing,marker};
}
