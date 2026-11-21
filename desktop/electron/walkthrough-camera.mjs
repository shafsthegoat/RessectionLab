import {Vector3,Quaternion,Ray} from 'three';

const point=values=>new Vector3(...values);
const radians=degrees=>degrees*Math.PI/180;

// Select only the optical azimuth. The same accepted capsules used by native
// collision and drawing determine whether a parked instrument masks the work.
export function clearWorkingView({camera,fieldMm,aspect,azimuthDeg,outward,slots,targets,raisedSheet=[],supportPair}){
  const target=point(camera.target),position=point(camera.position),axis=point(outward).normalize();
  const distance=position.distanceTo(target),direction=position.clone().sub(target).normalize();
  const reference=Math.abs(axis.dot(new Vector3(0,1,0)))>.9?new Vector3(1,0,0):new Vector3(0,1,0);
  const parked=[],parkedDirections=[];
  for(const [side,slot] of Object.entries(slots)){
    if(slot.inserted)continue;
    const q=new Quaternion(...slot.quaternion).normalize(),origin=point(slot.position);
    const lateral=origin.clone().sub(target);lateral.addScaledVector(axis,-lateral.dot(axis));
    if(lateral.length()>0)parkedDirections.push(lateral.normalize());
    for(const capsule of slot.contactGeometry??[]){
      const world=v=>point(v).applyQuaternion(q).add(origin);
      parked.push({side,part:capsule.part,a:world(capsule.a),b:world(capsule.b),radius:capsule.radiusMm});
    }
  }
  const subjects=[target,...targets.map(point)],sheetPoints=raisedSheet.map(point);
  const mirrorDistance = degree => Math.min(...[azimuthDeg,-azimuthDeg].map(a=>Math.abs(((degree-a+540)%360)-180)));
  const candidates=[...new Set([azimuthDeg,-azimuthDeg,90,-90,130,-130,45,-45,180,0])];
  let best;
  for(const candidate of candidates){
    const outwardView=direction.clone().applyAxisAngle(axis,radians(candidate-azimuthDeg));
    const origin=target.clone().addScaledVector(outwardView,distance);
    const forward=outwardView.clone().negate(),right=forward.clone().cross(reference).normalize(),up=right.clone().cross(forward).normalize();
    let closest=Infinity,frameMargin=Infinity,obstruction=null;
    for(const subject of subjects){
      const relative=subject.clone().sub(origin),depth=relative.dot(forward);
      const halfField=fieldMm*.5*depth/distance;
      const nx=relative.dot(right)/(halfField*aspect),ny=relative.dot(up)/halfField;
      frameMargin=Math.min(frameMargin,1-Math.abs(nx),1-Math.abs(ny));
      const length=relative.length(),ray=new Ray(origin,relative.normalize());
      for(const c of parked){
        const onRay=new Vector3(),separation=Math.sqrt(ray.distanceSqToSegment(c.a,c.b,onRay,new Vector3()));
        if(onRay.distanceTo(origin)>length+c.radius)continue;
        const clearance=separation-c.radius;
        if(clearance<closest){closest=clearance;obstruction=c.side+' / '+c.part;}
      }
    }
    const delta=mirrorDistance(candidate)+(candidate===azimuthDeg?0:.01);
    const lateral=outwardView.clone().addScaledVector(axis,-outwardView.dot(axis)).normalize();
    const parkedSide=Math.max(0,...parkedDirections.map(dir=>dir.dot(lateral)));
    const apertureRay=new Ray(origin,forward);
    let sheetClearance=Infinity;
    for(const center of sheetPoints){
      const toward=center.clone().sub(origin).dot(forward);
      if(toward>0&&toward<distance)sheetClearance=Math.min(sheetClearance,apertureRay.distanceToPoint(center));
    }
    const sheetPenalty=Math.max(0,8-sheetClearance);
    let supportSeparationMm=null,supportPenalty=0;
    if(supportPair){
      const [a,b]=supportPair.map(point);
      const screen=p=>{
        const v=p.clone().sub(origin),scale=distance/v.dot(forward);
        return new Vector3(v.dot(right)*scale,v.dot(up)*scale,0);
      };
      supportSeparationMm=screen(a).distanceTo(screen(b));
      supportPenalty=Math.max(0,a.distanceTo(b)-supportSeparationMm);
    }
    const score=Math.min(24,closest)-supportPenalty+Math.min(0,frameMargin-.1)*70-parkedSide*24-sheetPenalty*5-delta/180;
    if(!best||score>best.score)best={azimuthDeg:candidate,score,
      parkedClearanceMm:Number.isFinite(closest)?closest:null,frameMargin,nearestParkedPart:obstruction,supportSeparationMm,
      apertureSheetClearanceMm:Number.isFinite(sheetClearance)?sheetClearance:null};
  }
  return best;
}
