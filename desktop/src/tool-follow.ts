import {createContext} from 'react';
import type {Point3} from './viewer/coordinates';
import type {SurgicalView} from './surgical-types';
import type {ViewerVolume} from './viewer/contracts';
import {rasAffine} from './viewer/coordinates.ts';
import {axisGridFrame,labelDistance,labeledCells,type LabelDistance} from './viewer/toolNavigation.ts';

const same=(a:readonly number[]|null,b:readonly number[]|null)=>!!a&&!!b&&a.every((value,i)=>Math.abs(value-b[i])<1e-6);

/** Linked-cursor following for an instrument tip (RAS mm). Each new tip moves
 * the cursor. When following stops, the cursor returns to where the user left
 * it before following began, unless the user has moved it since or the
 * study/scene (`key`) changed. */
export function createCursorFollower(read:()=>Point3|null,write:(point:Point3)=>void){
  let origin:Point3|null=null,applied:Point3|null=null,scope:string|null=null;
  return {
    follow(tip:Point3,key:string){
      if(!tip.every(Number.isFinite))return;
      if(scope!==key){origin=null;applied=null;scope=key;}
      if(same(applied,tip)&&same(read(),tip))return;
      if(applied===null){const current=read();origin=current?[...current] as Point3:null;}
      applied=[...tip] as Point3;write(applied);
    },
    stop(key:string){
      const restore=scope===key&&origin&&applied&&same(read(),applied)?origin:null;
      origin=null;applied=null;
      if(restore)write(restore);
    },
  };
}

/** The accepted tip to follow: the selected hand when its instrument is
 * inserted, otherwise the only inserted instrument. Withdrawn tools have no
 * tip in the field. */
export function acceptedToolTip(view:Pick<SurgicalView,'slots'>|null|undefined,selected:string):{hand:string;position:Point3}|null{
  if(!view)return null;
  const preferred=view.slots[selected];
  if(preferred?.inserted&&preferred.position.every(Number.isFinite))return {hand:selected,position:[...preferred.position] as Point3};
  const inserted=Object.entries(view.slots).filter(([,slot])=>slot.inserted&&slot.position.every(Number.isFinite));
  return inserted.length===1?{hand:inserted[0][0],position:[...inserted[0][1].position] as Point3}:null;
}

export type TipTargetMeasure=(tip:Point3)=>LabelDistance|null;

/** Exact signed distance from a RAS tip to the original declared target cells
 * of the source volume. Null when no target layer is present or the source
 * grid is oblique (its cells are not axis-aligned boxes). */
export function sourceTargetMeasure(volume:ViewerVolume|null|undefined,targetNames:readonly string[]):TipTargetMeasure|null{
  if(!volume||!targetNames.length)return null;
  let frame:ReturnType<typeof axisGridFrame>=null;
  try{frame=axisGridFrame(rasAffine(volume.affine,volume.frame),volume.shape);}catch{return null;}
  if(!frame)return null;
  const count=volume.shape[0]*volume.shape[1]*volume.shape[2];
  const tests=volume.compartments.filter(layer=>targetNames.includes(layer.name)).flatMap(layer=>{
    const labels=layer.labelSet===undefined?undefined:volume.labelSets?.find(set=>set.id===layer.labelSet)?.data;
    if(labels&&labels.length===count&&layer.labelValue!==undefined){const value=layer.labelValue;return [(i:number)=>labels[i]===value];}
    return layer.mask.length===count?[(i:number)=>layer.mask[i]!==0]:[];
  });
  if(!tests.length)return null;
  const member=(i:number)=>tests.some(test=>test(i));
  let cells:Int32Array|null=null;
  const {grid,fromRas}=frame;
  return tip=>{
    cells??=labeledCells(count,member);
    if(!cells.length)return null;
    return labelDistance(grid,member,fromRas(tip),{complete:true,cells,maxMm:80});
  };
}

/** Supplied by the Rehearse workspace to overlays rendered inside the viewer. */
export const TipNavigationContext=createContext<{measure:TipTargetMeasure}|null>(null);
