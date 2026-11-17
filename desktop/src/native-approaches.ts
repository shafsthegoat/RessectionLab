import type {Point3} from './viewer/coordinates';
import type {ViewerRoute} from './viewer/contracts';
export interface ApproachStructure {
  layerId:string;label:string;kind:string;available:boolean;
  clearanceLowerBoundMm?:number|null;intersectsOrUnresolved?:boolean;confirmedIntersection?:boolean;
  closestPointRasMm?:Point3;nearestPoint?:Point3;[key:string]:unknown;
}
export interface NativeApproach {
  id:string;name:string;sourceSnapshotId:string;parameterHash:string;
  entry:Point3;target:Point3;radiusMm:number;direction:Point3;
  entryPatch?:{center:Point3;normal:Point3;radiusMm:number};
  constraints:{maxLengthMm?:number;minClearanceMm?:number;excludedLayerIds?:string[]};
  assessmentStatus:'current'|'stale'|'unassessed';
  measurements?:{lengthMm:number;orientation:{azimuthDeg:number;elevationDeg:number};structures:ApproachStructure[];
    coverage:Record<string,unknown>;targets:Array<Record<string,unknown>>;normalTraversal:Record<string,unknown>;
    status:'known-conflict'|'incomplete'|'no-detected-conflict';unknowns:string[];assumptions:string[];constraintFailures?:string[];[key:string]:unknown};
  [key:string]:unknown;
}
export interface ApproachResponse {snapshotId:string;approaches:NativeApproach[];selectedId?:string;proposalReport?:{evaluated:number;accepted:number;rejected:number;reasons:unknown;selectionRule:string;whyFewer?:string|null};}
export const approachStatus=(a:NativeApproach)=>a.assessmentStatus!=='current'?a.assessmentStatus:a.measurements?.status??'unassessed';

/** Separate observations from evidence gaps and user-declared constraint failures. */
export function approachAssessment(a:NativeApproach){
  const m=a.assessmentStatus==='current'?a.measurements:undefined;
  const structures=m?.structures??[],assessed=structures.filter(s=>s.available),intersections=assessed.filter(s=>s.confirmedIntersection);
  const observed=intersections.length?'intersection':assessed.length?'no-intersection':'unavailable';
  const observedLabel=observed==='intersection'?'Detected anatomical intersection':observed==='no-intersection'?'No detected intersection in assessed anatomy':'Geometric measurement unavailable';
  const unavailable=structures.filter(s=>!s.available),unresolved=assessed.filter(s=>s.intersectsOrUnresolved&&!s.confirmedIntersection);
  const limitations=[...(m?.unknowns??[])];
  if(m&&!assessed.length)limitations.push('No supplied anatomical structure could be measured');
  for(const s of unavailable)if(!limitations.some(text=>text.includes(s.layerId)||text.includes(s.label)))limitations.push(`${s.label}: measurement unavailable`);
  for(const s of unresolved)if(!limitations.some(text=>text.includes(s.layerId)||text.includes(s.label)))limitations.push(`${s.label}: conservative contact unresolved`);
  if(m?.coverage?.withinReferenceEnvelope===false&&!limitations.some(text=>text.includes('outside reference')))limitations.push('Corridor envelope extends outside the reference image field');
  const incomplete=!!m&&(m.status==='incomplete'||limitations.length>0||unavailable.length>0||unresolved.length>0);
  const completeness=a.assessmentStatus==='stale'?'stale':!m?'unavailable':incomplete?'incomplete':'complete';
  const completenessLabel=completeness==='stale'?'Source changed · reassessment required':completeness==='unavailable'?'Assessment unavailable':incomplete?'Incomplete evidence':'Recorded evidence assessed';
  const finite=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value);
  const bounds=assessed.map(s=>s.clearanceLowerBoundMm).filter(finite);
  const targets=(m?.targets??[]).filter(t=>t.available&&finite(t.coveredVoxelCenterVolumeMm3));
  return {observed,observedLabel,completeness,completenessLabel,assessedCount:assessed.length,totalCount:structures.length,
    intersections,unavailable,unresolved,limitations:limitations.map(text=>structures.reduce((result,s)=>result.replaceAll(`${s.layerId}:`,`${s.label}:`),text)),
    constraintFailures:m?.constraintFailures??[],minimumBoundMm:bounds.length?Math.min(...bounds):null,
    targetReachedMm3:targets.length?targets.reduce((sum,t)=>sum+Number(t.coveredVoxelCenterVolumeMm3),0):null,
    traversalMm3:m?.normalTraversal?.available&&finite(m.normalTraversal.conservativeVolumeUpperBoundMm3)?m.normalTraversal.conservativeVolumeUpperBoundMm3:null};
}

export function approachRoutes(approaches:NativeApproach[],selectedId:string|null):ViewerRoute[]{const displayed=approaches.slice(0,3),selected=approaches.find(a=>a.id===selectedId);if(selected&&!displayed.includes(selected))displayed[2]=selected;return displayed.map((a,index)=>({route_id:a.id,comparisonSlot:(['A','B','C'] as const)[index],scope:'geometric-approach',label:a.name,selected:a.id===selectedId,entry_mm:a.entry,target_mm:a.target,
  tool:{working_length_mm:Math.hypot(...a.target.map((v,i)=>v-a.entry[i])),tip_length_mm:0,shaft_radius_mm:a.radiusMm,tip_radius_mm:a.radiusMm},
  window:a.entryPatch?{center_mm:a.entryPatch.center,normal_inward:a.entryPatch.normal,radius_mm:a.entryPatch.radiusMm}:undefined}));}

/** Round down so a displayed lower bound never overstates source clearance. */
export const lowerBoundText=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?(Math.floor(value*10)/10).toFixed(1):'—';
