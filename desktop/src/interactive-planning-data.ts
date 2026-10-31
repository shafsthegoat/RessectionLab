import type {Point3} from './viewer/coordinates';
import type {SurgicalPose,SurgicalResponse,SurgicalView} from './surgical-types';
export interface InteractivePlanningConstraints {
  target:{centerMm:Point3;radiusMm:number};
  entry:{centerMm:Point3;radiusMm:number};
  tool:{instrument:'cusa';radiusMm:number};
  requiredEvidence:string[];candidateCount:number;nonTargetClearanceMm:number;
  minEntrySeparationMm?:number;minAngleDeg?:number;maxCandidates?:number;maxWallSeconds?:number;
}
export interface InteractivePlanningAvailability {available:boolean;reason:string|null;defaults:InteractivePlanningConstraints|null;evidence:Record<string,unknown>;scope:string}
export interface InteractivePlanningRoute {
  id:string;label:string;entryPose:SurgicalPose;targetPose:SurgicalPose;outsidePose:SurgicalPose;
  actionProgram:unknown[];metrics:Record<string,unknown>;separationComparisons?:Array<{routeId:string;entrySeparationMm:number;angleDeg:number}>;requiredTargetRemoval?:{sourceCells:number[];additionalNonTargetTraversalAllowedMm3:number;status:string};scope:string;
}
export interface InteractivePlanningLimit {name:string;positionMm:Point3;reason:string}
export interface InteractivePlanningReport {
  schema:string;status:string;current?:boolean;revision?:number;historyCount?:number;sourceBranchId?:string;sourcePosition?:number;applications?:unknown[];sourceSceneId:string;startStateId:string;prefixActions:number;
  targetSelection?:Record<string,unknown>;constraints:InteractivePlanningConstraints;evidence:Record<string,unknown>;routes:InteractivePlanningRoute[];
  missingRequiredEvidence?:string[];rejections:Array<Record<string,unknown>>;reportSeal:string;limitingStructures?:InteractivePlanningLimit[];
}
export interface InteractivePlanningOverlay {
  binding:string;constraints:InteractivePlanningConstraints;routes:InteractivePlanningRoute[];selectedId:string|null;
  limitingStructures:InteractivePlanningLimit[];
}
export function planningBinding(response:SurgicalResponse|null|undefined):string {
  return response?JSON.stringify([response.caseHash,response.view.scene.sceneId,response.view.stateId,response.branchId,response.position]):'';
}
/** Stable constraints comparison never substitutes for the explicit draft revision. */
export function planningConstraintKey(c:InteractivePlanningConstraints):string {
  const canonical=(value:unknown):unknown=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b)).map(([key,item])=>[key,canonical(item)])):value;
  return JSON.stringify(canonical({...c,requiredEvidence:[...c.requiredEvidence].sort()}));
}
export function planningDraftIssue(c:InteractivePlanningConstraints):string|null {
  for(const [name,region] of [['Target',c.target],['Entry',c.entry]] as const){
    if(region.centerMm.length!==3||region.centerMm.some(n=>!Number.isFinite(n)))return `${name} needs three finite RAS coordinates.`;
    if(!Number.isFinite(region.radiusMm)||region.radiusMm<.5||region.radiusMm>30)return `${name} radius must be between 0.5 and 30 mm.`;
  }
  if(!Number.isFinite(c.tool.radiusMm)||c.tool.radiusMm<.1||c.tool.radiusMm>6)return 'CUSA radius must be between 0.1 and 6 mm.';
  if(!Number.isFinite(c.nonTargetClearanceMm)||c.nonTargetClearanceMm<0||c.nonTargetClearanceMm>2)return 'Mapped non-target clearance must be between 0 and 2 mm.';
  if(!Number.isInteger(c.candidateCount)||c.candidateCount<2||c.candidateCount>6)return 'Request between 2 and 6 approaches.';
  return null;
}
export function currentPlanningReport(response:SurgicalResponse|null|undefined,draft:InteractivePlanningConstraints,submitted:{binding:string;revision:number}|null,revision:number):InteractivePlanningReport|null {
  const report=response?.interactivePlanning;
  if(!response||!report||report.current===false||!submitted||submitted.binding!==planningBinding(response)||submitted.revision!==revision||
    response.readOnly||response.playback||response.verification?.readOnly||report.sourceSceneId!==response.view.scene.sceneId||
    report.startStateId!==response.view.stateId||report.prefixActions!==response.position||planningConstraintKey(report.constraints)!==planningConstraintKey(draft))return null;
  return report;
}
export function selectedPlanningRoute(report:InteractivePlanningReport|null,id:string|null):InteractivePlanningRoute|null {
  return report?.routes.find(route=>route.id===id)??null;
}
export function planningOverlayForView(overlay:InteractivePlanningOverlay|null,response:SurgicalResponse|null,view:SurgicalView|null):InteractivePlanningOverlay|undefined {
  return overlay&&response&&view&&overlay.binding===planningBinding(response)&&response.view===view?overlay:undefined;
}
export const PLANNING_ROUTE_COLORS=['#70e0ff','#ffcf70','#caa1ff','#9beb9b','#ff94c2','#d3e7ef'];

export function planningRoutePoints(route:InteractivePlanningRoute):Point3[]{
  return [route.outsidePose.position,route.entryPose.position,route.targetPose.position];
}
export interface PlanningSliceSegment {id:string;color:string;selected:boolean;points:[Point3,Point3]}
/** Every shown segment intersects this MRI slab; full projected routes would imply false slice occupancy. */
export function planningSliceSegments(overlay:InteractivePlanningOverlay,axis:0|1|2,position:number,halfThickness:number):PlanningSliceSegment[]{
  const out:PlanningSliceSegment[]=[];
  if(!Number.isFinite(position)||!Number.isFinite(halfThickness)||halfThickness<0)return out;
  overlay.routes.forEach((route,index)=>{
    const points=planningRoutePoints(route);
    for(let i=1;i<points.length;i++){
      const a=points[i-1],b=points[i],delta=b[axis]-a[axis];let low=0,high=1;
      if(Math.abs(delta)<1e-12){if(Math.abs(a[axis]-position)>halfThickness)continue;}
      else {const t1=(position-halfThickness-a[axis])/delta,t2=(position+halfThickness-a[axis])/delta;low=Math.max(0,Math.min(t1,t2));high=Math.min(1,Math.max(t1,t2));if(low>high)continue;}
      const at=(t:number)=>a.map((n,j)=>j===axis?position:n+t*(b[j]-n)) as Point3;
      out.push({id:`${route.id}:${i}`,color:PLANNING_ROUTE_COLORS[index%PLANNING_ROUTE_COLORS.length],selected:route.id===overlay.selectedId,points:[at(low),at(high)]});
    }
  });return out;
}
export function planningSphereSliceRadius(center:Point3,radius:number,axis:0|1|2,position:number):number|null {
  const d=center[axis]-position;return Math.abs(d)<=radius?Math.sqrt(Math.max(0,radius*radius-d*d)):null;
}
