import type {Point3} from './viewer/coordinates';
import type {SurgicalResponse,SurgicalView} from './surgical-types';
import type {InteractivePlanningOverlay,InteractivePlanningRoute} from './interactive-planning-data';
import {planningBinding} from './interactive-planning-data.ts';
import {cranialComparison} from './cranial-workflow-data.ts';

export interface CranialRecordedPath {actionIndex:number;operation:string;slot:string;instrument:string;status:'accepted'|'partial';stateBefore:string;stateAfter:string;pointsRasMm:Point3[]}
export interface CranialSourceStructure {id:string;label:string;kind:'neural'|'nonneural';evidenceKind:'generated-engineering';reviewStatus:string;provenance:string;sourceCells:number[];pointsRasMm:Point3[];displayTruncated:boolean;removedMm3:number;cutSourceMm3:number}
export interface CranialRecordedGeometry {
  schema:'resectionlab.recorded-cranial-geometry.v1';caseHash:string;sourceSceneId:string;sourceStateId:string;rootSourceStateId:string;sceneId:string;policyHash:string;branchId:string;finalStateId:string;nativeActions:number;actionHistoryHash:string;geometrySeal:string;
  harmBounds:{maximumNeuralRemovalMm3:number;maximumNeuralCutSourceMm3:number;maximumBoneRemovalMm3:number;maximumNonbonePreparationRemovalMm3:number;maximumModeledElapsedSeconds:number;maximumStructureRemovalMm3:Record<string,number>;maximumStructureCutSourceMm3:Record<string,number>};
  cleanup:{debrisPresentMm3:number;debrisRemovedMm3:number;fluidPresentMm3:number;activeBleedingSources:number};
  acceptedActions:Array<{actionIndex:number;operation:string;slot:string;instrument:string;status:'accepted'|'partial';stateBefore:string;stateAfter:string;elapsedSeconds:number}>;acceptedStateIds:string[];
  paths:CranialRecordedPath[];omittedPartialActions:number[];structures:CranialSourceStructure[];targetSourceCells:number[];targetPointsRasMm:Point3[];residualSourceCells:number[];residualPointsRasMm:Point3[];targetDisplayTruncated:boolean;scope:string;
}
const point=(p:unknown):p is Point3=>Array.isArray(p)&&p.length===3&&p.every(n=>typeof n==='number'&&Number.isFinite(n));
const points=(p:unknown):p is Point3[]=>Array.isArray(p)&&p.length<=4096&&p.every(point);
const cells=(v:unknown):v is number[]=>Array.isArray(v)&&v.length<=2_000_000&&v.every((x,i)=>Number.isSafeInteger(x)&&x>=0&&(i===0||x>v[i-1]));
/** Backend cold validation remeasures these facts. The renderer also joins all
 * displayed identities and rejects edited/mismatched source or endpoint data. */
export function checkedCranialGeometry(response:SurgicalResponse,row:NonNullable<SurgicalResponse['cranialPlanning']>['alternatives'][number]):CranialRecordedGeometry|null {
  const g=row.recordedGeometry,c=response.cranialPlanning;
  if(!g||!c||g.schema!=='resectionlab.recorded-cranial-geometry.v1'||g.caseHash!==response.caseHash||g.sourceSceneId!==c.sourceSceneId||g.sourceStateId!==c.sourceStateId||g.sceneId!==row.sceneId||g.finalStateId!==row.finalStateId||g.branchId!==row.branchId||g.nativeActions!==row.nativeActions||g.policyHash!==c.procedurePolicy?.policyHash||!g.geometrySeal||!g.actionHistoryHash||!g.rootSourceStateId||!Array.isArray(g.paths)||g.paths.length>row.nativeActions||!Array.isArray(g.structures)||g.structures.length>64||!points(g.targetPointsRasMm)||!points(g.residualPointsRasMm)||!cells(g.targetSourceCells)||!cells(g.residualSourceCells)||g.targetSourceCells.length!==row.goalAssessment.originalTargetCells||g.residualSourceCells.length!==row.goalAssessment.residualOriginalTargetCells)return null;
  const targetCells=new Set(g.targetSourceCells);
  if(!Array.isArray(g.acceptedStateIds)||g.acceptedStateIds.length!==row.nativeActions+1||g.acceptedStateIds[0]!==g.rootSourceStateId||g.acceptedStateIds.at(-1)!==g.finalStateId||!Array.isArray(g.acceptedActions)||g.acceptedActions.length!==row.nativeActions||g.acceptedActions.some((a,i)=>a.actionIndex!==i||!['accepted','partial'].includes(a.status)||a.stateBefore!==g.acceptedStateIds[i]||a.stateAfter!==g.acceptedStateIds[i+1]||!Number.isFinite(a.elapsedSeconds)||a.elapsedSeconds<0))return null;
  if(!g.cleanup||![g.cleanup.debrisPresentMm3,g.cleanup.debrisRemovedMm3,g.cleanup.fluidPresentMm3,g.cleanup.activeBleedingSources].every(v=>typeof v==='number'&&Number.isFinite(v)&&v>=0)||!Array.isArray(g.omittedPartialActions)||g.omittedPartialActions.some(i=>!Number.isSafeInteger(i)||i<0||i>=row.nativeActions)||g.targetPointsRasMm.length!==Math.min(g.targetSourceCells.length,4096)||g.residualPointsRasMm.length!==Math.min(g.residualSourceCells.length,4096)||g.targetDisplayTruncated!==(g.targetSourceCells.length>4096)||g.residualSourceCells.some(c=>!targetCells.has(c)))return null;
  if(g.paths.some(p=>!Number.isSafeInteger(p.actionIndex)||p.actionIndex<0||p.actionIndex>=row.nativeActions||!['accepted','partial'].includes(p.status)||!p.stateBefore||!p.stateAfter||!points(p.pointsRasMm)||p.pointsRasMm.length<2||g.acceptedStateIds[p.actionIndex]!==p.stateBefore||g.acceptedStateIds[p.actionIndex+1]!==p.stateAfter))return null;
  if(g.structures.some(s=>!s.id||!['neural','nonneural'].includes(s.kind)||s.evidenceKind!=='generated-engineering'||!cells(s.sourceCells)||!points(s.pointsRasMm)||s.pointsRasMm.length!==Math.min(s.sourceCells.length,4096)||s.displayTruncated!==(s.sourceCells.length>4096)||s.removedMm3!==(row.sequenceOutcome.dimensions.structureRemovedMm3 as Record<string,{cumulative:number}>|undefined)?.[s.id]?.cumulative||s.cutSourceMm3!==(row.sequenceOutcome.dimensions.structureCutSourceMm3 as Record<string,{cumulative:number}>|undefined)?.[s.id]?.cumulative))return null;
  return g;
}
export function cranialRecordedOverlay(response:SurgicalResponse|null,view:SurgicalView|null):InteractivePlanningOverlay|undefined {
  if(!response||!view||response.view!==view)return;
  const comparison=cranialComparison(response);if(!comparison||!['resectionlab.cranial-strategy-comparison.v2','resectionlab.cranial-strategy-comparison.v3'].includes(comparison.schema))return;
  const current=comparison.alternatives.find(r=>r.branchId===response.branchId);
  // The intact common checkpoint or an exact saved branch/replay prefix owns
  // the overlay. A user-edited continuation cannot retain recorded guides.
  const source=response.branchId===comparison.sourceBranchId&&view.scene.sceneId===comparison.sourceSceneId&&view.stateId===comparison.sourceStateId&&response.position===0;
  if(!source&&(!current||view.scene.sceneId!==current.sceneId||response.position>current.nativeActions||(!response.playback&&view.stateId!==current.finalStateId&&view.stateId!==current.recordedGeometry?.rootSourceStateId&&!current.recordedGeometry?.acceptedStateIds?.includes(view.stateId))))return;
  const selected=current??comparison.alternatives.find(r=>r.branchId===(comparison.historicalInspectionOnly?comparison.historicalSelectedBranchId:comparison.selectedBranchId));
  if(!selected)return;
  const g=checkedCranialGeometry(response,selected);if(!g)return;
  const routes:InteractivePlanningRoute[]=[];
  comparison.alternatives.forEach((row,colorIndex)=>{
    const geometry=checkedCranialGeometry(response,row);if(!geometry)return;
    for(const path of geometry.paths.filter(p=>p.instrument==='cusa'&&p.operation==='apply')){
      const at=(position:Point3)=>({position,quaternion:[0,0,0,1] as [number,number,number,number]});
      routes.push({id:`${row.branchId}:${path.actionIndex}`,label:row.label,entryPose:at(path.pointsRasMm[0]),outsidePose:at(path.pointsRasMm[0]),targetPose:at(path.pointsRasMm.at(-1)!),pointsRasMm:path.pointsRasMm,selected:row.branchId===selected.branchId,colorIndex,actionProgram:[],metrics:{},scope:g.scope});
    }
  });
  const center=g.targetPointsRasMm[0]??[0,0,0];
  return {binding:planningBinding(response),constraints:{target:{centerMm:center,radiusMm:0},entry:{centerMm:center,radiusMm:0},tool:{instrument:'cusa',radiusMm:1.5},requiredEvidence:[],candidateCount:comparison.alternatives.length,nonTargetClearanceMm:0},routes,selectedId:null,limitingStructures:[],recorded:{comparisonSeal:comparison.comparisonSeal,geometrySeal:g.geometrySeal,policyHash:g.policyHash,selectedBranchId:selected.branchId,structures:g.structures,targetPointsRasMm:g.targetPointsRasMm,residualPointsRasMm:g.residualPointsRasMm,scope:g.scope}};
}
