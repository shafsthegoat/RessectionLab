import type {SurgicalResponse,SurgicalView,SurgicalAction} from './surgical-types';
import type {Point3} from './viewer/coordinates';
import type {CranialSourceStructure} from './cranial-recorded-geometry';
import {planningBinding} from './interactive-planning-data.ts';
import type {InteractivePlanningOverlay} from './interactive-planning-data';

export interface SequentialAttempt {
  attemptIndex:number;nativePositionBefore:number;nativePositionAfter:number;kind:string;skillId:string;
  action:SurgicalAction;toolPositionBefore:Point3;toolPositionAfter:Point3;instrumentBefore:string;instrumentAfter:string;
  effectiveParametersBefore:Record<string,number>;effectiveParametersAfter:Record<string,number>;
  status:string;reason:string|null;reasonCode:string|null;stateBefore:string;stateAfter:string;elapsedSeconds:number;
}
export interface SequentialEpisode {
  branchId:string;label:string;method:string;seed:number;checkpoint:{path:string;sha256:string;selectedUpdates:number}|null;
  originalEpisode:{path:string;sha256:string};nativeActions:number;acceptedStateIds:string[];finalStateId:string;
  snapshotHash:string;actionHistoryHash:string;attempts:SequentialAttempt[];checkedStop:boolean;
  endpoint:{classification:string;completedTask:boolean;completeOriginalTargetRemoval:boolean;validWithdrawal:boolean;cleanupComplete:boolean;truncated:boolean;configHash:string};
  progress:{originalTargetTreatmentMm3:number;residualOriginalTargetCells:number;originalTargetCells:number;originalTargetRemovedMm3:number};
  neuralRemovedMm3:number;structureRemovedMm3:Record<string,number>;structureCutSourceMm3:Record<string,number>;
  modeledSeconds:number;computation:{physicalCalls:number;planningPhysicalCalls:number;executionPhysicalCalls:number;wallSeconds:number;tier:{name:string;max_physical_calls:number;max_wall_seconds:number};previewSkillAttempts:number;previewRefusals:number;canceledRequests:number};
  cleanup:{debrisMm3:number;fluidMm3:number;debrisCollectedMm3:number};originalSkillIds:string[];
}
export interface SequentialEpisodePacket {
  schema:'resectionlab.recorded-sequential-episodes.v1';caseHash:string;sourceSceneId:string;sourceStateId:string;policyHash:string;configHash:string;parent:number;tier:string;scope:string;contentHash:string;
  geometry:{structures:CranialSourceStructure[];targetSourceCells:number[];targetPointsRasMm:Point3[]};episodes:SequentialEpisode[];
}
const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const hash=(v:unknown)=>typeof v==='string'&&/^sha256:[0-9a-f]{64}$/.test(v);
const fileHash=(v:unknown)=>typeof v==='string'&&/^[0-9a-f]{64}$/.test(v);
const count=(v:unknown):v is number=>Number.isSafeInteger(v)&&(v as number)>=0;
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
const point=(v:unknown):v is Point3=>Array.isArray(v)&&v.length===3&&v.every(x=>typeof x==='number'&&Number.isFinite(x));
/** Display-only export. Native branch restoration remains the execution authority. */
export function sequentialEpisodePacket(artifact:unknown,response:SurgicalResponse|null):SequentialEpisodePacket|null {
  if(!object(artifact)||!object(artifact.sequentialEpisodes)||!response)return null;
  const p=artifact.sequentialEpisodes;
  if(p.schema!=='resectionlab.recorded-sequential-episodes.v1'||p.caseHash!==response.caseHash||p.sourceSceneId!==response.view.scene.sceneId||!hash(p.sourceStateId)||!hash(p.policyHash)||p.policyHash!==response.planningAdmission?.policyHash||!hash(p.configHash)||!hash(p.contentHash)||!count(p.parent)||typeof p.scope!=='string'||!object(p.geometry)||!Array.isArray(p.geometry.structures)||!Array.isArray(p.geometry.targetSourceCells)||!Array.isArray(p.geometry.targetPointsRasMm)||p.geometry.targetSourceCells.length!==p.geometry.targetPointsRasMm.length||!p.geometry.targetPointsRasMm.every(point)||!Array.isArray(p.episodes)||p.episodes.length!==5)return null;
  if(p.geometry.structures.some(s=>!object(s)||typeof s.id!=='string'||!Array.isArray(s.sourceCells)||!s.sourceCells.every(count)||!Array.isArray(s.pointsRasMm)||s.pointsRasMm.length!==s.sourceCells.length||!s.pointsRasMm.every(point)))return null;
  const signatures=new Set(p.episodes.map(r=>object(r)?`${r.method}:${r.seed}`:''));
  if(!['learned:17','learned:43','greedy:0','search:0','fixed-narrow-greedy:0'].every(s=>signatures.has(s)))return null;
  const ids=new Set<string>();
  for(const item of p.episodes){
    if(!object(item)||typeof item.branchId!=='string'||ids.has(item.branchId)||!['learned','greedy','search','fixed-narrow-greedy'].includes(String(item.method))||!count(item.seed)||!count(item.nativeActions)||!response.branches.some(b=>b.id===item.branchId&&b.steps===item.nativeActions)||!hash(item.finalStateId)||!hash(item.snapshotHash)||!hash(item.actionHistoryHash)||!Array.isArray(item.acceptedStateIds)||item.acceptedStateIds.length!==item.nativeActions+1||item.acceptedStateIds[0]!==p.sourceStateId||item.acceptedStateIds.at(-1)!==item.finalStateId||!item.acceptedStateIds.every(hash)||!Array.isArray(item.attempts)||item.attempts.length>512||!object(item.endpoint)||item.endpoint.configHash!==p.configHash||!object(item.progress)||!object(item.cleanup)||!object(item.computation)||!object(item.originalEpisode)||!fileHash(item.originalEpisode.sha256))return null;
    if(item.method==='learned'&&(!object(item.checkpoint)||!fileHash(item.checkpoint.sha256)||!count(item.checkpoint.selectedUpdates)||item.checkpoint.selectedUpdates===0))return null;
    if(![item.progress.originalTargetTreatmentMm3,item.progress.originalTargetRemovedMm3,item.neuralRemovedMm3,item.modeledSeconds,item.cleanup.debrisMm3,item.cleanup.fluidMm3,item.cleanup.debrisCollectedMm3,item.computation.wallSeconds].every(finite)||!count(item.progress.originalTargetCells)||!count(item.progress.residualOriginalTargetCells)||!object(item.structureRemovedMm3)||!Object.values(item.structureRemovedMm3).every(finite)||!object(item.structureCutSourceMm3)||!Object.values(item.structureCutSourceMm3).every(finite)||!object(item.computation.tier)||item.computation.tier.name!==p.tier||![item.computation.physicalCalls,item.computation.planningPhysicalCalls,item.computation.executionPhysicalCalls,item.computation.previewRefusals,item.computation.previewSkillAttempts,item.computation.canceledRequests,item.computation.tier.max_physical_calls].every(count)||!finite(item.computation.tier.max_wall_seconds))return null;
    let position=0;
    for(const [index,a] of item.attempts.entries()){
      if(!object(a)||a.attemptIndex!==index||a.nativePositionBefore!==position||!['accepted','partial','rejected'].includes(String(a.status))||!object(a.action)||!point(a.toolPositionBefore)||!point(a.toolPositionAfter)||!object(a.effectiveParametersBefore)||!object(a.effectiveParametersAfter)||a.stateBefore!==item.acceptedStateIds[position])return null;
      if(a.status!=='rejected')position++;
      if(a.nativePositionAfter!==position||a.stateAfter!==item.acceptedStateIds[position])return null;
    }
    if(position!==item.nativeActions)return null;ids.add(item.branchId);
  }
  return p as unknown as SequentialEpisodePacket;
}
export function sequentialEpisodeOverlay(packet:SequentialEpisodePacket|null,response:SurgicalResponse|null,view:SurgicalView|null):InteractivePlanningOverlay|null {
  if(!packet||!response||!view||response.view!==view)return null;
  const row=packet.episodes.find(r=>r.branchId===response.branchId);if(!row)return null;
  if(response.playback?response.playback.branchId!==row.branchId:row.acceptedStateIds[response.position]!==view.stateId)return null;
  const structures=packet.geometry.structures.map(s=>({...s,removedMm3:row.structureRemovedMm3[s.id]??0,cutSourceMm3:row.structureCutSourceMm3[s.id]??0}));
  const removed=new Set(view.removed),residual=packet.geometry.targetPointsRasMm.filter((_,i)=>!removed.has(packet.geometry.targetSourceCells[i]));
  const at=(position:Point3)=>({position,quaternion:[0,0,0,1] as [number,number,number,number]});
  const center=packet.geometry.targetPointsRasMm[0]??[0,0,0];
  return {binding:planningBinding(response),constraints:{target:{centerMm:center,radiusMm:0},entry:{centerMm:center,radiusMm:0},tool:{instrument:'cusa',radiusMm:1.5},requiredEvidence:[],candidateCount:packet.episodes.length,nonTargetClearanceMm:0},
    routes:row.attempts.filter(a=>a.status!=='rejected'&&a.action.operation==='apply').map(a=>({id:`${row.branchId}:${a.attemptIndex}`,label:row.label,entryPose:at(a.toolPositionBefore),outsidePose:at(a.toolPositionBefore),targetPose:at(a.toolPositionAfter),pointsRasMm:[a.toolPositionBefore,a.toolPositionAfter],selected:true,actionProgram:[],metrics:{},scope:packet.scope})),selectedId:null,limitingStructures:[],
    recorded:{comparisonSeal:packet.contentHash,geometrySeal:row.actionHistoryHash,policyHash:packet.policyHash,selectedBranchId:row.branchId,structures,targetPointsRasMm:packet.geometry.targetPointsRasMm,residualPointsRasMm:residual,scope:packet.scope}};
}
