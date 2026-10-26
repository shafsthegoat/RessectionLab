import type {SurgicalResponse} from './surgical-types';
export interface PatientCusaExecution {schema:string;method:'bc'|'ppo';branchId:string;parentPosition:number;sourceSceneId:string;sourceStateId:string;modelSha256:string;checkpointFileSha256:string;trainingReceiptSha256:string;observationSchema:string;optimizerUpdates:number;actorDecisions:string[];deltaTargetMm3:number;deltaNonTargetMm3:number;cleanupComplete:boolean;admittedNonStopActions:number;status:string}
const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const hash=(v:unknown)=>typeof v==='string'&&/^[0-9a-f]{64}$/.test(v);
const sealedHash=(v:unknown)=>typeof v==='string'&&/^sha256:[0-9a-f]{64}$/.test(v);
const count=(v:unknown)=>Number.isSafeInteger(v)&&(v as number)>=0;
const actions=new Set(['inward-0.5','inward-1','inward-1.5','inward-2','left-0.5','right-0.5','apply-here','inspect','STOP']);
/** Display records never grant policy or native-engine authority. */
export function patientCusaExecutions(artifact:unknown,response:SurgicalResponse|null):PatientCusaExecution[]{
  if(!object(artifact)||!Array.isArray(artifact.learnedExecutions)||artifact.learnedExecutions.length>8||!response)return[];
  const rows=artifact.learnedExecutions;
  if(rows.some(row=>!object(row)||row.schema!=='resectionlab.patient-cusa-learning.v1'||!['bc','ppo'].includes(String(row.method))||
    typeof row.branchId!=='string'||!response.branches.some(b=>b.id===row.branchId)||!count(row.parentPosition)||
    (row.parentPosition as number)>response.branches.find(b=>b.id===row.branchId)!.steps||
    row.sourceSceneId!==response.view.scene.sceneId||!sealedHash(row.sourceStateId)||
    !hash(row.modelSha256)||!hash(row.checkpointFileSha256)||!sealedHash(row.trainingReceiptSha256)||
    row.observationSchema!=='surgical-rl.v2-tool-frame-task-v1'||!count(row.optimizerUpdates)||
    (row.method==='ppo'&&(row.optimizerUpdates as number)<1)||(row.method==='bc'&&row.optimizerUpdates!==0)||
    !Array.isArray(row.actorDecisions)||row.actorDecisions.length>6||!row.actorDecisions.length||!row.actorDecisions.every(a=>typeof a==='string'&&actions.has(a))||
    !count(row.admittedNonStopActions)||typeof row.cleanupComplete!=='boolean'||typeof row.status!=='string'||
    typeof row.deltaTargetMm3!=='number'||!Number.isFinite(row.deltaTargetMm3)||row.deltaTargetMm3<0||
    typeof row.deltaNonTargetMm3!=='number'||!Number.isFinite(row.deltaNonTargetMm3)||row.deltaNonTargetMm3<0||
    (row.status==='success'&&(!row.cleanupComplete||(row.admittedNonStopActions as number)<2||row.deltaTargetMm3<30||row.deltaNonTargetMm3>20))))return[];
  if(new Set(rows.map(row=>row.branchId)).size!==rows.length)return[];
  return rows as unknown as PatientCusaExecution[];
}
