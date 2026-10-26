import type {SurgicalResponse} from './surgical-types';

export interface LearnedExecution {
  method:'frozen_behavior_cloning'|'ppo_initialized_from_behavior_cloning';branchId:string;
  checkpointSha256:string;observationSchema:string;actualActorChoices:string[];
  actualSupervisedOptimizerSteps?:number;actualPpoOptionTransitions?:number;seed?:number;
  auditPassed:boolean;prefixActions:number;finalStateId:string;
}
const object=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const options=new Set(['coagulate_center','coagulate_lower','coagulate_forward','aspirate_pool','wait','finish_verify']);
/** Descriptive experiment provenance only; native replay retains its own authority. */
export function learnedExecution(value:unknown,response:SurgicalResponse|null):LearnedExecution|null {
  if(!object(value)||!['resectionlab.patient-rescue-bc.v1','resectionlab.patient-rescue-ppo-bc-audit.v1'].includes(String(value.schema))||
    !['frozen_behavior_cloning','ppo_initialized_from_behavior_cloning'].includes(String(value.method))||
    typeof value.branchId!=='string'||!response?.branches.some(b=>b.id===value.branchId)||
    typeof value.checkpointSha256!=='string'||!(/^[0-9a-f]{64}$/).test(value.checkpointSha256)||
    value.observationSchema!=='resectionlab.prepared-rescue-observation.v2'||
    !Number.isSafeInteger(value.prefixActions)||(value.prefixActions as number)<0||
    (value.prefixActions as number)>response.branches.find(b=>b.id===value.branchId)!.steps||
    typeof value.finalStateId!=='string'||!/^sha256:[0-9a-f]{64}$/.test(value.finalStateId)||
    value.samePreparedInjuryOnly!==true||value.promoted!==false||typeof value.auditPassed!=='boolean'||
    !Array.isArray(value.actualActorChoices)||!value.actualActorChoices.length||value.actualActorChoices.length>12||
    !value.actualActorChoices.every(action=>typeof action==='string'&&options.has(action)))return null;
  for(const key of ['actualSupervisedOptimizerSteps','actualPpoOptionTransitions','seed'])
    if(value[key]!==undefined&&(!Number.isSafeInteger(value[key])||(value[key] as number)<0))return null;
  if(value.method==='ppo_initialized_from_behavior_cloning'&&
    (value.schema!=='resectionlab.patient-rescue-ppo-bc-audit.v1'||!(typeof value.actualPpoOptionTransitions==='number'&&value.actualPpoOptionTransitions>0)))return null;
  if(value.method==='frozen_behavior_cloning'&&
    (value.schema!=='resectionlab.patient-rescue-bc.v1'||value.actualRlOptionTransitions!==0||!(typeof value.actualSupervisedOptimizerSteps==='number'&&value.actualSupervisedOptimizerSteps>0)))return null;
  return value as unknown as LearnedExecution;
}
