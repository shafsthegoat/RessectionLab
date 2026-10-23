import type {SurgicalResponse} from './surgical-types';

export interface RescueLearningRow {
  branchId:string;method:'untrained'|'scripted'|'trained'|'search-selected';seed:number;
  success:boolean;stateId:string;reward:number;decisions:number;reason:string;
  modeledField:[number,number,number,number];
}
export interface RescueLearningReport {status:string|null;assessments:RescueLearningRow[];limitations:string[];settings:{train_decisions:number;search_decisions:number}}
const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const count=(v:unknown):v is number=>Number.isSafeInteger(v)&&(v as number)>=0;
/** Imported experiment metadata is descriptive, never replay authority. */
export function rescueLearningReport(value:unknown,response:SurgicalResponse|null):RescueLearningReport|null {
  if(!object(value)||value.schema!=='resectionlab.patient-rescue-learning.v1'||!object(value.settings)||
      !count(value.settings.train_decisions)||!count(value.settings.search_decisions)||
      !Array.isArray(value.assessments)||!value.assessments.length||value.assessments.length>32||
      !Array.isArray(value.limitations)||value.limitations.length>32||
      !value.limitations.every(v=>typeof v==='string'&&v.length<2048))return null;
  const branches=new Set(response?.branches.map(b=>b.id));
  const rows=value.assessments;
  if(rows.some(r=>!object(r)||typeof r.branchId!=='string'||!branches.has(r.branchId)||
      !['untrained','scripted','trained','search-selected'].includes(String(r.method))||
      !count(r.seed)||typeof r.success!=='boolean'||typeof r.stateId!=='string'||!/^sha256:[0-9a-f]{64}$/.test(r.stateId)||
      typeof r.reward!=='number'||!Number.isFinite(r.reward)||!count(r.decisions)||
      typeof r.reason!=='string'||r.reason.length>256||!Array.isArray(r.modeledField)||r.modeledField.length!==4||
      !r.modeledField.every(v=>typeof v==='number'&&Number.isFinite(v)&&v>=0)))return null;
  if(new Set(rows.map(r=>`${r.method}:${r.seed}`)).size!==rows.length)return null;
  if(rows.some(r=>rows.some(other=>other.branchId===r.branchId&&other.stateId!==r.stateId)))return null;
  return {status:typeof value.status==='string'?value.status:null,
    assessments:rows as RescueLearningRow[],limitations:value.limitations as string[],
    settings:{train_decisions:value.settings.train_decisions,search_decisions:value.settings.search_decisions}};
}

export function learningEndpointVerified(row:RescueLearningRow,response:SurgicalResponse|null):boolean {
  const fluid=response?.view.metrics.fluidPresentMm3,outlets=response?.view.metrics.activeVesselWounds;
  return !!response&&!response.verification?.readOnly&&!response.readOnly&&
    response.branchId===row.branchId&&response.view.stateId===row.stateId&&
    typeof fluid==='number'&&Math.abs(fluid-row.modeledField[0])<=1e-9&&outlets===row.modeledField[1]&&
    response.position===response.branches.find(b=>b.id===row.branchId)?.steps;
}
