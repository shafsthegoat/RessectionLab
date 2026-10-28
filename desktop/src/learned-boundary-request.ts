import type {SurgicalResponse} from './surgical-types';
import {reconstructedPrefix} from './surgical-authority.ts';

/** Respect the imported-checkpoint admission sequence before seeking a skill. */
export function learnedBoundaryRequest(response:SurgicalResponse,branchId:string,position:number):{command:string;branchId:string;position?:number}|null{
  const branch=response.branches.find(b=>b.id===branchId);
  if(!branch||!Number.isSafeInteger(position)||position<0||position>branch.steps)throw Error('Learned policy boundary is unavailable');
  if(response.branchId!==branchId)return{command:response.verification?.readOnly?'inspectArchive':'selectBranch',branchId,...(response.verification?.readOnly?{}:{position})};
  if(response.verification?.readOnly&&!reconstructedPrefix(response))return{command:'seekPrefix',branchId,position:0};
  if(response.position===position&&!response.playback)return null;
  return{command:response.verification?.readOnly?'seekPrefix':'selectBranch',branchId,position};
}
