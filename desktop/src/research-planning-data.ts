import type {SurgicalPlanAlternative} from './surgical-types';

/** Align alternatives by decision count, counting rejected commands separately
 * from the native accepted-action timeline. No new simulation is run by replay. */
export function planReplayPosition(plan:{prefixActions:number;decisions:Array<{status:string}>},decision:number):number {
  if(!Number.isInteger(decision)||decision<0||decision>Math.max(6,plan.decisions.length)||!Number.isInteger(plan.prefixActions)||plan.prefixActions<0)throw Error('Invalid research replay position');
  return plan.prefixActions+plan.decisions.slice(0,decision).filter(row=>row.status!=='rejected').length;
}

/** Recover annotations after reopen or external timeline playback/scrubbing.
 * Refused commands share a physical boundary; an explicit selection there can
 * still be retained by the caller rather than inferred from physical state. */
export function planDecisionAtPosition(plan:SurgicalPlanAlternative,position:number):number {
  if(!Number.isInteger(position)||position<0)throw Error('Invalid native replay position');
  if(position<=plan.prefixActions)return 0;
  let accepted=plan.prefixActions,decision=0;
  for(const row of plan.decisions){
    if(row.status!=='rejected')accepted++;
    if(accepted>position)break;
    decision++;
  }
  return decision;
}

export function planLabel(plan:SurgicalPlanAlternative):string {
  return plan.label??(plan.method==='checkpoint'?'Frozen learner · seed 17':plan.method==='greedy'?'Observation-only greedy':'Immediate STOP');
}

export function planDecisionReason(plan:SurgicalPlanAlternative,decision:number):string {
  if(decision===0)return plan.route?'Shared source-anatomy starting state':'Shared post-exposure starting state';
  const row=plan.decisions[Math.min(decision,plan.decisions.length)-1];
  if(!row)return plan.stopReason.replaceAll('_',' ');
  if(row.actionId==='STOP')return 'Policy selected STOP; left tool deactivated. Residual target remains unassessed.';
  if(row.status==='rejected')return `Execution refused: ${row.reason??'geometric constraint'}`;
  return `${row.actionId} · ${row.status}${row.reason?` · ${row.reason}`:''}${decision>=plan.decisions.length?` · ${plan.stopReason.replaceAll('_',' ')}`:''}`;
}
