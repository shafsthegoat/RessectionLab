import type {SurgicalPlanAlternative} from './surgical-types';

/** Align alternatives by decision count, counting rejected commands separately
 * from the native accepted-action timeline. No new simulation is run by replay. */
export function planReplayPosition(plan:SurgicalPlanAlternative,decision:number):number {
  if(!Number.isInteger(decision)||decision<0||decision>6||!Number.isInteger(plan.prefixActions)||plan.prefixActions<0)throw Error('Invalid research replay position');
  return plan.prefixActions+plan.decisions.slice(0,decision).filter(row=>row.status!=='rejected').length;
}

export function planDecisionReason(plan:SurgicalPlanAlternative,decision:number):string {
  if(decision===0)return 'Shared post-exposure starting state';
  const row=plan.decisions[Math.min(decision,plan.decisions.length)-1];
  if(!row)return plan.stopReason.replaceAll('_',' ');
  if(row.actionId==='STOP')return 'Policy selected STOP; left tool deactivated. Residual target remains unassessed.';
  if(row.status==='rejected')return `Execution refused: ${row.reason??'geometric constraint'}`;
  return `${row.actionId} · ${row.status}${row.reason?` · ${row.reason}`:''}${decision>=plan.decisions.length?` · ${plan.stopReason.replaceAll('_',' ')}`:''}`;
}
