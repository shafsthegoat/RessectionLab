import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {learnedBoundaryRequest} from './learned-boundary-request';
import {learnedSectorExecutions} from './sector-learning-data';

export function LearnedSectorExecutionPanel({c,artifact,pause}:{c:SurgicalController;artifact:unknown;pause:()=>void}){
  const [requested,setRequested]=useState<{branchId:string;position:number}|null>(null);
  useEffect(()=>setRequested(null),[c.result?.caseHash]);
  useEffect(()=>{
    if(!requested||!c.result||c.pending)return;
    const next=learnedBoundaryRequest(c.result,requested.branchId,requested.position);
    if(!next){setRequested(null);return;}
    const {command,...args}=next;void c.run(command,args).then(ok=>{if(!ok)setRequested(null);});
  },[requested,c.result,c.pending]);
  const rows=learnedSectorExecutions(artifact,c.result);
  if(!rows.length)return null;
  return <details className="research-planning learned-sector-execution" open><summary>Learned hand and sector decisions</summary>
    <p>The frozen actor chose an access hand, near/far sector options and STOP from each current observation. Native preview, commit and independent replay check the actual tool actions.</p>
    <p className="surgical-warning">Development simulation. Published tool/goal coordinates can support source-specific shortcuts; these results establish neither generalization nor clinical validity. PPO has not improved on BC in the current matched trials.</p>
    <div className="rescue-comparison"><table aria-label="Learned sector outcomes"><caption>Additional target / non-target after access · mm³</caption><thead><tr><th>Policy</th><th>Target</th><th>Non-target</th><th>Task gate</th><th>Cleanup</th></tr></thead><tbody>{rows.map(row=><tr key={row.branchId}><th><button data-learned-sector-branch={row.branchId} data-learned-sector-start={row.sectorStartPosition} data-learned-sector-end={row.nativeActions} disabled={c.pending||!!requested} onClick={()=>{pause();setRequested({branchId:row.branchId,position:0});}}>Inspect {row.optimizerUpdates>0?'PPO':'BC'} sector policy</button></th><td>{row.deltaTargetMm3.toFixed(2)}</td><td>{row.deltaNonTargetMm3.toFixed(2)}</td><td>{row.taskGateMet?'Passed':`${row.status} · unpassed`}</td><td>{row.cleanupComplete?'Complete':'Incomplete'}</td></tr>)}</tbody></table></div>
    <p className="surgical-note">Playback shows the saved actor-selected native history. Run frozen sector checkpoint below performs new inference.</p>
    {rows.map(row=><details key={row.branchId}><summary>Policy decisions and checkpoint</summary><p>Recorded approach: {row.executionAccessMethod.replaceAll('_',' ').replaceAll('-',' ')}.</p><p>{row.highLevelDecisions.join(' → ')}.</p><p>Sector target: near {row.perSectorRemovedMm3.near.toFixed(2)} / far {row.perSectorRemovedMm3.far.toFixed(2)} mm³. Total non-target: {row.totalNonTargetMm3.toFixed(2)} mm³.</p><p>{row.optimizerUpdates} recorded PPO update rounds · {row.nativeActions} native actions · exact replay verified.</p><p>Checkpoint SHA-256: {row.checkpointFileSha256}</p><p>Actor SHA-256: {row.actorSha256}</p><p>Fitting receipt: {row.trainingReceiptSha256}</p><p>Observation contract: {row.observationSchema}</p></details>)}
  </details>;
}
