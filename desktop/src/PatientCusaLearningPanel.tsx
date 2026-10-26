import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {learnedBoundaryRequest} from './learned-boundary-request';
import {patientCusaExecutions} from './patient-cusa-learning-data';

export function PatientCusaLearningPanel({c,artifact,pause}:{c:SurgicalController;artifact:unknown;pause:()=>void}){
  const [requested,setRequested]=useState<{branchId:string;position:number}|null>(null);
  useEffect(()=>setRequested(null),[c.result?.caseHash]);
  useEffect(()=>{
    if(!requested||!c.result||c.pending)return;
    const next=learnedBoundaryRequest(c.result,requested.branchId,requested.position);
    if(!next){setRequested(null);return;}
    const {command,...args}=next;void c.run(command,args).then(ok=>{if(!ok)setRequested(null);});
  },[requested,c.result,c.pending]);
  const rows=patientCusaExecutions(artifact,c.result);if(!rows.length)return null;
  return <details className="research-planning patient-cusa-learning" open><summary>Learned CUSA continuation after selected access</summary>
    <p>Frozen actors chose tool-frame CUSA strokes and STOP from each current observation. Every effect used the shared surgical engine and independently replayed native actions.</p>
    <p className="surgical-warning">Development simulation. This does not establish superiority, transfer or clinical validity; inspect the matched baselines and failed gates in the experiment receipt.</p>
    <table aria-label="Learned CUSA outcomes"><caption>Additional modeled removal after the recorded access · mm³</caption><thead><tr><th>Policy</th><th>Target</th><th>Non-target</th><th>Task gate</th><th>Cleanup</th></tr></thead><tbody>{rows.map(row=><tr key={row.branchId} aria-selected={c.result?.branchId===row.branchId}><th><button data-policy-branch={row.branchId} data-policy-position={row.parentPosition} disabled={c.pending||!!requested} onClick={()=>{pause();setRequested({branchId:row.branchId,position:row.parentPosition});}}>Inspect {row.method==='ppo'?'PPO':'BC'} CUSA execution</button></th><td>{row.deltaTargetMm3.toFixed(2)}</td><td>{row.deltaNonTargetMm3.toFixed(2)}</td><td>{row.status==='success'?'Passed':`${row.status} · unpassed`}</td><td>{row.cleanupComplete?'Complete':'Incomplete'}</td></tr>)}</tbody></table>
    <p className="surgical-note">Inspection opens the actual policy starting boundary. Play replay animates recorded instruments and changing anatomy; it runs no new policy inference. Skull access and missing critical anatomy remain unsupported.</p>
    {rows.map(row=><details key={row.branchId}><summary>{row.method==='ppo'?'PPO':'BC'} checkpoint and decisions</summary><p>{row.method==='ppo'?`${row.optimizerUpdates} recorded PPO update rounds`:'Supervised behavior cloning; no RL update'}. {row.actorDecisions.join(' → ')}.</p>{row.status!=='success'&&<p role="status">Task gate unpassed · {row.stopReason??row.status}. {row.failureReasons?.map(reason=>reason.replaceAll('_',' ')).join('; ')}</p>}<p>Checkpoint file SHA-256: {row.checkpointFileSha256}</p><p>Actor SHA-256: {row.modelSha256}</p><p>Fitting receipt: {row.trainingReceiptSha256}</p><p>Observation: {row.observationSchema} · nine choices, 22 features per choice.</p></details>)}
  </details>;
}
