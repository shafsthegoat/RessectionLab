import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {learnedExecution} from './learned-execution-data';
import {learnedBoundaryRequest} from './learned-boundary-request';

export function LearnedExecutionPanel({c,report,pause}:{c:SurgicalController;report:unknown;pause:()=>void}){
  const [requested,setRequested]=useState<{branchId:string;position:number}|null>(null);
  useEffect(()=>setRequested(null),[c.result?.caseHash]);
  useEffect(()=>{
    if(!requested||!c.result||c.pending)return;
    const next=learnedBoundaryRequest(c.result,requested.branchId,requested.position);
    if(!next){setRequested(null);return;}
    const {command,...args}=next;void c.run(command,args).then(ok=>{if(!ok)setRequested(null);});
  },[requested,c.result,c.pending]);
  const data=learnedExecution(report,c.result);if(!data)return null;
  const ppo=data.method==='ppo_initialized_from_behavior_cloning';
  return <details className="research-planning learned-execution" open><summary>Recorded learned execution · {ppo?'PPO after behavior cloning':'Behavior cloning'}</summary>
    <p>{ppo?`${data.actualPpoOptionTransitions} actual PPO option transitions`:`${data.actualSupervisedOptimizerSteps} supervised optimizer steps; zero RL transitions`}{data.seed!==undefined?` · seed ${data.seed}`:''}.</p>
    <p className="surgical-warning">One prepared development injury. This does not demonstrate glioma resection, transfer, clinical validity or an improvement over behavior cloning. Checkpoint is not promoted.</p>
    <p>Frozen policy decisions were made from each current observed state in the shared surgical engine. {data.auditPassed?'The saved audit reports a matching cold-load policy rollout and native replay.':'The independent audit gate is unpassed.'}</p>
    <button disabled={c.pending||!!requested} onClick={()=>{pause();setRequested({branchId:data.branchId,position:data.prefixActions});}}>Inspect recorded {ppo?'PPO':'BC'} execution</button>
    <p className="surgical-note">Playback shows recorded authoritative actions and modeled effects. It performs no new policy inference.</p>
    <details><summary>Checkpoint and observed decisions</summary><p>Checkpoint SHA-256: {data.checkpointSha256}</p><p>Observation: {data.observationSchema}</p><ol>{data.actualActorChoices.map((action,index)=><li key={index}>{action.replaceAll('_',' ')}</li>)}</ol></details>
  </details>;
}
