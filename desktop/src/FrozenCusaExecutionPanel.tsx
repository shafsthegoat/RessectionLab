import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';

export function FrozenCusaExecutionPanel({c,pending,onRun,onRunSector,pause}:{c:SurgicalController;pending:boolean;onRun?:(branchId:string)=>Promise<void>;onRunSector?:(method:string)=>Promise<void>;pause:()=>void}){
  const comparison=c.result?.planning;
  const intact=c.view?.stateId===comparison?.startStateId&&c.result?.branches.some(item=>item.id===c.result?.branchId&&item.steps===0);
  const [branch,setBranch]=useState('');
  useEffect(()=>{
    if(comparison?.task==='anatomy-access')setBranch(comparison.alternatives.some(p=>p.branchId===c.result?.branchId)?c.result!.branchId:comparison.alternatives[0]?.branchId??'');
  },[comparison?.startStateId,comparison?.totalWallSeconds,c.result?.caseHash]);
  if(!onRun||comparison?.task!=='anatomy-access')return null;
  return <section className="frozen-cusa-execution" data-source-scene={c.view?.scene.sceneId} aria-label="Execute frozen CUSA policy">
    <h3>Run a learned continuation</h3>
    <p>Choose an access approach and a compatible local checkpoint. The frozen actor makes new decisions from current simulator observations, then saves its actions for 3D playback. No training runs.</p>
    <label>Starting approach<select aria-label="Frozen policy access approach" value={branch} disabled={pending||c.pending} onChange={e=>setBranch(e.target.value)}>{comparison.alternatives.map(plan=><option key={plan.branchId} value={plan.branchId}>{plan.label??plan.method}</option>)}</select></label>
    <button disabled={pending||c.pending||!branch} onClick={()=>{pause();void onRun(branch);}}>{pending?'Executing frozen CUSA policy…':'Run frozen CUSA checkpoint'}</button>
    {onRunSector&&!intact&&<button disabled={pending||c.pending} onClick={()=>{pause();const start=c.result?.branches.find(item=>item.steps===0);void c.run(start?'selectBranch':'reset',start?{branchId:start.id,position:0}:{});}}>Return to intact source for policy</button>}
    {onRunSector&&<button disabled={pending||c.pending||!branch||!intact} onClick={()=>{const plan=comparison.alternatives.find(plan=>plan.branchId===branch);if(plan){pause();void onRunSector(plan.method);}}}>Run frozen sector checkpoint</button>}
    {onRunSector&&<p className="surgical-note">The sector hierarchy starts at the common intact source and chooses its access hand before entering. Return to the source start before running it.</p>}
    <p className="surgical-note">Development simulation · 30 mm³ additional target, two admitted actions, at most 20 mm³ additional non-target, checked withdrawal. Unpassed episodes retain their failed gate.</p>
  </section>;
}
