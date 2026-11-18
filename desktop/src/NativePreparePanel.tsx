import {useState} from 'react';
import type {PreparationPlan,PreparationStage} from './native-anatomy-data';
import {isOperationCancelled,operationMessage} from './operation-feedback';
import type {NativeWorkspace} from './use-native-workspace';

const stageStatuses:Record<string,string>={
  ready:'Ready',cached:'Reused',available:'Available',missing_input:'Missing input',
  unsupported_input:'Unsupported input',unavailable:'Unavailable',blocked:'Blocked',unprocessed:'Needs confirmation',
  retry_available:'CPU retry available',failed:'Failed',empty_output:'Empty output',interrupted:'Interrupted',complete:'Completed',
};
const retryable=new Set(['failed','empty_output','retry_available']);

export function NativePreparePanel({workspace:w,onReviewAnatomy}:{workspace:NativeWorkspace;onReviewAnatomy:()=>void}){
  const [tumorIndication,setTumorIndication]=useState('');
  const [fullHeadConfirmed,setFullHeadConfirmed]=useState(false);
  const [includeDuralProxy,setIncludeDuralProxy]=useState(false);
  const [report,setReport]=useState<{scope:string;plan:PreparationPlan;outcomes?:PreparationStage[]}|null>(null);
  const [running,setRunning]=useState(false),[cancelled,setCancelled]=useState(false);
  const [error,setError]=useState('');
  const b=w.binding,api=w.api;
  if(!b||!api||!b.document.series.length)return null;
  const blocked=w.busy||!!w.draft||w.planning.pending;
  const options={
    ...(tumorIndication?{tumorIndication}:{}),
    ...(fullHeadConfirmed?{fullHeadConfirmed}:{}),
    ...(includeDuralProxy?{includeDuralProxy}:{}),
  };
  const planScope=(snapshotId:string)=>JSON.stringify([b.collectionId,snapshotId,options]);
  const plan=report?.scope===planScope(b.document.snapshot_id)?report.plan:null;
  const interrupted=plan?.stages.some(stage=>stage.status==='interrupted'&&stage.retryable!==false)??false;
  const hasReady=interrupted||(plan?.stages.some(stage=>stage.status==='ready')??false);
  const shownStages=new Map(plan?.stages.map(stage=>[stage.id,stage]));
  if(plan)for(const outcome of report?.outcomes??[]){
    const proposed=shownStages.get(outcome.id);
    shownStages.set(outcome.id,{...proposed,...outcome,reason:outcome.reason||proposed?.reason});
  }
  const hasFailures=plan?.stages.some(stage=>stage.retryable!==false&&retryable.has(stage.status))??false;
  function updateOption(update:()=>void){update();setCancelled(false);setError('');}
  function prepare(action:'plan'|'run',retryFailed=false){
    if(action==='run'&&!plan)return;
    setError('');setCancelled(false);
    void w.change(async()=>{
      const collectionId=b!.collectionId;
      const review=()=>api!.anatomyPrepare({collectionId,action:'plan',...options});
      setRunning(action==='run');
      w.setMessage(action==='run'?'Preparing anatomy…':'Reviewing available anatomy stages…');
      try{
        const result=action==='plan'?await review():await w.runPreparation({
          ...w.args(),planId:plan!.id,retryFailed,...options,
        });
        setReport({scope:planScope(result.document.snapshot_id),plan:result.plan,outcomes:result.outcomes});
        w.setMessage(action==='plan'?'Preparation plan available':'Preparation complete · Review the stage outcomes');
        return result;
      }catch(failure){
        const cancelled=isOperationCancelled(failure);
        const summary=cancelled?'Preparation cancelled':operationMessage(failure);
        if(action==='run'){
          setRunning(false);
          setCancelled(cancelled);
          try{
            const result=await review();
            setReport({scope:planScope(result.document.snapshot_id),plan:result.plan});
            if(!cancelled)setError(summary);
            w.setMessage(cancelled?'Preparation cancelled · Completed stages saved':'Preparation interrupted · Review the available stages');
            return result;
          }catch(refreshFailure){
            setReport(null);
            setError(summary+' · Could not refresh preparation. '+operationMessage(refreshFailure));
            return null;
          }
        }
        if(!cancelled)setError(summary);
        w.setMessage(cancelled?'Preparation review cancelled':'Preparation request failed');
        return null;
      }finally{setRunning(false);}
    },{align:false});
  }
  return <details className="unified-details preparation-panel" aria-label="Anatomy preparation" open={running||!!report||!!error||undefined}>
    <summary><strong>Prepare anatomy</strong><span>Models & supplied labels</span></summary>
    <details>
      <summary>Preparation options</summary>
      <label>Tumor model indication<select aria-label="Tumor model indication" disabled={blocked} value={tumorIndication} onChange={e=>updateOption(()=>setTumorIndication(e.target.value))}>
        <option value="">Use supplied lesion masks</option>
        <option value="meningioma">Meningioma · T1 post-contrast</option>
        <option value="monai-brats-0.5.4">Glioma compartments · T1 post-contrast, T1, T2 and FLAIR</option>
        <option value="glioma-gbm">Glioblastoma · T1 post-contrast</option>
        <option value="glioma-lgg">Lower-grade glioma · FLAIR</option>
      </select></label>
      <label className="intake-confirm"><input type="checkbox" checked={fullHeadConfirmed} disabled={blocked} onChange={e=>updateOption(()=>setFullHeadConfirmed(e.target.checked))}/>
        Confirm full-head MRI coverage for head and cranial tissue estimates</label>
      <label className="intake-confirm"><input type="checkbox" checked={includeDuralProxy} disabled={blocked} onChange={e=>updateOption(()=>setIncludeDuralProxy(e.target.checked))}/>
        Include a geometric dural envelope where supported</label>
    </details>
    {!running&&<button disabled={blocked} onClick={()=>prepare('plan')}>Review preparation plan</button>}
    {plan&&<ul className="preparation-stages">{[...shownStages.values()].map(({id,label,status,reason})=>
      <li key={id} data-preparation-stage={id} data-stage-status={status}>
        <div><strong>{label}</strong><span>{stageStatuses[status]??status.replaceAll('_',' ')}</span></div>
        {reason&&<p>{reason}</p>}
      </li>
    )}</ul>}
    {running&&<div className="preparation-progress" role="status">
      <progress max={1} value={w.progress?.fraction}/>
      <span>{w.message}</span>
      <button onClick={w.cancel}>Cancel preparation</button>
    </div>}
    {cancelled&&<p role="status">Preparation cancelled. Completed stages remain available.</p>}
    {error&&<p role="alert">{error}</p>}
    {hasReady&&<button disabled={blocked} onClick={()=>prepare('run')}>{cancelled||interrupted?'Resume preparation':'Prepare anatomy'}</button>}
    {hasFailures&&<button disabled={blocked} onClick={()=>prepare('run',true)}>Retry failed stages</button>}
    {!!b.document.layers.length&&<button disabled={blocked} onClick={onReviewAnatomy}>Inspect anatomy</button>}
  </details>;
}
