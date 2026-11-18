import {useState} from 'react';
import type {PreparationPlan} from './native-anatomy-data';
import {isOperationCancelled,operationMessage} from './operation-feedback';
import type {NativeWorkspace} from './use-native-workspace';

const stageStatuses:Record<string,string>={
  ready:'Ready',cached:'Reused',available:'Available',missing_input:'Missing input',
  unsupported_input:'Unsupported input',unavailable:'Unavailable',blocked:'Blocked',
  failed:'Failed',empty_output:'Empty output',complete:'Complete',completed:'Complete',
};
const retryable=new Set(['failed','empty_output']);

export function NativePreparePanel({workspace:w,onReviewAnatomy}:{workspace:NativeWorkspace;onReviewAnatomy:()=>void}){
  const [tumorIndication,setTumorIndication]=useState('');
  const [fullHeadConfirmed,setFullHeadConfirmed]=useState(false);
  const [includeDuralProxy,setIncludeDuralProxy]=useState(false);
  const [report,setReport]=useState<{scope:string;plan:PreparationPlan}|null>(null);
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
  const hasReady=plan?.stages.some(stage=>stage.status==='ready')??false;
  const hasFailures=plan?.stages.some(stage=>retryable.has(stage.status))??false;
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
        setReport({scope:planScope(result.document.snapshot_id),plan:result.plan});
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
  return <section className="unified-details preparation-panel" aria-label="Anatomy preparation">
    <h2>Prepare anatomy</h2>
    <details>
      <summary>Preparation options</summary>
      <label>Tumor model indication<select aria-label="Tumor model indication" disabled={blocked} value={tumorIndication} onChange={e=>updateOption(()=>setTumorIndication(e.target.value))}>
        <option value="">Use supplied lesion masks</option>
        <option value="meningioma">Meningioma · T1 post-contrast</option>
        <option value="glioma-gbm">Glioblastoma · T1 post-contrast</option>
        <option value="glioma-lgg">Lower-grade glioma · FLAIR</option>
      </select></label>
      <label className="intake-confirm"><input type="checkbox" checked={fullHeadConfirmed} disabled={blocked} onChange={e=>updateOption(()=>setFullHeadConfirmed(e.target.checked))}/>
        Include estimates from eligible full-head MRI</label>
      <label className="intake-confirm"><input type="checkbox" checked={includeDuralProxy} disabled={blocked} onChange={e=>updateOption(()=>setIncludeDuralProxy(e.target.checked))}/>
        Include a geometric dural envelope where supported</label>
    </details>
    {!running&&<button disabled={blocked} onClick={()=>prepare('plan')}>Review preparation plan</button>}
    {plan&&<ul className="preparation-stages">{plan.stages.map(stage=><li key={stage.id} data-preparation-stage={stage.id} data-stage-status={stage.status}>
      <div><strong>{stage.label}</strong><span>{stageStatuses[stage.status]??stage.status.replaceAll('_',' ')}</span></div>
      {stage.reason&&<p>{stage.reason}</p>}
    </li>)}</ul>}
    {running&&<div className="preparation-progress" role="status">
      <progress max={1} value={w.progress?.fraction}/>
      <span>{w.message}</span>
      <button onClick={w.cancel}>Cancel preparation</button>
    </div>}
    {cancelled&&<p role="status">Preparation cancelled. Completed stages remain available.</p>}
    {error&&<p role="alert">{error}</p>}
    {hasReady&&<button disabled={blocked} onClick={()=>prepare('run')}>{cancelled?'Resume preparation':'Prepare anatomy'}</button>}
    {hasFailures&&<button disabled={blocked} onClick={()=>prepare('run',true)}>Retry failed stages</button>}
    {!!b.document.layers.length&&<button disabled={blocked} onClick={onReviewAnatomy}>Inspect anatomy</button>}
  </section>;
}
