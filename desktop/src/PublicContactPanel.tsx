import {ReplayStepControl} from './ReplayStepControl';
import type {PublicContactView} from './public-contact-data';
import type {ContactGoalId,ContactSelector} from './public-contact-types';
export function PublicContactPanel({view,step,selector,goalId,busy,unavailableReason,visible,onSelector,onGoal,onExecute,onStep,onShow,onSource,onFocusGoal}:{
 view:PublicContactView|null;step:number;selector:ContactSelector;goalId:ContactGoalId;busy:boolean;unavailableReason?:string;visible:boolean;
 onSelector:(v:ContactSelector)=>void;onGoal:(v:ContactGoalId)=>void;onExecute:()=>void;onStep:(v:number)=>void;onShow:()=>void;onSource:()=>void;onFocusGoal:()=>void;
}){
 const e=view?.episode,f=e?.replayFrames[step],prefix=view?.prefixes[step];
 return <section className="episode-panel" aria-label="Public retained-contact task">
  <div className="planning-title"><span className="eyebrow">GENERATED PUBLIC GOAL</span><h2>Retain & touch</h2><p>Open a path, then touch the declared cell with the geometric probe while leaving that cell intact. This models contact only; it does not measure force or identify anatomy.</p></div>
  <label className="field-label" htmlFor="contact-goal">Public goal</label><select id="contact-goal" value={goalId} disabled={busy} onChange={event=>onGoal(event.target.value as ContactGoalId)}><option value="near">Near goal · native cell 6, 6, 3</option><option value="costly">Costly goal · native cell 6, 6, 7</option></select>
  <label className="field-label" htmlFor="contact-selector">Selection method</label><select id="contact-selector" value={selector} disabled={busy} onChange={event=>onSelector(event.target.value as ContactSelector)}><option value="scripted">Scripted contact demonstration</option><option value="SEARCH">Existing bounded search</option></select>
  <button className="primary-button" disabled={busy||!!unavailableReason} onClick={onExecute}>{busy?'Executing / checking…':'Execute public-contact task'}</button>
  {unavailableReason&&<p className="muted-note" role="status">{unavailableReason}</p>}
  <p className="instrument-note">At most two actions. Contact with a retained goal earns 1; all removal and motion incur cost. STOP is allowed. No learned policy or training runs. The nominal-target annotation is fixture background and earns no reward here.</p>
  {view&&e&&f&&prefix&&<>
   <div className="episode-status" role="status"><strong>Generated public contact · native replay checked</strong><span>{e.publicGoal.goalId} goal · {e.selector} · {e.history.length} actions · {e.replayFrames.length} recorded frames</span></div>
   <p className="instrument-note">This contact episode is temporary and cannot yet be saved in a workspace. Vessel annotation evaluation and actor comparisons are unavailable for this task.</p>
   <div className="episode-replay-buttons"><button className="outline-button" onClick={onFocusGoal} disabled={busy}>Focus public goal</button><button className="outline-button" onClick={onShow} disabled={busy||visible}>Show replay</button><button className="text-button" onClick={onSource} disabled={!visible}>Source view</button></div>
   <p>Public goal RAS+ mm: {e.publicGoal.rasMm.map(n=>n.toFixed(1)).join(', ')}. The marker is a declared task location, not an anatomical label.</p>
   <ReplayStepControl id="contact-replay-step" requestedStep={step} appliedStep={step} stepCount={view.frames.length-1} disabled={busy} onRequest={onStep}/>
   <div className="episode-replay-buttons"><button disabled={busy||step===0} onClick={()=>onStep(step-1)}>Previous frame</button><button disabled={busy||step===view.frames.length-1} onClick={()=>onStep(step+1)}>Next frame</button></div>
   <p className="episode-pose"><strong>{f.phase==='initial'?'Initial tissue':f.phase==='stop'?'STOP · committed terminal action':`${f.mode==='probe'?'Probe · geometric contact':'Aspiration · modeled removal'} / ${f.phase}`}</strong>{f.tipRasMm&&<span>Tip RAS+ mm: {f.tipRasMm.map(n=>n.toFixed(2)).join(', ')} · {f.toolId}</span>}</p>
   <dl className="episode-quantities"><div><dt>Goal at this recorded frame</dt><dd>{!prefix.retained?'Removed · no retained-contact credit':prefix.contactedAndRetained?'Retained and contacted':'Retained · not yet contacted'}</dd></div><div><dt>All tissue removed at frame</dt><dd>{prefix.removedVolumeMm3.toFixed(2)} mm³</dd></div></dl>
   <h3>Complete episode outcome</h3><dl className="episode-quantities"><div><dt>Modeled return</dt><dd>{view.outcome.modeledReturn.toFixed(3)}</dd></div><div><dt>Total removal</dt><dd>{view.outcome.totalRemovedMm3.toFixed(2)} mm³</dd></div><div><dt>Round-trip tool path</dt><dd>{view.outcome.completePathMm.toFixed(2)} mm</dd></div><div><dt>Effort and removal cost</dt><dd>{view.outcome.totalCost.toFixed(3)}</dd></div><div><dt>Termination</dt><dd>{view.outcome.termination==='STOP'?'STOP':'Two-action limit'}</dd></div></dl>
   <h3>Recorded actions</h3><ol className="episode-timeline">{e.history.map((h,i)=><li key={i}><button disabled={busy} onClick={()=>onStep(e.replayFrames.findIndex(row=>row.actionIndex===i))}><span>{i+1}. {h.interaction_mode==='stop'?'STOP':h.interaction_mode==='probe'?'Probe':'Aspirate'}</span><small>{h.removed_indices_native.length} cells removed · reward {h.reward.toFixed(3)}</small></button></li>)}</ol>
   <details className="episode-evidence"><summary>Task definition & evidence</summary><p>The strategy was sealed before execution. No private-reference scoring occurred. Rejected pre-opening probe is a fixture check, not STOP. No patient admission or clinical validation.</p><p>Exact recorded insertion and reversed withdrawal; no interpolated effects. The original source and fixture annotations remain unchanged.</p><p>{e.interpretation}</p><dl className="episode-identities"><dt>Episode</dt><dd>{e.episodeId}</dd><dt>Public objective</dt><dd>{e.publicGoal.objectiveHash}</dd><dt>Native source</dt><dd>{e.sourceHash}</dd><dt>Context</dt><dd>{e.taskContract.contextVersion}</dd></dl></details>
  </>}
 </section>;
}
