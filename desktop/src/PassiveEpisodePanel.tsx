import type {PersistentPassiveView} from './persistent-passive-view';
import {ReplayStepControl} from './ReplayStepControl';

export function PassiveEpisodePanel({view,step,busy,visible,onExecute,onStep,onShow,onSource,unavailableReason}:{
  view:PersistentPassiveView|null;step:number;busy:boolean;visible:boolean;
  onExecute:()=>void;onStep:(value:number)=>void;onShow:()=>void;onSource:()=>void;
  unavailableReason?:string;
}){
  const frame=view?.frames[step],row=step>0?view?.episode.history[step-1]:null;
  return <section className="episode-panel" aria-label="Generated persistent passive episode">
    <div className="planning-title"><span className="eyebrow">GENERATED EXAMPLE</span>
      <h2>Two tools, one episode</h2><p>Inspect six recorded states. Both tools remain visible after withdrawal. This is not patient anatomy or force-validated surgery.</p></div>
    <button className="primary-button" disabled={busy||!!unavailableReason} onClick={onExecute}>
      {busy?'Executing / checking…':'Execute generated passive episode'}</button>
    {unavailableReason&&<p role="status" className="muted-note">{unavailableReason}</p>}
    {view&&frame&&<>
      <div className="episode-status" role="status"><strong>{view.origin==='reopened'?'Saved episode checked':'Generated replay ready'}</strong>
        <span>5 actions · 6 recorded states · two tools</span></div>
      <div className="episode-replay-buttons"><button className="outline-button" onClick={onShow} disabled={busy||visible}>Show this episode</button>
        <button className="text-button" onClick={onSource} disabled={!visible}>Source view</button></div>
      <ReplayStepControl id="passive-replay-step" requestedStep={step} appliedStep={step} stepCount={5} disabled={busy} onRequest={onStep}/>
      <div className="episode-replay-buttons"><button className="outline-button" disabled={busy||step===0} onClick={()=>onStep(step-1)}>Previous state</button>
        <button className="outline-button" disabled={busy||step===5} onClick={()=>onStep(step+1)}>Next state</button></div>
      <p className="episode-pose"><strong>{frame.phase.replaceAll('_',' ')} · t={frame.elapsedGeneratedSeconds.toFixed(1)} generated seconds</strong>
        <span>Moving probe: {frame.recordedTools[0].tipRasMm.map(n=>n.toFixed(1)).join(', ')} RAS+ mm · {frame.recordedTools[0].retainedAtDepth?'retained':'at access, still present'}</span>
        <span>Second tool: {frame.recordedTools[1].tipRasMm.map(n=>n.toFixed(1)).join(', ')} RAS+ mm · {frame.recordedTools[1].retainedAtDepth?'retained':'at access, still present'}</span></p>
      <p className="instrument-note">{frame.inspectionStatus==='not_acquired'?'Inspection information is not yet available.':
        `Generated inspection status: ${frame.inspectionStatus}. This is not a sensor measurement.`}</p>
      {row?.stop_reason&&<p className="instrument-note">STOP: {row.stop_reason.replaceAll('_',' ')}.</p>}
      <p className="instrument-note">These are recorded states, with a declared time schedule. No continuous motion, tissue removal or injury outcome is shown.</p>
      <h3>Committed actions</h3><ol className="episode-timeline">{view.episode.history.map((action,index)=><li key={index}>
        <button className={step===index+1?'selected':''} disabled={busy} onClick={()=>onStep(index+1)} aria-current={step===index+1?'step':undefined}>
          <span>{index+1}. {action.action_id.replaceAll('_',' ')}</span>
          <small>{action.duration_generated_seconds.toFixed(1)} declared seconds · no tissue removal</small></button></li>)}</ol>
      <details className="episode-evidence"><summary>Evidence and limits</summary><p>Generated analytic free tunnel and generic, source-bound tool geometry. Both returned tools remain represented at their access poses. Primary geometry checks pair clearance; the independent check covers tissue motion only. No retraction, brain shift, force, injury or clinical outcome was validated. The inspection event is a generated protocol input; hidden simulator properties are not deployment inputs.</p>
        <p>Source: {view.episode.sourceHash}</p><p>Episode: {view.episode.episodeId}</p></details>
    </>}
  </section>;
}
