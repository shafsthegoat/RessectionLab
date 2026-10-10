import { ReplayStepControl } from "./ReplayStepControl";
import type { EpisodeView } from "./episode-data";
import type { DevelopmentEpisodeRequest } from "./episode-types";
export function EpisodePanel({view,step,selector,busy,unavailableReason,onSelector,onExecute,onStep,onShow,onSource,visible}: {
  view:EpisodeView|null;step:number;selector:DevelopmentEpisodeRequest["selector"];busy:boolean;unavailableReason?:string;
  onSelector:(value:DevelopmentEpisodeRequest["selector"])=>void;onExecute:()=>void;onStep:(value:number)=>void;
  onShow:()=>void;onSource:()=>void;visible:boolean;
}) {
  const episode=view?.episode, frame=episode?.replayFrames[step], replay=view?.frames[step];
  return <section className="episode-panel" aria-label="Generated development episode">
    <div className="planning-title"><span className="eyebrow">GENERATED SOFTWARE FIXTURE</span><h2>Execute & replay</h2>
      <p>One persistent cavity, aspiration and a non-removing geometric probe. No patient data or clinical validation.</p></div>
    <label className="field-label" htmlFor="episode-selector">Episode selector</label>
    <select id="episode-selector" value={selector} disabled={busy} onChange={event=>onSelector(event.target.value as DevelopmentEpisodeRequest["selector"])}>
      <option value="scripted">Scripted interaction sequence</option><option value="SEARCH">Existing bounded search</option>
    </select>
    <button className="primary-button" disabled={busy || !!unavailableReason} onClick={onExecute}>{busy?"Executing / checking…":"Execute generated episode"}</button>
    {unavailableReason && <p role="status" className="muted-note">{unavailableReason}</p>}
    <p className="instrument-note">Search uses the same backend state and objective. No learned policy is selected here.</p>
    {episode && frame && replay && <>
      <div className="episode-status" role="status"><strong>Generated · executed · geometry checked</strong><span>{episode.selector} · {episode.history.length} committed actions · {episode.replayFrames.length} recorded frames</span></div>
      <div className="episode-replay-buttons"><button className="outline-button" onClick={onShow} disabled={busy || visible}>Show this episode</button><button className="text-button" onClick={onSource} disabled={!visible}>Source view</button></div>
      <ReplayStepControl id="episode-replay-step" requestedStep={step} appliedStep={step} stepCount={episode.replayFrames.length-1} disabled={busy} onRequest={onStep}/>
      <div className="episode-replay-buttons"><button className="outline-button" disabled={busy || step===0} onClick={()=>onStep(step-1)}>Previous frame</button><button className="outline-button" disabled={busy || step===episode.replayFrames.length-1} onClick={()=>onStep(step+1)}>Next frame</button></div>
      <p className="episode-pose"><strong>{frame.phase === "initial" ? "Initial tissue" : frame.phase === "stop" ? "STOP · committed terminal action" : `${frame.mode === "probe" ? "Probe · geometric contact" : "Aspiration · modeled removal"} / ${frame.phase}`}</strong>
        {frame.tipRasMm && <span>Tip RAS+ mm: {frame.tipRasMm.map(n=>n.toFixed(2)).join(", ")} · {frame.toolId}</span>}</p>
      <dl className="episode-quantities"><div><dt>Target removed</dt><dd>{replay.removedTargetVolumeMm3.toFixed(2)} mm³</dd></div><div><dt>Other tissue removed</dt><dd>{replay.removedNormalVolumeMm3.toFixed(2)} mm³</dd></div><div><dt>Target remaining</dt><dd>{replay.residualTargetVolumeMm3.toFixed(2)} mm³</dd></div><div><dt>Probe contact</dt><dd>{view.probeCounts[step]} cells contacted during probing</dd></div></dl>
      <p className="instrument-note">Recorded backend boundaries only. Withdrawal reverses the recorded path. No interpolation, extra tissue effects or elapsed-time animation.</p>
      <h3>Action timeline</h3><ol className="episode-timeline">{episode.history.map((action,index)=>{
        const first=episode.replayFrames.findIndex(f=>f.actionIndex===index);
        return <li key={index}><button className={frame.actionIndex===index?"selected":""} disabled={busy} onClick={()=>onStep(first)} aria-current={frame.actionIndex===index?"step":undefined}>
          <span>{index+1}. {action.interaction_mode === "stop" ? "STOP" : action.interaction_mode === "probe" ? "Probe" : "Aspirate"}</span><small>{action.removed_indices_native.length} cells removed · committed</small></button></li>;
      })}</ol>
      <details open className="episode-evidence"><summary>Rejected attempt & evidence</summary>
        {episode.attemptDiagnostics.map((attempt,i)=><p key={i}><strong>Rejected probe · no state change</strong><br/>{attempt.reason.replaceAll("_"," ")} · 0 cells removed. This was not STOP.</p>)}
        <p>{String(episode.sequentialEffect.explanation)}</p><p>{episode.interpretation}</p>
        <p><strong>Unsupported:</strong> {episode.unsupported.map(v=>v.replaceAll("_"," ")).join(", ")}.</p>
        <p>Patient admission: no. Clinical validation: no. Contact is modeled geometry, not an acquired measurement.</p>
        <dl className="episode-identities"><dt>Episode</dt><dd>{episode.episodeId}</dd><dt>Source case</dt><dd>{episode.caseHash}</dd><dt>Native source</dt><dd>{episode.sourceHash}</dd><dt>Current replay state</dt><dd>{frame.stateAfter}</dd></dl>
      </details>
    </>}
  </section>;
}
