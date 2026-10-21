import type {RehearsalController} from './use-rehearsal';
import type {RehearsalEpisode,RehearsalMetrics} from './rehearsal-types';
import './RehearsalPanel.css';
const number=(value:unknown,digits=1)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'Unknown';
const words=(value:unknown)=>typeof value==='string'?value.replaceAll('_',' ').replaceAll('-',' '):'';
function finalMetrics(episode:RehearsalEpisode):RehearsalMetrics{return episode.replayFrames[episode.replayFrames.length-1].metrics;}
function outcome(episode:RehearsalEpisode){const last=episode.replayFrames.at(-1);return words(episode.stopReason??last?.stopReason??last?.metrics.stopReason??(last?.kind==='stop'?'Stopped':'Bounded sequence complete'));}
export function RehearsalPanel({controller:c,unavailableReason}:{controller:RehearsalController;unavailableReason?:string}) {
  const episode=c.activeEpisode,frame=episode?.replayFrames[c.frameIndex],tool=frame?.tools.find(t=>t.toolId===c.selectedTool),
    end=episode?episode.replayFrames.length-1:0,live=c.selectedBranch==='live',stopped=!!c.episode?.stopReason||c.episode?.replayFrames.at(-1)?.metrics.terminated===true||c.episode?.replayFrames.at(-1)?.kind.toLowerCase()==='stop';
  const actions=c.suggestedActions.filter(a=>!a.action.toolId||a.action.toolId===c.selectedTool);
  return <section className="rehearsal-panel" aria-label="Source-bound surgical research rehearsal">
    <div className="rehearsal-heading"><span className="eyebrow">POST-EXPOSURE RESEARCH</span><h2>Rehearse a bounded episode</h2>
      <p>Two persistent tools share one action clock. Aspiration removes tissue; field management records contact and manages a simplified field.</p></div>
    <div className="rehearsal-evidence"><span>Patient source</span><span>Simplified physics</span><span>Vessels unknown</span><span>Function unknown</span></div>
    {!episode&&<div className="rehearsal-start"><button className="primary-button" disabled={c.pending||!!unavailableReason} onClick={()=>void c.create()}>{c.pending?'Preparing source-bound episode…':'Prepare rehearsal'}</button>
      <p>Begins after an assumed exposure. No skull opening or approach safety is inferred from this MRI.</p><button className="text-button" disabled={c.pending||!!unavailableReason} onClick={()=>void c.inspect()}>Reopen saved rehearsal</button></div>}
    {unavailableReason&&<p className="muted-note">{unavailableReason}</p>}
    {episode&&frame&&<>
      <div className="rehearsal-scene-title"><strong>{live?'Working episode':c.comparison?.alternatives.find(a=>a.id===c.selectedBranch)?.label}</strong>
        <button className="text-button" onClick={c.visible?c.hide:c.show}>{c.visible?'Show source anatomy':'Show rehearsal'}</button></div>
      <div className="rehearsal-player"><button onClick={c.playing?c.pause:c.play} disabled={c.pending||end===0} aria-label={c.playing?'Pause rehearsal':'Play rehearsal'}>{c.playing?'Pause':'Play'}</button>
        <input type="range" min="0" max={end} step="1" value={c.frameIndex} disabled={c.pending||end===0} onChange={e=>c.setFrameIndex(+e.target.value)} aria-label="Rehearsal frame"/>
        <output>{c.frameIndex} / {end}</output></div>
      <div className="rehearsal-frame-note"><span>{words(frame.kind)} · {number(frame.elapsedSeconds)} simulated s</span><span>{c.frameIndex===end?'Latest state':'Recorded state'}</span></div>
      <div className="rehearsal-tools" role="group" aria-label="Select persistent instrument">{frame.tools.map(t=><button key={t.toolId} aria-pressed={c.selectedTool===t.toolId} onClick={()=>c.selectTool(t.toolId)} className={`rehearsal-tool ${t.mode}`}>
        <i/><span><strong>{t.mode==='aspiration'?'Aspirator':'Field manager'}</strong><small>{t.active?'Active':'Held · inactive'}</small></span></button>)}</div>
      {tool&&<p className="rehearsal-pose">Tip {tool.tipRasMm.map(n=>n.toFixed(1)).join(' / ')} mm · RAS</p>}
      <div className="rehearsal-metrics"><div><span>Target removed</span><strong>{number(frame.metrics.targetRemovedMm3)} <small>mm³</small></strong></div>
        <div><span>Target remaining</span><strong>{number(frame.metrics.residualTargetMm3)} <small>mm³</small></strong></div>
        <div><span>Non-target removed</span><strong>{number(frame.metrics.nonTargetRemovedMm3)} <small>mm³</small></strong></div>
        <div><span>Modeled field burden</span><strong>{number(frame.metrics.fieldBurden,3)} <small>model units</small></strong></div></div>
      {live&&<details className="rehearsal-controls" open={!c.comparison}><summary>Bounded actions</summary><p>Actions commit from the latest saved state. Scrubbing only changes the view.</p>
        <div className="rehearsal-action-grid">{actions.map(a=><button key={a.id} disabled={c.pending||stopped} onClick={()=>void c.step(a.action)}>{a.label}</button>)}</div>
        <button className="primary-button" disabled={c.pending||stopped} onClick={()=>void c.compare()}>{c.pending?'Checking episode…':'Compare two continuations'}</button></details>}
      {!live&&<button className="outline-button" onClick={()=>c.selectBranch('live')}>Return to working episode</button>}
      {c.comparison&&<div className="rehearsal-comparison"><h3>Two continuations, one saved state</h3><p>Both preserve the same source, tool poses and prior effects.</p>
        <div className="rehearsal-branch-head"><span>Outcome</span>{c.comparison.alternatives.map(a=><button key={a.id} aria-pressed={c.selectedBranch===a.id} onClick={()=>c.selectBranch(a.id)}>{a.label}<small>{c.selectedBranch===a.id?'Viewing replay':'View replay'}</small></button>)}</div>
        {([['Target removed','targetRemovedMm3','mm³'],['Non-target removed','nonTargetRemovedMm3','mm³'],['Residual target','residualTargetMm3','mm³'],['Field burden','fieldBurden','units']] as const).map(([label,key,unit])=><div className="rehearsal-compare-row" key={key}><span>{label}</span>{c.comparison!.alternatives.map(a=><strong key={a.id}>{number(finalMetrics(a.episode)[key],key==='fieldBurden'?3:1)}<small> {unit}</small></strong>)}</div>)}
        <div className="rehearsal-compare-row"><span>Anatomical conflicts</span>{c.comparison.alternatives.map(a=><span key={a.id}>{number(finalMetrics(a.episode).anatomicalConflicts)}</span>)}</div>
        <div className="rehearsal-compare-row"><span>Initial exposure</span>{c.comparison.alternatives.map(a=><span key={a.id}>Assumed · {number(finalMetrics(a.episode).assumedExposureTissueMm3)} mm³</span>)}</div>
        <div className="rehearsal-compare-row"><span>Vascular events</span>{c.comparison.alternatives.map(a=><span key={a.id}>{finalMetrics(a.episode).vascularEvents===null?'Unknown':number(finalMetrics(a.episode).vascularEvents)}</span>)}</div>
        <div className="rehearsal-compare-row"><span>Motor / language</span><span>Unknown</span><span>Unknown</span></div>
        <div className="rehearsal-compare-row"><span>Stop reason</span>{c.comparison.alternatives.map(a=><span key={a.id}>{outcome(a.episode)}</span>)}</div>
      </div>}
      <details className="rehearsal-timeline"><summary>Action timeline · {episode.history.length} committed actions</summary><ol>
        {episode.replayFrames.filter((f,i,all)=>i===0||f.actionIndex!==all[i-1].actionIndex||i===all.length-1).map(f=><li key={f.frameIndex}><button aria-current={f.frameIndex===c.frameIndex?'step':undefined} onClick={()=>c.setFrameIndex(f.frameIndex)}><span>{f.frameIndex===0?'Initial assumed exposure':words(f.kind)}</span><small>{number(f.elapsedSeconds)} s · frame {f.frameIndex}</small></button></li>)}</ol></details>
      <details className="rehearsal-limits"><summary>Source, assumptions and limits</summary><p>Original acquired MRI and source segmentations remain unchanged. Violet marks assumed initial access; mint marks new modeled removal; amber marks field-management contact. Neither exposure nor contact establishes a safe corridor.</p>
        <p>Physics are simplified and uncalibrated. Field management models local contact and a debris proxy. Irrigation flow, fluid balance, thermal injury, vessel sealing, blood loss and neurological outcomes are not modeled. Vessels and motor/language function remain unknown when they were not acquired.</p>
        <p className="rehearsal-id">Case {episode.sourceBinding.caseHash}<br/>State {frame.stateAfter}</p></details>
    </>}
    {c.error&&<div className="rehearsal-error" role="alert">{c.error}</div>}
    <p className="rehearsal-research-note">Research simulation for specialist review · not a surgical guide</p>
  </section>;
}
