import type {SurgicalController} from './use-surgical';
import type {ReactNode} from 'react';
import {fullCaseStateSummary,recordedFullCasePhase} from './full-case-presentation';
import {surgicalPlaybackPosition} from './surgical-playback';
const amount=(value:number|null|undefined,digits=2)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'Unknown';

export function FullCaseRehearsalPanel({c,pause,onFrameField,fieldDisplayControls}:{c:SurgicalController;pause:()=>void;onFrameField?:()=>void;fieldDisplayControls?:ReactNode}) {
  const v=c.view,r=c.result;if(!v)return null;
  const summary=fullCaseStateSummary(v),record=r?.fullCase,readOnly=!!r?.playback,
    position=surgicalPlaybackPosition(r?.playback,r?.position??0),
    phase=recordedFullCasePhase(record,position);
  const run=(scenario:'primary'|'vascular')=>{pause();c.cancel();c.startFullCase(scenario);};
  return <section className="full-case-rehearsal" aria-label="Full Case Rehearsal">
    <strong>Full Case Rehearsal</strong>
    <p className="surgical-note">Compute the rehearsal in bounded slices, recording each action for replay. Pause stops after the current slice finishes.</p>
    <div className="surgical-actions">
      <button disabled={c.pending||readOnly} onClick={()=>run('primary')}>{record?.scenario==='primary'?'Resume primary rehearsal':'Start primary rehearsal'}</button>
      <button disabled={c.pending||readOnly} onClick={()=>run('vascular')}>{record?.scenario==='vascular'?'Resume vascular injury / rescue':'Start vascular injury / rescue'}</button>
      {c.computingFullCase&&<button onClick={c.pauseFullCase}>Pause computation</button>}
      {onFrameField&&<button onClick={onFrameField}>Frame operative field</button>}
    </div>
    {fieldDisplayControls}
    {readOnly&&<p className="surgical-note">Read-only observed microstep. Return to a recorded action boundary to resume computation.</p>}
    {c.computingFullCase&&<p role="status">Computing {c.computingFullCase==='primary'?'primary rehearsal':'vascular injury / rescue'} · two actions per slice. Each next slice waits for the accepted display.</p>}
    {!c.computingFullCase&&c.simulationPending&&<p className="surgical-note">Current bounded request continues; no further slices will start automatically.</p>}
    {record&&<>
      <p role="status">{record.scenario==='primary'?'Primary rehearsal':'Vascular injury / rescue'} · {record.status.replaceAll('_',' ')}{phase?` · viewing ${phase.label}`:''}</p>
      {record.stopReason&&<p className="surgical-warning">{record.stopReason.replaceAll('_',' ')}</p>}
      <div className="full-case-phases" aria-label="Recorded rehearsal phases">{record.phases.map((item,index)=><button key={`${item.id}-${index}`} disabled={c.pending||item.endPosition<1} aria-current={phase===item?'step':undefined} onClick={()=>{pause();void c.run('seek',{position:item.endPosition});}}>{item.label}<small>States {item.startPosition}–{item.endPosition}</small></button>)}</div>
      {record.limitations?.length&&<details><summary>Rehearsal limits</summary><ul>{record.limitations.map((limit,i)=><li key={i}>{limit}</li>)}</ul></details>}
    </>}
    <div className="surgical-metrics" aria-label="Recorded source target accounting">
      <div><small>Original modeled target · mm³</small><strong>{amount(summary.initialTargetMm3)}</strong></div>
      <div><small>Target removed · mm³</small><strong>{amount(summary.targetRemovedMm3)}</strong></div>
      <div><small>Target remaining · mm³</small><strong>{amount(summary.targetRemainingMm3)}</strong></div>
      <div><small>Target removed · %</small><strong>{amount(summary.targetRemovedPercent)}</strong></div>
    </div>
    {record?.fullAnnotationTargetMm3!==undefined&&<p className="surgical-note">Full source annotation: {amount(record.fullAnnotationTargetMm3)} mm³. Displayed percentages use the modeled source grid target.</p>}
    <div className="surgical-metrics" aria-label="Recorded normal and unknown material accounting">
      <div><small>Non-target normal material removed · mm³</small><strong>{amount(summary.normalRemovedMm3)}</strong></div>
      <div><small>Mapped unknown material removed · mm³</small><strong>{amount(summary.unknownRemovedMm3)}</strong></div>
      <div><small>Normal-material exposure · cell·s</small><strong>{amount(summary.normalExposureCellSeconds)}</strong></div>
      <div><small>Unknown-material exposure · cell·s</small><strong>{amount(summary.unknownExposureCellSeconds)}</strong></div>
    </div>
    <p className="surgical-note">Normal removal excludes target labels; exposure is grouped by mapped material. Missing anatomy and functional injury remain UNKNOWN.</p>
    <div className="surgical-metrics" aria-label="Recorded bone wounds and fluid">
      <div><small>Bone removed · mm³</small><strong>{amount(summary.boneRemovedMm3)}</strong></div>
      <div><small>Modeled wounds / active outlets</small><strong>{amount(summary.woundCount,0)} / {amount(summary.activeWounds,0)}</strong></div>
      <div><small>Nonpatent local vessel faces</small><strong>{amount(summary.nonpatentFaces,0)}</strong></div>
      <div><small>Spatial fluid / outlet reservoirs · mm³</small><strong>{amount(summary.spatialFluidMm3)} / {amount(summary.reservoirFluidMm3)}</strong></div>
      <div><small>Modeled blood generated · mm³</small><strong>{amount(summary.modeledBloodGeneratedMm3)}</strong></div>
      <div><small>Fluid suctioned · mm³</small><strong>{amount(summary.fluidSuctionedMm3)}</strong></div>
      <div><small>Elapsed simulation · s</small><strong>{amount(summary.elapsedSeconds)}</strong></div>
    </div>
    <p className="surgical-note">Modeled blood uses the wound-source ledger; suction can include irrigation fluid. Blue glyphs scale with each cell's recorded fluid volume, not a reconstructed fluid surface. Reservoirs are counted separately; field clearance does not restore vessel patency.</p>
  </section>;
}
