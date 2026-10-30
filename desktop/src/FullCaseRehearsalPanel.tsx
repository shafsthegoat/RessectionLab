import type {SurgicalController} from './use-surgical';
import type {ReactNode} from 'react';
import {fullCaseStateSummary,recordedFullCasePhase} from './full-case-presentation';
import {surgicalPlaybackPosition} from './surgical-playback';
const amount=(value:number|null|undefined,digits=2)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'Unknown';

export function FullCaseRehearsalPanel({c,pause,onFrameField,fieldDisplayControls}:{c:SurgicalController;pause:()=>void;onFrameField?:()=>void;fieldDisplayControls?:ReactNode}) {
  const v=c.view,r=c.result;if(!v)return null;
  const summary=fullCaseStateSummary(v),record=r?.fullCase,readOnly=!!r?.readOnly||!!r?.playback||!!r?.verification?.readOnly,
    position=surgicalPlaybackPosition(r?.playback,r?.position??0),
    phase=recordedFullCasePhase(record,position);
  const run=(scenario:'primary'|'vascular')=>{pause();c.cancel();c.startFullCase(scenario);};
  const readiness=c.fullCaseReadiness,recovery=readiness?.recovery;
  return <section className="full-case-rehearsal" aria-label="Full Case Rehearsal">
    <strong>Full Case Rehearsal</strong>
    <p className="surgical-note">Compute the rehearsal in bounded slices, recording each action for replay. Pause stops after the current slice finishes.</p>
    <p role="status" className={readiness?.status==='ready'?'surgical-note':'surgical-warning'}>{readiness?.status==='ready'?'Ready for full-case rehearsal':`Full-case rehearsal blocked${readiness?.blockers[0]?.message?`: ${readiness.blockers[0].message}`:': readiness for this case state is unavailable.'}`}</p>
    {readiness&&<details><summary>Readiness details</summary>
      {readiness.blockers.length?<ul>{readiness.blockers.map(blocker=><li key={blocker.code}>{blocker.message}{blocker.details.length>0&&<ul>{blocker.details.map((detail,index)=><li key={index}>{detail}</li>)}</ul>}</li>)}</ul>:<p>Required evidence is available for this case state.</p>}
      <p className="surgical-note">Protection contract: {readiness.protectionContractId??'None attached'}</p>
    </details>}
    {recovery&&<button disabled={c.pending||readOnly||!readiness.availableActions.includes(recovery.action)} onClick={()=>{pause();void c.run(recovery.action==='seekInitial'?'seek':'attachProtection',recovery.action==='seekInitial'?{position:0}:{});}}>{recovery.label}</button>}
    <div className="surgical-actions">
      <button disabled={c.pending||!c.canStartFullCase('primary')} onClick={()=>run('primary')}>{record?.scenario==='primary'?'Resume primary rehearsal':'Start primary rehearsal'}</button>
      <button disabled={c.pending||!c.canStartFullCase('vascular')} onClick={()=>run('vascular')}>{record?.scenario==='vascular'?'Resume vascular injury / rescue':'Start vascular injury / rescue'}</button>
      {c.computingFullCase&&<button onClick={c.pauseFullCase}>Pause computation</button>}
      {onFrameField&&<button onClick={onFrameField}>Frame operative field</button>}
    </div>
    {fieldDisplayControls}
    {readOnly&&<p className="surgical-note">{r?.verification?.status==='reconstructed-prefix'?'Read-only reconstructed prefix. Verify the full archive before computation.':r?.verification?.readOnly?'Imported endpoint requires replay verification before computation.':'Read-only observed microstep. Return to a recorded action boundary to resume computation.'}</p>}
    {c.computingFullCase&&<p role="status">Computing {c.computingFullCase==='primary'?'primary rehearsal':'vascular injury / rescue'} · two actions per slice. Each next slice waits for the accepted display.</p>}
    {!c.computingFullCase&&c.simulationPending&&<p className="surgical-note">Current bounded request continues; no further slices will start automatically.</p>}
    {record&&<>
      <p role="status">{record.scenario==='primary'?'Primary rehearsal':'Vascular injury / rescue'} · {record.status.replaceAll('_',' ')}{phase?` · viewing ${phase.label}`:''}</p>
      {record.stopReason&&<p className="surgical-warning">{record.stopReason.replaceAll('_',' ')}</p>}
      <label className="full-case-phase-jump">Recorded phase
        <select aria-label="Jump to recorded rehearsal phase" disabled={c.pending} value="" onChange={event=>{const item=record.phases[Number(event.target.value)];if(item&&item.endPosition>0){pause();void c.run('seek',{position:item.endPosition});}}}>
          <option value="" disabled>Jump to a recorded phase…</option>
          {record.phases.map((item,index)=><option key={`${item.id}-${index}`} value={index} disabled={item.endPosition<1}>{item.label} · states {item.startPosition}–{item.endPosition}</option>)}
        </select>
      </label>
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
