import type {SurgicalController} from './use-surgical';
import type {ReactNode} from 'react';
import {recordedFullCasePhase} from './full-case-presentation';
import {surgicalPlaybackPosition} from './surgical-playback';

export function FullCaseRehearsalPanel({c,pause,onFrameField,fieldDisplayControls}:{c:SurgicalController;pause:()=>void;onFrameField?:()=>void;fieldDisplayControls?:ReactNode}) {
  const v=c.view,r=c.result;if(!v)return null;
  const record=r?.fullCase,readOnly=!!r?.readOnly||!!r?.playback||!!r?.verification?.readOnly,
    position=surgicalPlaybackPosition(r?.playback,r?.position??0),
    phase=recordedFullCasePhase(record,position);
  const run=(scenario:'primary'|'vascular')=>{pause();c.cancel();c.startFullCase(scenario);};
  const readiness=c.fullCaseReadiness,recovery=readiness?.recovery;
  return <section className="full-case-rehearsal" aria-label="Full Case Rehearsal">
    <strong>Full Case Rehearsal</strong>
    <p role="status" className={!readOnly&&readiness?.status==='ready'?'surgical-note':'surgical-warning'}>{readOnly?(r?.verification?.readOnly?'Full-case rehearsal blocked: verify the saved replay first.':'Full-case rehearsal blocked: return to a recorded boundary first.'):readiness?.status==='ready'?'Ready for full-case rehearsal':`Full-case rehearsal blocked${readiness?.blockers[0]?.message?`: ${readiness.blockers[0].message}`:': readiness for this case state is unavailable.'}`}</p>
    {recovery&&<button disabled={c.pending||readOnly||!readiness.availableActions.includes(recovery.action)} onClick={()=>{pause();void c.run(recovery.action==='seekInitial'?'seek':'attachProtection',recovery.action==='seekInitial'?{position:0}:{});}}>{recovery.label}</button>}
    <div className="surgical-actions">
      <button className="primary-button" disabled={c.pending||!c.canStartFullCase('primary')} onClick={()=>run('primary')}>{record?.scenario==='primary'?'Resume primary rehearsal':'Start primary rehearsal'}</button>
      {c.computingFullCase&&<button onClick={c.pauseFullCase}>Pause computation</button>}
      {onFrameField&&<button onClick={onFrameField}>Frame operative field</button>}
    </div>
    <details><summary>Rehearsal scenarios and evidence</summary>
    {readiness&&<details className="readiness-details"><summary>Readiness details</summary>
      {readiness.blockers.length?<ul>{readiness.blockers.map(blocker=><li key={blocker.code}>{blocker.message}{blocker.details.length>0&&<ul>{blocker.details.map((detail,index)=><li key={index}>{detail}</li>)}</ul>}</li>)}</ul>:<p>Required evidence is available for this case state.</p>}
      <p className="surgical-note">Protection contract: {readiness.protectionContractId??'None attached'}</p>
    </details>}
    <details><summary>Vascular injury / rescue scenario</summary>
      <button disabled={c.pending||!c.canStartFullCase('vascular')} onClick={()=>run('vascular')}>{record?.scenario==='vascular'?'Resume vascular injury / rescue':'Start vascular injury / rescue'}</button>
    </details>
    </details>
    {fieldDisplayControls}
    {c.computingFullCase&&<p role="status">Computing {c.computingFullCase==='primary'?'primary rehearsal':'vascular injury / rescue'} · two actions per slice. Each next slice waits for the accepted display.</p>}
    {!c.computingFullCase&&c.simulationPending&&<p className="surgical-note">Current bounded request continues; no further slices will start automatically.</p>}
    {record&&<>
      <p role="status">{record.scenario==='primary'?'Primary rehearsal':'Vascular injury / rescue'} · {c.computingFullCase?'computing':c.simulationPending?'request finishing':['ready','running'].includes(record.status)?'computation paused':record.status.replaceAll('_',' ')}{phase?` · viewing ${phase.label}`:''}</p>
      {record.stopReason&&<p className="surgical-warning">{record.stopReason.replaceAll('_',' ')}</p>}
      <details><summary>Recorded phases and rehearsal limits</summary>
      <p>Recorded director status: {record.status.replaceAll('_',' ')}</p>
      <label className="full-case-phase-jump">Recorded phase
        <select aria-label="Jump to recorded rehearsal phase" disabled={c.pending} value="" onChange={event=>{const item=record.phases[Number(event.target.value)];if(item&&item.endPosition>0){pause();void c.run('seek',{position:item.endPosition});}}}>
          <option value="" disabled>Jump to a recorded phase…</option>
          {record.phases.map((item,index)=><option key={`${item.id}-${index}`} value={index} disabled={item.endPosition<1}>{item.label} · states {item.startPosition}–{item.endPosition}</option>)}
        </select>
      </label>
      {record.limitations?.length&&<details><summary>Rehearsal limits</summary><ul>{record.limitations.map((limit,i)=><li key={i}>{limit}</li>)}</ul></details>}
      </details>
    </>}

  </section>;
}
