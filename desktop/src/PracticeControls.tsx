import {useRef, useState} from 'react';
import type {KeyboardEvent, ReactNode} from 'react';
import type {SurgicalAction, SurgicalOutcome, SurgicalPose} from './surgical-types';
import type {SurgicalController} from './use-surgical';
import {CranialPracticeTask} from './CranialPracticeTask';
import {PracticeHoldControls} from './PracticeHoldControls';
import {amount} from './SurgicalOutcomeSummary';
import {jogPracticePose, practiceDrag, practiceJogKey, practiceWithdrawalPose} from './practice-input';

const coordinates = (point: number[]) => point.map(n => amount(n, 2)).join(', ');

export function PracticeControls({c, blocked, settings, params, duration}: {
  c: SurgicalController;
  blocked: boolean;
  settings: ReactNode;
  params: Record<string, number>;
  duration: number;
}) {
  const drag = useRef<{id: number; x: number; y: number; pose: SurgicalPose; depth: boolean} | null>(null);
  const [continuous,setContinuous] = useState(false);
  const [request, setRequest] = useState<{action: SurgicalAction; position: number; previousOutcome: SurgicalOutcome | null} | null>(null);
  const result = c.result, view = c.view, slot = view?.slots[c.slot];
  if (!result || !view || !slot) return null;
  const tool = view.catalog.find(item => item.id === slot.instrument);
  const parent = result.practiceEpisode!.parent;
  const outcome = result.outcome;
  const actual = slot.position;
  const draft = c.pose;
  const dot = slot.quaternion.reduce((sum,value,i) => sum + value * draft.quaternion[i], 0);
  const tiltDegrees = 2 * Math.acos(Math.min(1,Math.abs(dot))) * 180 / Math.PI;
  const proposed = actual.some((n, i) => Math.abs(n - draft.position[i]) > .0001) ||
    tiltDegrees > .01 || c.path.length > 0;
  const withdrawal = practiceWithdrawalPose(result,c.slot) ?? (proposed ? draft : undefined);
  const send = (operation: string, instrument?: string) => {
    if (blocked) return;
    const action: SurgicalAction = operation === 'exchange' ? {slot: c.slot, operation, instrument, duration} : c.action(operation, params, duration);
    if (operation === 'withdraw') {
      if (!withdrawal) return;
      action.pose = withdrawal;
    }
    setRequest({action, position:result.position, previousOutcome:result.outcome});
    void c.runPracticeAction(action);
  };
  function jogKey(event: KeyboardEvent<HTMLButtonElement>) {
    if (blocked || event.altKey || event.metaKey || event.ctrlKey) return;
    const jog = practiceJogKey(event.key, event.shiftKey);
    if (jog) {
      event.preventDefault();
      c.setPose(jogPracticePose(draft, jog));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (!event.repeat) send('move');
    } else if (event.key === 'Escape') {
      event.preventDefault();
      c.cancel();
    }
  }
  const lastRequested = request?.action.path?.at(-1)?.position;
  const attempted = !!request && !!outcome && request.previousOutcome !== outcome && request.action.slot === c.slot &&
    result.position === request.position + (outcome.status === 'rejected' ? 0 : 1);
  return <section className="practice-controls" aria-label="Instrument practice" aria-busy={c.pending}
    data-practice-episode={result.branchId} data-practice-state-id={view.stateId} data-practice-action-status={outcome?.status ?? 'ready'}
    data-practice-accepted-position={JSON.stringify(actual)} data-practice-proposed-position={JSON.stringify(draft.position)}>
    <CranialPracticeTask result={result}/>
    <fieldset disabled={blocked} className="surgical-controls">
      <label>Instrument<select aria-label="Practice instrument" value={slot.instrument} onChange={e => send('exchange', e.target.value)}>
        {view.catalog.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label>
      <div className="practice-tools">
        <button onClick={() => send('place')}>Place tool</button>
        <button onClick={() => send('move')}>Move tool</button>
        <button className="primary-button" onClick={() => send('apply')}>Apply tool</button>
        <button onClick={() => send(slot.active ? 'deactivate' : 'activate')}>{slot.active ? 'Deactivate tool' : 'Activate tool'}</button>
        {tool?.effect === 'retract' && <button onClick={() => send('hold')}>Hold tissue</button>}
        {tool?.effect === 'grasp' && <><button onClick={() => send('grasp')}>Grasp tissue</button><button onClick={() => send('extract')}>Extract tissue</button></>}
        <button onClick={() => send('inspect')}>Observe interval</button>
        <button onClick={() => send('release')}>Release tool</button>
        <button disabled={!withdrawal} title={withdrawal ? `Check the route to ${coordinates(withdrawal.position)} mm RAS` : 'Place the instrument outside or propose an exterior tip pose first'} onClick={() => send('withdraw')}>Withdraw tool</button>
      </div>
    </fieldset>
    <div role="group" aria-label="Instrument control mode" className="practice-mode">
      <button type="button" aria-pressed={!continuous} disabled={blocked} onClick={()=>setContinuous(false)}>Jog and commit</button>
      <button type="button" aria-pressed={continuous} disabled={blocked} onClick={()=>{if(!continuous)c.cancel();setContinuous(true);}}>Hold controls</button>
    </div>
    {continuous ? <PracticeHoldControls c={c} blocked={blocked} params={params} onRequest={(action,origin)=>{
      setRequest({action,position:origin.position,previousOutcome:origin.outcome});
    }}/> :
    <fieldset disabled={blocked} className="surgical-controls">
      <button type="button" aria-label="Instrument jog pad" aria-describedby="practice-jog-help" className="practice-jog"
        onKeyDown={jogKey}
        onPointerDown={event => {
          if (blocked || event.button !== 0 || !event.isPrimary) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = {id:event.pointerId, x:event.clientX, y:event.clientY, pose:draft, depth:event.shiftKey};
        }}
        onPointerMove={event => {
          const start = drag.current;
          if (!blocked && start?.id === event.pointerId && event.buttons === 1)
            c.setPose(jogPracticePose(start.pose, practiceDrag(event.clientX - start.x, event.clientY - start.y, start.depth)));
        }}
        onPointerUp={() => {drag.current = null;}} onLostPointerCapture={() => {drag.current = null;}}
        onPointerCancel={() => {drag.current = null;c.cancel();}}>
        <strong>Jog instrument</strong><span>Drag across shaft · Shift-drag for depth</span>
      </button>

    </fieldset>}
    {slot.inserted&&!withdrawal&&<p className="surgical-warning">Propose an exterior pose for checked withdrawal. This hand has no recorded exterior placement.</p>}
    <div className="practice-pose" aria-live="polite">
      <p><strong>Accepted tip</strong> {coordinates(actual)} mm RAS · {slot.active ? 'active' : 'inactive'}</p>
      {proposed && <p><strong>Proposed tip</strong> {coordinates(draft.position)} mm RAS{tiltDegrees > .01 ? ' · '+amount(tiltDegrees,1)+'° from accepted' : ''} · {c.path.length ? c.path.length+' stroke points' : 'uncommitted'} <button disabled={blocked} onClick={c.cancel}>Discard</button></p>}
      {attempted && lastRequested && outcome.status !== 'accepted' && <p>Requested tip {coordinates(lastRequested)} mm RAS</p>}
    </div>
    {!continuous&&<p id="practice-jog-help" className="surgical-note">On the jog pad, arrows shift 0.5 mm, Page Down / Up advances or retracts, and Shift + arrows tilts 2°. Enter moves to the proposal. Escape discards it.</p>}
    {settings}
      <details><summary>Source checkpoint and model limits</summary>
        <p>Parent {parent.branchId} · action {parent.position}</p><p>{parent.stateId}</p>
        <p>Volumes use the engine source grid. Unsupported neural membership, vascular risk and functional outcomes are unknown. Neurological outcome probabilities are unavailable.</p>
      </details>
  </section>;
}
