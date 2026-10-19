import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {SurgicalPlanAlternative} from './surgical-types';
import {planDecisionReason,planReplayPosition} from './research-planning-data';

export function ResearchPlanningPanel({c,pause}:{c:SurgicalController;pause:()=>void}) {
  const [acknowledged,setAcknowledged]=useState(false),[decision,setDecision]=useState(0);
  const readiness=c.result?.planningAvailability,comparison=c.result?.planning;
  useEffect(()=>{setAcknowledged(false);setDecision(0);},[c.view?.scene.sceneId]);
  useEffect(()=>{setDecision(0);},[comparison?.startStateId,comparison?.totalWallSeconds]);
  const disabled=c.pending||!acknowledged||!readiness?.available;
  function compare(opponent:'stop'|'checkpoint') {
    pause();void c.run('plan',{acknowledgeResearchUnknowns:acknowledged,opponent});
  }
  function show(plan:SurgicalPlanAlternative,step=decision) {
    pause();setDecision(step);void c.run('selectBranch',{branchId:plan.branchId,position:planReplayPosition(plan,step)});
  }
  const active=comparison?.alternatives.find(p=>p.branchId===c.result?.branchId);
  return <details className="research-planning"><summary>Research Planning · compare continuations</summary>
    <p>{readiness?.scope??'Inspect a source-bound environment before planning.'}</p>
    <p className="surgical-warning">Development simulation. Missing vessels and functional anatomy remain unknown. Neurological harm is unvalidated.</p>
    {!readiness?.available&&<p role="status">Planning unavailable: {readiness?.reasons.join(' ')??'Environment evidence is unavailable.'}</p>}
    <label><input aria-label="Acknowledge research planning assumptions" type="checkbox" checked={acknowledged} disabled={c.pending} onChange={e=>setAcknowledged(e.target.checked)}/>I acknowledge the post-exposure setup, missing anatomy and uncalibrated simulation.</label>
    <div className="surgical-actions"><button disabled={disabled} onClick={()=>compare('stop')}>Compare greedy / STOP</button><button disabled={disabled||!readiness?.checkpointAvailable} onClick={()=>compare('checkpoint')}>Compare greedy / saved learner</button></div>
    {!readiness?.checkpointAvailable&&<p className="surgical-note">Saved learner unavailable: {readiness?.checkpointReason??readiness?.reasons.join(' ')}</p>}
    {readiness?.checkpointAvailable&&<p className="surgical-note">Seed 17 is fixed in advance. This learner previously removed no additional target in both development assessments; it is not promoted.</p>}
    {comparison&&<>
      <p><strong>{comparison.sourceRole}</strong> · {comparison.totalWallSeconds.toFixed(2)} s total · {comparison.optimizerUpdates} optimizer updates</p>
      <p className="surgical-note">Both alternatives start at the same recorded state. Selecting one uses the existing 3D/MPR replay, tool poses and cavity. The source branch is preserved.</p>
      <label>Aligned decision {decision} / 6<input aria-label="Research planning decision" type="range" min="0" max="6" value={decision} disabled={c.pending||!active} onChange={e=>active&&show(active,+e.target.value)}/></label>
      <div className="research-plan-alternatives">{comparison.alternatives.map(plan=><article key={plan.branchId}>
        <button disabled={c.pending||!c.result?.branches.some(b=>b.id===plan.branchId)} aria-pressed={c.result?.branchId===plan.branchId} onClick={()=>show(plan)}>{plan.method==='checkpoint'?'Frozen learner · seed 17':plan.method==='greedy'?'Observation-only greedy':'Immediate STOP'} · replay</button>
        <p>{planDecisionReason(plan,decision)}</p>
        <dl><div><dt>Additional target</dt><dd>{plan.incrementalEndpoints.tumor_removed_mm3.toFixed(2)} mm³</dd></div>
          <div><dt>Additional non-target tissue</dt><dd>{plan.incrementalEndpoints.healthy_removed_mm3.toFixed(2)} mm³</dd></div>
          <div><dt>Cumulative target removal</dt><dd>{(100*plan.fullTargetFractionRemoved).toFixed(3)}%</dd></div>
          <div><dt>Reachable target removal</dt><dd>Unknown · no certified reachable set</dd></div>
          <div><dt>Modeled vessel damage</dt><dd>{plan.incrementalEndpoints.vessel_damage_mm2.toFixed(3)} mm² · coverage incomplete</dd></div>
          <div><dt>Protected-mask removal</dt><dd>{plan.incrementalEndpoints.protected_damage_mm3.toFixed(3)} mm³ · supplied estimates only</dd></div>
          <div><dt>Instrument travel / simulated time</dt><dd>{plan.travelMm.toFixed(2)} mm / {plan.incrementalEndpoints.elapsed_seconds.toFixed(2)} s</dd></div>
          <div><dt>Invalid actions / early STOP</dt><dd>{plan.invalidActions} / {plan.earlyStop?'yes':'no'}</dd></div>
          <div><dt>End reason</dt><dd>{plan.stopReason.replaceAll('_',' ')}{plan.truncated?' · truncated, incomplete plan':''}</dd></div>
          <div><dt>Execution / fresh replay</dt><dd>{plan.cost.executionCalls} / {plan.cost.freshReplayCalls} calls · {(plan.cost.planningSeconds+plan.cost.freshReplaySeconds).toFixed(2)} s</dd></div>
        </dl>
        <details><summary>Plan provenance and checked decisions</summary><p>{plan.verification.scope}</p><p>Exact state and measured endpoint vector: {plan.verification.exactStateMatch&&plan.verification.endpointVectorMatch?'match':'unverified'}.</p><p>Unmeasured endpoints: {plan.unassessedEndpoints.join(', ')||'none in this engineering vector'}.</p><p>Checkpoint: {plan.checkpointSha256??'none'}</p><p>Decision seal: {plan.planSeal}</p><ol>{plan.decisions.map(row=><li key={row.index}>{row.actionId} · {row.status} · reward {row.reward.toFixed(5)}{row.reason&&` · ${row.reason}`}</li>)}</ol></details>
      </article>)}</div>
      <details><summary>Evidence limits</summary><ul>{comparison.limitations.map(text=><li key={text}>{text}</li>)}</ul><p>Source scene: {comparison.sourceSceneId}</p><p>Exclusion contract: {comparison.guardContractId}</p></details>
    </>}
  </details>;
}
