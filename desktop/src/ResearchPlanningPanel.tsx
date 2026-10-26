import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {SurgicalPlanAlternative} from './surgical-types';
import {planDecisionReason,planReplayPosition,planLabel,planDecisionAtPosition} from './research-planning-data';
import {surgicalPlaybackPosition} from './surgical-playback';
import {ModeledOutcomesPanel} from './ModeledOutcomesPanel';

export function ResearchPlanningPanel({c,pause}:{c:SurgicalController;pause:()=>void}) {
  const [acknowledged,setAcknowledged]=useState(false),[brainCorridor,setBrainCorridor]=useState(false),[decision,setDecision]=useState(0);
  const readiness=c.result?.planningAvailability,comparison=c.result?.planning;
  useEffect(()=>{setAcknowledged(false);setBrainCorridor(false);setDecision(0);},[c.view?.scene.sceneId]);
  useEffect(()=>{setDecision(0);},[comparison?.startStateId,comparison?.totalWallSeconds]);
  const disabled=c.pending||!acknowledged||!readiness?.available;
  function compare(opponent:'stop'|'checkpoint'|'vascular-rescue'|'anatomy-access') {
    pause();void c.run('plan',{acknowledgeResearchUnknowns:acknowledged,opponent,...(opponent==='anatomy-access'?{allowModeledBrainCorridor:brainCorridor}:{})});
  }
  function show(plan:SurgicalPlanAlternative,step=decision) {
    const bounded=Math.min(step,plan.decisions.length);
    pause();void c.run('selectBranch',{branchId:plan.branchId,position:planReplayPosition(plan,bounded)}).then(ok=>{if(ok)setDecision(bounded);});
  }
  const active=comparison?.alternatives.find(p=>p.branchId===c.result?.branchId);
  const acceptedPosition=surgicalPlaybackPosition(c.result?.playback,c.result?.position??0);
  const beforeComparison=!!active&&acceptedPosition<active.prefixActions;
  useEffect(()=>{
    if(!active||c.pending)return;
    if(planReplayPosition(active,Math.min(decision,active.decisions.length))!==acceptedPosition)
      setDecision(planDecisionAtPosition(active,acceptedPosition));
  },[active,acceptedPosition,c.pending,decision]);
  const rescue=comparison?.task==='vascular-rescue';
  const access=comparison?.task==='anatomy-access';
  const vesselsMapped=!!c.view?.scene.regions.some(region=>region.material==='vessel');
  const protectionMapped=!!c.view?.protection?.structures.some(structure=>structure.sourceCells>0);
  const rescueContext=rescue||readiness?.rescueAvailable;
  const maxDecisions=Math.max(6,...(comparison?.alternatives.map(plan=>plan.decisions.length)??[]));
  return <details className="research-planning"><summary>Research Planning · compare approaches</summary>
    <p>Generate source-bound approaches through three different anatomy directions. Each executes in its own branch of this surgical environment, with the same starting state and 48 actions per approach and 120 seconds total.</p>
    <p>{rescue||readiness?.rescueAvailable?'Compare three recorded-source rescue sequences from the same verified injury boundary. No learning runs.':readiness?.scope??'Inspect a source-bound environment before planning.'}</p>
    <p className="surgical-warning">Development simulation. Missing vessels and functional anatomy remain unknown. Neurological harm is unvalidated.</p>
    {!rescue&&!readiness?.rescueAvailable&&!readiness?.available&&<p role="status">Planning unavailable: {readiness?.reasons.join(' ')??'Environment evidence is unavailable.'}</p>}
    <label><input aria-label="Acknowledge research planning assumptions" type="checkbox" checked={acknowledged} disabled={c.pending} onChange={e=>setAcknowledged(e.target.checked)}/>I acknowledge the declared starting state, missing anatomy and uncalibrated simulation.</label>
    <button disabled={c.pending||!acknowledged||!readiness?.accessAvailable} onClick={()=>compare('anatomy-access')}>Generate and execute three access approaches</button>
    <label><input aria-label="Allow modeled brain corridors" type="checkbox" checked={brainCorridor} disabled={c.pending} onChange={e=>setBrainCorridor(e.target.checked)}/>Include modeled corridors through the mapped brain layer (which may be an unreviewed estimate), capped at 200 mm³ non-target removal per approach. Skull access and missing critical anatomy remain unsupported.</label>
    {!readiness?.accessAvailable&&<p className="surgical-note">Access approaches unavailable: {readiness?.accessReason??'Source-anatomy readiness has not been checked.'}</p>}
    {!rescueContext&&<div className="surgical-actions"><button disabled={disabled} onClick={()=>compare('stop')}>Compare greedy / STOP</button><button disabled={disabled||!readiness?.checkpointAvailable} onClick={()=>compare('checkpoint')}>Compare greedy / saved learner</button></div>}
    {readiness?.rescueAvailable&&<button disabled={c.pending||!acknowledged} onClick={()=>compare('vascular-rescue')}>Execute three vascular rescue strategies</button>}
    {!rescueContext&&!readiness?.checkpointAvailable&&<p className="surgical-note">Saved learner unavailable: {readiness?.checkpointReason??readiness?.reasons.join(' ')}</p>}
    {!rescueContext&&readiness?.checkpointAvailable&&<p className="surgical-note">Seed 17 is fixed in advance. This learner previously removed no additional target in both development assessments; it is not promoted.</p>}
    {comparison&&<>
      <p><strong>{comparison.sourceRole}</strong> · {comparison.totalWallSeconds.toFixed(2)} s total · {comparison.optimizerUpdates} optimizer updates</p>
      {access&&<p role="status" className="surgical-warning">{comparison.status==='three_executable_source_approaches'?'Three bounded source-anatomy approaches executed.':comparison.status==='insufficient_distinct_routes'?`${comparison.alternatives.length} of 3 directions executed; the three-approach gate remains unmet.`:'Three bounded source-tissue routes executed; skull entry and craniotomy remain unsupported.'}</p>}
      {access&&!!comparison.directionFailures?.length&&<ul>{comparison.directionFailures.map(failure=><li key={failure.nativeAxis}>Direction {failure.nativeAxis}: {failure.reason}</li>)}</ul>}
      {access&&<div className="rescue-comparison"><table aria-label="Access approach outcomes"><caption>Executed model outputs · mm³ and simulated seconds</caption><thead><tr><th>Approach</th><th>Target removed</th><th>Non-target incl. bone</th><th>Residual target</th><th>Time</th></tr></thead><tbody>{comparison.alternatives.map(plan=><tr key={plan.branchId} aria-selected={c.result?.branchId===plan.branchId}><th><button disabled={c.pending} onClick={()=>show(plan)}>{planLabel(plan)}</button></th><td>{plan.incrementalEndpoints.tumor_removed_mm3.toFixed(2)}</td><td>{plan.summary?.nonTargetRemovedMm3?.toFixed(2)??'Unknown'}</td><td>{plan.summary?.residualTargetMm3?.toFixed(2)??'Unknown'}</td><td>{plan.incrementalEndpoints.elapsed_seconds.toFixed(1)}</td></tr>)}</tbody></table><p className="surgical-note">Each row opens its actual instrument actions and evolving anatomy. Bounded endpoints retain residual target. Missing skull, vessels and functional anatomy limit feasibility assessment.</p></div>}
      <p className="surgical-note">All alternatives start at the same recorded state. Selecting one uses the 3D/MPR replay, tool poses and cavity. {rescue?'These scripted rescue sequences compare intervention order and delay after the same modeled vessel injury. They do not compare tumor access corridors.':'The source branch is preserved.'}</p>
      {rescue&&<div className="rescue-comparison"><table aria-label="Vascular rescue outcomes"><caption>Model outputs · mm³ and simulated seconds</caption><thead><tr><th>Strategy</th><th>Wound</th><th>Retained fluid</th><th>Time</th></tr></thead><tbody>{comparison.alternatives.map(plan=><tr key={plan.branchId} aria-selected={c.result?.branchId===plan.branchId}><th><button disabled={c.pending} onClick={()=>show(plan)}>{plan.method==='prompt-coagulation'?'Coagulation first':plan.method==='suction-first'?'Suction first':'Delayed'}</button></th><td>{plan.summary?.additionalWoundFluidMm3?.toFixed(3)}</td><td>{plan.summary?.fluidPresentMm3?.toFixed(3)}</td><td>{plan.incrementalEndpoints.elapsed_seconds.toFixed(0)}</td></tr>)}</tbody></table><p className="surgical-note">Retained fluid includes outlet reservoirs. Select a row to inspect its real trajectory.</p></div>}
      <label>{beforeComparison?`Shared approach · comparison starts at action ${active!.prefixActions}`:`Aligned command ${decision} / ${maxDecisions}`}<input aria-label="Research planning decision" type="range" min="0" max={maxDecisions} value={decision} disabled={c.pending||!active||beforeComparison} onChange={e=>active&&show(active,+e.target.value)}/></label>
      <div className="research-plan-alternatives">{comparison.alternatives.map(plan=><article key={plan.branchId}>
        <button disabled={c.pending||!c.result?.branches.some(b=>b.id===plan.branchId)} aria-pressed={c.result?.branchId===plan.branchId} onClick={()=>show(plan)}>{planLabel(plan)} · replay</button>
        <p>{beforeComparison?'Replaying the shared approach before the comparison boundary.':planDecisionReason(plan,decision)}</p>
        {plan.route&&<p className="surgical-note">Source direction {plan.route.nativeAxis} · brain-surface entry {plan.route.surfacePose?.position.map(v=>v.toFixed(1)).join(', ')??'unavailable'} RAS mm · source geometry {plan.route.sourceSceneId??comparison.sourceSceneId}</p>}
        <details open={rescue?undefined:true}><summary>Outcome details</summary>
        <dl><div><dt>Additional target</dt><dd>{plan.incrementalEndpoints.tumor_removed_mm3.toFixed(2)} mm³</dd></div>
          <div><dt>Additional non-target soft tissue (excl. bone)</dt><dd>{plan.incrementalEndpoints.healthy_removed_mm3.toFixed(2)} mm³</dd></div>
          {access&&<div><dt>Source bone removed</dt><dd>{plan.summary?.boneRemovedMm3==null?'Unavailable · skull access unsupported':`${plan.summary.boneRemovedMm3.toFixed(2)} mm³`}</dd></div>}
          <div><dt>Cumulative target removal</dt><dd>{(100*plan.fullTargetFractionRemoved).toFixed(3)}%</dd></div>
          <div><dt>Reachable target removal</dt><dd>Unknown · no certified reachable set</dd></div>
          <div><dt>Modeled vessel damage</dt><dd>{vesselsMapped?`${plan.incrementalEndpoints.vessel_damage_mm2.toFixed(3)} mm² · coverage incomplete`:'Unavailable · no mapped vessel anatomy'}</dd></div>
          <div><dt>Protected-mask removal</dt><dd>{protectionMapped?`${plan.incrementalEndpoints.protected_damage_mm3.toFixed(3)} mm³ · supplied estimates only`:'Unavailable · no declared protection masks'}</dd></div>
          <div><dt>Instrument travel / simulated time</dt><dd>{plan.travelMm.toFixed(2)} mm / {plan.incrementalEndpoints.elapsed_seconds.toFixed(2)} s</dd></div>
          {rescue&&<><div><dt>Additional modeled wound fluid</dt><dd>{plan.summary?.additionalWoundFluidMm3?.toFixed(3)} mm³</dd></div><div><dt>Retained fluid incl. reservoirs / active outlets</dt><dd>{plan.summary?.fluidPresentMm3?.toFixed(3)} mm³ / {plan.summary?.activeVesselWounds}</dd></div><div><dt>Residual tumor</dt><dd>{plan.summary?.residualTargetMm3?.toFixed(2)} mm³</dd></div><div><dt>Protected-structure encounters</dt><dd>Unknown · complete anatomy unavailable</dd></div></>}
          <div><dt>Invalid actions / early STOP</dt><dd>{plan.invalidActions} / {plan.earlyStop?'yes':'no'}</dd></div>
          <div><dt>End reason</dt><dd>{plan.stopReason.replaceAll('_',' ')}{plan.truncated?' · truncated, incomplete plan':''}</dd></div>
          <div><dt>Execution / verification replay</dt><dd>{plan.cost.executionCalls} / {plan.cost.freshReplayCalls} calls · {(plan.cost.planningSeconds+plan.cost.freshReplaySeconds).toFixed(2)} s</dd></div>
        </dl>
        <ModeledOutcomesPanel outcome={plan.modeledOutcomes}/>
        <details><summary>Plan provenance and checked decisions</summary><p>{plan.verification.scope}</p><p>Exact state and measured endpoint vector: {plan.verification.exactStateMatch&&plan.verification.endpointVectorMatch?'match':'unverified'}.</p><p>Unmeasured endpoints: {plan.unassessedEndpoints.join(', ')||'none in this engineering vector'}.</p><p>Checkpoint: {plan.checkpointSha256??'none'}</p><p>Decision seal: {plan.planSeal}</p><ol>{plan.decisions.map(row=><li key={row.index}>{row.actionId} · {row.status}{!rescue&&` · reward ${row.reward.toFixed(5)}`}{row.reason&&` · ${row.reason}`}</li>)}</ol></details>
        </details>
      </article>)}</div>
      <details><summary>Evidence limits</summary><ul>{comparison.limitations.map(text=><li key={text}>{text}</li>)}</ul><p>Source scene: {comparison.sourceSceneId}</p><p>Exclusion contract: {comparison.guardContractId}</p></details>
    </>}
  </details>;
}
