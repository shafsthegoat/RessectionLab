import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {SurgicalSectorPlan} from './surgical-types';
import {planReplayPosition} from './research-planning-data';
import {ModeledOutcomesPanel} from './ModeledOutcomesPanel';

export function SectorStrategiesPanel({c,pause}:{c:SurgicalController;pause:()=>void}){
  const [acknowledged,setAcknowledged]=useState(false);
  useEffect(()=>setAcknowledged(false),[c.result?.caseHash]);
  const source=c.result?.planning;
  const comparison=c.result?.sectorPlanning;
  if(source?.task!=='anatomy-access'&&!comparison)return null;
  const intact=!!source&&c.view?.stateId===source.startStateId;
  const show=(plan:SurgicalSectorPlan,boundary:'sector'|'source'|'endpoint')=>{pause();void c.run('selectBranch',{branchId:plan.branchId,position:boundary==='endpoint'?planReplayPosition(plan,plan.decisions.length):boundary==='sector'?(plan.sourceAccessPrefixActions??0):0});};
  return <details className="research-planning sector-strategies" open><summary>Resection order and tool arrangement</summary>
    <p>{source?.sourceRole?.startsWith('Joint source-only shortlist')?
      'This saved joint comparison executed one fixed far-first sector order with left CUSA for each retained access route. Each action checked the full tool and other hand in the existing engine.':
      'Compare source-defined target sectors in different orders with left or right CUSA. Each action checks the full tool and other hand in the existing engine.'}</p>
    <p className="surgical-warning">Partial anatomy and uncalibrated mechanics. These are bounded tasks with substantial residual target; complete surgery and clinical feasibility remain unassessed.</p>
    <label><input type="checkbox" checked={acknowledged} disabled={c.pending} onChange={e=>setAcknowledged(e.target.checked)}/>Use the recorded source anatomy and research assumptions for this comparison.</label>
    {!intact&&source?.alternatives[0]&&<button disabled={c.pending} onClick={()=>{pause();const start=c.result?.branches.find(item=>item.steps===0);void c.run(start?'selectBranch':'reset',start?{branchId:start.id,position:0}:{});}}>Return to common source start</button>}
    <button disabled={c.pending||!acknowledged||!intact} onClick={()=>{pause();void c.run('plan',{opponent:'anatomy-strategy',acknowledgeResearchUnknowns:true});}}>Generate sector and hand strategies</button>
    {comparison&&<>
      <p role="status">{comparison.alternatives.filter(plan=>plan.continuationOutcome?.taskGateMet).length} / {comparison.alternatives.length} attempted strategies passed the declared task gate.</p>
      <p className="surgical-note">The gate requires 30 mm³ additional target, 10 mm³ in each declared sector, at most 20 mm³ additional and 200 mm³ total non-target removal, and checked cleanup.</p>
      <p className="surgical-note">A row opens the recorded sector starting boundary. The full timeline retains the intact source and access actions.</p>
      <div className="rescue-comparison"><table aria-label="Sector strategy outcomes"><caption>Additional modeled removal after access · mm³</caption><thead><tr><th>Strategy</th><th>Hand / order</th><th>Target</th><th>Non-target</th><th>Gate / cleanup</th></tr></thead><tbody>{comparison.alternatives.map(plan=><tr key={plan.branchId} aria-selected={plan.branchId===c.result?.branchId}><th><button disabled={c.pending} data-sector-branch={plan.branchId} data-sector-start={plan.sourceAccessPrefixActions??0} data-sector-end={plan.decisions.length} data-sector-hand={plan.strategy?.hand} data-sector-order={plan.strategy?.sectorOrder.join(',')} data-sector-cleanup={String(!!plan.continuationOutcome?.cleanupComplete)} data-sector-gate={String(!!plan.continuationOutcome?.taskGateMet)} onClick={()=>show(plan,'sector')}>{plan.label??plan.method}</button></th><td>{plan.strategy?.hand??'Unknown'} · {plan.strategy?.sectorOrder.join(' → ')??'Unknown'}</td><td>{plan.continuationOutcome?.deltaTargetMm3.toFixed(2)??'Unavailable'}</td><td>{plan.continuationOutcome?.deltaNonTargetMm3.toFixed(2)??'Unavailable'}</td><td>{plan.continuationOutcome?.taskGateMet?'Passed':'Unpassed'} / {plan.continuationOutcome?.cleanupComplete?'Complete':'Incomplete'}</td></tr>)}</tbody></table></div>
      {comparison.alternatives.map(plan=><details key={plan.branchId}><summary>{plan.label??plan.method} · evidence</summary>
        <button disabled={c.pending} onClick={()=>show(plan,'endpoint')}>Inspect strategy endpoint</button>
        <button disabled={c.pending} onClick={()=>show(plan,'source')}>Replay from intact source</button>
        <p>End status: {plan.status}. {plan.failureReasons.map(reason=>reason.replaceAll('_',' ')).join('; ')}</p>
        <p>Observed first progress: {plan.continuationOutcome?.observedFirstProgressOrder?.join(' → ')||'None'}.</p>
        <p>Per-sector additional target: {Object.entries(plan.continuationOutcome?.perSectorRemovedMm3??{}).map(([name,value])=>`${name}: ${value.toFixed(2)} mm³`).join(' · ')||'Unavailable'}.</p>
        <p>{plan.decisions.length} recorded decisions · {plan.cost.executionCalls} committed action calls · {plan.cost.previewCalls} previews · {plan.cost.freshReplayCalls} verification calls.</p>
        {!!plan.escapePreflights?.length&&<details className="sector-exit-checks"><summary>Exit checks before sector actions</summary>
          <p>Each proposed sector option was checked with withdrawal on a separate native branch before admission. Refused options leave the current accepted state unchanged.</p>
          <ul>{plan.escapePreflights.map((probe,index)=><li key={`${probe.optionId}-${index}`}>{probe.optionId} at action {probe.atActionPosition}: {probe.status==='passed'?'Checked exit available':'Option refused'}{probe.reason?` · ${probe.reason.replaceAll('_',' ')}`:''}. {probe.previewCalls} discarded probe previews and {probe.committedCalls} discarded probe transitions.</li>)}</ul>
          <p>{plan.cost.totalPhysicalCalls??plan.cost.previewCalls+plan.cost.executionCalls+plan.cost.freshReplayCalls+(plan.cost.preflightPreviewCalls??0)+(plan.cost.preflightCommittedCalls??0)} total native calls, including discarded probes and verification.</p>
        </details>}
        <ModeledOutcomesPanel outcome={plan.modeledOutcomes}/>
        <details><summary>Ordered native decisions</summary><ol>{plan.decisions.map(row=><li key={row.index}>{row.actionId??`${row.action.slot}: ${row.action.instrument??row.action.operation}`} · {row.action.duration.toFixed(2)} s · {row.status}{row.reason?` · ${row.reason}`:''}</li>)}</ol></details>
      </details>)}
      <details><summary>Source and uncertainty</summary><p>Source scene: {comparison.sourceSceneId}</p><p>Source access comparison: {comparison.sourceAccessComparisonSeal}</p><ul>{comparison.limitations.map(text=><li key={text}>{text}</li>)}</ul></details>
    </>}
  </details>;
}
