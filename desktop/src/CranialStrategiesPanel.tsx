import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {WorkflowAlternatives,WorkflowPlanningShell} from './WorkflowAlternatives';
import type {WorkflowPlanningPresentation} from './WorkflowAlternatives';
import {cranialAvailability,cranialComparison,cranialGenerationIssue,cranialWorkflowAlternative,cranialTargetBasisLabel} from './cranial-workflow-data';
const amount=(v:unknown)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(2):'Unassessed';
const quantityNames:Record<string,string>={toolExchanges:'Instrument exchanges',acceptedActions:'Accepted actions',brainStrainSquaredMm3S:'Accumulated strain exposure',brainStressSquaredPa2Mm3S:'Accumulated stress exposure',thermalPeakC:'Peak temperature',thermalExposureMm3KS:'Thermal exposure',bloodGeneratedMm3:'Modeled blood generation',motorEncounterMm3:'Motor mapping',languageEncounterMm3:'Language mapping'};
export function CranialStrategiesPanel({c,pause,play,stage,onCompared,onInstalled}:WorkflowPlanningPresentation&{c:SurgicalController;pause:()=>void;play:(branchId:string)=>void}) {
  const [ack,setAck]=useState(false),[selected,setSelected]=useState<string|null>(null);
  const a=cranialAvailability(c.result),comparison=cranialComparison(c.result),issue=cranialGenerationIssue(c.result);
  useEffect(()=>{setAck(false);setSelected(null);},[c.result?.caseHash,a?.sourceSceneId,a?.sourceStateId]);
  const choose=async(branchId:string)=>{
    const row=comparison?.alternatives.find(x=>x.branchId===branchId);if(!row||c.pending)return;
    pause();c.clearPreview();setSelected(branchId);
    if(await c.run('selectBranch',{branchId:row.branchId,position:row.nativeActions}))onInstalled?.();
  };
  return <WorkflowPlanningShell stage={stage} title="Opening, preparation and target strategy" className="cranial-strategies">
    <div hidden={!!stage&&stage!=='plan'}>
      <p>Generate complete sequences through a source-bound opening, supported preparation, target work and instrument exit. All opening and preparation costs remain in the whole sequence.</p>
      {!!a?.roots.length&&<ul aria-label="Source-bound opening options">{a.roots.map(root=><li key={root.rootId}><strong>{root.label}</strong> · {root.openingMethod} · {root.available?'Available for checked planning':'Blocked'}{root.blockers.length>0&&<ul>{root.blockers.map((b,i)=><li key={`${b.code}:${i}`}>{b.message} <small>({b.code})</small></li>)}</ul>}</li>)}</ul>}
      {issue&&<p role="status" className="surgical-warning">{issue}</p>}
      {!!a?.blockers.length&&<ul aria-label="Cranial planning blockers">{a.blockers.map((b,i)=><li key={`${b.code}:${i}`}>{b.message} <small>({b.code})</small></li>)}</ul>}
      <label><input type="checkbox" aria-label="Acknowledge cranial research context" checked={ack} disabled={c.pending} onChange={e=>setAck(e.target.checked)}/>I understand the declared research context and unavailable clinical outcomes.</label>
      <p className="surgical-note">Acknowledgement does not override missing anatomy, protection or UNKNOWN exclusions. Prior scalp exposure cost may remain unknown.</p>
      <button className="primary-button" disabled={c.pending||!!c.preview||!ack||!!issue} onClick={()=>{pause();void c.run('plan',{opponent:'anatomy-cranial-strategy',acknowledgeResearchUnknowns:true,planningHorizon:5,beamWidth:3}).then(ok=>{if(ok)onCompared?.();});}}>Generate strategies</button>
      {c.pending&&<p role="status">Checking opening, preparation, target work and complete exit…</p>}
    </div>
    <div hidden={!!stage&&stage!=='compare'}>
      {c.result?.cranialPlanning&&!comparison&&<p role="alert" className="surgical-warning">The cranial comparison could not be bound to its source roots, complete ledger and saved native branches. Regenerate from a qualified source.</p>}
      {!c.result?.cranialPlanning&&<p role="status">Generate a checked comparison from Plan. No complete tumor-treatment sequence has been established.</p>}
      {comparison&&<>
        <p className="surgical-note">Recorded comparison. Current branch edits do not recalculate these results; generate again after changes.</p>
        <p>{comparison.decision.replaceAll('_',' ')}. Bone clearance alone is not target-task completion.</p>
        <WorkflowAlternatives rows={comparison.alternatives.map(row=>cranialWorkflowAlternative(row,comparison.selectedBranchId))} selectedId={selected??comparison.selectedBranchId} disabled={c.pending||!!c.preview} onSelect={id=>void choose(id)}/>
        {!comparison.alternatives.length&&<p>No recorded complete or progress sequence was returned. This bounded search does not prove anatomical impossibility.</p>}
        {!!comparison.failures.length&&<ul aria-label="Strategy refusals">{comparison.failures.map((f,i)=><li key={`${f.code}:${i}`}>{f.message} <small>({f.code})</small></li>)}</ul>}
        {comparison.alternatives.map(row=><details key={row.branchId}><summary>{row.label} · whole sequence ledger and replay</summary>
          <dl><div><dt>Opening bone removed</dt><dd>{amount(row.ledger.boneRemovedMm3)} mm³</dd></div><div><dt>Preparation nonbone removed</dt><dd>{amount(row.ledger.preparationNonboneRemovedMm3)} mm³</dd></div><div><dt>Work non-target removed</dt><dd>{amount(row.ledger.workNonTargetRemovedMm3)} mm³</dd></div><div><dt>Whole non-target removed</dt><dd>{amount(row.ledger.wholeNonTargetRemovedMm3)} mm³</dd></div>
            <div><dt>Original target basis</dt><dd>{cranialTargetBasisLabel(row.goalAssessment.targetBasis)} · {row.goalAssessment.originalTargetCells} cells · {amount(row.goalAssessment.originalTargetVolumeMm3)} mm³</dd></div><div><dt>Original target removed</dt><dd>{amount(row.goalAssessment.originalTargetRemovedMm3)} mm³</dd></div><div><dt>Credited target treatment</dt><dd>{amount(row.goalAssessment.originalTargetTreatmentMm3)} mm³ · {(row.goalAssessment.originalTargetTreatmentFraction*100).toFixed(1)}%</dd></div></dl>
          <p>Whole-body exit {row.wholeExitComplete&&row.sequenceOutcome.withdrawalComplete?'complete':'incomplete'}. {row.ledger.priorExposureCost===undefined?'Prior exposure cost is unknown.':String(row.ledger.priorExposureCost)}</p>
          <table aria-label={`${row.label} shared consequences`}><thead><tr><th>Quantity</th><th>Whole sequence</th><th>Evidence</th></tr></thead><tbody>{Object.entries(row.sequenceOutcome.dimensions).filter(([,q])=>q&&typeof q==='object'&&'unit' in q).map(([key,q])=>{const value=q as {unit:string;status:string;cumulative:number|null};return <tr key={key}><th>{quantityNames[key]??key.replaceAll(/([A-Z])/g,' $1')}</th><td>{amount(value.cumulative)} {value.unit}</td><td>{value.status}</td></tr>;})}</tbody></table>
          <p>Unavailable evidence: {row.sequenceOutcome.missingEvidence.join('; ')||'Completeness is not established'}.</p>
          <button disabled={c.pending||!!c.preview} onClick={()=>{pause();play(row.branchId);}}>Replay {row.label}</button>
          <p className="surgical-note">Playback uses this saved native branch. Edits require a new comparison; recorded outcomes do not recalculate. Engineering task completion does not grant patient qualification.</p>
        </details>)}
        <p>Whole search: {comparison.cost.physicalCalls} physical calls ({comparison.cost.previewCalls} previews + {comparison.cost.directStepCalls} direct steps) · {amount(comparison.cost.wallSeconds)} s including all roots and export/replay.</p>
        <ul>{comparison.limitations.map((line,i)=><li key={i}>{line}</li>)}</ul>
      </>}
    </div>
  </WorkflowPlanningShell>;
}
