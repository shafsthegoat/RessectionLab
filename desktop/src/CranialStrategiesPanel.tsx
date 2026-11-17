import type {Point3} from './viewer/coordinates';
import {checkedCranialGeometry} from './cranial-recorded-geometry';
import {declaredStructureColor} from './interactive-planning-data';
import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {WorkflowAlternatives,WorkflowPlanningShell} from './WorkflowAlternatives';
import type {WorkflowPlanningPresentation} from './WorkflowAlternatives';
import {cranialAvailability,cranialComparison,cranialGenerationIssue,cranialWorkflowAlternative,cranialTargetBasisLabel,cranialAlternativeLabel} from './cranial-workflow-data';
const amount=(v:unknown)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(2):'Unassessed';
const quantityNames:Record<string,string>={neuralRemovedMm3:'Direct neural removal',boneRemovedMm3:'Bone removal',nonbonePreparationRemovedMm3:'Nonneural preparation removal',protectedRemovedMm3:'Protected source removal',toolExchanges:'Instrument exchanges',acceptedActions:'Accepted actions',brainStrainSquaredMm3S:'Accumulated strain exposure',brainStressSquaredPa2Mm3S:'Accumulated stress exposure',thermalPeakC:'Peak temperature',thermalExposureMm3KS:'Thermal exposure',bloodGeneratedMm3:'Modeled blood generation',motorEncounterMm3:'Motor mapping',languageEncounterMm3:'Language mapping'};
export function CranialStrategiesPanel({c,pause,play,stage,onCompared,onInstalled,onCursor}:WorkflowPlanningPresentation&{c:SurgicalController;pause:()=>void;play:(branchId:string)=>void;onCursor?:(point:Point3)=>void}) {
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
      {a?.procedurePolicy&&<p aria-label="Active procedure policy">Procedure: {a.procedurePolicy.policyId??'Unsupported evidence'} · {a.procedurePolicy.schema??'version unavailable'} · start {a.procedurePolicy.startStage??'unavailable'}. Brainstem and cerebellar traversal are excluded. Neural and bone costs have separate limits.</p>}
      {!!a?.roots.length&&<ul aria-label="Source-bound opening options">{a.roots.map(root=><li key={root.rootId}><strong>{root.label}</strong> · {root.openingMethod} · {root.available?'Available for checked planning':'Blocked'}{root.blockers.length>0&&<ul>{root.blockers.map((b,i)=><li key={`${b.code}:${i}`}>{b.message} <small>({b.code})</small></li>)}</ul>}</li>)}</ul>}
      {issue&&<p role="status" className="surgical-warning">{issue}</p>}
      {!!a?.blockers.length&&<ul aria-label="Cranial planning blockers">{a.blockers.map((b,i)=><li key={`${b.code}:${i}`}>{b.message} <small>({b.code})</small></li>)}</ul>}
      <label><input type="checkbox" aria-label="Acknowledge cranial research context" checked={ack} disabled={c.pending} onChange={e=>setAck(e.target.checked)}/>I understand the declared research context and unavailable clinical outcomes.</label>
      <p className="surgical-note">Acknowledgement does not override missing anatomy, protection or UNKNOWN exclusions. Prior scalp exposure cost may remain unknown.</p>
      {comparison?.sourceBranchId&&issue&&<button disabled={c.pending||!!c.preview} onClick={()=>{pause();c.clearPreview();void c.run('selectBranch',{branchId:comparison.sourceBranchId,position:0});}}>Open common planning checkpoint</button>}
      <button className="primary-button" disabled={c.pending||!!c.preview||!ack||!!issue} onClick={()=>{pause();void c.run('plan',{opponent:'anatomy-cranial-strategy',acknowledgeResearchUnknowns:true,planningHorizon:5,beamWidth:3}).then(ok=>{if(ok)onCompared?.();});}}>Generate strategies</button>
      {c.pending&&<p role="status">Checking opening, preparation, target work and complete exit…</p>}
    </div>
    <div hidden={!!stage&&stage!=='compare'}>
      {c.result?.cranialPlanning&&!comparison&&<p role="alert" className="surgical-warning">The cranial comparison could not be bound to its source roots, complete ledger and saved native branches. Regenerate from a qualified source.</p>}
      {!c.result?.cranialPlanning&&<p role="status">Generate a checked comparison from Plan. No complete tumor-treatment sequence has been established.</p>}
      {comparison&&<>
        <p className="surgical-note">Recorded comparison. Current branch edits do not recalculate these results; generate again after changes.</p>
        <p>{comparison.decision.replaceAll('_',' ')}. Compare treatment and harm against the complete original target.</p>
        {comparison.procedurePolicy&&<p aria-label="Recorded procedure policy">Policy {comparison.procedurePolicy.policyId}. Maximize treatment within declared harm bounds; retain original target residual.</p>}
        {comparison.alternatives[0]?.recordedGeometry?.harmBounds&&<p aria-label="Declared anatomical harm budgets">Declared neural removal budget: <strong>{amount(comparison.alternatives[0].recordedGeometry.harmBounds.maximumNeuralRemovalMm3)} mm³</strong>; neural cut-source budget {amount(comparison.alternatives[0].recordedGeometry.harmBounds.maximumNeuralCutSourceMm3)} mm³. Bone limit {amount(comparison.alternatives[0].recordedGeometry.harmBounds.maximumBoneRemovalMm3)} mm³. These are engineering bounds.</p>}
        <table className="cranial-comparison-table" aria-label="Checked cranial alternatives"><caption>Whole sequence · mm³ and modeled seconds</caption><thead><tr><th>Plan / tip RAS mm</th><th>Treated / residual</th><th>Neural</th><th>Bone / time</th></tr></thead><tbody>{comparison.alternatives.map((row,index)=><tr key={row.branchId} aria-selected={row.branchId===(selected??comparison.selectedBranchId)}><th><button disabled={c.pending||!!c.preview} onClick={()=>void choose(row.branchId)}>Inspect path {index+1}</button><small>{row.recordedGeometry?.paths.filter(p=>p.instrument==='cusa'&&p.operation==='apply').at(-1)?.pointsRasMm.at(-1)?.map(n=>n.toFixed(1)).join(' / ')??row.label}{row.branchId===comparison.selectedBranchId?' · policy choice':''}</small></th><td>{amount(row.goalAssessment.originalTargetTreatmentMm3)} / {amount(row.goalAssessment.residualOriginalTargetMm3)}</td><td>{amount(row.ledger.neuralRemovedMm3)}</td><td>{amount(row.ledger.boneRemovedMm3)} / {amount(row.ledger.modeledElapsedSeconds)} s</td></tr>)}</tbody></table>
        {onCursor&&comparison.alternatives[0]?.recordedGeometry?.structures.some(s=>s.id==='motor-cortex'&&s.pointsRasMm.length>0)&&<button onClick={()=>{const p=comparison.alternatives[0]?.recordedGeometry?.structures.find(s=>s.id==='motor-cortex')?.pointsRasMm[0];if(p)onCursor(p);}}>Center generated motor anatomy</button>}
        {comparison.alternatives.some(row=>row.recordedGeometry)&&<details className="cranial-recorded-legend" aria-label="Recorded route and source anatomy legend"><summary>Recorded paths and generated anatomy legend</summary>
          <p><strong>Recorded target-work paths</strong> · bright line: inspected branch; dim lines: other checked alternatives. Pink: original/residual target. Lines come from accepted action history; native playback animates the physical tools and tissue.</p>
          <p>Declared generated motor anatomy · cyan: motor cortex; violet: corticospinal tract. Overlapping memberships remain separate. Red: brainstem; amber: cerebellum. These source memberships do not predict paralysis; encounter and functional consequences remain unavailable.</p>
          <p>Policy-selected alternative: <strong>{cranialAlternativeLabel(comparison.alternatives.find(row=>row.branchId===comparison.selectedBranchId)??comparison.alternatives[0])}</strong>. A checked STOP with residual is distinct from task completion.</p>
        </details>}
        <WorkflowAlternatives rows={comparison.alternatives.map(row=>cranialWorkflowAlternative(row,comparison.selectedBranchId))} selectedId={selected??comparison.selectedBranchId} disabled={c.pending||!!c.preview} onSelect={id=>void choose(id)}/>
        {!comparison.alternatives.length&&<p>No recorded complete or progress sequence was returned. This bounded search does not prove anatomical impossibility.</p>}
        {!!comparison.failures.length&&<ul aria-label="Strategy refusals">{comparison.failures.map((f,i)=><li key={`${f.code}:${i}`}>{f.message} <small>({f.code})</small>{f.constraintViolations&&<ul>{f.constraintViolations.map((v,j)=><li key={j}>{quantityNames[v.metric]??v.metric.replace('structure:','').replace(':removedMm3',' direct removal').replace(':cutSourceMm3',' cut-source incidence')}: proposed {amount(v.actual)} / bound {amount(v.maximum)} {v.unit??''}</li>)}</ul>}</li>)}</ul>}
        {comparison.alternatives.map(row=><details key={row.branchId}><summary>{cranialAlternativeLabel(row)} · whole sequence ledger and replay</summary>
          <dl><div><dt>Opening bone removed</dt><dd>{amount(row.ledger.boneRemovedMm3)} mm³</dd></div><div><dt>Preparation nonbone removed</dt><dd>{amount(row.ledger.preparationNonboneRemovedMm3)} mm³</dd></div><div><dt>Work non-target removed</dt><dd>{amount(row.ledger.workNonTargetRemovedMm3)} mm³</dd></div><div><dt>Whole non-target removed</dt><dd>{amount(row.ledger.wholeNonTargetRemovedMm3)} mm³</dd></div>
            <div><dt>Direct neural removal</dt><dd>{amount(row.ledger.neuralRemovedMm3)} mm³</dd></div>
            <div><dt>Original target basis</dt><dd>{cranialTargetBasisLabel(row.goalAssessment.targetBasis)} · {row.goalAssessment.originalTargetCells} cells · {amount(row.goalAssessment.originalTargetVolumeMm3)} mm³</dd></div><div><dt>Original target removed</dt><dd>{amount(row.goalAssessment.originalTargetRemovedMm3)} mm³</dd></div><div><dt>Credited target treatment</dt><dd>{amount(row.goalAssessment.originalTargetTreatmentMm3)} mm³ · {(row.goalAssessment.originalTargetTreatmentFraction*100).toFixed(1)}%</dd></div></dl>
          {row.recordedGeometry&&c.result&&checkedCranialGeometry(c.result,row)&&<details className="cranial-geometry-evidence"><summary>Recorded route geometry and generated memberships</summary>
            <p>{row.recordedGeometry.scope}</p>
            <p>Actual cleanup: {amount(row.recordedGeometry.cleanup.debrisRemovedMm3)} mm³ debris collected; {amount(row.recordedGeometry.cleanup.debrisPresentMm3)} mm³ debris and {amount(row.recordedGeometry.cleanup.fluidPresentMm3)} mm³ fluid remain. Active bleeding sources: {row.recordedGeometry.cleanup.activeBleedingSources}.</p>
            <ul>{row.recordedGeometry.structures.map(s=><li key={s.id}><span style={{color:declaredStructureColor(s.id)}}>{s.label}</span> · {s.sourceCells.length} independently declared source cells · direct removal {amount(s.removedMm3)} mm³ · cut-source incidence {amount(s.cutSourceMm3)} mm³ · {s.reviewStatus}{onCursor&&s.pointsRasMm.length>0&&<button onClick={()=>onCursor(s.pointsRasMm[0])}>Center {s.label}</button>}{s.displayTruncated?' · display truncated; exact memberships retained':''}</li>)}</ul>
            <p>Overlaps are retained and structure quantities must not be summed. {row.recordedGeometry.omittedPartialActions.length?`${row.recordedGeometry.omittedPartialActions.length} partial action paths omitted because exact endpoint geometry is unavailable.`:'All represented motion endpoints come from complete accepted actions or recorded native contact dwells.'}</p>
            <ol aria-label={`${row.branchId} accepted physical actions`}>{row.recordedGeometry.acceptedActions.map(action=><li key={action.actionIndex}>Action {action.actionIndex+1} · {action.instrument} · {action.operation} · {action.status} · {amount(action.elapsedSeconds)} modeled seconds</li>)}</ol>
            <p>Source: <code>{row.recordedGeometry.sourceSceneId}</code> · case <code>{row.recordedGeometry.caseHash}</code> · policy <code>{row.recordedGeometry.policyHash}</code> · accepted endpoint <code>{row.recordedGeometry.finalStateId}</code> · history <code>{row.recordedGeometry.actionHistoryHash}</code> · geometry <code>{row.recordedGeometry.geometrySeal}</code>.</p>
          </details>}
          {row.endpointAssessment&&<section aria-label={`${row.branchId} common endpoint`}>
            <p><strong>{row.endpointAssessment.classification.replaceAll('-',' ')}</strong> · original target fully removed: {row.endpointAssessment.completeOriginalTargetRemoval?'yes':'no'} · common task completed: {row.endpointAssessment.completedTask?'yes':'no'} · truncation: {row.endpointAssessment.truncated?'yes':'no'}.</p>
            <p>Cleanup {row.endpointAssessment.cleanupComplete?'complete':`pending: ${row.endpointAssessment.pending.join(', ')}`} · debris {amount(row.endpointAssessment.debrisMm3)} mm³ · fluid {amount(row.endpointAssessment.fluidMm3)} mm³. Config: <code>{row.endpointAssessment.configHash}</code>.</p>
          </section>}
          <p>Whole-body exit {row.wholeExitComplete&&row.sequenceOutcome.withdrawalComplete?'complete':'incomplete'}. {row.ledger.priorExposureCost===undefined?'Prior exposure cost is unknown.':String(row.ledger.priorExposureCost)}</p>
          <table aria-label={`${row.label} shared consequences`}><thead><tr><th>Quantity</th><th>Whole sequence</th><th>Evidence</th></tr></thead><tbody>{Object.entries(row.sequenceOutcome.dimensions).filter(([,q])=>q&&typeof q==='object'&&'unit' in q).map(([key,q])=>{const value=q as {unit:string;status:string;cumulative:number|null};return <tr key={key}><th>{quantityNames[key]??key.replaceAll(/([A-Z])/g,' $1')}</th><td>{amount(value.cumulative)} {value.unit}</td><td>{value.status}</td></tr>;})}</tbody></table>
          <p>Unavailable evidence: {row.sequenceOutcome.missingEvidence.join('; ')||'Completeness is not established'}.</p>
          {!!row.sequenceOutcome.dimensions.structureRemovedMm3&&<ul aria-label={`${row.label} structure removal`}>{Object.entries(row.sequenceOutcome.dimensions.structureRemovedMm3).map(([name,q])=><li key={name}>{name}: {amount(q.cumulative)} mm³ · {q.status}</li>)}</ul>}
          <button disabled={c.pending||!!c.preview} onClick={()=>{pause();play(row.branchId);}}>Replay {cranialAlternativeLabel(row)}</button>
          <p className="surgical-note">Playback uses this saved native branch. Edits require a new comparison; recorded outcomes do not recalculate. Engineering task completion does not grant patient qualification.</p>
        </details>)}
        <p>Whole search: {comparison.cost.physicalCalls} physical calls ({comparison.cost.previewCalls} previews + {comparison.cost.directStepCalls} direct steps) · {amount(comparison.cost.wallSeconds)} s including all roots and export/replay.</p>
        <ul>{comparison.limitations.map((line,i)=><li key={i}>{line}</li>)}</ul>
      </>}
    </div>
  </WorkflowPlanningShell>;
}
