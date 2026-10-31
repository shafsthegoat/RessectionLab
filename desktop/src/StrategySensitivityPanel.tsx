import {useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {TwoToolComparison} from './two-tool-data';
import type {Point3} from './viewer/coordinates';
import {currentSensitivity,failedCommandPoint,executionIssueText,executionFailureText} from './strategy-sensitivity-data';

export function StrategySensitivityPanel({c,report,locked,ack,pause,onSelect,onCursor}:{c:SurgicalController;report:TwoToolComparison;
  locked:boolean;ack:boolean;pause:()=>void;onSelect:(id:string)=>void;onCursor?:(p:Point3)=>void}){
  const [offset,setOffset]=useState(report.executionSensitivity?.study.assessment.controls.offsetMm??1);
  const study=currentSensitivity(report,offset),a=study?.assessment;
  const issue=!Number.isFinite(offset)||offset<0||offset>2||offset+(report.publicPlan.constraints.encounter?.marginMm??0)>5;
  const label=(id:string)=>report.publicPlan.alternatives.find(r=>r.id===id)?.label??id;
  return <details className="strategy-sensitivity"><summary>Execution-offset sensitivity</summary>
    <p className="surgical-note">A declared stress test shifts every commanded pose of both tools by one shared RAS offset for the complete sequence. Anatomy stays fixed. These offsets are research assumptions; this study supplies no patient-specific execution, registration or segmentation error calibration.</p>
    <label>Command offset · mm<input aria-label="Execution command offset mm" type="number" min={0} max={2} step={.25} value={offset} disabled={locked} onChange={e=>setOffset(Number(e.target.value))}/></label>
    {issue&&<p role="alert">Use 0–2 mm within the software stress bound; context margin plus offset must be at most 5 mm.</p>}
    <button disabled={locked||!ack||issue||!!c.preview} onClick={()=>{pause();void c.run('analyzeStrategySensitivity',{expectedPlanSeal:report.reportSeal,offsetMm:offset,acknowledgeResearchUnknowns:true});}}>Assess declared execution offsets</button>
    {a&&<><p role="status" aria-label="Execution sensitivity decision">{a.scenarioCount} declared {a.scenarioCount===1?'scenario':'scenarios'} · nominal {label(a.policies.nominal)} · fixed margin {label(a.policies.fixedMargin)} · all tested offsets {label(a.policies.allDeclaredScenarios)}.</p>
      <p className="surgical-note">{a.policies.allDeclaredScenarios==='stop'?(report.publicPlan.constraints.minimumTargetMm3===0?'STOP already meets the declared progress requirement without new actions.':'No held operative program meets the task in every tested offset. STOP preserves the baseline.'):'This held program meets the task in every listed offset.'} Intervening offsets, changing errors, rotations and anatomical uncertainty remain unassessed. Scenario counts are not probabilities.</p>
      <p>Fixed nominal clearance: additional shaft margin {offset.toFixed(2)} mm; non-target {a.fixedMarginDefinition.nonTargetClearanceMm.toFixed(2)} mm; supplied context {a.fixedMarginDefinition.contextMarginMm.toFixed(2)} mm; additional protected-structure margin {a.fixedMarginDefinition.protectedExtraMm.toFixed(2)} mm.</p>
      <button disabled={locked} onClick={()=>{c.clearPreview();onSelect(a.policies.allDeclaredScenarios);}}>Select result for all tested offsets</button>
      <p className="surgical-note">Preview and installation use the nominal program. Offset runs are retained as counterfactual evidence. Failed-command markers show the requested pose. A failed action can stop before that pose.</p>
      {!!a.preferenceChanges.length&&<p>Preference changes in: {a.preferenceChanges.map(p=>p.scenarioId).join(', ')}.</p>}
      <table aria-label="Execution offset outcomes"><thead><tr><th>Program / RAS offset mm</th><th>Execution and modeled result</th></tr></thead><tbody>{a.candidates.flatMap(candidate=>candidate.outcomes.map(outcome=>{
        const point=failedCommandPoint(outcome);return <tr key={candidate.id+outcome.scenarioId} data-offset-case={candidate.id+':'+outcome.scenarioId}><th>{label(candidate.id)} · {outcome.scenarioId}<br/>{outcome.translationRasMm.map(v=>v.toFixed(2)).join(' / ')}</th><td><p>{outcome.completeExecution&&outcome.withdrawalComplete?'Complete through withdrawal':`Stopped in ${outcome.failure?.phase??'execution'}`}. {executionIssueText(outcome)}.</p><p>Target {outcome.summary.additionalTargetMm3.toFixed(2)} mm³ · context near {outcome.anatomicalEncounter.nearMm3.toFixed(2)} mm³.</p>{outcome.summary.additionalNonTargetMm3>0&&<p>Additional non-target removal {outcome.summary.additionalNonTargetMm3.toFixed(2)} mm³.</p>}{outcome.failure&&<p>{executionFailureText(outcome.failure.reason)}</p>}{point&&onCursor&&<button disabled={locked} aria-label={`Show failed commanded pose ${candidate.id} ${outcome.scenarioId}`} onClick={()=>onCursor(point)}>Show requested pose</button>}</td></tr>;
      }))}</tbody></table>
      <details><summary>Offset evidence and receipts</summary><p>{a.scope}</p><pre>{JSON.stringify(study,null,2)}</pre></details>
    </>}
  </details>;
}
