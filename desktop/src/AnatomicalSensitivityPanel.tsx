import {useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {TwoToolComparison} from './two-tool-data';
import type {Point3} from './viewer/coordinates';
import {currentAnatomicalSensitivity,anatomicalIssueText,contextCellPoint} from './anatomical-sensitivity-data';

export function AnatomicalSensitivityPanel({c,report,locked,ack,pause,onSelect,onCursor}:{c:SurgicalController;report:TwoToolComparison;
  locked:boolean;ack:boolean;pause:()=>void;onSelect:(id:string)=>void;onCursor?:(p:Point3)=>void}){
  const [step,setStep]=useState(report.anatomicalSensitivity?.study.assessment.controls.contextVoxelShift??1);
  const study=currentAnatomicalSensitivity(report,step),a=study?.assessment;
  const issue=!Number.isInteger(step)||step<0||step>2;
  const label=(id:string)=>report.publicPlan.alternatives.find(r=>r.id===id)?.label??id;
  return <details className="anatomical-sensitivity"><summary>Context-label sensitivity</summary>
    <p className="surgical-note">Test the same complete maneuvers with the supplied context label shifted on its source voxel grid. Tissue, target and tool commands stay fixed. These are declared label-location stress tests with no patient error calibration. Source images and annotations stay unchanged.</p>
    <label>Context shift · source voxels<input aria-label="Context label voxel shift" type="number" min={0} max={2} step={1} value={step} disabled={locked} onChange={e=>setStep(Number(e.target.value))}/></label>
    <p className="surgical-note">Nominal plus both directions of each source grid axis; at most 2 mm physical shift and 5 mm combined fixed margin. Subvoxel changes, diagonals, rotations and shape changes remain untested. Shifted-in cells without source support remain unknown.</p>
    {issue&&<p role="alert">Use an integer 0–2 voxel shift.</p>}
    <button disabled={locked||!ack||issue||!!c.preview} onClick={()=>{pause();void c.run('analyzeStrategySensitivity',{expectedPlanSeal:report.reportSeal,contextVoxelShift:step,acknowledgeResearchUnknowns:true});}}>Assess declared context shifts</button>
    {a&&<><p role="status" aria-label="Context sensitivity decision">{a.scenarioCount} declared {a.scenarioCount===1?'scenario':'scenarios'} · nominal {label(a.policies.nominal)} · fixed context margin {label(a.policies.fixedContextMargin)} · all tested label shifts {label(a.policies.allDeclaredScenarios)}.</p>
      <p>Maximum tested shift {a.scenarioConfiguration.maximumTranslationMm.toFixed(2)} mm. Fixed nominal context margin {a.fixedMarginDefinition.contextMarginMm.toFixed(2)} mm.</p>
      <p className="surgical-note">Any new CUSA work in hypothetical context and unknown encountered or worked cells disqualify a maneuver. Work membership counts whole cells, including partial contact; it is not allocated aspiration capacity or injury. Historical removal remains disclosed. The fixed-margin comparator does not bound changes in the labels of manipulated cells. Scenario counts are not clinical probabilities.</p>
      <button disabled={locked} onClick={()=>{c.clearPreview();onSelect(a.policies.allDeclaredScenarios);}}>Select result for all tested label shifts</button>
      <p className="surgical-note">Selection, preview and installation use the original nominal program. The label scenarios remain counterfactual evidence. {a.policies.allDeclaredScenarios==='stop'?(report.publicPlan.constraints.minimumTargetMm3===0?'STOP meets the declared task without new actions.':'No held operative program meets every listed scenario; STOP preserves the baseline.'):'The selected held program meets every listed scenario.'}</p>
      {!!a.preferenceChanges.length&&<p>Preference changes in: {a.preferenceChanges.map(p=>p.scenarioId).join(', ')}.</p>}
      <table aria-label="Context label outcomes"><thead><tr><th>Program / source shift</th><th>Modeled context consequence</th></tr></thead><tbody>{a.candidates.flatMap(candidate=>candidate.outcomes.map(outcome=>{
        const scenario=a.scenarioConfiguration.scenarios.find(s=>s.id===outcome.scenarioId)!;
        const point=outcome.encounter.nearPositionsMm[0];
        const scene=c.result?.view.scene;
        const unknownPoint=scene&&scene.sceneId===report.sourceSceneId?contextCellPoint(outcome.encounter.unknownEncounteredCells[0],scene):null;
        const workPoint=scene&&scene.sceneId===report.sourceSceneId?contextCellPoint(outcome.encounter.newContextWorkCells[0],scene):null;
        return <tr key={candidate.id+outcome.scenarioId} data-label-case={candidate.id+':'+outcome.scenarioId}><th>{label(candidate.id)} · {outcome.scenarioId}<br/>Grid {scenario.voxelShift.join(' / ')}<br/>RAS mm {scenario.translationRasMm.map(v=>v.toFixed(2)).join(' / ')}</th><td><p>{anatomicalIssueText(outcome.violations)}.</p><p>Near {outcome.encounter.nearMm3.toFixed(2)} mm³ · cells receiving new CUSA work {outcome.encounter.newContextWorkMm3.toFixed(2)} mm³ · new removal {outcome.encounter.newRemovedContextMm3.toFixed(2)} mm³.</p><p>Unknown encountered/worked {outcome.encounter.unknownEncounteredMm3.toFixed(2)} mm³ · historical removed context {outcome.encounter.baselineRemovedContextMm3.toFixed(2)} mm³ · context shifted out of field {scenario.outOfFieldContextCells} cells.</p>{point&&onCursor&&<button disabled={locked} onClick={()=>onCursor(point)}>Show hypothetical context cell {candidate.id} {outcome.scenarioId}</button>}{unknownPoint&&onCursor&&<button disabled={locked} onClick={()=>onCursor(unknownPoint)}>Show unknown context cell {candidate.id} {outcome.scenarioId}</button>}{workPoint&&onCursor&&<button disabled={locked} onClick={()=>onCursor(workPoint)}>Show manipulated context cell {candidate.id} {outcome.scenarioId}</button>}</td></tr>;
      }))}</tbody></table>
      <details><summary>Label-shift evidence and receipts</summary><p>{a.scope}</p><pre>{JSON.stringify(study,null,2)}</pre></details>
    </>}
  </details>;
}
