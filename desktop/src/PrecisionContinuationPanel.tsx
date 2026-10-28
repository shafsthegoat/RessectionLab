import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {precisionContinuationDisplay,precisionGenerationEligibility} from './precision-continuation-data';
import {stagedStrategyDisplay} from './staged-strategy-data';
const mm=(v:number)=>v.toFixed(1);

export function PrecisionContinuationPanel({c,artifact,pause,play}:{c:SurgicalController;artifact:unknown;pause:()=>void;play:(branchId:string)=>void}){
  const [acknowledged,setAcknowledged]=useState(false);
  useEffect(()=>setAcknowledged(false),[c.result?.caseHash,c.result?.view.scene.sceneId]);
  const record=artifact&&typeof artifact==='object'?artifact as Record<string,unknown>:null;
  const recorded=Object.hasOwn(c.result??{},'precisionPlanning')||!!record&&(
    Object.hasOwn(record,'precisionPlanning')||Object.hasOwn(record,'precisionSummary'));
  const data=precisionContinuationDisplay(artifact,c.result),staged=stagedStrategyDisplay(artifact,c.result);
  if(!recorded&&!staged)return null;
  const eligibility=precisionGenerationEligibility(artifact,c.result);
  if(recorded&&!data)return <p role="alert" className="surgical-warning">Precision continuation evidence could not be bound to this scene and its native branches. Earlier staged evidence remains available above.</p>;
  const inspect=(branchId:string,position:number)=>{pause();void c.run('selectBranch',{branchId,position});};
  const disabled=c.pending||c.result?.verification?.status==='unverified-import'||c.result?.readOnly;
  return <details className="research-planning precision-continuation"><summary>Precision continuation{data?` · +${mm(data.additionalTargetMm3)} mm³ target`:''}</summary>
    <p className="surgical-warning">Research behavior on partial anatomy using legacy instantaneous CUSA with V2 mechanics. Skull, complete vessels and functional mapping remain unavailable. These effects are not clinical risk estimates.</p>
    {!data?<>
      <p>Continue from a checked staged endpoint with a fixed 0.6 mm tip and bounded source-derived re-entry proposals. Native checks and checked withdrawal decide retained stages. No actor or optimizer runs.</p>
      <label><input type="checkbox" aria-label="Acknowledge precision continuation assumptions" checked={acknowledged} disabled={c.pending} onChange={e=>setAcknowledged(e.target.checked)}/> I acknowledge the missing anatomy and research-only assumptions.</label>
      <button disabled={c.pending||!acknowledged||!eligibility.ready} onClick={()=>{pause();void c.run('plan',{opponent:'anatomy-precision-reentry',acknowledgeResearchUnknowns:true});}}>Generate precision continuation</button>
      {!eligibility.ready&&<p role="status">{eligibility.reason}</p>}
    </>:<>
      <p>STOP at source: whole target {mm(data.sourceOutcome.wholeTargetRemovedMm3)} mm³ · whole non-target {mm(data.sourceOutcome.wholeNonTargetRemovedMm3)} mm³ · larger task {data.sourceOutcome.taskGateMet?'passed':'unpassed'}.</p>
      <p>Earlier study and export qualification is separate. This continuation does not establish whether the original study met its export or timing requirements; any recorded failures remain unchanged.</p>
      <p>Continuation effects: +{mm(data.additionalTargetMm3)} mm³ target · +{mm(data.additionalNonTargetMm3)} mm³ non-target.</p>
      <p>Whole final approach: target {mm(data.outcome.wholeTargetRemovedMm3)} mm³ · non-target {mm(data.outcome.wholeNonTargetRemovedMm3)} mm³ · residual target {mm(data.outcome.residualTargetMm3)} mm³ · {(data.outcome.sourceTargetFractionRemoved*100).toFixed(2)}% source target removed.</p>
      <p>{data.newStageCount} new stages · native actions {data.sourceNativeActions} → {data.finalNativeActions} · checked cleanup {data.outcome.cleanupComplete?'complete':'incomplete'} · exact native replay verified.</p>
      <p>{data.cost.physicalCalls} / {data.cost.maxPhysicalCalls} additional controller calls including source and final replay · {data.wallSeconds.toFixed(1)} wall seconds / 90 s controller limit · zero optimizer updates. Prior planning, desktop admission and saved-file reopen have separate costs. Stop reason: {data.stopReason.replaceAll('_',' ')}. This bounded result does not establish global infeasibility.</p>
      <div className="surgical-actions"><button disabled={disabled} data-precision-branch={data.sourceBranchId} data-precision-position={data.sourceNativeActions} onClick={()=>inspect(data.sourceBranchId,data.sourceNativeActions)}>Inspect STOP endpoint</button>
        <button disabled={disabled} data-precision-branch={data.finalBranchId} data-precision-position={data.finalNativeActions} onClick={()=>inspect(data.finalBranchId,data.finalNativeActions)}>Inspect precision endpoint</button>
        <button disabled={disabled||data.finalNativeActions===0} onClick={()=>play(data.finalBranchId)}>Play recorded native actions</button></div>
      <details><summary>Retained re-entry stages</summary>{data.stages.length?<ol>{data.stages.map(stage=><li key={stage.stage}>
        Stage {stage.stage+1} · {stage.configuration} · whole stage target +{mm(stage.wholeTargetIncrementMm3)} mm³, including entry and dwell +{mm(stage.entryTargetIncrementMm3)} mm³ · brain +{mm(stage.wholeBrainIncrementMm3)} mm³.
        <p>Source proposal endpoint [{stage.candidate.endpoint.position.map(mm).join(', ')}] RAS mm · radial offset {mm(stage.candidate.radialFromOriginalRayMm)} mm · external depth {mm(stage.candidate.externalDepthMm)} mm.</p>
        <button disabled={disabled} onClick={()=>inspect(data.finalBranchId,stage.nativeActions)}>Inspect work endpoint at action {stage.nativeActions}</button>
        <button disabled={disabled} onClick={()=>inspect(data.finalBranchId,stage.finalNativeActions)}>Inspect checked exit at action {stage.finalNativeActions}</button>
        <p>Checked exit state {stage.checkedExitStateId}.</p>
      </li>)}</ol>:<p>No new work stage was retained. The STOP baseline and measured planning cost still apply.</p>}</details>
      <details><summary>Task gates and original baseline</summary>
        <p>Larger task {data.outcome.taskGateMet?'passed':'unpassed'} · failed gates: {data.outcome.failedGates.join(', ')||'none'}.</p>
        <p>Final zones: {Object.entries(data.outcome.zoneTargetRemovedMm3).map(([zone,value])=>`${zone}: ${mm(value)}`).join(' · ')} mm³.</p>
        <p>Final configurations: {Object.entries(data.outcome.configurationTargetRemovedMm3).map(([configuration,value])=>`${configuration}: ${mm(value)}`).join(' · ')||'none'} mm³.</p>
        <p>Original 48-action task {data.outcome.legacyTask.taskGateMet?'passed':'unpassed'} · post-access target +{mm(data.outcome.legacyTask.additionalTargetMm3)} mm³ · post-access non-target +{mm(data.outcome.legacyTask.additionalNonTargetMm3)} mm³.</p>
        <p>Original access baseline non-target {mm(data.originalBaselineNonTargetMm3)} mm³. The continuation non-target gate continues to use this baseline.</p>
        <p>Whole non-target ≤200 mm³ and ≤0.2 × whole target; post-access non-target ≤20 mm³; accepted actions ≤240. Source zone, configuration, cleanup and replay gates remain required.</p>
      </details>
      <details><summary>Recorded limitations and identity</summary><ul>{data.limitations.map((note,i)=><li key={i}>{note}</li>)}</ul>
        <p>Scene {data.sourceSceneId} · source comparison {data.sourceComparisonSeal} · precision comparison {data.comparisonSeal}.</p></details>
    </>}
  </details>;
}
