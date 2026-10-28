import type {SurgicalController} from './use-surgical';
import {retractionStudyDisplay} from './retraction-study-data';

export function RetractionStudyPanel({c,pause}:{c:SurgicalController;pause:()=>void}){
  const data=retractionStudyDisplay(c.result);
  if(!data)return c.result?.retractionStudy?<p role="alert">Paired-tool study could not be bound to the native branches.</p>:null;
  const disabled=c.pending||c.result?.verification?.status==='unverified-import'||c.result?.readOnly;
  const baseline=data.variants.find(row=>!row.assisted);
  const sameRemoval=baseline&&data.variants.filter(row=>row.assisted).every(row=>
    row.outcome.wholeTargetRemovedMm3===baseline.outcome.wholeTargetRemovedMm3&&
    row.outcome.wholeNonTargetRemovedMm3===baseline.outcome.wholeNonTargetRemovedMm3);
  return <details className="research-planning retraction-study"><summary>Recorded bimanual study · {data.variants.filter(r=>r.outcome.taskGateMet).length} / {data.variants.length} larger task gates passed</summary>
    <p>These are recorded native programs on one access route, with and without a fixed catalog retractor. Playback does not run new inference. Different tool arrangements are not distinct successful surgical strategies.</p>
    <p className="surgical-warning">{sameRemoval?'Retraction has no demonstrated removal benefit in these development runs. The matched unassisted controller remains the baseline.':'Compare the recorded assistance effects, task gates and cost against the unassisted baseline.'} Anatomy lacks skull, complete vessels and functional mapping. Clinical outcomes are unavailable.</p>
    <p>Legacy instantaneous CUSA with existing V2 coupled tissue mechanics. Fixed blade radius 3 mm; shaft radius 0.5 mm and length 40 mm. Mechanical effects are uncalibrated engineering predictions.</p>
    <p>{data.physicalCalls} public native calls · {data.wallSeconds.toFixed(1)} wall seconds in this study receipt. Replayed programs and new planning are identified separately below; zero optimizer updates.</p>
    <section aria-label="Bimanual native study outcomes">{data.variants.map(row=><article key={row.branchId} data-bimanual-branch={row.branchId}>
      <h4>{row.assisted?'CUSA with held retractor':'Unassisted CUSA'} · {row.status.replaceAll('_',' ')}</h4>
      <p>Whole tumor removal {row.outcome.wholeTargetRemovedMm3.toFixed(1)} mm³ · whole non-target {row.outcome.wholeNonTargetRemovedMm3.toFixed(1)} mm³ · residual target {row.outcome.residualTargetMm3.toFixed(1)} mm³.</p>
      <p>{row.nativeActions} accepted native actions · final tool cleanup {row.outcome.cleanupComplete?'complete':'incomplete'}. Larger task {row.outcome.taskGateMet?'passed':'unpassed'} · failed {row.outcome.failedGates.join(', ')||'none'}.</p>
      <p>Action-boundary peak brain displacement {row.sampledMechanicalPeaks.brainMaxDisplacementMm.toFixed(3)} mm; final residual displacement {row.finalMetrics.brainMaxDisplacementMm.toPrecision(3)} mm. Tool cleanup does not imply tissue returned exactly to its source position.</p>
      <p>Contact {row.finalMetrics.contactCellSeconds.toFixed(2)} cell seconds · {row.finalMetrics.elapsedSeconds.toFixed(2)} simulated seconds.</p>
      <p>{row.costMode==='recorded_program_replay'?`Fresh replay only: ${row.publicCalls} calls. Original planning and verification used ${row.originalPlanningAndReplayCalls} calls; no new strategy inference was performed.`:`Planning and independent native replay: ${row.publicCalls} calls.`}</p>
      {row.outsideRecoveryOffsetMm>0&&<p>Placement recovered {row.outsideRecoveryOffsetMm} mm farther outside along the same ray with unchanged blade geometry and native collision checks. Outside-image anatomy remains unknown.</p>}
      {row.failureReason&&<p>Recorded failure: {row.failureReason}</p>}
      <div className="surgical-actions">{row.phases.map((phase,index)=><button key={index}
        disabled={disabled} data-bimanual-position={phase.position} data-bimanual-phase={phase.label}
        onClick={()=>{pause();void c.run('selectBranch',{branchId:row.branchId,position:phase.position});}}>
        {phase.label} · action {phase.position}</button>)}</div>
    </article>)}</section>
    <p>Use the native timeline to pause, inspect both tools and changing anatomy, rewind and play recorded microsteps. Source {data.sourceStateId} · comparison {data.comparisonSeal}.</p>
  </details>;
}
