import type {SurgicalController} from './use-surgical';
import {fullCaseStateSummary} from './full-case-presentation';
export const amount=(value:unknown,digits=2)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'Unknown';
const metricAmount=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(2):'Unknown';
const evidenceLabels:Record<string,string>={brainStrainSquaredMm3S:'accumulated strain exposure',brainStressSquaredPa2Mm3S:'accumulated stress exposure',thermalPeakC:'peak temperature',thermalExposureMm3KS:'thermal exposure',bloodGeneratedMm3:'modeled blood generation',motorEncounterMm3:'motor mapping',languageEncounterMm3:'language mapping'};
const membraneStages:Record<string,string>={BOUND:'Membrane opening not started',BONE_WORK:'Bone stage; membrane retained',MEMBRANE_READY:'Bone cleared; membrane retained',MEMBRANE_CUTTING:'Membrane cutting; tissue retained',CUT_RETAINED:'Boundary cut; membrane retained',HELD:'Membrane held or lifted',RELEASED_IN_FIELD:'Membrane released in field',EXTRACTED:'Membrane extracted; instrument exit pending',WITHDRAWN:'Membrane extracted and instruments withdrawn'};
const openingStages:Record<string,string>={BOUND:'Opening not started',KERF_IN_PROGRESS:'Bone cutting / separation',GRASPED_OR_TRANSPORTED:'Flap held or lifted',DETACHED_IN_FIELD:'Flap released in field',EVACUATED:'Flap removed; withdrawal pending',HOLDER_WITHDRAWN:'Flap removed and instruments withdrawn',MILLING_BOUND:'Milling not started',MILLING_IN_PROGRESS:'Milling bone',MILLING_DEBRIS_RETAINED:'Bone removed; debris collection pending',MILLING_CLEARED:'Bone and debris cleared; withdrawal pending',MILLING_WITHDRAWN:'Bone and debris cleared; instruments withdrawn'};

/** One readout for the anatomy actually displayed. Missing values stay unknown. */
export function SurgicalOutcomeSummary({c}:{c:SurgicalController}) {
  const v=c.preview??c.view,r=c.result;if(!v)return null;
  const summary=fullCaseStateSummary(v),record=r?.fullCase;
  const opening=v.craniotomy;
  const stage=v.cranialStages?.schema==='resectionlab.cranial-stages.v1'&&v.cranialStages.stateId===v.stateId&&v.cranialStages.sceneId===v.scene.sceneId?v.cranialStages:null;
  const ledger=r?.sequenceOutcome?.stateId===v.stateId&&r.sequenceOutcome.sceneId===v.scene.sceneId?r.sequenceOutcome:null;
  return <section className="surgical-outcome" aria-label="Modeled outcome summary">
    <h3>{c.preview?'Preview modeled outcome':'Modeled outcome'}</h3>
    <div className="surgical-metrics">
      <div><small>Target removed · mm³</small><strong data-surgical-metric="targetRemovedMm3">{metricAmount(summary.targetRemovedMm3)}</strong></div>
      <div><small>Target remaining · mm³</small><strong>{amount(summary.targetRemainingMm3)}</strong></div>
      <div><small>Other removed · mm³</small><strong data-surgical-metric="nonTargetRemovedMm3">{metricAmount(v.metrics.nonTargetRemovedMm3)}</strong></div>
      <div><small>Modeled field fluid · mm³</small><strong data-surgical-metric="fluidPresentMm3">{metricAmount(v.metrics.fluidPresentMm3)}</strong></div>
    </div>
    <p className="surgical-note">{summary.targetRemovedPercent===null?'Unknown fraction':amount(summary.targetRemovedPercent)+'%'} of {amount(summary.initialTargetMm3)} mm³ modeled target. Functional injury remains Unknown.</p>
    {opening&&<div aria-label="Native bone opening">
      <div className="surgical-metrics">
        {opening.method==='milling'?<>
        <div><small>Milled bone · mm³</small><strong data-craniotomy-metric="milledBoneMm3">{metricAmount(opening.milledBoneMm3)}</strong></div>
        <div><small>Bone debris collected · mm³</small><strong data-craniotomy-metric="debrisRemovedMm3">{metricAmount(opening.milling?.debrisRemovedMm3)}</strong></div>
        <div><small>Bone debris remaining · mm³</small><strong data-craniotomy-metric="debrisPresentMm3">{metricAmount(opening.milling?.debrisPresentMm3)}</strong></div>
        </>:<>
        <div><small>Kerf removed · mm³</small><strong data-craniotomy-metric="kerfRemovedMm3">{metricAmount(opening.kerfRemovedMm3)}</strong></div>
        <div><small>Flap evacuated · mm³</small><strong data-craniotomy-metric="flapEvacuatedMm3">{metricAmount(opening.flapEvacuatedMm3)}</strong></div>
        </>}
        <div><small>Opening bone removed · mm³</small><strong data-craniotomy-metric="declaredBoneRemovedMm3">{metricAmount(opening.declaredBoneRemovedMm3)}</strong></div>
      </div>
      <p>Opening stage: {Object.hasOwn(openingStages,opening.phase)?openingStages[opening.phase]:'Unknown'}.</p>
      <p>Aperture: {opening.apertureEvacuated===true?'Evacuated':opening.apertureEvacuated===false?'Not evacuated':'Unknown'}. Opening with both hands withdrawn: {opening.openingComplete===true?'Complete':opening.openingComplete===false?'Incomplete':'Unknown'}.</p>
      {opening.method==='milling'&&<p>Bone debris escaped: {metricAmount(opening.milling?.debrisEscapedMm3)} mm³. {stage?'Bone clearance alone does not permit work-region entry.':'Dural opening and tumor entry remain unsupported.'}</p>}
      <p className="surgical-note">Bone-opening research model. Tumor-task completion and clinical safety remain unassessed.</p>
    </div>}
    {stage&&<div aria-label="Native membrane stage">
      <p>Membrane stage: {Object.hasOwn(membraneStages,stage.phase)?membraneStages[stage.phase]:'Unknown'}.</p>
      <div className="surgical-metrics">
        <div><small>Cut boundary edges</small><strong>{amount(stage.cutBoundaryEdges,0)} / {amount(stage.requiredBoundaryEdges,0)}</strong></div>
        <div><small>Membrane removed · mm³</small><strong data-membrane-metric="removedMembraneMm3">{metricAmount(stage.membrane?.removedMembraneMm3)}</strong></div>
      </div>
      <p>Work-region entry: {stage.assessmentCurrent===true?(stage.workEntrySupported===true?'Permitted by native stage checks':'Not permitted'):'Unassessed'}. Both instruments withdrawn: {stage.assessmentCurrent===true?(stage.wholeExitComplete===true?'Yes':'No'):'Unknown'}.</p>
      <p className="surgical-note">{stage.startingExposure==='conditional_post_scalp_exposure'?'Conditional post-scalp exposure; scalp opening was not simulated and prior exposure cost unknown.':'Starting exposure is unsupported or unknown.'} Source-bound research model; tumor-task completion and clinical safety remain unassessed.</p>
      <p className="surgical-note">Unknowns: {stage.sourceUnknowns?.join('; ')||'Source coverage and clinical consequences remain unassessed'}.</p>
    </div>}
    <details><summary>Source target, material and fluid accounting</summary>
    {record?.fullAnnotationTargetMm3!==undefined&&<p className="surgical-note">Full source annotation: {amount(record.fullAnnotationTargetMm3)} mm³. Displayed percentages use the modeled source grid target.</p>}
    <div className="surgical-metrics" aria-label="Recorded normal and unknown material accounting">
      <div><small>Non-target normal material removed · mm³</small><strong>{amount(summary.normalRemovedMm3)}</strong></div>
      <div><small>Mapped unknown material removed · mm³</small><strong>{amount(summary.unknownRemovedMm3)}</strong></div>
      <div><small>Normal-material exposure · cell·s</small><strong>{amount(summary.normalExposureCellSeconds)}</strong></div>
      <div><small>Unknown-material exposure · cell·s</small><strong>{amount(summary.unknownExposureCellSeconds)}</strong></div>
    </div>
    <p className="surgical-note">Normal removal excludes target labels; exposure is grouped by mapped material. Missing anatomy and functional injury remain UNKNOWN.</p>
    <div className="surgical-metrics" aria-label="Recorded bone wounds and fluid">
      <div><small>Bone removed · mm³</small><strong>{amount(summary.boneRemovedMm3)}</strong></div>
      <div><small>Modeled wounds / active outlets</small><strong>{amount(summary.woundCount,0)} / {amount(summary.activeWounds,0)}</strong></div>
      <div><small>Nonpatent local vessel faces</small><strong>{amount(summary.nonpatentFaces,0)}</strong></div>
      <div><small>Spatial fluid / outlet reservoirs · mm³</small><strong>{amount(summary.spatialFluidMm3)} / {amount(summary.reservoirFluidMm3)}</strong></div>
      <div><small>Modeled blood generated · mm³</small><strong>{amount(summary.modeledBloodGeneratedMm3)}</strong></div>
      <div><small>Fluid suctioned · mm³</small><strong>{amount(summary.fluidSuctionedMm3)}</strong></div>
      <div><small>Elapsed simulation · s</small><strong>{amount(summary.elapsedSeconds)}</strong></div>
    </div>
    <p className="surgical-note">Modeled blood uses the wound-source ledger; suction can include irrigation fluid. Blue glyphs scale with each cell's recorded fluid volume, not a reconstructed fluid surface. Reservoirs are counted separately; field clearance does not restore vessel patency.</p>
      <p>Contact · cell·s: <span data-surgical-metric="contactCellSeconds">{amount(v.metrics.contactCellSeconds)}</span></p>
    </details>
    <details><summary>Complete-sequence breakdown</summary>
      {ledger?<>
        <p>{ledger.accounting}</p>
        <table aria-label="Complete sequence quantities"><thead><tr><th>Quantity</th><th>Whole sequence</th><th>Units</th></tr></thead><tbody>
          {([['targetRemovedMm3','Target removed'],['nonTargetRemovedMm3','Other tissue removed'],['elapsedSeconds','Modeled procedure time'],['toolExchanges','Instrument exchanges'],['acceptedActions','Accepted actions']] as const).map(([key,label])=><tr key={key}><th>{label}</th><td>{amount(ledger.dimensions[key].cumulative)}</td><td>{ledger.dimensions[key].unit}</td></tr>)}
        </tbody></table>
        <p>Withdrawal {ledger.withdrawalComplete?'complete':'incomplete'}. Remaining owned target {amount(ledger.target.residualMm3)} mm³.</p>
        <p>Original target remaining {amount(ledger.originalTarget.residualMm3)} mm³ · {ledger.originalTarget.fractionRemoved===null?'Unknown fraction':amount(ledger.originalTarget.fractionRemoved*100)+'% removed'}.</p>
        {ledger.originalTarget.reason&&<p>{ledger.originalTarget.reason}</p>}
        {ledger.target.residualCentroidRasMm&&<p>Residual centroid · RAS mm: {ledger.target.residualCentroidRasMm.map(n=>amount(n)).join(', ')}.</p>}
        {!!ledger.target.octants?.length&&<p>Residual by source-centroid octant · mm³: {ledger.target.octants.filter(o=>o.residualMm3>0).map(o=>`${o.region}: ${amount(o.residualMm3)}`).join(' · ')||'none'}.</p>}
        <p>Unavailable measures: {ledger.missingEvidence.map(key=>evidenceLabels[key]??key).join(', ')||'None in the declared dimensions'}.</p>
        {!!ledger.protection.missingRequired.length&&<p className="surgical-warning">Missing protection: {ledger.protection.missingRequired.join(', ')}.</p>}
        <ul>{ledger.limitations.map(line=><li key={line}>{line}</li>)}</ul>
      </>:<><p>Calculate the ledger for this exact accepted checkpoint, including all access and preparation.</p><button disabled={c.pending||!!c.preview||!!r?.verification?.readOnly||!!r?.playback?.readOnly} onClick={()=>void c.run('inspect',{includeSequenceOutcome:true})}>Calculate sequence breakdown</button></>}
    </details>
  </section>;
}
