import type {SurgicalController} from './use-surgical';
import {fullCaseStateSummary} from './full-case-presentation';
export const amount=(value:unknown,digits=2)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'Unknown';
const metricAmount=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(2):'Unknown';

/** One readout for the anatomy actually displayed. Missing values stay unknown. */
export function SurgicalOutcomeSummary({c}:{c:SurgicalController}) {
  const v=c.preview??c.view,r=c.result;if(!v)return null;
  const summary=fullCaseStateSummary(v),record=r?.fullCase;
  return <section className="surgical-outcome" aria-label="Modeled outcome summary">
    <h3>{c.preview?'Preview modeled outcome':'Modeled outcome'}</h3>
    <div className="surgical-metrics">
      <div><small>Target removed · mm³</small><strong data-surgical-metric="targetRemovedMm3">{metricAmount(summary.targetRemovedMm3)}</strong></div>
      <div><small>Target remaining · mm³</small><strong>{amount(summary.targetRemainingMm3)}</strong></div>
      <div><small>Other removed · mm³</small><strong data-surgical-metric="nonTargetRemovedMm3">{metricAmount(v.metrics.nonTargetRemovedMm3)}</strong></div>
      <div><small>Modeled field fluid · mm³</small><strong data-surgical-metric="fluidPresentMm3">{metricAmount(v.metrics.fluidPresentMm3)}</strong></div>
    </div>
    <p className="surgical-note">{summary.targetRemovedPercent===null?'Unknown fraction':amount(summary.targetRemovedPercent)+'%'} of {amount(summary.initialTargetMm3)} mm³ modeled target. Functional injury remains Unknown.</p>
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
  </section>;
}
