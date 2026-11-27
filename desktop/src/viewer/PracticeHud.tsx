import {useMemo} from 'react';
import {surgicalFieldAmounts} from '../full-case-presentation';
import type {SurgicalInteraction, SurgicalView} from '../surgical-types';
import {practiceStatus,type ScenarioLine} from '../operative-scenarios';
import {ScenarioReadout} from './ScenarioReadout';

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const quantity = (n: unknown, unit: string) => finite(n) ? `${+n.toFixed(2)} ${unit}` : 'Unknown';

/** Render the receipt installed with the visible scene, including during
 * display preparation when a newer backend state may already exist. Closed, it
 * is one status line at the pane edge (clock, target removed, blood loss and
 * the most urgent finding); the full readout opens as a disclosure. */
export function PracticeHud({view, interaction, membrane, scenarios=[]}: {view:SurgicalView; interaction:SurgicalInteraction; membrane?:string|null; scenarios?:readonly ScenarioLine[]}) {
  const hud = interaction.practiceHud;
  const field=useMemo(()=>surgicalFieldAmounts(view),[view]);
  if (!hud) return null;
  const total = hud.originalTargetVolumeMm3;
  const remaining = view.originalTargetResidualMm3;
  const removed = finite(total) && finite(remaining) ? total - remaining : null;
  const fraction = finite(removed) && finite(total) && total > 0 ? 100 * removed / total : null;
  const hand = interaction.selectedSlot;
  const slot = view.slots[hand];
  const active = interaction.effectiveActivation?.[hand] ?? slot?.active;
  const outcome = interaction.outcome;
  const acceptedEvents=outcome&&outcome.status!=='rejected'?outcome.events:[];
  const irrigating=acceptedEvents.some(event=>event.kind==='irrigation'&&Number(event.volumeMm3)>0);
  const suctioning=acceptedEvents.some(event=>event.kind==='aspiration'&&Number(event.fluid)>0);
  const coagulating=acceptedEvents.some(event=>event.kind==='local_energy'&&Number(event.exposure)>0);
  const activity=irrigating?'Irrigation delivered':suctioning?'Free fluid suctioned':coagulating?'Coagulation contact':null;
  const sources=view.sources?.filter(source=>source.bleedingActive===true||source.bleedingActive===false);
  const emitting=sources?.filter(source=>source.bleedingActive===true).length;
  const status = interaction.recordedFrame ? 'Observed frame' : outcome?.status === 'partial' ? 'Partial action' :
    outcome?.status === 'rejected' ? 'Rejected action' : outcome?.status === 'accepted' ? 'Accepted action' : 'Verified checkpoint';
  const line=practiceStatus(view,total);
  return <details className="operative-readout practice-hud" aria-label="Practice metrics" data-operative-readout-state={view.stateId}
    onDoubleClick={event=>event.stopPropagation()} onPointerDown={event=>event.stopPropagation()}>
    <summary title={[line.elapsed,line.removed+' removed',line.bloodLoss.replace(/^loss /,'blood loss '),line.urgent?.text,line.urgent?.detail].filter(Boolean).join(' · ')+' — show the full practice readout'}
      data-urgent={line.urgent?.tone}>
      <span>{line.elapsed}</span><span>{line.removed}</span><span>{line.bloodLoss}</span>
      {line.urgent&&<b data-tone={line.urgent.tone}>{line.urgent.text}</b>}
      {line.urgent?.detail&&<small data-tone={line.urgent.tone}>{line.urgent.detail}</small>}
    </summary>
    <div className="practice-hud-body">
    <strong>{status} · {quantity(view.metrics.elapsedSeconds,'s')} · {hand} {active ? 'active' : 'inactive'}</strong>
    {membrane&&<span>{membrane}</span>}
    {(activity||!!sources?.length)&&<span>{[activity,sources?.length?`${emitting} of ${sources.length} modeled sources emitting`:null].filter(Boolean).join(' · ')}</span>}
    <dl>
      <div><dt>Original target removed</dt><dd>{quantity(removed,'mm³')}</dd></div>
      <div><dt>Original target remaining</dt><dd>{quantity(remaining,'mm³')}</dd></div>
      <div><dt>Non-target brain removed</dt><dd>{quantity(hud.nonTargetBrainRemovedMm3,'mm³')}</dd></div>
      <div><dt>Free-field fluid</dt><dd>{quantity(field.spatialFluidMm3,'mm³')}</dd></div>
      <div><dt>Outlet reservoir</dt><dd>{quantity(field.reservoirFluidMm3,'mm³')}</dd></div>
      <div><dt>Retained fluid total</dt><dd>{quantity(field.retainedFluidMm3,'mm³')}</dd></div>
      <div><dt>Field debris</dt><dd>{quantity(view.metrics.debrisMm3,'mm³')}</dd></div>
      {finite(view.metrics.brainMaxDisplacementMm) && <div><dt>Max tissue shift</dt><dd>{quantity(view.metrics.brainMaxDisplacementMm,'mm')}</dd></div>}
    </dl>
    {finite(fraction) && <span>{+fraction.toFixed(1)}% removed of {quantity(total,'mm³')} original target</span>}
    <ScenarioReadout lines={scenarios}/>
    </div>
  </details>;
}
