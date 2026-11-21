import type {SurgicalInteraction, SurgicalView} from '../surgical-types';

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const quantity = (n: unknown, unit: string) => finite(n) ? `${+n.toFixed(2)} ${unit}` : 'Unknown';

/** Render the receipt installed with the visible scene, including during
 * display preparation when a newer backend state may already exist. */
export function PracticeHud({view, interaction, membrane}: {view:SurgicalView; interaction:SurgicalInteraction; membrane?:string|null}) {
  const hud = interaction.practiceHud;
  if (!hud) return null;
  const total = hud.originalTargetVolumeMm3;
  const remaining = view.originalTargetResidualMm3;
  const removed = finite(total) && finite(remaining) ? total - remaining : null;
  const fraction = finite(removed) && finite(total) && total > 0 ? 100 * removed / total : null;
  const hand = interaction.selectedSlot;
  const slot = view.slots[hand];
  const active = interaction.effectiveActivation?.[hand] ?? slot?.active;
  const outcome = interaction.outcome;
  const status = interaction.recordedFrame ? 'Observed frame' : outcome?.status === 'partial' ? 'Partial action' :
    outcome?.status === 'rejected' ? 'Rejected action' : outcome?.status === 'accepted' ? 'Accepted action' : 'Verified checkpoint';
  return <section className="operative-readout practice-hud" aria-label="Practice metrics" data-operative-readout-state={view.stateId}>
    <strong>{status} · {quantity(view.metrics.elapsedSeconds,'s')} · {hand} {active ? 'active' : 'inactive'}</strong>
    {membrane&&<span>{membrane}</span>}
    <dl>
      <div><dt>Original target removed</dt><dd>{quantity(removed,'mm³')}</dd></div>
      <div><dt>Original target remaining</dt><dd>{quantity(remaining,'mm³')}</dd></div>
      <div><dt>Non-target brain removed</dt><dd>{quantity(hud.nonTargetBrainRemovedMm3,'mm³')}</dd></div>
      <div><dt>Field fluid</dt><dd>{quantity(view.metrics.fluidPresentMm3,'mm³')}</dd></div>
      <div><dt>Field debris</dt><dd>{quantity(view.metrics.debrisMm3,'mm³')}</dd></div>
      {finite(view.metrics.brainMaxDisplacementMm) && <div><dt>Max tissue shift</dt><dd>{quantity(view.metrics.brainMaxDisplacementMm,'mm')}</dd></div>}
    </dl>
    {finite(fraction) && <span>{+fraction.toFixed(1)}% removed of {quantity(total,'mm³')} original target</span>}
  </section>;
}
