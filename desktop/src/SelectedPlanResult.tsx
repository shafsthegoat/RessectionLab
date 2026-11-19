import type {SelectedPlanDecision} from './surgical-types';

export function SelectedPlanResult({decision}:{decision:SelectedPlanDecision}) {
  const {measured,forecast}=decision;
  return <section aria-label="Selected route engine result" className="environment-route-decision">
    <h3>STOP before access</h3>
    <p>{decision.patientId} · {decision.routeName} · {decision.targetLayerId}</p>
    <p>{measured.actions} engine actions · {measured.elapsedSeconds} s · {measured.insertedTools} tools inserted</p>
    <p>{measured.targetRemovedMm3} mm³ target removed · {measured.nonTargetRemovedMm3} mm³ mapped non-target removed.
      {' '}<strong>{measured.originalTargetResidualMm3.toLocaleString(undefined,{maximumFractionDigits:2})} mm³ target remains</strong> on the engine grid.</p>
    <p>{forecast.reason} Treatment, vascular injury and functional outcome remain unassessed.</p>
    <details><summary>Decision and preparation record</summary>
      <p>{decision.geometry.lengthMm.toFixed(2)} mm proposed path · {decision.preparation.gridSpacingMm} mm research grid · {Object.values(decision.preparation.layers).map(layer=>layer.layerId).join(', ')}.</p>
      <p>STOP forecast · {forecast.preparationActions} preparation actions and {forecast.withdrawalActions} withdrawal actions.
        No tool entered. No physical treatment candidates were examined.</p>
      {decision.geometry.unknowns.length>0&&<ul>{decision.geometry.unknowns.map(item=><li key={item}>{item}</li>)}</ul>}
      <p>Admission checks · {decision.admissionCodes.join(', ')}</p>
      <p>Study revision <code>{decision.sourceSnapshotId}</code></p>
      <p>Plan revision <code>{decision.planRevision}</code></p>
      <p>Route <code>{decision.routeId}</code> · branch {decision.branchId}</p>
      <p>Engine state <code>{decision.stateId}</code></p>
      <p>Decision <code>{decision.decisionId}</code></p>
    </details>
  </section>;
}
