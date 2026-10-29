import type {SurgicalController} from './use-surgical';
import {consequenceComparisonData} from './consequence-comparison-data';

export function ConsequenceComparisonPanel({c,pause}:{c:SurgicalController;pause:()=>void}){
  const data=consequenceComparisonData(c.result);
  if(!data)return c.result?.consequenceComparison?<p role="alert">Consequence comparison could not be bound to the current source branches.</p>:null;
  const ids=[...new Set(data.samples.map(s=>s.branchId))];
  const disabled=c.pending||c.result?.verification?.status==='unverified-import'||c.result?.readOnly;
  const range=(branch:string,key:string,scope:'cumulative'|'incremental')=>{
    const values=data.samples.filter(s=>s.branchId===branch).map(s=>key==='residualTargetMm3'?s.vector.residualTargetMm3:s.vector.dimensions[key]?.[scope]);
    if(values.some(v=>v==null)||!values.length)return 'Unknown';
    const numbers=values as number[],lo=Math.min(...numbers),hi=Math.max(...numbers);
    const format=(x:number)=>Math.abs(x)>=1?x.toFixed(2):x.toPrecision(3);
    return lo===hi?format(lo):`${format(lo)} – ${format(hi)}`;
  };
  return <details className="research-planning consequence-comparison"><summary>Recorded consequence comparison · {data.scenarios.length} declared parameter worlds</summary>
    <p>These are research choices among executed native programs. Model consequences have no calibrated injury meaning. Missing motor, language, vascular or acquired-shift evidence stays unknown.</p>
    <p>Source annotation costs include the whole exposure. Target gain is additional removal after the exact common prefix. Ranges describe declared relaxation scenarios, not confidence intervals or independent patients.</p>
    {data.decisions.map(decision=><section key={decision.policy.id} aria-label={`Consequence decision ${decision.policy.id}`}>
      <h4>{decision.policy.id.replaceAll('-',' ')}</h4>
      <p data-consequence-decision={decision.policy.id} data-consequence-selected={decision.selectedBranchId??''}>
        {decision.stopFallback?'STOP with residual':decision.selectedBranchId?'Conditional research choice':'No executable choice'}: {decision.selectedBranchId??'unavailable'} · declared resection task {decision.taskSatisfied?'met':'not met'}.
      </p>
      <p>Paired-scenario Pareto continuations: {decision.pairedScenarioParetoFront.join(', ')||'none'}. No scalar reward or clinical probabilities.</p>
      <details><summary>Constraints and unavailable evidence</summary><pre>{JSON.stringify(decision.policy,null,2)}</pre>
        <ul>{Object.entries(decision.violations).map(([branch,rows])=><li key={branch}>{branch}: {[...new Set(rows.map(r=>r.metric?`${r.metric} · ${r.reason.replaceAll('_',' ')}`:r.reason.replaceAll('_',' ')))].join('; ')||'No declared violation'}</li>)}</ul>
      </details>
    </section>)}
    <div role="region" aria-label="Consequence vectors"><table><thead><tr><th>Program</th><th>Additional target · mm³</th><th>Remaining target · mm³</th><th>Whole non-target · mm³</th><th>Brain strain² exposure · mm³·s</th><th>Total modeled time · s</th></tr></thead><tbody>
      {ids.map(branch=><tr key={branch} data-consequence-branch={branch}><th>{branch}</th>
        <td>{range(branch,'targetRemovedMm3','incremental')}</td><td>{range(branch,'residualTargetMm3','cumulative')}</td><td>{range(branch,'nonTargetRemovedMm3','cumulative')}</td>
        <td>{range(branch,'brainStrainSquaredMm3S','cumulative')}</td><td>{range(branch,'elapsedSeconds','cumulative')}</td></tr>)}
    </tbody></table></div>
    <div className="surgical-actions">{ids.map(branch=>{const sample=data.samples.find(s=>s.branchId===branch&&s.scenarioId==='nominal')!;return <button key={branch} disabled={disabled}
      data-consequence-replay={branch} data-consequence-position={sample.vector.nativeActions}
      onClick={()=>{pause();void c.run('selectBranch',{branchId:branch,position:sample.vector.nativeActions});}}>Inspect {branch}</button>;})}</div>
    <details><summary>Scenario assumptions and limitations</summary><ul>{data.limitations.map((x,i)=><li key={i}>{x}</li>)}</ul><pre>{JSON.stringify(data.scenarios,null,2)}</pre></details>
    <p>Branch selection replays the nominal native scene. Counterfactual worlds are recorded sensitivity results. Physiological RL and patient-response validation remain on hold.</p>
  </details>;
}
