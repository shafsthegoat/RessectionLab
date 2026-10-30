import {useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {consequenceComparisonData,consequenceAnalysisData,consequenceEvidenceMetrics,defaultConsequencePolicyInput,sameConsequencePolicyInput,validConsequencePolicyInput} from './consequence-comparison-data';
import type {ConsequenceComparison,ConsequencePolicyInput,ConsequenceDecision,ConsequenceViolation} from './consequence-comparison-data';

const metricNames:Record<string,string>={targetRemovedMm3:'Additional target removal',nonTargetRemovedMm3:'Cumulative non-target removal',elapsedSeconds:'Total modeled duration',supplyDeficitSeconds:'Vascular supply deficit',motorEncounterMm3:'Motor encounter',languageEncounterMm3:'Language encounter',patientShiftErrorMm:'Acquired patient shift error',brainStrainSquaredMm3S:'Brain strain² exposure',brainStressSquaredPa2Mm3S:'Brain stress² exposure'};
const format=(x:number)=>Math.abs(x)>=1?x.toFixed(2):x===0?'0':x.toPrecision(3);
export type ConsequencePolicyDraft={minimumAdditionalTargetMm3:string;maximumCumulativeNonTargetMm3:string;requiredMetrics:ConsequencePolicyInput['requiredMetrics'];maxBrainStrainSquaredMm3S:string;maxBrainStressSquaredPa2Mm3S:string};
export const consequencePolicyDraft=(input:ConsequencePolicyInput):ConsequencePolicyDraft=>({...input,minimumAdditionalTargetMm3:String(input.minimumAdditionalTargetMm3),maximumCumulativeNonTargetMm3:String(input.maximumCumulativeNonTargetMm3),maxBrainStrainSquaredMm3S:input.maxBrainStrainSquaredMm3S===null?'':String(input.maxBrainStrainSquaredMm3S),maxBrainStressSquaredPa2Mm3S:input.maxBrainStressSquaredPa2Mm3S===null?'':String(input.maxBrainStressSquaredPa2Mm3S)});
export function consequencePolicyFromDraft(draft:ConsequencePolicyDraft):ConsequencePolicyInput|null {
  const number=(value:string)=>value.trim()===''?NaN:Number(value);
  const input={minimumAdditionalTargetMm3:number(draft.minimumAdditionalTargetMm3),maximumCumulativeNonTargetMm3:number(draft.maximumCumulativeNonTargetMm3),requiredMetrics:draft.requiredMetrics,
    maxBrainStrainSquaredMm3S:draft.maxBrainStrainSquaredMm3S.trim()===''?null:number(draft.maxBrainStrainSquaredMm3S),maxBrainStressSquaredPa2Mm3S:draft.maxBrainStressSquaredPa2Mm3S.trim()===''?null:number(draft.maxBrainStressSquaredPa2Mm3S)};
  return validConsequencePolicyInput(input)?input:null;
}
function violationText(row:ConsequenceViolation){
  const metric=row.metric?(metricNames[row.metric]??row.metric):'';
  if(row.reason==='declared_research_constraint'&&row.actual!==undefined&&row.value!==undefined)return `${metric} ${format(row.actual)}; requires ${row.operator==='at_most'?'≤':'≥'} ${format(row.value)} (${row.scope})`;
  return `${metric?metric+' · ':''}${row.reason==='unknown_required_consequence'?'Unknown required evidence':row.reason.replaceAll('_',' ')}`;
}
function Exclusions({rows,worldCount}:{rows:ConsequenceViolation[];worldCount:number}){
  const groups=new Map<string,string[]>();
  for(const row of rows){const text=violationText(row);groups.set(text,[...(groups.get(text)??[]),row.scenario]);}
  return <ul>{[...groups].map(([text,worlds])=><li key={text}>{text}<small>{new Set(worlds).size===worldCount?'All declared worlds':worlds.join(', ')}</small></li>)}</ul>;
}
function DecisionSummary({decision}:{decision:ConsequenceDecision}){
  return <><p>{decision.stopFallback?'STOP with residual':decision.selectedBranchId?'Conditional research choice':'No executable choice'}: {decision.selectedBranchId??'unavailable'} · declared research task {decision.taskSatisfied?'met':'not met'}.</p>
    <p>Eligible continuations: {decision.feasibleContinuations?.join(', ')||'none'}. Paired-world Pareto front: {decision.pairedScenarioParetoFront.join(', ')||'none'}.</p>
    {decision.pairedScenarioParetoFront.length>1&&<p>Unresolved tradeoff: no unique continuation. STOP retains residual target.</p>}
    <p>STOP cleanup {decision.stopExecutable?'executable':'unavailable'}. Its fallback may violate the removal objective or required evidence. No scalar reward, averaging or clinical probabilities.</p></>;
}
function ConsequenceEditor({c,pause,data}:{c:SurgicalController;pause:()=>void;data:ConsequenceComparison}){
  const analysis=consequenceAnalysisData(c.result,data),initial=analysis?.latest?.policyInput??defaultConsequencePolicyInput(data);
  const [draft,setDraft]=useState<ConsequencePolicyDraft|null>(()=>initial?consequencePolicyDraft(initial):null);
  const input=draft?consequencePolicyFromDraft(draft):null,latest=analysis?.latest;
  const matches=!!(input&&latest&&sameConsequencePolicyInput(input,latest.policyInput));
  const current=!!(analysis?.current&&matches),decision=current?latest?.decision:null;
  const disabled=!!(c.pending||c.result?.verification?.status==='unverified-import'||c.result?.readOnly);
  const ids=[...new Set(data.samples.map(s=>s.branchId))];
  const update=(key:keyof Omit<ConsequencePolicyDraft,'requiredMetrics'>,value:string)=>setDraft(previous=>previous?{...previous,[key]:value}:previous);
  const range=(branch:string,key:string,scope:'cumulative'|'incremental')=>{
    const values=data.samples.filter(s=>s.branchId===branch).map(s=>key==='residualTargetMm3'?s.vector.residualTargetMm3:s.vector.dimensions[key]?.[scope]);
    if(values.some(v=>v==null)||!values.length)return 'Unknown';
    const numbers=values as number[],lo=Math.min(...numbers),hi=Math.max(...numbers);
    return lo===hi?format(lo):`${format(lo)} – ${format(hi)}`;
  };
  return <section aria-label="Reassess checked consequences" className="consequence-editor">
    <h4>Reassess checked alternatives</h4>
    <p>One frozen common checkpoint, two existing native continuations and executed STOP cleanup. Edit a declared research task, then explicitly reassess these checked records. This runs no new optimization or physical actions.</p>
    <p>Counts and modeled duration are fixed comparison objectives. Budgets are research declarations with no clinical meaning. Source annotation costs include the shared exposure; additional target removal starts after the common prefix.</p>
    {!analysis&&<p role="alert">The analysis context could not be bound to the accepted case, scene and common checkpoint. Reassessment is unavailable.</p>}
    {draft?<fieldset disabled={disabled}><legend>Declared research task</legend>
      <label>Minimum additional target removal · mm³<input aria-label="Minimum additional target removal mm³" type="number" min="0" step="any" value={draft.minimumAdditionalTargetMm3} onChange={e=>update('minimumAdditionalTargetMm3',e.target.value)}/></label>
      <label>Cumulative non-target removal budget · mm³<input aria-label="Cumulative non-target removal budget mm³" type="number" min="0" step="any" value={draft.maximumCumulativeNonTargetMm3} onChange={e=>update('maximumCumulativeNonTargetMm3',e.target.value)}/></label>
      <p>The initial budget is the shared source-count exposure at STOP, including all non-target removal before this checkpoint.</p>
      <fieldset className="consequence-evidence"><legend>Required evidence · unavailable evidence remains Unknown</legend>{consequenceEvidenceMetrics.map(metric=><label key={metric}><span>{metricNames[metric]} · Unknown</span><input type="checkbox" aria-label={`Require ${metricNames[metric]} evidence`} checked={draft.requiredMetrics.includes(metric)} onChange={e=>{const checked=e.target.checked;setDraft(previous=>previous?{...previous,requiredMetrics:checked?[...previous.requiredMetrics,metric]:previous.requiredMetrics.filter(x=>x!==metric)}:previous);}}/></label>)}</fieldset>
      <details><summary>Advanced mechanical research caps</summary><p>Unvalidated model exposure quantities, not injury thresholds. Leave blank for no cap.</p>
        <label>Maximum cumulative brain strain² · mm³·s<input aria-label="Maximum cumulative brain strain squared mm³ seconds" type="number" min="0" step="any" value={draft.maxBrainStrainSquaredMm3S} onChange={e=>update('maxBrainStrainSquaredMm3S',e.target.value)}/></label>
        <label>Maximum cumulative brain stress² · Pa²·mm³·s<input aria-label="Maximum cumulative brain stress squared Pa² mm³ seconds" type="number" min="0" step="any" value={draft.maxBrainStressSquaredPa2Mm3S} onChange={e=>update('maxBrainStressSquaredPa2Mm3S',e.target.value)}/></label>
      </details>
    </fieldset>:<p role="alert">The frozen research task and shared source-count budget are unavailable.</p>}
    {draft&&!input&&<p role="alert">Enter finite values of zero or greater. Mechanical caps may be blank.</p>}
    {analysis&&!analysis.available&&<><p>{analysis.reason?.replaceAll('_',' ')||'Open the accepted common checkpoint before reassessing.'}</p><button disabled={disabled} onClick={()=>{pause();void c.run('selectBranch',{branchId:analysis.baseline.branchId});}}>Open common checkpoint</button></>}
    <button className="primary-button" disabled={disabled||!analysis?.available||!input} onClick={()=>{if(input&&analysis?.available){pause();void c.run('reevaluateConsequences',{expectedComparisonSeal:data.comparisonSeal,policyInput:input});}}}>Reassess checked alternatives</button>
    <div role="status" data-consequence-analysis-status={current?'current':latest?'historical':'unassessed'}>{current&&latest?<><strong>Current reassessment · revision {latest.revision}</strong>{decision&&<DecisionSummary decision={decision}/>}</>:<p>{latest?matches?'Previous reassessment is historical at this view. Return to the common checkpoint and explicitly reassess.':'Draft changed. Previous reassessment does not apply to these inputs.':'No user reassessment yet. Reassess to compute eligibility and the paired-world Pareto front.'}</p>}</div>
    <div role="region" aria-label="Checked alternative consequences" className="consequence-alternatives">{ids.map(branch=>{
      const sample=data.samples.find(s=>s.branchId===branch&&s.scenarioId==='nominal')!;
      return <article key={branch} data-consequence-analysis-branch={branch} aria-label={`Consequences for ${branch}`}>
        <h5>{branch===data.stopBranchId?`STOP · ${branch}`:branch}</h5>
        <dl>{[['Additional target · mm³','targetRemovedMm3','incremental'],['Residual target · mm³','residualTargetMm3','cumulative'],['Cumulative non-target · mm³','nonTargetRemovedMm3','cumulative'],['Modeled duration · s','elapsedSeconds','cumulative']].map(([label,metric,scope])=><div key={metric}><dt>{label}</dt><dd>{range(branch,metric,scope as 'cumulative'|'incremental')}</dd></div>)}</dl>
        <p>{input?.requiredMetrics.length?input.requiredMetrics.map(metric=><span className="consequence-unknown" key={metric}>{metricNames[metric]}: Unknown</span>):'No evidence required. Unavailable evidence remains Unknown.'}</p>
        {decision?(decision.violations[branch]?.length?<Exclusions rows={decision.violations[branch]} worldCount={data.scenarios.length}/>:<p>{branch===data.stopBranchId?decision.stopExecutable?'Executable cleanup; no eligible continuation claim':'Cleanup is not executable':'Eligible in every declared world'}</p>):<p>Reassessment required</p>}
        {branch===data.stopBranchId&&decision&&<p>Cleanup {decision.stopExecutable?'executable':'unavailable'}; residual target remains.</p>}
        <button disabled={disabled} data-consequence-replay={branch} data-consequence-position={sample.vector.nativeActions} onClick={()=>{pause();void c.run('selectBranch',{branchId:branch,position:sample.vector.nativeActions});}}>Inspect {branch}</button>
      </article>;
    })}</div>
    <p>Declared worlds: {data.scenarios.map(s=>`${s.id} (relaxation ×${s.relaxationTimeScale})`).join('; ')}. Ranges span these sensitivity worlds, not confidence intervals or independent patients.</p>
    {latest&&<details><summary>Previous reassessment record · revision {latest.revision}{current?' · current':' · historical'}</summary><p>{current?'Bound to this accepted common checkpoint.':'Historical record only; no current decision is implied.'}</p><pre>{JSON.stringify(latest.policyInput,null,2)}</pre><DecisionSummary decision={latest.decision}/><p>Optimizer updates: 0 · new physical actions: 0 · clinical validation: false.</p></details>}
  </section>;
}
export function ConsequenceComparisonPanel({c,pause}:{c:SurgicalController;pause:()=>void}){
  const data=consequenceComparisonData(c.result);
  if(!data)return c.result?.consequenceComparison?<p role="alert">Consequence comparison could not be bound to the current source branches.</p>:null;
  return <details className="research-planning consequence-comparison"><summary>Recorded consequence comparison · {data.scenarios.length} declared parameter worlds</summary>
    <ConsequenceEditor key={`${c.result?.caseHash}:${data.comparisonSeal}`} c={c} pause={pause} data={data}/>
    <details data-consequence-frozen-report><summary>Original frozen experiment · read-only report, declaration and decisions</summary>
      <p>These are research choices among executed native programs. Model consequences have no calibrated injury meaning. Missing motor, language, vascular or acquired-shift evidence stays unknown.</p>
      {data.decisions.map(decision=><section key={decision.policy.id} aria-label={`Consequence decision ${decision.policy.id}`}><h4>{decision.policy.id.replaceAll('-',' ')}</h4><p data-consequence-decision={decision.policy.id} data-consequence-selected={decision.selectedBranchId??''}>{decision.stopFallback?'STOP with residual':decision.selectedBranchId?'Conditional research choice':'No executable choice'}: {decision.selectedBranchId??'unavailable'} · declared resection task {decision.taskSatisfied?'met':'not met'}.</p><p>Paired-scenario Pareto continuations: {decision.pairedScenarioParetoFront.join(', ')||'none'}. No scalar reward or clinical probabilities.</p><details><summary>Constraints and unavailable evidence</summary><pre>{JSON.stringify(decision.policy,null,2)}</pre><ul>{Object.entries(decision.violations).map(([branch,rows])=><li key={branch}>{branch}: {[...new Set(rows.map(r=>r.metric?`${r.metric} · ${r.reason.replaceAll('_',' ')}`:r.reason.replaceAll('_',' ')))].join('; ')||'No declared violation'}</li>)}</ul></details></section>)}
      <div role="region" aria-label="Consequence vectors"><table><thead><tr><th>Program</th><th>Additional target · mm³</th><th>Remaining target · mm³</th><th>Whole non-target · mm³</th><th>Brain strain² exposure · mm³·s</th><th>Total modeled time · s</th></tr></thead><tbody>{[...new Set(data.samples.map(sample=>sample.branchId))].map(branch=><tr key={branch} data-consequence-branch={branch}><th>{branch}</th>{[['targetRemovedMm3','incremental'],['residualTargetMm3','cumulative'],['nonTargetRemovedMm3','cumulative'],['brainStrainSquaredMm3S','cumulative'],['elapsedSeconds','cumulative']].map(([metric,scope])=>{const values=data.samples.filter(sample=>sample.branchId===branch).map(sample=>metric==='residualTargetMm3'?sample.vector.residualTargetMm3:sample.vector.dimensions[metric]?.[scope as 'cumulative'|'incremental']);const known=values.length>0&&values.every((value):value is number=>typeof value==='number');return <td key={metric}>{known?Math.min(...values)===Math.max(...values)?format(values[0]):`${format(Math.min(...values))} – ${format(Math.max(...values))}`:'Unknown'}</td>;})}</tr>)}</tbody></table></div>
      <details><summary>Frozen declaration</summary><pre>{JSON.stringify(data.declaration??null,null,2)}</pre></details>
      <details><summary>Complete original report</summary><pre>{JSON.stringify(c.result?.consequenceComparison,null,2)}</pre></details>
      <details><summary>Scenario assumptions and limitations</summary><ul>{data.limitations.map((x,i)=><li key={i}>{x}</li>)}</ul><pre>{JSON.stringify(data.scenarios,null,2)}</pre></details>
    </details>
    <p>Branch selection replays the nominal native scene. Counterfactual worlds are recorded sensitivity results. Physiological RL and patient-response validation remain on hold.</p>
  </details>;
}
