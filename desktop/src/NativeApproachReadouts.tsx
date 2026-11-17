import type {ApproachInputComparisonReport,ApproachProposalReport,ApproachSensitivity,ApproachScenarioSet,ApproachSourceReading,ApproachScenarioGuard} from './native-approaches';
import {lowerBoundText} from './native-approaches';

const metricLabels:Record<string,string>={
  lengthMm:'Corridor length',
  targetReachMm3:'Target voxel centres reached',
  nonTargetTraversalMm3:'Non-target brain traversal upper bound',
  vascularClearanceMm:'Vascular clearance lower bound',
  tractClearanceMm:'Tract clearance lower bound',
  otherCriticalClearanceMm:'Other critical-label clearance lower bound',
};

function measurement(key:string,value:number|null,units?:string){
  if(value===null||!Number.isFinite(value))return 'Unknown';
  const number=key.endsWith('ClearanceMm')?lowerBoundText(value):value.toFixed(1);
  return number+' '+(units??(key.endsWith('Mm3')?'mm³':'mm'));
}

function point(values:number[]){return values.map(v=>v.toFixed(1)).join(', ');}

function ScenarioGuard({guard}:{guard:ApproachScenarioGuard}){
  return <div className="approach-scenario-guard">
    {guard.vetoScenarioIds.length>0&&<p>Hard-constraint veto in {guard.vetoScenarioIds.join(', ')}.</p>}
    {guard.incompleteScenarioIds.length>0&&<p>Incomplete scenario coverage in {guard.incompleteScenarioIds.join(', ')}.</p>}
  </div>;
}

function SourceReadings({readings,labels}:{readings:Record<string,ApproachSourceReading>;labels:Record<string,string>}){
  return <details><summary>Per-source clearance, contact and coverage</summary>
    <dl className="approach-value-list approach-source-values">{Object.entries(readings).map(([id,[coverage,clearance,contact,count]])=><div key={id}>
      <dt>{labels[id]??id}</dt><dd>{coverage==='unavailable'?'Unassessed':<>{measurement('sourceClearanceMm',clearance)} lower bound<br/>{contact==='detected'?'Contact detected':contact==='unresolved'?'Contact unresolved':'No detected contact'}{count!==null&&` (${count.toLocaleString()})`}<br/>{coverage==='full-field'?'Whole envelope in supplied field':'Partial field · outside unassessed'}</>}</dd>
    </div>)}</dl>
  </details>;
}

export function ApproachProposalDetails({report,targetLabel,labels}:{report:ApproachProposalReport;targetLabel?:string;labels:Record<string,string>}){
  const reasons=report.reasons;
  return <details className="approach-proposal-report" data-proposal-status={report.assessmentStatus??'current'}>
    <summary>{report.accepted} alternatives retained · sampled entries</summary>
    {report.assessmentStatus==='stale'&&<p role="status">Recorded comparison refers to older source anatomy. Generate alternatives for this revision.</p>}
    {report.assessmentStatus==='historical'&&<p role="status">{report.invalidationReason??'The saved corridors have changed since these alternatives were frozen.'}</p>}
    {targetLabel&&<p>Frozen proposal target · {targetLabel}</p>}
    <p>{report.selectionRule}</p>
    {report.whyFewer&&<p>{report.whyFewer}</p>}
    <p>{report.enumerated??report.evaluated} sampled · {report.evaluated} measured · {report.rejected} rejected</p>
    {typeof reasons==='string'?<p>{reasons}</p>:reasons&&typeof reasons==='object'?<ul>{Object.entries(reasons).map(([reason,count])=><li key={reason}>{reason} ({String(count)})</li>)}</ul>:null}
    {report.candidates&&<details>
      <summary>All {report.candidates.length} sampled paths</summary>
      <ol className="approach-diagnostics-list">{report.candidates.map(candidate=><li key={candidate.id} data-sampled-disposition={candidate.disposition}>
        <details><summary>{candidate.disposition.replaceAll('-', ' ')} · {point(candidate.entry)} RAS mm</summary>
          <p>Target (RAS mm) {point(candidate.target)} · capsule radius {candidate.radiusMm} mm</p>
          {candidate.pareto===true&&<p>Pareto frontier within its comparable evidence group.</p>}
          {!!candidate.detectedConflictLayerIds?.length&&<p>Detected contact · {candidate.detectedConflictLayerIds.map(id=>labels[id]??id).join(', ')}</p>}
          {candidate.reasons?.map(reason=><p key={reason}>{reason}</p>)}
          {candidate.scenarioGuard&&<ScenarioGuard guard={candidate.scenarioGuard}/>}
          {candidate.objectives&&<dl className="approach-value-list">{Object.entries(candidate.objectives).map(([metric,value])=><div key={metric}><dt>{metricLabels[metric]??metric}</dt><dd>{measurement(metric,value)}</dd></div>)}</dl>}
          {candidate.sourceReadings&&<SourceReadings readings={candidate.sourceReadings} labels={labels}/>}
          {report.finalistOutcomes?.[candidate.id]&&<div><h4>Worst assessed values</h4><dl className="approach-value-list">{Object.entries(report.finalistOutcomes[candidate.id].metrics).map(([metric,value])=><div key={metric}><dt>{metricLabels[metric]??metric}</dt><dd>{measurement(metric,value.worst,value.units)}</dd></div>)}</dl>
            <p>{report.finalistOutcomes[candidate.id].violatedScenarioIds.length} scenarios with constraint violations · {report.finalistOutcomes[candidate.id].unassessedScenarioIds.length} unassessed scenarios</p></div>}
        </details>
      </li>)}</ol>
    </details>}
  </details>;
}

export function ApproachSensitivityDetails({sensitivity,scenarioSet,labels,referenceBound=false}:{sensitivity:ApproachSensitivity;scenarioSet?:ApproachScenarioSet;labels:Record<string,string>;referenceBound?:boolean}){
  const metrics=Object.entries(sensitivity.metrics);
  const nominalSet=scenarioSet?.id===sensitivity.scenarioSetId?scenarioSet:sensitivity.scenarioSet;
  return <section className="approach-sensitivity" data-scenario-set={sensitivity.scenarioSetId}>
    <h3>Declared sensitivity</h3>
    <p>{sensitivity.scenarioCount} declared scenarios. {sensitivity.meaning??'Worst values use assessed scenarios, including nominal. The empirical tail uses only complete perturbed values.'}</p>
    {referenceBound&&<p>Target and brain labels are bound to the fixed reference scan. Their reach and traversal values stay fixed while the scenarios move registered sources and test critical-label boundaries.</p>}
    <p>These are geometric sensitivity estimates. Postoperative deficit probabilities are unavailable.</p>
    <div className="approach-metric-results">
      {metrics.map(([key,metric])=><article key={key}>
        <h4>{metricLabels[key]??key}</h4>
        <dl className="approach-value-list">
          <div><dt>Nominal</dt><dd>{measurement(key,metric.nominal,metric.units)}</dd></div>
          <div><dt>Worst measured</dt><dd>{measurement(key,metric.worst,metric.units)}</dd></div>
          <div><dt>Adverse {Math.round(metric.tailFraction*100)}% mean</dt><dd>{measurement(key,metric.tailMean,metric.units)}</dd></div>
        </dl>
        <p>{metric.assessedCount} assessed · {metric.unknownCount} unknown{metric.worstScenarioId&&' · '+metric.worstScenarioId}</p>
      </article>)}
    </div>
    <details><summary>Scenarios and unassessed results</summary>
      {(nominalSet?.assumptions??sensitivity.assumptions??[]).map(assumption=><p key={assumption}>{assumption}</p>)}
      <p className="approach-identity">Scenario set {sensitivity.scenarioSetId}</p>
      <ol className="approach-diagnostics-list">{sensitivity.scenarios.map(scenario=>{const definition=nominalSet?.scenarios.find(row=>row.id===scenario.id);return <li key={scenario.id}>
        <details><summary>{scenario.id} · {scenario.status.replaceAll('-', ' ')}</summary>
          {definition&&<p>Critical-label boundary padding {definition.boundaryMm??0} mm{Object.entries(definition.sourceTranslationsMm??{}).map(([source,offset])=><span key={source}><br/>{source} translation (RAS mm) {point(offset)}</span>)}</p>}
          {scenario.conflictLayerIds?.length? <p>Detected contact · {scenario.conflictLayerIds.map(id=>labels[id]??id).join(', ')}</p>:null}
          {scenario.constraintFailures?.map(failure=><p key={failure}>{failure}</p>)}
          {scenario.unknowns?.map((unknown,index)=><p key={index}>{unknown}</p>)}
          <dl className="approach-value-list">{Object.entries(scenario.values).map(([key,value])=><div key={key}><dt>{metricLabels[key]??key}</dt><dd>{measurement(key,value)}</dd></div>)}</dl>
          {scenario.sourceReadings&&<SourceReadings readings={scenario.sourceReadings} labels={labels}/>}
        </details>
      </li>})}</ol>
    </details>
  </section>;
}

export function ApproachInputComparison({report,labels,targetLabel}:{report:ApproachInputComparisonReport;labels:Record<string,string>;targetLabel?:string}){
  return <details className="approach-input-result" data-input-comparison={report.assessmentStatus??'current'}>
    <summary>Rich and restricted comparison results</summary>
    {report.assessmentStatus==='stale'&&<p role="status">Recorded comparison refers to older source anatomy. Rerun to assess this revision.</p>}
    {targetLabel&&<p>Frozen comparison target · {targetLabel}</p>}
    <p>{report.candidateBudget} sampled entries per arm · {report.scenarioSet.scenarios.length} shared scenarios</p>
    {report.candidateFreeze&&<p className="approach-identity">Frozen candidate identity {report.candidateFreeze.id}</p>}
    {report.guardProtection&&<p>{report.guardProtection}</p>}
    <p>Both arms use the same imperfect geometric reference. Operative permissions and clinical outcomes are unassessed.</p>
    {(['rich','restricted'] as const).map(kind=>{const arm=report.arms[kind];return <section key={kind} data-input-arm={kind}>
      <h3>{kind==='rich'?'Rich inputs':'Restricted inputs'}</h3>
      <p>{arm.nominated.length} nominated · {arm.refusalCount} reference refusals · {arm.abstentionCount} abstentions</p>
      <details><summary>Inputs available before nomination</summary><ul>{arm.inputWhitelist.map(input=><li key={input}>{input}</li>)}</ul>
        <p>Source scans · {arm.sourceSeriesIds.join(', ')||'None'}</p><p>Labels · {arm.layerIds.map(id=>labels[id]??id).join(', ')||'None'}</p>
        <p>Tract bundles · {arm.tractIds.map(id=>labels[id]??id).join(', ')||'None'}</p>
      </details>
      <ol className="approach-diagnostics-list">{arm.nominated.map((candidate,index)=>{
        const geometry=arm.candidates.find(row=>row.id===candidate.candidateId);
        return <li key={candidate.candidateId}><details><summary>Nomination {index+1} · {candidate.status==='refused'?'Reference veto':'Geometrically eligible'}</summary>
          {geometry&&<p>Entry (RAS mm) {point(geometry.entry)} · capsule radius {geometry.radiusMm} mm</p>}
          {candidate.reasons.map(reason=><p key={reason}>{reason}</p>)}
          {candidate.scenarioGuard&&<ScenarioGuard guard={candidate.scenarioGuard}/>}
          {!!candidate.detectedConflictLayerIds?.length&&<p>Detected contact · {candidate.detectedConflictLayerIds.map(id=>labels[id]??id).join(', ')}</p>}
          {candidate.objectives&&<dl className="approach-value-list">{Object.entries(candidate.objectives).map(([key,value])=><div key={key}><dt>{metricLabels[key]??key}</dt><dd>{measurement(key,value)}</dd></div>)}</dl>}
          {candidate.sensitivity&&<ApproachSensitivityDetails sensitivity={candidate.sensitivity} scenarioSet={report.scenarioSet} labels={labels}/>}
        </details></li>;
      })}</ol>
    </section>})}
  </details>;
}
