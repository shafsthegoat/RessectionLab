import type {ApproachConsequenceEvidence,ApproachConsequenceMetric,NativeApproach} from './native-approaches';
import type {ViewerRoute} from './viewer/contracts';

const geometryRows=[
  'lengthMm','targetVoxelCenterCoverageMm3','targetPointInsideLabel',
  'nonTargetTraversalUpperBoundMm3','tractBundlesEncountered',
];
const mechanismGroups=[
  ['Functional','motorEncounterMm3','languageEncounterMm3'],
  ['Vascular','bloodGeneratedMm3','supplyDeficitSeconds','drainageDeficitSeconds'],
  ['Mechanical','brainStressSquaredPa2Mm3S','brainStrainSquaredMm3S','patientShiftErrorMm'],
  ['Clinical','clinicalDeficitProbability'],
];
const units:Record<string,string>={mm3:'mm³','mm3*s':'mm³·s','Pa2*mm3*s':'Pa²·mm³·s'};
const readable=(s:string)=>s.replace(/[_-]/g,' ');
const number=new Intl.NumberFormat(undefined,{maximumFractionDigits:1});
function reading(row?:ApproachConsequenceMetric){
  if(row?.value==null)return 'Unknown';
  if(typeof row.value==='boolean')return row.value?'Inside':'Outside';
  return number.format(row.value)+(row.status==='partial_coverage'?'*':'');
}
function quantity(row?:ApproachConsequenceMetric){
  return row?.value==null?'Unknown':`${reading(row)}${row.unit==='mm3'?' mm³':''}`;
}
function metricLabel(row:ApproachConsequenceMetric){
  return row.unit&&row.unit!=='1'&&row.unit!=='boolean'
    ?`${row.label} (${units[row.unit]??row.unit})`:row.label;
}

export function NativeConsequenceReadout({packet,approaches,comparison}:{packet:ApproachConsequenceEvidence;approaches:NativeApproach[];comparison:ViewerRoute[]}){
  const evidenceById=new Map(packet.routes.map(route=>[route.approachId,route]));
  const approachesById=new Map(approaches.map(approach=>[approach.id,approach]));
  const routes=comparison.flatMap(({route_id,comparisonSlot})=>{
    const route=evidenceById.get(route_id),approach=approachesById.get(route_id);
    return route&&route.comparisonSlot===comparisonSlot&&route.targetLayerId===packet.targetLayerId
      &&route.targetCandidateBinding===packet.targetCandidateBinding&&route.sourceSnapshotId===packet.sourceSnapshotId
      &&approach?.assessmentStatus==='current'&&approach.sourceSnapshotId===route.sourceSnapshotId
      &&approach.parameterHash===route.parameterHash&&approach.measurementHash===route.measurementHash?[route]:[];
  });
  const first=routes[0];
  if(!first?.target)return null;
  const selectedId=comparison.find(route=>route.selected)?.route_id;
  const selected=routes.find(route=>route.approachId===selectedId);
  return <section className="approach-consequence-comparison" aria-label="Native consequence comparison"
    data-native-consequence-comparison data-source-snapshot={packet.sourceSnapshotId} data-target-binding={packet.targetCandidateBinding}>
    <h2>Consequence evidence</h2>
    <p className="approach-consequence-disposition">Reassess before treatment</p>
    <p>{first.target.label} · {quantity(first.dimensions.targetLabelVolumeMm3)} labelled membership · {readable(first.target.evidenceClass)}</p>
    <table aria-label="Geometric evidence by approach">
      <thead><tr><th scope="col">Derived geometry</th>{routes.map(route=><th scope="col" title={route.name} key={route.approachId} data-native-consequence-route={route.approachId}>{route.comparisonSlot}</th>)}</tr></thead>
      <tbody>{geometryRows.map(key=>first.dimensions[key]&&<tr key={key}><th scope="row">{metricLabel(first.dimensions[key])}</th>
        {routes.map(route=><td key={route.approachId} title={route.dimensions[key]?.reason??undefined}>{reading(route.dimensions[key])}</td>)}</tr>)}</tbody>
    </table>
    {routes.some(route=>geometryRows.some(key=>route.dimensions[key]?.status==='partial_coverage'))&&<p>* Measured within partial source coverage.</p>}
    <p>Approached volume counts labelled voxel centres inside the corridor. Resection and residual after treatment are unknown.</p>
    {selected&&<>
    <p>{selected.researchDisposition.reasons.slice(0,2).join(' ')}</p>
    <details>
      <summary>Functional, vascular and mechanical evidence</summary>
      <dl className="approach-value-list">{mechanismGroups.map(([group,...metrics])=><div key={group}><dt>{group}</dt>
        <dd>{metrics.filter(key=>selected.dimensions[key]).map(key=><span key={key} title={selected.dimensions[key].reason??undefined}>
          {selected.dimensions[key].label} · {reading(selected.dimensions[key])}<br/>
        </span>)}</dd></div>)}</dl>
    </details>
    <details>
      <summary>Coverage, warnings and registration · {selected.comparisonSlot}</summary>
      <ul>{selected.warnings.criticalContacts.map(row=><li key={row.layerId}>{row.label} · {readable(row.contact)} contact</li>)}
        {[...new Set([...selected.warnings.constraintFailures,...selected.warnings.missingEvidence])].map(reason=><li key={reason}>{reason}</li>)}</ul>
      {selected.registrations.map(row=><p key={row.sourceSeriesId}>{row.sourceSeriesId} · {readable(row.status)}{row.method&&` · ${row.method}`}.
        Sampled landmark residual {reading(row.landmarkResidual)}{row.landmarkResidual.value!=null?' mm':''}. Whole-field error {reading(row.wholeFieldError)}.</p>)}
      {!!selected.uncertainty.scenarioCount&&<p>{selected.uncertainty.scenarioCount} declared geometry scenarios, including nominal.</p>}
      {selected.entrySupport&&<p>Entry surface · {selected.entrySupport.label} · {selected.entrySupport.sourceSeries}</p>}
      {selected.uncertainty.entryRegistration&&<p>Entry registration sensitivity · {readable(selected.uncertainty.entryRegistration.status)}
        {selected.uncertainty.entryRegistration.scenarioCount>0&&` · ${selected.uncertainty.entryRegistration.assessedCount}/${selected.uncertainty.entryRegistration.scenarioCount} translated entry scenarios assessed`}.</p>}
      {selected.uncertainty.limitations.map(reason=><p key={reason}>{reason}</p>)}
    </details>
    {selected.sources.some(source=>source.kind==='tractography')&&<details>
      <summary>Supplied streamline encounters · {selected.comparisonSlot}</summary>
      <ul>{selected.sources.filter(source=>source.kind==='tractography').map(source=><li key={source.layerId}>
        {source.label} · {reading(source.readouts.intersectedStreamlineCount)} streamlines · {readable(source.coverage.status)}
      </li>)}</ul>
    </details>}
    </>}
  </section>;
}
