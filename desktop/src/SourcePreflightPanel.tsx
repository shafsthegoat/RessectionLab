import type {SourcePreflight} from './interactive-planning-data';

const labels:Record<string,string>={
  'source-role':'Source provenance and anatomy bindings','source-bytes':'Native source bytes',
  'native-replay':'Recorded native replay','model-versions':'Instrument geometry and shared clock',
  'withdrawn-baseline':'Withdrawn instruments','executed-exposure':'Executed exposure',
  'static-baseline':'Static geometry','declared-task':'Bounded planning task','evidence-review':'Anatomical review',
};
export function SourcePreflightPanel({report}:{report:SourcePreflight}){
  const source=report.sourceEvidence;
  return <details aria-label="Planning readiness" open>
    <summary>Source and checkpoint readiness</summary>
    <p><strong>{report.planningAdmitted?'The existing task admission is verified at this checkpoint.':'Planning readiness is incomplete.'}</strong>{!report.planningAdmitted&&' Complete the outstanding source and task checks before planning.'}</p>
    {source&&<>
      <p>{source.patientId} · {source.sourceRole} · {source.studyTimepoint}</p>
      <p>Modalities: {source.images.map(row=>row.modality).join(', ')}.</p>
      <ul>{Object.entries(source.anatomy).map(([name,row])=><li key={name}>{name}: {row.kind} · {row.reviewStatus}. {row.coverage}</li>)}</ul>
      <p>Missing evidence: {source.unknownEvidence.join(', ')}.</p>
    </>}
    <p>Planning requires a verified task and checkpoint. This inspection leaves anatomy and instrument actions unchanged.</p>
    <ul>{report.checks.map(row=><li key={row.id} data-preflight-check={row.id} data-preflight-status={row.status}>
      <strong>{labels[row.id]??row.id}: {row.status==='passed'?'checked':row.status.replaceAll('-',' ')}</strong>. {row.description}
    </li>)}</ul>
    <p className="surgical-note">{report.scope}</p>
  </details>;
}
