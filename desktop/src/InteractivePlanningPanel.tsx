import {useEffect,useRef,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {Point3} from './viewer/coordinates';
import {currentPlanningReport,currentSourcePreflight,planningBinding,planningDraftIssue,selectedPlanningRoute,PLANNING_ROUTE_COLORS} from './interactive-planning-data';
import {SourcePreflightPanel} from './SourcePreflightPanel';
import type {InteractivePlanningConstraints,InteractivePlanningReport} from './interactive-planning-data';

const numberLabel=(n:unknown,digits=1)=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(digits):'Unassessed';
function maneuverabilityLabel(value:unknown):string {
  if(value&&typeof value==='object'){
    const p=value as Record<string,unknown>;
    if(Number.isInteger(p.acceptedCount)&&Number.isInteger(p.attemptedCount))return `${p.acceptedCount}/${p.attemptedCount}`;
  }
  return 'Unassessed';
}
export function RegionEditor({name,value,onChange,cursor,onCursor,disabled}:{name:string;value:InteractivePlanningConstraints['target'];onChange:(v:InteractivePlanningConstraints['target'])=>void;cursor:Point3|null;onCursor?:(p:Point3)=>void;disabled:boolean}){
  return <fieldset className="approach-region" disabled={disabled}><legend>{name}</legend>
    <div className="approach-coordinates">{['R','A','S'].map((axis,i)=><label key={axis}>{axis} · mm<input aria-label={`${name} ${axis} coordinate`} type="number" step="0.5" value={value.centerMm[i]} onChange={e=>{const p=[...value.centerMm] as Point3;p[i]=e.target.valueAsNumber;onChange({...value,centerMm:p});}}/></label>)}</div><label className="approach-radius">Radius · mm<input aria-label={`${name} radius`} type="number" min="0.5" max="30" step="0.5" value={value.radiusMm} onChange={e=>onChange({...value,radiusMm:e.target.valueAsNumber})}/></label>
    <div className="surgical-actions"><button disabled={!cursor} onClick={()=>cursor&&onChange({...value,centerMm:[...cursor]})}>Use linked MRI cursor</button><button disabled={!onCursor||value.centerMm.some(n=>!Number.isFinite(n))} onClick={()=>onCursor?.(value.centerMm)}>Show in linked views</button></div>
  </fieldset>;
}
export function InteractivePlanningPanel({c,cursor=null,onCursor,pause}:{c:SurgicalController;cursor?:Point3|null;onCursor?:(p:Point3)=>void;pause:()=>void}){
  const a=c.result?.interactivePlanningAvailability,stored=c.result?.interactivePlanning;
  const qualification=a?.admission;
  const sourcePreflight=currentSourcePreflight(c.result);
  const historicalExposure=(stored?.evidence.historicalExposure??a?.evidence?.historicalExposure) as {targetRemovedMm3?:number;nonTargetRemovedMm3?:number}|undefined;
  const savedCurrent=stored?.current===true&&stored.sourceSceneId===c.view?.scene.sceneId&&stored.startStateId===c.view?.stateId&&stored.prefixActions===c.result?.position;
  return <details className="interactive-planning" open><summary>Plan an approach · post-exposure</summary>
    {sourcePreflight&&<SourcePreflightPanel report={sourcePreflight}/>}
    {qualification&&<details aria-label="Patient task evidence"><summary>Patient and task evidence</summary>
      <p>{qualification.sourceEvidence.patientId} · {qualification.sourceEvidence.sourceRole} research · {qualification.sourceEvidence.studyTimepoint}</p>
      <p className="surgical-note">Planning uses source annotations, model estimates and the completed native state. Image-only planning and clinical outcomes are unsupported.</p>
      <p>Acquired modalities: {qualification.sourceEvidence.images.map(image=>image.modality).join(', ')}.</p>
      <ul>{Object.entries(qualification.sourceEvidence.anatomy).map(([name,row])=><li key={name}>{name}: {row.kind} · {row.reviewStatus}. {row.coverage}</li>)}</ul>
      <p>Unknown evidence: {qualification.sourceEvidence.unknownEvidence.join(', ')}.</p>
      <p className="surgical-note">Tool geometry revision {qualification.models.toolMechanicsRevision} · shared clock revision {qualification.models.physicalClockRevision}. Qualification applies to this declared task and checkpoint.</p>
    </details>}
    {historicalExposure&&<p className="surgical-note" aria-label="Prior source exposure cost">Prior source exposure: {numberLabel(historicalExposure.targetRemovedMm3,0)} mm³ target + {numberLabel(historicalExposure.nonTargetRemovedMm3,0)} mm³ non-target removed before route planning.</p>}
    {!a?.available||!a.defaults?<p role="status">{a?.reason??'Inspect a verified patient checkpoint to check planning availability.'}</p>:<PlanningDraft key={planningBinding(c.result)} c={c} cursor={cursor} onCursor={onCursor} pause={pause} defaults={savedCurrent?stored!.constraints:a.defaults} savedCurrent={savedCurrent?stored:null}/>}
    {stored&&!savedCurrent&&<details className="approach-history" data-planning-status="historical"><summary>Saved approach comparison · historical</summary><p className="surgical-note">{stored.routes.length} routes from a different planning checkpoint. Editing and installation require that checkpoint.</p><ul>{stored.routes.map(route=><li key={route.id}>{route.label} · {numberLabel(route.metrics.pathLengthMm)} mm · {numberLabel(route.metrics.actionCount,0)} actions</li>)}</ul><button disabled={c.pending||!stored.sourceBranchId||!Number.isInteger(stored.sourcePosition)||!c.result?.branches.some(b=>b.id===stored.sourceBranchId&&b.steps>=stored.sourcePosition!)} onClick={()=>{pause();void c.run('selectBranch',{branchId:stored.sourceBranchId,position:stored.sourcePosition});}}>Open planning checkpoint</button><details><summary>Saved evidence and receipts</summary><pre>{JSON.stringify(stored,null,2)}</pre></details></details>}
  </details>;
}
function PlanningDraft({c,cursor,onCursor,pause,defaults,savedCurrent}:{savedCurrent:InteractivePlanningReport|null|undefined;c:SurgicalController;cursor:Point3|null;onCursor?:(p:Point3)=>void;pause:()=>void;defaults:InteractivePlanningConstraints}){
  const [draft,setDraft]=useState(()=>structuredClone(defaults)),[revision,setRevision]=useState(0),[ack,setAck]=useState(false);
  const [submitted,setSubmitted]=useState<{binding:string;revision:number;previousReport:InteractivePlanningReport|null|undefined}|null>(()=>savedCurrent?{binding:planningBinding(c.result),revision:0,previousReport:null}:null);
  const [selectedId,setSelectedId]=useState<string|null>(null),[previewed,setPreviewed]=useState<string|null>(null);
  const latest=useRef({revision,binding:planningBinding(c.result)});latest.current={revision,binding:planningBinding(c.result)};
  const rawReport=currentPlanningReport(c.result,draft,submitted,revision);
  const report=rawReport&&rawReport!==submitted?.previousReport?rawReport:null;
  const selected=selectedPlanningRoute(report,selectedId);
  const contactCount=selected?.requiredTargetRemoval?.sourceCells.length;
  const sourceCellMm3=c.view?.scene.resolutionMm.reduce((volume,spacing)=>volume*spacing,1);
  const contactMm3=contactCount!==undefined&&sourceCellMm3!==undefined?contactCount*sourceCellMm3:null;
  const a=c.result!.interactivePlanningAvailability!,issue=planningDraftIssue(draft);
  const immutable=!!c.result?.readOnly||!!c.result?.playback||!!c.result?.verification?.readOnly;
  const evidence=a.evidence??{},protections=Array.isArray(evidence.protections)?evidence.protections as Array<Record<string,unknown>>:[];
  const unknown=Array.isArray(evidence.unknownEvidence)?evidence.unknownEvidence.map(String):['vessels','functional','shift','skull'];
  const requiredOptions=[...new Set([...unknown,...protections.flatMap(p=>typeof p.id==='string'?[p.id]:[])])];
  const edit=(next:InteractivePlanningConstraints)=>{setDraft(next);setRevision(v=>v+1);setSelectedId(null);setPreviewed(null);c.setPlanningOverlay(null);c.clearPreview();};
  useEffect(()=>{
    if(issue||immutable){c.setPlanningOverlay(null);return;}
    c.setPlanningOverlay({binding:planningBinding(c.result),constraints:draft,routes:report?.routes??[],selectedId:selected?.id??null,limitingStructures:report?.limitingStructures??[]});
    return ()=>c.setPlanningOverlay(null);
  },[draft,issue,immutable,report,selected?.id]);
  useEffect(()=>{if(!c.preview)setPreviewed(null);},[c.preview]);
  function generate(){
    if(issue||!ack)return;pause();c.pausePlayback();c.clearPreview();setPreviewed(null);setSelectedId(null);
    const binding=planningBinding(c.result),submittedRevision=revision,previousReport=c.result?.interactivePlanning;
    setSubmitted(null);
    void c.run('generateApproaches',{constraints:draft,acknowledgeResearchUnknowns:true}).then(ok=>{
      if(ok&&latest.current.binding===binding&&latest.current.revision===submittedRevision)setSubmitted({binding,revision:submittedRevision,previousReport});
    });
  }
  function choose(id:string){pause();c.clearPreview();setPreviewed(null);setSelectedId(id);const route=report?.routes.find(r=>r.id===id);if(route)onCursor?.(route.entryPose.position);}
  function preview(){if(!selected||!report)return;pause();void c.run('previewApproach',{candidateId:selected.id,expectedPlanSeal:report.reportSeal}).then(ok=>{if(ok)setPreviewed(selected.id);});}
  function apply(){if(!selected||!report||previewed!==selected.id||!c.preview||c.approachPreview?.candidateId!==selected.id||c.approachPreview.reportSeal!==report.reportSeal)return;pause();void c.run('applyApproach',{candidateId:selected.id,expectedPlanSeal:report.reportSeal});}
  return <div className="approach-draft" data-planning-status={report?'current':'draft'} data-planning-source={report?.sourceSceneId} data-planning-state={report?.startStateId} data-planning-seal={report?.reportSeal}>
    <details className="approach-constraints" open={!report}><summary>Target, entry and CUSA constraints</summary><div className="approach-constraints-body">
    <p className="surgical-note">Choose a retained target and an allowable entry patch in the existing exposure. Routes stop before active CUSA removal.</p>
    <RegionEditor name="Target region" value={draft.target} onChange={target=>edit({...draft,target})} cursor={cursor} onCursor={onCursor} disabled={c.pending||immutable}/>
    <RegionEditor name="Allowable entry region" value={draft.entry} onChange={entry=>edit({...draft,entry})} cursor={cursor} onCursor={onCursor} disabled={c.pending||immutable}/>
    <fieldset className="approach-tool" disabled={c.pending||immutable}><legend>CUSA and protection</legend>
      <label>Tip radius · mm<input aria-label="Planning CUSA tip radius" type="number" min={a.admission?.task.toolRadiusMm??.1} max={a.admission?.task.toolRadiusMm??6} disabled={!!a.admission} step="0.1" value={draft.tool.radiusMm} onChange={e=>edit({...draft,tool:{instrument:'cusa',radiusMm:e.target.valueAsNumber}})}/></label>
      {a.admission&&<p className="surgical-note">This task binds the fixed catalog CUSA geometry.</p>}
      <label>Mapped non-target clearance · mm<input aria-label="Mapped non-target clearance mm" type="number" min="0" max="2" step="0.1" value={draft.nonTargetClearanceMm} onChange={e=>edit({...draft,nonTargetClearanceMm:e.target.valueAsNumber})}/></label>
      <p className="surgical-note">Research buffer around retained source non-target geometry. This does not establish a clinical safe margin.</p>
      <p className="surgical-note">Stock shaft: 0.5 mm radius · 40 mm working length. Source protection guards remain active.</p>
      {requiredOptions.length>0&&<><span>Require evidence before generating</span><div className="approach-evidence">{requiredOptions.map(id=><label key={id}><input type="checkbox" aria-label={`Require ${id} evidence`} checked={draft.requiredEvidence.includes(id)} onChange={e=>edit({...draft,requiredEvidence:e.target.checked?[...draft.requiredEvidence,id]:draft.requiredEvidence.filter(v=>v!==id)})}/>{protections.find(p=>p.id===id)?.name as string??id.replaceAll('_',' ')}</label>)}</div></>}
      <label>Approaches requested<select aria-label="Number of approaches" value={draft.candidateCount} onChange={e=>edit({...draft,candidateCount:+e.target.value})}>{[2,3,4,5,6].map(n=><option key={n}>{n}</option>)}</select></label>
    </fieldset>

    </div></details>
    <p className="surgical-warning">Research geometry only. {unknown.length?`Unassessed evidence: ${unknown.join(', ')}. `:''}Clearance, local maneuverability and neurological harm are unvalidated.</p>
    <label><input aria-label="Acknowledge interactive planning unknowns" type="checkbox" checked={ack} disabled={c.pending||immutable} onChange={e=>{setAck(e.target.checked);if(!e.target.checked){setPreviewed(null);c.clearPreview();}}}/>I accept these research limits for this planning draft.</label>
    {issue&&<p role="alert" className="surgical-error">{issue}</p>}
    <button className="primary-button" disabled={c.pending||immutable||!!issue||!ack||!!c.preview} onClick={generate}>Generate approaches</button>

    {!report&&submitted===null&&c.result?.interactivePlanning&&<p className="surgical-note">The draft changed. Generate again to compare and install current approaches.</p>}
    {report&&<>
      {report.decision&&<p role="status" aria-label="Planning decision">{report.decision.operation==='STOP'?'STOP · no executable approach meets this draft.':'Compare approaches · access and withdrawal passed the declared planning model.'}</p>}
      <p role="status">{report.routes.length} / {draft.candidateCount} checked candidates · {report.routes.length<2?'No distinct approach comparison available.':report.routes.length<draft.candidateCount?'Only these candidates met the current geometry constraints.':'Select a route to inspect it in 3D and linked MRI.'}</p>
      {report.routes.length===0&&!!report.missingRequiredEvidence?.length&&<p role="alert" className="surgical-error">Approaches refused: required evidence is unavailable for {report.missingRequiredEvidence.join(', ')}.</p>}
      {report.routes.length>0&&<div className="approach-table"><table aria-label="Interactive approach comparison"><caption>Entry angle relative to the first route</caption><thead><tr><th>Route</th><th>Travel / actions</th><th>Angle</th><th>Probes</th></tr></thead><tbody>{report.routes.map((route,i)=><tr key={route.id} data-planning-candidate={route.id} aria-selected={selected?.id===route.id}><th><button disabled={c.pending} aria-label={route.label} aria-pressed={selected?.id===route.id} onClick={()=>choose(route.id)}><span style={{color:PLANNING_ROUTE_COLORS[i%PLANNING_ROUTE_COLORS.length]}}>● </span>{i+1}</button></th><td>{numberLabel(route.metrics.pathLengthMm)} mm / {numberLabel(route.metrics.actionCount,0)}</td><td>{numberLabel(route.separationComparisons?.[0]?.angleDeg??0)}°</td><td>{maneuverabilityLabel(route.metrics.localManeuverability)}</td></tr>)}</tbody></table></div>}
      {selected&&<section className="approach-selection" aria-label="Selected approach"><strong>{selected.label}</strong><p className="surgical-note">Path {numberLabel(selected.metrics.pathLengthMm)} mm · entry angle {numberLabel(selected.separationComparisons?.[0]?.angleDeg??0)}° · {numberLabel(selected.metrics.actionCount,0)} actions.</p>{contactCount!==undefined&&<p className="surgical-note">Target boundary contact assumption: {contactCount} source cells ({numberLabel(contactMm3)} mm³). Tissue remains retained.</p>}<p className="surgical-note">Inactive approach · additional target removal {numberLabel(selected.metrics.targetRemovedMm3)} mm³ · non-target traversal {numberLabel(selected.metrics.nonTargetTraversalMm3)} mm³. Four local geometric probes do not characterize all maneuvers. MRI guides show current slice intersections.</p><div className="surgical-actions"><button disabled={c.pending} onClick={()=>onCursor?.(selected.targetPose.position)}>Show target</button><button disabled={c.pending||!ack} onClick={preview}>Preview checked approach</button><button className="primary-button" disabled={c.pending||!ack||previewed!==selected.id||!c.preview||c.approachPreview?.candidateId!==selected.id||c.approachPreview.reportSeal!==report.reportSeal} onClick={apply}>Install checked branch</button></div>{previewed===selected.id&&c.preview&&<p role="status">Preview shown · anatomy is uncommitted. Install creates the checked branch.</p>}</section>}
      {report.limitingStructures?.length?<p className="surgical-note">Limiting structures: {[...new Set(report.limitingStructures.map(s=>s.name))].join(', ')}. Amber markers identify limiting source cells.</p>:<p className="surgical-note">No located limiting structure was supplied. Missing anatomy remains unknown.</p>}
      {report.rejections.length>0&&<details><summary>Refused route samples · {report.rejections.length}</summary><ul>{report.rejections.slice(0,12).map((rejection,i)=><li key={i}>{String(rejection.reason??rejection.message??JSON.stringify(rejection)).replaceAll('_',' ')}</li>)}</ul></details>}
      <details><summary>Evidence and technical receipts</summary><p>{a.scope}</p><p>Source scene: <code>{report.sourceSceneId}</code></p><p>Checkpoint: <code>{report.startStateId}</code></p><p>Plan seal: <code>{report.reportSeal}</code></p><pre>{JSON.stringify({publicObservation:report.publicObservation,publicPlanSeal:report.publicPlanSeal,decision:report.decision,evidence:report.evidence,targetSelection:report.targetSelection,constraints:report.constraints,routes:report.routes.map(({id,metrics,scope,separationComparisons})=>({id,metrics,scope,separationComparisons})),previewReceipt:c.approachPreview?.receipt},null,2)}</pre></details>
    </>}
  </div>;
}
