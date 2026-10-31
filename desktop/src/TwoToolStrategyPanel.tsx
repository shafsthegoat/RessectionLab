import {useEffect,useRef,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {planningBinding,planningDraftIssue} from './interactive-planning-data';
import {RegionEditor} from './InteractivePlanningPanel';
import type {Point3} from './viewer/coordinates';
import {strategyControlKey,strategyControls} from './two-tool-data';
import type {TwoToolControls,TwoToolComparison,ContextOption,AnatomicalEncounter} from './two-tool-data';

const n=(value:number|undefined)=>typeof value==='number'?value.toFixed(2):'Unassessed';
const labels:Record<string,string>={inspect:'Inspection',access:'CUSA access','position-secondary':'Suction positioned',operate:'After finite contact',reassess:'Reassessment',withdraw:'Complete withdrawal'};
export function TwoToolStrategyPanel({c,pause,cursor=null,onCursor}:{c:SurgicalController;pause:()=>void;cursor?:Point3|null;onCursor?:(p:Point3)=>void}){
  const a=c.result?.strategyPlanningAvailability,report=c.result?.strategyPlanning;
  const current=report?.current===true&&report.startStateId===c.result?.view.stateId&&report.sourceSceneId===c.result?.view.scene.sceneId;
  return <details className="two-tool-planning"><summary>Compare two-tool strategies</summary>
    <p className="surgical-note">Inspect → CUSA and suction positioning → finite contact → reassess → withdraw or STOP. Source annotations and uncalibrated research mechanics support this comparison.</p>
    {a?.available&&a.defaults?<StrategyDraft key={planningBinding(c.result)} c={c} pause={pause} cursor={cursor} onCursor={onCursor} defaults={current?strategyControls(report!):a.defaults} saved={current?report:null} contexts={a.contextOptions??[]}/>:<p role="status">{a?.reason??'Select a qualified native planning checkpoint.'}</p>}
    {report&&!current&&<details open><summary>Saved two-tool comparison</summary><p>{report.publicPlan.decision.operation} · {report.publicPlan.alternatives.length} recorded alternatives.</p>
      <button disabled={c.pending||!report.sourceBranchId} onClick={()=>{pause();void c.run('selectBranch',{branchId:report.sourceBranchId,position:report.sourcePosition});}}>Open strategy checkpoint</button>
      {report.applications?.map((app,i)=>{const phase=app.receipt.phases.find(p=>p.id==='reassess');return phase&&<button key={i} disabled={c.pending} onClick={()=>{pause();void c.run('selectBranch',{branchId:app.branchId,position:phase.position});}}>Inspect {app.candidateId} reassessment</button>;})}
    </details>}
  </details>;
}
function StrategyDraft({c,pause,cursor,onCursor,defaults,saved,contexts}:{c:SurgicalController;pause:()=>void;cursor:Point3|null;onCursor?:(p:Point3)=>void;defaults:TwoToolControls;saved:TwoToolComparison|null|undefined;contexts:ContextOption[]}){
  const [draft,setDraft]=useState(()=>structuredClone(defaults)),[revision,setRevision]=useState(0),[submitted,setSubmitted]=useState<number|null>(saved?0:null);
  const latest=useRef(revision);latest.current=revision;
  const [ack,setAck]=useState(false),[selected,setSelected]=useState<string|null>(null),[phase,setPhase]=useState('operate');
  const raw=c.result?.strategyPlanning;
  const report=raw?.current&&submitted===revision&&strategyControlKey(strategyControls(raw))===strategyControlKey(draft)?raw:null;
  const plan=report?.publicPlan,id=selected??plan?.decision.selectedId,row=plan?.alternatives.find(r=>r.id===id);
  const contextRefusals=(plan?.rejections??[]).filter(r=>typeof r.reason==='string'&&r.reason.startsWith('ANATOMICAL_CONTEXT_LIMIT')&&r.anatomicalEncounter).map(r=>({route:String(r.routeId),receipt:r.anatomicalEncounter as AnatomicalEncounter}));
  const contextPoint=row?.anatomicalEncounter?.nearPositionsMm?.[0]??contextRefusals.flatMap(r=>r.receipt.nearPositionsMm??[])[0];
  const locked=c.pending||!!c.result?.readOnly||!!c.result?.playback||!!c.result?.verification?.readOnly;
  const preview=c.preview&&c.approachPreview?.kind==='strategy'&&c.approachPreview.candidateId===id&&c.approachPreview.reportSeal===report?.reportSeal;
  // One primary approach can still supply two distinct timed strategies.
  const geometryIssue=planningDraftIssue(draft.geometry.candidateCount===1?{...draft.geometry,candidateCount:2}:draft.geometry);
  useEffect(()=>{
    if(geometryIssue||c.result?.readOnly||c.result?.playback||c.result?.verification?.readOnly)return;
    const route=plan?.chosenPair?.route;
    const points=[...(row?.anatomicalEncounter?.nearPositionsMm??[]),...contextRefusals.flatMap(r=>r.receipt.nearPositionsMm??[])];
    const unique=points.filter((p,i)=>points.findIndex(q=>JSON.stringify(q)===JSON.stringify(p))===i).slice(0,12);
    c.setPlanningOverlay({binding:planningBinding(c.result),constraints:draft.geometry,routes:route?[route]:[],selectedId:route?.id??null,limitingStructures:[...(plan?.approaches.limitingStructures??[]),...unique.map(positionMm=>({name:'Supplied context within declared tool margin',positionMm,reason:'Source-cell centre; geometric near-encounter, not injury'}))]});
    return ()=>c.setPlanningOverlay(null);
  },[draft.geometry,report,selected,geometryIssue]);
  function edit(next:TwoToolControls){setDraft(next);setRevision(v=>v+1);setSelected(null);c.clearPreview();}
  const issue=!!geometryIssue||!Number.isFinite(draft.shortSeconds)||!Number.isFinite(draft.longSeconds)||draft.shortSeconds<.1||draft.longSeconds>2||draft.shortSeconds>=draft.longSeconds||!Number.isFinite(draft.minimumTargetMm3)||draft.minimumTargetMm3<0||draft.minimumTargetMm3>8;
  const contextIssue=!!draft.encounter&&(!contexts.some(x=>x.layerId===draft.encounter!.layerId)||!Number.isFinite(draft.encounter.marginMm)||draft.encounter.marginMm<0||draft.encounter.marginMm>5||!Number.isFinite(draft.encounter.maxNearMm3)||draft.encounter.maxNearMm3<0||draft.encounter.maxNearMm3>1000);
  const context=contexts.find(x=>x.layerId===draft.encounter?.layerId);
  async function generate(){if(issue||contextIssue||!ack)return;pause();c.clearPreview();const rev=revision;setSubmitted(null);const ok=await c.run('generateStrategies',{constraints:draft,acknowledgeResearchUnknowns:true});if(ok&&latest.current===rev){setSubmitted(rev);setSelected(null);}}
  return <section aria-label="Two-tool strategy draft" data-strategy-seal={report?.reportSeal}>
    <details><summary>Target and shared entry patch</summary><RegionEditor name="Strategy target" value={draft.geometry.target} onChange={target=>edit({...draft,geometry:{...draft.geometry,target}})} cursor={cursor} onCursor={onCursor} disabled={locked}/><RegionEditor name="Shared entry patch" value={draft.geometry.entry} onChange={entry=>edit({...draft,geometry:{...draft.geometry,entry}})} cursor={cursor} onCursor={onCursor} disabled={locked}/><p className="surgical-note">Both instrument entry points must remain in this requested patch inside the qualified exposure.</p></details>
    <fieldset disabled={locked}><legend>Bounded comparison</legend>
      <label>Short contact · seconds<input aria-label="Short contact seconds" type="number" min="0.1" max="2" step="0.05" value={draft.shortSeconds} onChange={e=>edit({...draft,shortSeconds:e.target.valueAsNumber})}/></label>
      <label>Longer contact · seconds<input aria-label="Longer contact seconds" type="number" min="0.1" max="2" step="0.05" value={draft.longSeconds} onChange={e=>edit({...draft,longSeconds:e.target.valueAsNumber})}/></label>
      <label>Comparison target progress · mm³<input aria-label="Strategy target progress mm3" type="number" min="0" max="8" step="0.5" value={draft.minimumTargetMm3} onChange={e=>edit({...draft,minimumTargetMm3:e.target.valueAsNumber})}/></label>
      <p className="surgical-note">This is a research comparison threshold. The source tumor annotation does not define a clinically approved removal boundary. Durations request CUSA activation intervals on the shared clock. Tool radii remain 1.5 mm; no new non-target removal is allowed.</p>
      <label>Supplied anatomical context<select aria-label="Anatomical context layer" value={draft.encounter?.layerId??''} onChange={e=>{if(e.target.value)edit({...draft,encounter:{layerId:e.target.value,marginMm:draft.encounter?.marginMm??0,maxNearMm3:draft.encounter?.maxNearMm3??0}});else{const {encounter,...rest}=draft;edit(rest);}}}><option value="">No additional context constraint</option>{contexts.map(x=><option key={x.layerId} value={x.layerId}>{x.description} · label {x.labelValue}</option>)}</select></label>
      {draft.encounter&&<><p className="surgical-note">{context?.reviewStatus}. {context?.coverage}. This counts unique retained source context cells near the complete shafts and working parts across access, operation and withdrawal. Source context can include tumor; it does not establish healthy tissue or neurological function.</p>
        <label>Context research margin · mm<input aria-label="Context margin mm" type="number" min="0" max="5" step="0.5" value={draft.encounter.marginMm} onChange={e=>edit({...draft,encounter:{...draft.encounter!,marginMm:e.target.valueAsNumber}})}/></label>
        <label>Allowed near-context volume · mm³<input aria-label="Context near volume limit mm3" type="number" min="0" max="1000" step="1" value={draft.encounter.maxNearMm3} onChange={e=>edit({...draft,encounter:{...draft.encounter!,maxNearMm3:e.target.valueAsNumber}})}/></label>
        <p className="surgical-note">These are user-declared geometric research constraints, without a calibrated clinical tolerance. Zero permits no retained source context cell in the expanded tool envelope. Historical removal stays separate; outside the source image grid remains unassessed.</p></>}
      <label><input aria-label="Require functional evidence for strategies" type="checkbox" checked={draft.geometry.requiredEvidence.includes('functional')} onChange={e=>edit({...draft,geometry:{...draft.geometry,requiredEvidence:e.target.checked?[...draft.geometry.requiredEvidence,'functional']:draft.geometry.requiredEvidence.filter(x=>x!=='functional')}})}/>Require functional evidence</label>
      <label><input aria-label="Acknowledge two-tool research scope" type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/>I accept the limited manipulation task and missing anatomical evidence for this research draft.</label>
    </fieldset>
    {issue&&<p role="alert">{geometryIssue??'Use two increasing durations from 0.1 to 2 seconds and a progress threshold from 0 to 8 mm³.'}</p>}
    {contextIssue&&<p role="alert">Select available source context, a margin from 0 to 5 mm and a near-volume limit from 0 to 1000 mm³.</p>}
    <button className="primary-button" disabled={locked||issue||contextIssue||!ack||!!c.preview} onClick={()=>void generate()}>Compare executable strategies</button>
    {plan&&<><p role="status" aria-label="Two-tool decision">{plan.decision.operation} · {plan.decision.selectedId==='stop'?'STOP is selected.':`Modeled preference: ${plan.alternatives.find(r=>r.id===plan.decision.selectedId)?.label}.`}</p>
      <p className="surgical-note">{plan.decision.operation==='COMPARE'?'The preference meets the declared progress and optional context limits, then minimizes modeled time.':plan.decision.reason==='PROGRESS_ALREADY_SATISFIED'?'STOP already meets this research threshold without new actions.':plan.decision.reason==='COMPUTATION_BUDGET'?'The computation budget ended this comparison; feasibility remains unresolved.':plan.decision.reason==='REQUIRED_EVIDENCE_UNAVAILABLE'?'Required evidence is missing; STOP preserves the current state.':plan.decision.reason==='ANATOMICAL_CONTEXT_LIMIT'?'The declared source-context limit excludes the tested access or complete sequences. STOP preserves the residual source target and current burdens.':'The bounded search found no complete sequence meeting this draft. STOP preserves the current state.'} Visibility and clinical outcomes are unassessed.</p>
      {!!plan.approaches.missingRequiredEvidence?.length&&<p role="alert">Required evidence unavailable: {plan.approaches.missingRequiredEvidence.join(', ')}. No tissue-effect plan is offered.</p>}
      {!!contextRefusals.length&&<ul aria-label="Source context refusals">{contextRefusals.map((r,i)=><li key={i}>{r.route}: {n(r.receipt.nearMm3)} mm³ of retained supplied context within {n(r.receipt.marginMm)} mm of the tools exceeds this draft's {n(r.receipt.maxNearMm3)} mm³ limit. Source-cell centres are marked in the linked views.</li>)}</ul>}
      {draft.encounter&&<button disabled={locked||!contextPoint||!onCursor} onClick={()=>contextPoint&&onCursor?.([...contextPoint])}>Show supplied context in linked views</button>}
      <table aria-label="Two-tool strategy comparison"><thead><tr><th>Strategy</th><th>Target / pending mm³</th><th>Time s</th><th>Follow-up pose</th>{draft.encounter&&<th>Context near mm³</th>}</tr></thead><tbody>{plan.alternatives.map(item=><tr key={item.id} data-strategy-candidate={item.id} aria-selected={row?.id===item.id}><th><button disabled={locked} aria-label={`Select ${item.label}`} onClick={()=>{c.clearPreview();setSelected(item.id);}}>{item.label}</button></th><td>{n(item.summary.additionalTargetMm3)} / {n(item.summary.cusaPendingMm3)}</td><td>{n(item.summary.elapsedSeconds)}</td><td>{item.reassessment?.probeStatus??'STOP'}</td>{draft.encounter&&<td>{n(item.anatomicalEncounter?.nearMm3)} · {item.anatomicalEncounter?.withinLimit?'within limit':'exceeds limit'}</td>}</tr>)}</tbody></table>
      {row&&<section aria-label="Selected two-tool strategy"><strong>{row.label}</strong><p>Additional non-target removal {n(row.summary.additionalNonTargetMm3)} mm³. Fluid removed {n(row.summary.fluidRemovedMm3)} mm³. Residual source target {n(row.summary.residualTargetMm3)} mm³.</p>
        <p className="surgical-note">{row.id==='stop'?'STOP executes no new instrument actions and preserves existing removal, pending work and tool states.':'Access, persistent paired-tool geometry and complete withdrawal were executed in the declared model. CUSA may stop spatially while its activation interval continues; pending capacity persists.'}</p>
        {row.anatomicalEncounter&&<p aria-label="Anatomical encounter consequence">Historical source-context removal {n(row.anatomicalEncounter.baselineRemovedContextMm3)} mm³; new removal {n(row.anatomicalEncounter.newRemovedContextMm3)} mm³. Whole-sequence near-context union {n(row.anatomicalEncounter.nearMm3)} mm³ at {n(row.anatomicalEncounter.marginMm)} mm margin, limit {n(row.anatomicalEncounter.maxNearMm3)} mm³. Shaft {n(row.anatomicalEncounter.shaftNearMm3)} / working parts {n(row.anatomicalEncounter.workingNearMm3)} mm³; overlapping cells count once in the union. {row.feasible?'Within the declared limit.':'This sequence exceeds the declared limit and cannot be installed.'}</p>}
        {row.id!=='stop'&&<label>Preview checkpoint<select aria-label="Strategy preview checkpoint" value={phase} disabled={locked||row.id==='stop'} onChange={e=>{setPhase(e.target.value);c.clearPreview();}}>{Object.entries(labels).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>}
        <div className="surgical-actions"><button disabled={locked||!ack||!row.feasible} onClick={()=>{pause();void c.run('previewStrategy',{candidateId:row.id,expectedPlanSeal:report!.reportSeal,phase});}}>Preview strategy checkpoint</button><button disabled={locked||!ack||!preview||!row.feasible} onClick={()=>{pause();void c.run('applyStrategy',{candidateId:row.id,expectedPlanSeal:report!.reportSeal});}}>Install strategy branch</button></div>
        {preview&&<p role="status">Uncommitted checkpoint shown. Installation creates the complete checked sequence through withdrawal.</p>}
      </section>}
      <details><summary>Strategy evidence and receipts</summary><p>{plan.scope}</p><pre>{JSON.stringify({decision:plan.decision,contextBinding:report?.contextBinding,publicInputHash:plan.publicObservation.inputHash,publicPlanSeal:plan.publicPlanSeal,rejections:plan.rejections,alternatives:plan.alternatives,previewReceipt:c.approachPreview?.receipt},null,2)}</pre></details>
    </>}
  </section>;
}
