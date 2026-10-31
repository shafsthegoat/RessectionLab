import {useEffect,useRef,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {planningBinding,planningDraftIssue} from './interactive-planning-data';
import {RegionEditor} from './InteractivePlanningPanel';
import type {Point3} from './viewer/coordinates';
import {strategyControlKey} from './two-tool-data';
import type {TwoToolControls,TwoToolComparison} from './two-tool-data';

const n=(value:number|undefined)=>typeof value==='number'?value.toFixed(2):'Unassessed';
const labels:Record<string,string>={inspect:'Inspection',access:'CUSA access','position-secondary':'Suction positioned',operate:'After finite contact',reassess:'Reassessment',withdraw:'Complete withdrawal'};
export function TwoToolStrategyPanel({c,pause,cursor=null,onCursor}:{c:SurgicalController;pause:()=>void;cursor?:Point3|null;onCursor?:(p:Point3)=>void}){
  const a=c.result?.strategyPlanningAvailability,report=c.result?.strategyPlanning;
  const current=report?.current===true&&report.startStateId===c.result?.view.stateId&&report.sourceSceneId===c.result?.view.scene.sceneId;
  return <details className="two-tool-planning"><summary>Compare two-tool strategies</summary>
    <p className="surgical-note">Inspect → CUSA and suction positioning → finite contact → reassess → withdraw or STOP. Source annotations and uncalibrated research mechanics support this comparison.</p>
    {a?.available&&a.defaults?<StrategyDraft key={planningBinding(c.result)} c={c} pause={pause} cursor={cursor} onCursor={onCursor} defaults={current?report!.publicPlan.constraints:a.defaults} saved={current?report:null}/>:<p role="status">{a?.reason??'Select a qualified native planning checkpoint.'}</p>}
    {report&&!current&&<details open><summary>Saved two-tool comparison</summary><p>{report.publicPlan.decision.operation} · {report.publicPlan.alternatives.length} recorded alternatives.</p>
      <button disabled={c.pending||!report.sourceBranchId} onClick={()=>{pause();void c.run('selectBranch',{branchId:report.sourceBranchId,position:report.sourcePosition});}}>Open strategy checkpoint</button>
      {report.applications?.map((app,i)=>{const phase=app.receipt.phases.find(p=>p.id==='reassess');return phase&&<button key={i} disabled={c.pending} onClick={()=>{pause();void c.run('selectBranch',{branchId:app.branchId,position:phase.position});}}>Inspect {app.candidateId} reassessment</button>;})}
    </details>}
  </details>;
}
function StrategyDraft({c,pause,cursor,onCursor,defaults,saved}:{c:SurgicalController;pause:()=>void;cursor:Point3|null;onCursor?:(p:Point3)=>void;defaults:TwoToolControls;saved:TwoToolComparison|null|undefined}){
  const [draft,setDraft]=useState(()=>structuredClone(defaults)),[revision,setRevision]=useState(0),[submitted,setSubmitted]=useState<number|null>(saved?0:null);
  const latest=useRef(revision);latest.current=revision;
  const [ack,setAck]=useState(false),[selected,setSelected]=useState<string|null>(null),[phase,setPhase]=useState('operate');
  const raw=c.result?.strategyPlanning;
  const report=raw?.current&&submitted===revision&&strategyControlKey(raw.publicPlan.constraints)===strategyControlKey(draft)?raw:null;
  const plan=report?.publicPlan,id=selected??plan?.decision.selectedId,row=plan?.alternatives.find(r=>r.id===id);
  const locked=c.pending||!!c.result?.readOnly||!!c.result?.playback||!!c.result?.verification?.readOnly;
  const preview=c.preview&&c.approachPreview?.kind==='strategy'&&c.approachPreview.candidateId===id&&c.approachPreview.reportSeal===report?.reportSeal;
  // One primary approach can still supply two distinct timed strategies.
  const geometryIssue=planningDraftIssue(draft.geometry.candidateCount===1?{...draft.geometry,candidateCount:2}:draft.geometry);
  useEffect(()=>{
    if(geometryIssue||c.result?.readOnly||c.result?.playback||c.result?.verification?.readOnly)return;
    const route=plan?.chosenPair?.route;
    c.setPlanningOverlay({binding:planningBinding(c.result),constraints:draft.geometry,routes:route?[route]:[],selectedId:route?.id??null,limitingStructures:plan?.approaches.limitingStructures??[]});
    return ()=>c.setPlanningOverlay(null);
  },[draft.geometry,report,selected,geometryIssue]);
  function edit(next:TwoToolControls){setDraft(next);setRevision(v=>v+1);setSelected(null);c.clearPreview();}
  const issue=!!geometryIssue||!Number.isFinite(draft.shortSeconds)||!Number.isFinite(draft.longSeconds)||draft.shortSeconds<.1||draft.longSeconds>2||draft.shortSeconds>=draft.longSeconds||!Number.isFinite(draft.minimumTargetMm3)||draft.minimumTargetMm3<0||draft.minimumTargetMm3>8;
  async function generate(){if(issue||!ack)return;pause();c.clearPreview();const rev=revision;setSubmitted(null);const ok=await c.run('generateStrategies',{constraints:draft,acknowledgeResearchUnknowns:true});if(ok&&latest.current===rev){setSubmitted(rev);setSelected(null);}}
  return <section aria-label="Two-tool strategy draft" data-strategy-seal={report?.reportSeal}>
    <details><summary>Target and shared entry patch</summary><RegionEditor name="Strategy target" value={draft.geometry.target} onChange={target=>edit({...draft,geometry:{...draft.geometry,target}})} cursor={cursor} onCursor={onCursor} disabled={locked}/><RegionEditor name="Shared entry patch" value={draft.geometry.entry} onChange={entry=>edit({...draft,geometry:{...draft.geometry,entry}})} cursor={cursor} onCursor={onCursor} disabled={locked}/><p className="surgical-note">Both instrument entry points must remain in this requested patch inside the qualified exposure.</p></details>
    <fieldset disabled={locked}><legend>Bounded comparison</legend>
      <label>Short contact · seconds<input aria-label="Short contact seconds" type="number" min="0.1" max="2" step="0.05" value={draft.shortSeconds} onChange={e=>edit({...draft,shortSeconds:e.target.valueAsNumber})}/></label>
      <label>Longer contact · seconds<input aria-label="Longer contact seconds" type="number" min="0.1" max="2" step="0.05" value={draft.longSeconds} onChange={e=>edit({...draft,longSeconds:e.target.valueAsNumber})}/></label>
      <label>Comparison target progress · mm³<input aria-label="Strategy target progress mm3" type="number" min="0" max="8" step="0.5" value={draft.minimumTargetMm3} onChange={e=>edit({...draft,minimumTargetMm3:e.target.valueAsNumber})}/></label>
      <p className="surgical-note">This is a research comparison threshold. The source tumor annotation does not define a clinically approved removal boundary. Durations request CUSA activation intervals on the shared clock. Tool radii remain 1.5 mm; no new non-target removal is allowed.</p>
      <label><input aria-label="Require functional evidence for strategies" type="checkbox" checked={draft.geometry.requiredEvidence.includes('functional')} onChange={e=>edit({...draft,geometry:{...draft.geometry,requiredEvidence:e.target.checked?[...draft.geometry.requiredEvidence,'functional']:draft.geometry.requiredEvidence.filter(x=>x!=='functional')}})}/>Require functional evidence</label>
      <label><input aria-label="Acknowledge two-tool research scope" type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/>I accept the limited manipulation task and missing anatomical evidence for this research draft.</label>
    </fieldset>
    {issue&&<p role="alert">{geometryIssue??'Use two increasing durations from 0.1 to 2 seconds and a progress threshold from 0 to 8 mm³.'}</p>}
    <button className="primary-button" disabled={locked||issue||!ack||!!c.preview} onClick={()=>void generate()}>Compare executable strategies</button>
    {plan&&<><p role="status" aria-label="Two-tool decision">{plan.decision.operation} · {plan.decision.selectedId==='stop'?'STOP is selected.':`Modeled preference: ${plan.alternatives.find(r=>r.id===plan.decision.selectedId)?.label}.`}</p>
      <p className="surgical-note">{plan.decision.operation==='COMPARE'?'The preference meets the declared progress threshold, then minimizes modeled time.':plan.decision.reason==='PROGRESS_ALREADY_SATISFIED'?'STOP already meets this research threshold without new actions.':plan.decision.reason==='COMPUTATION_BUDGET'?'The computation budget ended this comparison; feasibility remains unresolved.':plan.decision.reason==='REQUIRED_EVIDENCE_UNAVAILABLE'?'Required evidence is missing; STOP preserves the current state.':'The bounded search found no complete sequence meeting this draft. STOP preserves the current state.'} Visibility and clinical outcomes are unassessed.</p>
      {!!plan.approaches.missingRequiredEvidence?.length&&<p role="alert">Required evidence unavailable: {plan.approaches.missingRequiredEvidence.join(', ')}. No tissue-effect plan is offered.</p>}
      <table aria-label="Two-tool strategy comparison"><thead><tr><th>Strategy</th><th>Target / pending mm³</th><th>Time s</th><th>Follow-up pose</th></tr></thead><tbody>{plan.alternatives.map(item=><tr key={item.id} data-strategy-candidate={item.id} aria-selected={row?.id===item.id}><th><button disabled={locked} aria-label={`Select ${item.label}`} onClick={()=>{c.clearPreview();setSelected(item.id);}}>{item.label}</button></th><td>{n(item.summary.additionalTargetMm3)} / {n(item.summary.cusaPendingMm3)}</td><td>{n(item.summary.elapsedSeconds)}</td><td>{item.reassessment?.probeStatus??'STOP'}</td></tr>)}</tbody></table>
      {row&&<section aria-label="Selected two-tool strategy"><strong>{row.label}</strong><p>Additional non-target removal {n(row.summary.additionalNonTargetMm3)} mm³. Fluid removed {n(row.summary.fluidRemovedMm3)} mm³. Residual source target {n(row.summary.residualTargetMm3)} mm³.</p>
        <p className="surgical-note">{row.id==='stop'?'STOP executes no new instrument actions and preserves existing removal, pending work and tool states.':'Access, persistent paired-tool geometry and complete withdrawal were executed in the declared model. CUSA may stop spatially while its activation interval continues; pending capacity persists.'}</p>
        {row.id!=='stop'&&<label>Preview checkpoint<select aria-label="Strategy preview checkpoint" value={phase} disabled={locked||row.id==='stop'} onChange={e=>{setPhase(e.target.value);c.clearPreview();}}>{Object.entries(labels).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>}
        <div className="surgical-actions"><button disabled={locked||!ack} onClick={()=>{pause();void c.run('previewStrategy',{candidateId:row.id,expectedPlanSeal:report!.reportSeal,phase});}}>Preview strategy checkpoint</button><button disabled={locked||!ack||!preview} onClick={()=>{pause();void c.run('applyStrategy',{candidateId:row.id,expectedPlanSeal:report!.reportSeal});}}>Install strategy branch</button></div>
        {preview&&<p role="status">Uncommitted checkpoint shown. Installation creates the complete checked sequence through withdrawal.</p>}
      </section>}
      <details><summary>Strategy evidence and receipts</summary><p>{plan.scope}</p><pre>{JSON.stringify({decision:plan.decision,publicInputHash:plan.publicObservation.inputHash,publicPlanSeal:plan.publicPlanSeal,rejections:plan.rejections,alternatives:plan.alternatives,previewReceipt:c.approachPreview?.receipt},null,2)}</pre></details>
    </>}
  </section>;
}
