import {CranialStrategiesPanel} from './CranialStrategiesPanel';
import {useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {Point3} from './viewer/coordinates';
import {PreparationPlanningPanel} from './PreparationPlanningPanel';
import {TwoToolStrategyPanel} from './TwoToolStrategyPanel';
import {InteractivePlanningPanel} from './InteractivePlanningPanel';
import {preferredWorkflowPlanner,workflowPlannerOptions,workflowReportCurrent} from './surgical-workflow-data';
import type {SurgicalWorkflowPlanner,SurgicalWorkflowStage} from './surgical-workflow-data';
import {StagedStrategiesPanel} from './StagedStrategiesPanel';
import {stagedStrategyDisplay} from './staged-strategy-data';

const stages:Array<{id:SurgicalWorkflowStage;label:string}>=[{id:'case',label:'Case'},{id:'plan',label:'Plan'},{id:'compare',label:'Compare'},{id:'rehearse',label:'Rehearse/edit'}];
const labels:Record<SurgicalWorkflowPlanner,string>={cranial:'Opening, preparation, target work and exit',staged:'Bounded staged source sequence · partial anatomy',preparation:'Preparation + complete two-tool sequence',strategy:'Two-tool sequence in existing exposure',approach:'Inactive approach only · no treatment'};
/** One mounted draft owns generation, editing and preview authority across stages. */
export function SurgicalWorkflow({c,stage,onStage,pause,cursor,onCursor,artifact,playRecorded=()=>{}}:{c:SurgicalController;stage:SurgicalWorkflowStage;onStage:(stage:SurgicalWorkflowStage)=>void;pause:()=>void;cursor?:Point3|null;onCursor?:(point:Point3)=>void;artifact?:unknown;playRecorded?:(branchId:string)=>void}) {
  const options=workflowPlannerOptions(c.result,artifact),preferred=preferredWorkflowPlanner(c.result,artifact);
  const [requested,setRequested]=useState<SurgicalWorkflowPlanner|null>(null);
  const planner=requested&&options.includes(requested)?requested:preferred;
  const r=c.result,v=c.view;if(!v)return null;
  const report=planner==='preparation'?r?.preparationPlanning:planner==='strategy'?r?.strategyPlanning:planner==='approach'?r?.interactivePlanning:null;
  const staged=planner==='staged'?stagedStrategyDisplay(artifact,c.result):null;
  const current=workflowReportCurrent(r,report),qualification=r?.interactivePlanningAvailability?.admission;
  const target=v.scene.regions.filter(region=>region.target).map(region=>region.name);
  const missing=[...new Set([
    ...Object.entries(v.scene.environment?.anatomy??{}).filter(([,status])=>!['supplied','mapped'].includes(status)).map(([name,status])=>`${name}: ${status}`),
    ...(v.protection?.unknownStructures??[]),...(v.protection?.missingRequired??[]),...(qualification?.sourceEvidence.unknownEvidence??[]),
  ])];
  const presentation={c,pause,cursor,onCursor,stage,onCompared:()=>onStage('compare'),onInstalled:()=>onStage('rehearse')};
  return <section className="surgical-workflow" aria-label="Case planning workflow" data-workflow-stage={stage} data-workflow-planner={planner??'unavailable'}>
    <nav aria-label="Surgical workflow stages">{stages.map((item,index)=><button key={item.id} aria-label={`${item.label} stage`} aria-current={stage===item.id?'step':undefined} onClick={()=>{pause();onStage(item.id);}}><span>{index+1}</span>{item.label}</button>)}</nav>
    <p className="surgical-note workflow-identity">{qualification?.sourceEvidence.patientId??v.scene.source?.caseId??'Current case'} · branch {r?.branchId} · action {r?.position??0}</p>
    <section hidden={stage!=='case'} aria-label="Case anatomy and evidence"><h3>Review this case</h3>
      <p>Inspect source planes and 3D anatomy in the linked views before planning.</p>
      <dl><div><dt>Declared target</dt><dd>{target.length?target.join(', '):'No target mapped'}</dd></div>
        <div><dt>Protected regions</dt><dd>{v.protection?.configured?v.protection.structures.map(row=>row.name).join(', ')||'No masks supplied':'No exclusion contract attached'}</dd></div>
        <div><dt>Missing or uncertain evidence</dt><dd>{missing.length?missing.join(', '):'Completeness has not been established'}</dd></div></dl>
      <p className="surgical-note">Mapped anatomy and constraints support engineering research. Missing anatomy remains unknown; this app does not establish clinical eligibility.</p>
      {qualification&&<details><summary>Source and task qualification</summary><p>{qualification.sourceEvidence.sourceRole} · {qualification.sourceEvidence.studyTimepoint}</p><ul>{Object.entries(qualification.sourceEvidence.anatomy).map(([name,row])=><li key={name}>{name}: {row.kind} · {row.reviewStatus}. {row.coverage}</li>)}</ul></details>}
      <button className="primary-button" disabled={c.pending} onClick={()=>onStage('plan')}>Review constraints and plan</button>
    </section>
    <div hidden={stage!=='plan'&&stage!=='compare'}>
      {options.length>1&&<label>Supported task<select aria-label="Supported planning task" disabled={c.pending||!!c.preview} value={planner??''} onChange={event=>{pause();c.clearPreview();setRequested(event.target.value as SurgicalWorkflowPlanner);onStage('plan');}}>{options.map(option=><option key={option} value={option}>{labels[option]}</option>)}</select></label>}
      {planner&&<p className="surgical-note">{labels[planner]}. {planner==='cranial'?'Compare complete native sequences across source-bound opening roots. Missing qualification blocks generation; bone clearance alone never completes the target task.':planner==='approach'?'Checks inactive placement and withdrawal; target removal is outside this task.':planner==='staged'?'Generate from an eligible intact paired source, compare complete recorded outcomes, then open a saved native branch for rehearsal. Playback does not recalculate a plan.':'Generate a bounded comparison, inspect complete costs, then preview and install a checked branch.'}</p>}
      {!!report&&!current&&<p role="status" className="surgical-warning">Saved comparison · historical checkpoint. Anatomy, tools, accepted actions or constraints may have changed. Open its saved checkpoint below or return to Plan and regenerate.</p>}
      {!planner&&<p role="status">No checked planning task is available at this checkpoint. Inspect source qualification or verify the saved replay. Rehearse/edit retains recorded playback and supported manual actions.</p>}
    </div>
    {planner==='cranial'&&<CranialStrategiesPanel {...presentation} play={playRecorded}/>}
    {planner==='preparation'&&<PreparationPlanningPanel {...presentation}/>}
    {planner==='strategy'&&<TwoToolStrategyPanel {...presentation}/>}
    {planner==='approach'&&<InteractivePlanningPanel {...presentation}/>}
    {planner==='staged'&&<StagedStrategiesPanel {...presentation} artifact={artifact} play={playRecorded}/>}
    {stage==='compare'&&<button disabled={c.pending} onClick={()=>{pause();onStage('rehearse');}}>Open recorded branch in rehearsal</button>}
    {stage==='rehearse'&&<p className="surgical-note">Playback shows accepted recorded actions. Pause at a boundary to edit; accepted edits continue as a native branch and require a new comparison.</p>}
    <details className="workflow-bindings"><summary>Case and checkpoint identity</summary><p>Scene: <code>{v.scene.sceneId}</code></p><p>Accepted state: <code>{v.stateId}</code></p>{report&&<p>Comparison: <code>{report.reportSeal}</code> · {current?'bound to this accepted checkpoint':'historical'}. Local draft edits still require regeneration.</p>}{staged&&<p>Recorded staged comparison: <code>{staged.comparisonSeal}</code>. Source state: <code>{staged.sourceStateId}</code>. Branch edits require a fresh search from its original source.</p>}</details>
  </section>;
}
