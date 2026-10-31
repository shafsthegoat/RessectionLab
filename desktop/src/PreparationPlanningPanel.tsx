import {useEffect,useRef,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {Point3} from './viewer/coordinates';
import {RegionEditor} from './InteractivePlanningPanel';
import {planningBinding} from './interactive-planning-data';
import type {InteractivePlanningOverlay} from './interactive-planning-data';
import {changePreparationFamily,currentPreparationReport,currentPreparationSuggestion,preparationActions,preparationApplicationLabel,preparationAttempt,preparationBaselineText,preparationCost,preparationDraftIssue,preparationFamilyName,
  preparationIncomplete,preparationLocked,preparationPhaseAvailable,preparationPointIsCell,
  preparationSuffix,preparationWidthOptions,snapPreparationPoint} from './preparation-planning-data';
import type {PreparationComparison,PreparationControls,PreparationFamily,PreparationPhase,PreparationPreview,PreparationResponse,PreparationSubmission} from './preparation-planning-data';
import type {ContextOption} from './two-tool-data';

const number=(n:number|undefined)=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(2):'—';
const pendingText:Record<string,string>={suggestPreparation:'Finding a public source-ray suggestion…',generatePreparations:'Comparing finite preparations and complete sequences…',previewPreparation:'Checking preparation preview…',applyPreparation:'Installing checked branch…'};

export function PreparationPlanningPanel({c,pause,cursor=null,onCursor}:{c:SurgicalController;pause:()=>void;cursor?:Point3|null;onCursor?:(p:Point3)=>void}){
  const response=c.result as PreparationResponse|null,a=response?.preparationPlanningAvailability,saved=response?.preparationPlanning;
  const bound=!!saved&&saved.current===true&&saved.sourceSceneId===response?.view.scene.sceneId&&saved.startStateId===response?.view.stateId&&saved.prefixActions===response?.position&&saved.sourcePosition===response?.position;
  return <details className="preparation-planning interactive-planning"><summary>Compare preparation and complete plans</summary>
    <p className="surgical-note">Compare declared local sweeps or paired working lanes along the same source ray, then test CUSA and suction access, finite contact and withdrawal. Preparation removes mapped non-target tissue and remains a cost even when the full sequence STOPs.</p>
    {a?.available&&a.defaults?<PreparationDraft key={planningBinding(response)} c={c} pause={pause} cursor={cursor} onCursor={onCursor} defaults={bound?saved!.controls:a.defaults} saved={bound?saved:null} contexts={a.contextOptions??[]}/>:<p role="status">{a?.reason??'Inspect a source-qualified native checkpoint to check preparation availability.'}</p>}
    {saved&&(!bound||preparationLocked(response)||!a?.available)&&<details open><summary>Saved preparation comparison</summary><p>{saved.rows.length} recorded width alternatives. {saved.scope}</p>
      {preparationBaselineText(saved)&&<p className="surgical-note" aria-label="Prior preparation burdens">{preparationBaselineText(saved)}</p>}
      <p>{preparationFamilyName(saved.controls.preparationFamily)}</p>
      <ul>{saved.rows.map(row=><li key={row.id}>{row.widthMm} mm {saved.controls.preparationFamily==='paired-lanes-v1'?'lane separation':'local sweep'} · preparation non-target {number(preparationCost(row)?.additionalNonTargetMm3)} mm³ · {number(preparationCost(row)?.elapsedSeconds)} s · {row.comparison?.publicPlan.decision.selectedId??'No complete preparation'}</li>)}</ul>
      <button disabled={c.pending||!saved.sourceBranchId} onClick={()=>{pause();c.clearPreview();void c.run('selectBranch',{branchId:saved.sourceBranchId,position:saved.sourcePosition});}}>Open preparation comparison checkpoint</button>
    </details>}
    {!!saved?.applications?.length&&<section className="approach-selection" aria-label="Installed preparation branches"><strong>Installed preparation branches</strong>
      <ul>{saved.applications.map((app,i)=><li key={`${app.branchId}:${app.revision}:${i}`} data-preparation-branch={app.branchId}>
        <button disabled={c.pending||!!response?.verification?.readOnly} aria-label={`${preparationApplicationLabel(app)} · ${app.branchId}`} onClick={()=>{pause();c.clearPreview();void c.run('selectBranch',{branchId:app.branchId});}}>{preparationApplicationLabel(app)}{app.phase==='withdraw'?' + complete sequence':''}</button>
        <p className="surgical-note">{preparationFamilyName(app.preparationFamily)}. Recorded preparation cost: {number(app.preparationSummary.additionalNonTargetMm3)} mm³ non-target + {number(app.preparationSummary.additionalTargetMm3)} mm³ target; {number(app.preparationSummary.elapsedSeconds)} s. Comparison revision {app.revision}.</p>
      </li>)}</ul>
    </section>}
  </details>;
}

function SourcePoint({name,value,onChange,cursor,onCursor,scene,disabled}:{name:string;value:Point3|null;onChange:(p:Point3)=>void;cursor:Point3|null;onCursor?:(p:Point3)=>void;scene:PreparationResponse['view']['scene'];disabled:boolean}){
  const snapped=cursor?snapPreparationPoint(cursor,scene):null;
  function update(axis:number,n:number){const point=[...(value??[NaN,NaN,NaN])] as Point3;point[axis]=n;onChange(point);}
  return <fieldset className="approach-region" disabled={disabled}><legend>{name}</legend>
    <div className="approach-coordinates">{['R','A','S'].map((axis,i)=><label key={axis}>{axis} · mm<input aria-label={`${name} ${axis} coordinate`} type="number" step="0.5" value={Number.isFinite(value?.[i])?value![i]:''} onChange={e=>update(i,e.target.valueAsNumber)} onBlur={()=>{if(value){const point=snapPreparationPoint(value,scene);if(point&&!point.every((n,j)=>n===value[j]))onChange(point);}}}/></label>)}</div>
    <div className="surgical-actions"><button disabled={!snapped} onClick={()=>snapped&&onChange([...snapped])}>Use linked MRI cursor for {name.toLowerCase()}</button><button disabled={!value||value.some(n=>!Number.isFinite(n))||!onCursor} onClick={()=>value&&onCursor?.([...value])}>Show {name.toLowerCase()} in linked views</button></div>
    <p className="surgical-note">Coordinates snap to current source-cell centres. The backend checks retained surface or target-boundary membership.</p>
  </fieldset>;
}

function PreparationDraft({c,pause,cursor,onCursor,defaults,saved,contexts}:{c:SurgicalController;pause:()=>void;cursor:Point3|null;onCursor?:(p:Point3)=>void;defaults:PreparationControls;saved:PreparationComparison|null|undefined;contexts:ContextOption[]}){
  const response=c.result as PreparationResponse|null;
  const [draft,setDraft]=useState(()=>structuredClone(defaults)),[revision,setRevision]=useState(0);
  const [submitted,setSubmitted]=useState<PreparationSubmission|null>(saved?{binding:planningBinding(response),revision:0}:null);
  const [ack,setAck]=useState(false),[selected,setSelected]=useState<string|null>(null),[phase,setPhase]=useState<PreparationPhase>('prepared');
  const [showRay,setShowRay]=useState(false),[notice,setNotice]=useState<string|null>(null);
  const revisionRef=useRef(0),bindingRef=useRef(planningBinding(response)),alive=useRef(true);
  bindingRef.current=planningBinding(response);
  const suggestionRequest=useRef<PreparationSubmission|null>(null);
  useEffect(()=>{alive.current=true;return ()=>{alive.current=false;};},[]);
  const locked=preparationLocked(response,c.pending),report=currentPreparationReport(response,draft,submitted,revision);
  const paired=draft.preparationFamily==='paired-lanes-v1';
  const id=selected??report?.policies.completePlanAware.selectedPreparationId??report?.policies.corridorFirst.selectedPreparationId??report?.rows[0]?.id;
  const row=report?.rows.find(r=>r.id===id),attempt=preparationAttempt(row),cost=preparationCost(row),suffix=preparationPhaseAvailable(row,'withdraw',draft.preparationFamily)?preparationSuffix(row):null;
  const suggestion=currentPreparationSuggestion(response);
  let issue=preparationDraftIssue(draft,contexts);
  if(!issue&&draft.sourceRay&&response&&!(['surfaceMm','sourceTargetMm'] as const).every(k=>preparationPointIsCell(draft.sourceRay![k],response.view.scene)))issue='Source ray points must be cell centres inside the current source grid.';
  const actions=preparationActions({response,draft,submitted,revision,candidateId:id,phase,preview:c.approachPreview as PreparationPreview|null,displayed:c.preview,pending:c.pending,acknowledged:ack,contexts});
  const canPreview=actions.previewAllowed,matchingPreview=actions.matchingPreview;
  const previewBusy=!!c.preview&&!matchingPreview;

  function edit(next:PreparationControls){revisionRef.current+=1;setRevision(revisionRef.current);setDraft(next);setSubmitted(null);setSelected(null);setShowRay(false);setNotice(null);suggestionRequest.current=null;c.clearPreview();}
  useEffect(()=>{
    const request=suggestionRequest.current;
    if(!request||!suggestion||request.binding!==bindingRef.current||request.revision!==revisionRef.current)return;
    suggestionRequest.current=null;
    edit({...draft,sourceRay:structuredClone(suggestion.sourceRay),target:structuredClone(suggestion.target)});
    setNotice('Suggested cells and common target region are ready for review.');
  },[response?.preparationSuggestion]);
  useEffect(()=>{
    if(!showRay||issue||locked||c.preview||!draft.sourceRay||!draft.target||!response)return;
    const start=draft.sourceRay.surfaceMm,end=draft.sourceRay.sourceTargetMm;
    const pose=(position:Point3)=>({position:[...position] as Point3,quaternion:[0,0,0,1] as [number,number,number,number]});
    const overlay:InteractivePlanningOverlay={binding:planningBinding(response),constraints:{target:draft.target,entry:{centerMm:start,radiusMm:draft.entryRadiusMm},tool:{instrument:'cusa',radiusMm:1.5},requiredEvidence:draft.requiredEvidence,candidateCount:2,nonTargetClearanceMm:0},
      routes:[{id:'declared-preparation-ray',label:'Declared source ray · not an executable approach',outsidePose:pose(start),entryPose:pose(start),targetPose:pose(end),actionProgram:[],metrics:{},scope:'Declared source-cell centre line; not a cleared trajectory.'}],selectedId:'declared-preparation-ray',limitingStructures:[]};
    c.setPlanningOverlay(overlay);return ()=>c.setPlanningOverlay(null);
  },[showRay,draft,issue,locked,c.preview]);
  async function suggest(){
    if(locked||!ack||c.preview)return;pause();c.clearPreview();suggestionRequest.current={binding:bindingRef.current,revision:revisionRef.current};setNotice(null);
    const ok=await c.run('suggestPreparation',{acknowledgeResearchUnknowns:true});
    if(!ok&&alive.current){suggestionRequest.current=null;setNotice('Source-ray suggestion did not complete.');}
  }
  async function generate(){
    if(locked||issue||!ack||c.preview)return;pause();c.clearPreview();setSubmitted(null);setNotice(null);
    const request={binding:bindingRef.current,revision:revisionRef.current};
    const ok=await c.run('generatePreparations',{constraints:structuredClone(draft),acknowledgeResearchUnknowns:true});
    if(alive.current&&ok&&request.binding===bindingRef.current&&request.revision===revisionRef.current){setSubmitted(request);setSelected(null);setPhase('prepared');}
  }
  function rayPoint(key:'surfaceMm'|'sourceTargetMm',point:Point3){edit({...draft,sourceRay:{surfaceMm:draft.sourceRay?.surfaceMm??[NaN,NaN,NaN],sourceTargetMm:draft.sourceRay?.sourceTargetMm??[NaN,NaN,NaN],[key]:point}});}
  const context=contexts.find(x=>x.layerId===draft.encounter?.layerId);
  return <section className="preparation-draft approach-draft" aria-label="Preparation comparison draft" data-preparation-seal={report?.reportSeal} data-preparation-family={draft.preparationFamily??'cardinal-v1'}>
    <fieldset disabled={locked}><legend>Source qualification and research acceptance</legend>
      <label><input aria-label="Acknowledge preparation research scope" type="checkbox" checked={ack} onChange={e=>{setAck(e.target.checked);c.clearPreview();}}/>I accept modeled non-target removal and missing skull, vascular and functional evidence for this research comparison.</label>
      <button disabled={!ack||!!c.preview} onClick={()=>void suggest()}>Suggest source ray from public anatomy</button>
    </fieldset>
    {response&&<><SourcePoint name="Source surface cell" value={draft.sourceRay?.surfaceMm??null} onChange={p=>rayPoint('surfaceMm',p)} cursor={cursor} onCursor={onCursor} scene={response.view.scene} disabled={locked}/>
      <SourcePoint name="Source target-boundary cell" value={draft.sourceRay?.sourceTargetMm??null} onChange={p=>rayPoint('sourceTargetMm',p)} cursor={cursor} onCursor={onCursor} scene={response.view.scene} disabled={locked}/></>}
    {draft.target?<RegionEditor name="Common target region" value={draft.target} onChange={target=>edit({...draft,target})} cursor={cursor} onCursor={onCursor} disabled={locked}/>:<button disabled={locked} onClick={()=>edit({...draft,target:{centerMm:[...(cursor??draft.sourceRay?.sourceTargetMm??[0,0,0])],radiusMm:6}})}>Define common target region</button>}
    <fieldset disabled={locked}><legend>Matched preparation limits</legend>
      <label>Preparation family<select aria-label="Preparation family" value={draft.preparationFamily??'cardinal-v1'} onChange={e=>edit(changePreparationFamily(draft,e.target.value as PreparationFamily))}><option value="cardinal-v1">Local cardinal sweeps</option><option value="paired-lanes-v1">Paired working lanes</option></select></label>
      <p>Fixed 1.5 mm CUSA tip and catalog shaft. Every alternative uses the same source ray, target region and limits.</p>
      {preparationWidthOptions(draft).map(width=><label key={width}><input aria-label={`Compare ${paired?'lane separation':'lateral sweep'} ${width} mm`} type="checkbox" checked={draft.widthsMm.includes(width)} disabled={!draft.widthsMm.includes(width)&&draft.widthsMm.length>=3} onChange={e=>edit({...draft,widthsMm:e.target.checked?[...draft.widthsMm,width].sort((a,b)=>a-b):draft.widthsMm.filter(w=>w!==width)})}/>{width} mm {paired?'lane separation':'local sweep'}</label>)}
      {([['maxNonTargetMm3','Preparation non-target limit · mm³',.1,200,1],['maxActions','Preparation action limit',6,128,1],['maxSeconds','Preparation time limit · s',2,180,1],['entryRadiusMm','Shared entry-patch radius · mm',.5,15,.5]] as const).map(([key,label,min,max,step])=><label key={key}>{label}<input aria-label={label} type="number" min={min} max={max} step={step} value={Number.isFinite(draft[key])?draft[key]:''} onChange={e=>edit({...draft,[key]:e.target.valueAsNumber})}/></label>)}
      <p className="surgical-note">{paired?'Paired lanes separate along the declared tool-frame Y direction. Each lane uses fixed 0.3 mm local cardinal passes, up to ten real passes before coincident paths are deduplicated. Zero separation still prepares one five-pass lane. Early target clipping may leave insufficient working room.':'A local sweep executes separate fixed-radius paths.'} Tool geometry and removal, action and duration caps remain unchanged. Both complete tools and withdrawal still require checked native execution. These are engineering limits, not tissue tolerances.</p>
    </fieldset>
    <fieldset disabled={locked}><legend>Complete-plan task, identical for every alternative</legend>
      {([['shortSeconds','Preparation comparison short contact seconds',.1,2,.05],['longSeconds','Preparation comparison longer contact seconds',.1,2,.05],['minimumTargetMm3','Preparation comparison target progress mm3',0,8,.5]] as const).map(([key,label,min,max,step])=><label key={key}>{label}<input aria-label={label} type="number" min={min} max={max} step={step} value={Number.isFinite(draft[key])?draft[key]:''} onChange={e=>edit({...draft,[key]:e.target.valueAsNumber})}/></label>)}
      <label><input aria-label="Require functional evidence for preparation comparison" type="checkbox" checked={draft.requiredEvidence.includes('functional')} onChange={e=>edit({...draft,requiredEvidence:e.target.checked?[...draft.requiredEvidence,'functional']:draft.requiredEvidence.filter(x=>x!=='functional')})}/>Require functional evidence</label>
      <label>Supplied anatomical context<select aria-label="Preparation context layer" value={draft.encounter?.layerId??''} onChange={e=>edit({...draft,encounter:e.target.value?{layerId:e.target.value,marginMm:draft.encounter?.marginMm??0,maxNearMm3:draft.encounter?.maxNearMm3??0}:null})}><option value="">No additional context constraint</option>{contexts.map(x=><option key={x.layerId} value={x.layerId}>{x.description} · label {x.labelValue}</option>)}</select></label>
      {draft.encounter&&<><p className="surgical-note">{context?.reviewStatus}. {context?.coverage}. Unique retained context cells near complete tools are a geometric endpoint, without an injury or neurological function estimate.</p>
        <label>Context margin · mm<input aria-label="Preparation context margin mm" type="number" min="0" max="5" step=".5" value={Number.isFinite(draft.encounter.marginMm)?draft.encounter.marginMm:''} onChange={e=>edit({...draft,encounter:{...draft.encounter!,marginMm:e.target.valueAsNumber}})}/></label>
        <label>Allowed near-context volume · mm³<input aria-label="Preparation context near volume mm3" type="number" min="0" max="1000" step="1" value={Number.isFinite(draft.encounter.maxNearMm3)?draft.encounter.maxNearMm3:''} onChange={e=>edit({...draft,encounter:{...draft.encounter!,maxNearMm3:e.target.valueAsNumber}})}/></label>
      </>}
    </fieldset>
    {issue&&<p role="alert">{issue}</p>}
    {notice&&<p role="status">{notice}</p>}
    {c.pending&&<p role="status">{pendingText[c.pendingCommand??'']??'Checking the current request…'}</p>}
    <div className="surgical-actions"><button disabled={locked||!!issue||!!c.preview||!onCursor} onClick={()=>{pause();setShowRay(true);if(draft.sourceRay)onCursor?.([...draft.sourceRay.surfaceMm]);}}>Show declared source ray in linked MRI views</button>
      <button className="primary-button" disabled={locked||!!issue||!ack||!!c.preview} onClick={()=>void generate()}>Compare preparations and complete plans</button></div>
    {report&&<>{preparationBaselineText(report)&&<p className="surgical-note" aria-label="Prior preparation burdens">{preparationBaselineText(report)}</p>}
      <p role="status" aria-label="Preparation policy decisions">Corridor-first: {report.policies.corridorFirst.selectedPreparationId??'STOP before preparation'}; complete-plan-aware: {report.policies.completePlanAware.selectedPreparationId??'STOP before preparation'}.</p>
      <p className="surgical-note">Both rules use the same finite offline table. Preparation costs include setup, finite tissue removal and parking; optional sequence costs are separate. No runtime efficiency or clinical preference is assessed.</p>
      {report.rows.some(preparationIncomplete)&&<p role="alert">A computation budget ended a search. Complete-plan feasibility remains unresolved.</p>}
      <div className="approach-table" role="region" aria-label="Preparation comparison table" tabIndex={0}><table aria-label="Preparation comparison"><thead><tr><th title={paired?'Lane-center separation along tool-frame Y':undefined}>{paired?'Lane offset':'Local sweep'}</th><th>Preparation</th><th>Complete sequence</th></tr></thead><tbody>{report.rows.map(item=>{const a=preparationAttempt(item),cost=preparationCost(item),plan=item.comparison?.publicPlan;return <tr key={item.id} data-preparation-candidate={item.id} aria-selected={item.id===id}><th><button disabled={locked} onClick={()=>{c.clearPreview();setSelected(item.id);setPhase('prepared');}} aria-label={`Select preparation width ${item.widthMm} mm`}>{item.widthMm} mm</button></th><td><strong>{number(cost?.additionalNonTargetMm3)} mm³</strong><p>{number(cost?.elapsedSeconds)} s</p><p>{plan&&Array.isArray(plan.approaches.routes)?plan.approaches.routes.length?'Checked route':'No checked route':'Not tested'}</p></td><td>{!item.preparation?'Evidence refusal':!item.preparation.publicPlan.attempts.length?'Generator geometric refusal':!a?'Preparation incomplete':preparationPhaseAvailable(item,'withdraw',report.controls.preparationFamily)?'Complete checked sequence':'STOP'}</td></tr>;})}</tbody></table></div>
      {row&&<section className="approach-selection" aria-label="Selected preparation"><strong>{preparationFamilyName(draft.preparationFamily)} · {row.widthMm} mm {paired?'lane separation':'local sweep'}</strong>
        {cost&&<p>{attempt?'Preparation':'Incomplete preparation attempt'}: {number(cost.additionalNonTargetMm3)} mm³ non-target + {number(cost.additionalTargetMm3)} mm³ target; {number(cost.elapsedSeconds)} s.</p>}
        {attempt?<>
          {suffix?<p>Optional complete sequence: {number(suffix.summary.additionalTargetMm3)} mm³ target progress, {number(suffix.summary.additionalNonTargetMm3)} mm³ additional non-target, {number(suffix.summary.elapsedSeconds)} s.</p>:<p>The full two-tool sequence STOPs for this row. Installing its preparation still incurs the displayed non-target removal and time.</p>}</>:
          <p>{row.preparation?.publicPlan.attempts.length?'Preparation did not complete.':'This cell was refused by the generator or required-evidence gate. It is not independent proof that no preparation is feasible.'}</p>}
        {!!row.missingEvidence?.length&&<p role="alert">Required evidence unavailable: {row.missingEvidence.join(', ')}.</p>}
        {row.reason&&<p role="status">Recorded refusal: {row.reason}</p>}
        <label>Preview and installation checkpoint<select aria-label="Preparation checkpoint phase" disabled={locked} value={phase} onChange={e=>{c.clearPreview();setPhase(e.target.value as PreparationPhase);}}><option value="prepared" disabled={!preparationPhaseAvailable(row,'prepared',draft.preparationFamily)}>Preparation complete · costs only</option><option value="withdraw" disabled={!preparationPhaseAvailable(row,'withdraw',draft.preparationFamily)}>Preparation + complete sequence + withdrawal</option></select></label>
        <div className="surgical-actions"><button disabled={!canPreview||previewBusy} onClick={()=>{pause();void c.run('previewPreparation',{candidateId:row.id,expectedPlanSeal:report.reportSeal,phase});}}>Preview preparation checkpoint</button>
          <button disabled={!actions.installAllowed} onClick={()=>{pause();void c.run('applyPreparation',{candidateId:row.id,expectedPlanSeal:report.reportSeal,phase});}}>{phase==='prepared'?'Install preparation branch':'Install complete plan branch'}</button></div>
        {previewBusy&&<button disabled={c.pending} onClick={()=>c.clearPreview()}>Cancel other preview</button>}
        {matchingPreview&&<p role="status">Uncommitted {phase==='prepared'?'preparation':'complete withdrawal'} checkpoint shown. Installation creates this checked native branch.</p>}
      </section>}
      <details><summary>Preparation evidence and failed attempts</summary><p>{report.scope}</p><pre>{JSON.stringify({reportSeal:report.reportSeal,revision:report.revision,policies:report.policies,rows:report.rows.map(({id,widthMm,reason,missingEvidence,preparation,comparison})=>({id,widthMm,reason,missingEvidence,preparation,comparison})),previewReceipt:(c.approachPreview as PreparationPreview|null)?.receipt},null,2)}</pre></details>
    </>}
  </section>;
}
