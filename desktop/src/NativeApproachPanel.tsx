import {useEffect,useState} from 'react';
import {ArrowDownToLine,Focus,Plus,Sparkles} from 'lucide-react';
import type {NativeWorkspace} from './use-native-workspace';
import type {Point3} from './viewer/coordinates';
import type {ApproachCoverage,ApproachSensitivityConfig,ApproachTargetOption,NativeApproach} from './native-approaches';
import {approachAssessment,approachStatus,defaultApproachSensitivity,lowerBoundText} from './native-approaches';
import {NativeStructureControls} from './NativeStructureControls';
import {ApproachProposalDetails,ApproachSensitivityDetails,ApproachInputComparison} from './NativeApproachReadouts';
import './native-approaches.css';
const mm=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(1):'—';
const bound=lowerBoundText;
const ras=(point:Point3)=>point.map(mm).join(', ');
const fieldCoverage=(field?:ApproachCoverage)=>field?.wholeEnvelopeWithinField===true?'Whole required envelope inside':field?.wholeEnvelopeWithinField===false?'Partial · outside unassessed':'Unknown';
type EditorFields={entry:Point3;target:Point3;radius:number;patchRadius:number;minimum:number;maximum:number;
  targetLayerId?:string;patchCenter?:Point3;patchNormal?:Point3;sensitivityConfig:ApproachSensitivityConfig};
function editorFields(selected:NativeApproach|null,entry:Point3,target:Point3,targetLayerId?:string):EditorFields{
  return {entry:[...(selected?.entry??entry)],target:[...(selected?.target??target)],radius:selected?.radiusMm??2,
    patchRadius:selected?.entryPatch?.radiusMm??15,minimum:selected?.constraints.minClearanceMm??0,maximum:selected?.constraints.maxLengthMm??150,
    targetLayerId:selected?selected.targetLayerId:targetLayerId,patchCenter:selected?.entryPatch?.center,patchNormal:selected?.entryPatch?.normal,
    sensitivityConfig:selected?.sensitivityConfig??defaultApproachSensitivity};
}
function inward(entry:Point3,target:Point3):Point3{
  const axis=target.map((v,i)=>v-entry[i]),length=Math.hypot(...axis);
  return length>0?axis.map(v=>v/length) as Point3:[0,0,-1];
}
function PointEditor({label,value,onChange}:{label:string;value:Point3;onChange:(p:Point3)=>void}){return <div className="approach-point"><span>{label} · RAS mm</span><div>{['R','A','S'].map((axis,i)=><label key={axis}>{axis}<input aria-label={`${label} ${axis}`} type="number" step="1" value={Number(value[i].toFixed(2))} onChange={e=>onChange(value.map((v,j)=>j===i?+e.target.value:v) as Point3)}/></label>)}</div></div>}
export function NativeApproachPanel({workspace:w}:{workspace:NativeWorkspace}){
  const p=w.planning,selected=p.selected,b=w.binding;
  const savedTask=p.report?.assessmentStatus==='current'?p.report.task:!p.report&&p.inputComparison?.assessmentStatus==='current'?p.inputComparison.task:undefined;
  const defaultTarget=p.targetOptions.find(option=>option.target!==null&&option.layerId===p.activeTargetLayerId);
  const newEditorKey=`${b?.collectionId}/${b?.document.snapshot_id}/new/${p.activeTargetLayerId??''}/${p.loading}`;
  const editorKey=selected?`${b?.collectionId}/${b?.document.snapshot_id}/${selected.id}/${selected.parameterHash}`:newEditorKey;
  const initialTarget=defaultTarget?.target??b?.document.display?.cursor_ras_mm??w.state.cursor??[0,0,40];
  const initialEntry=defaultTarget?.entry??(!defaultTarget?b?.document.display?.entry_ras_mm:undefined)??[initialTarget[0],initialTarget[1],initialTarget[2]+40];
  const initial=()=>{
    const base=editorFields(selected,initialEntry,initialTarget,defaultTarget?.layerId);
    const fields=!selected&&savedTask?{...base,
      entry:savedTask.entry??base.entry,target:savedTask.target??base.target,radius:savedTask.radiusMm??base.radius,
      patchCenter:savedTask.entryPatch?.center,patchNormal:savedTask.entryPatch?.normal,patchRadius:savedTask.entryPatch?.radiusMm??base.patchRadius,
      targetLayerId:savedTask.targetLayerId??undefined,sensitivityConfig:savedTask.sensitivityConfig??base.sensitivityConfig,
      minimum:savedTask.constraints?.minClearanceMm??base.minimum,maximum:savedTask.constraints?.maxLengthMm??base.maximum}:base;
    return {key:editorKey,fields,dirty:false,attempted:false};
  };
  const [editor,setEditor]=useState(initial);
  // A newly selected/source-revised object is also used on its first render,
  // before passive effects run, so an old draft cannot enter its request.
  const current=editor.key===editorKey?editor:initial(),fields=current.fields;
  const {entry,target,radius,patchRadius,minimum,maximum,targetLayerId,sensitivityConfig}=fields,dirty=!!selected&&current.dirty;
  const {translationMm,boundaryMm}=sensitivityConfig;
  useEffect(()=>{setEditor(value=>value.key===editorKey?value:initial())},[editorKey]);
  const blocked=w.busy||!!w.draft||p.pending||p.loading;
  const valid=[...entry,...target,radius,patchRadius,minimum,maximum,translationMm,boundaryMm].every(Number.isFinite)&&radius>0&&radius<=50&&patchRadius>=0&&patchRadius<=40&&minimum>=0&&minimum<=50&&maximum>=.1&&maximum<=1000&&translationMm>=0&&translationMm<=5&&boundaryMm>=0&&boundaryMm<=3&&Math.hypot(...target.map((v,i)=>v-entry[i]))>1e-6;
  function edit(change:Partial<EditorFields>){setEditor({...current,fields:{...fields,...change},dirty:true,attempted:false})}
  const payload=()=>{
    return {id:selected?.id,name:selected?.name??'Manual approach',entry,target,radiusMm:radius,targetLayerId:targetLayerId??null,sensitivityConfig,
      entryPatch:{center:fields.patchCenter??entry,normal:fields.patchNormal??inward(entry,target),radiusMm:patchRadius},
      constraints:{...selected?.constraints,minClearanceMm:minimum,maxLengthMm:maximum}};
  };
  async function commitEdits(){
    if(!dirty)return true;
    if(blocked||!valid)return false;
    const submitted=current;
    setEditor(value=>value.key===submitted.key&&value.fields===submitted.fields?{...value,attempted:true}:value);
    const result=await p.request({action:'upsert',approach:payload()});
    if(result)setEditor(value=>value.key===submitted.key&&value.fields===submitted.fields?{...value,dirty:false}:value);
    return !!result;
  }
  useEffect(()=>{
    if(!dirty||current.attempted||blocked||!valid)return;
    const timer=setTimeout(()=>{void commitEdits()},650);
    return()=>clearTimeout(timer);
  },[dirty,current.attempted,fields,blocked,valid,editorKey]);
  if(!b)return null;
  async function runComparison(action:'suggest'|'compareInputs'){
    if(!await commitEdits())return;
    const value=payload();
    const sameFrozenPatch=savedTask?.entry&&savedTask.targetLayerId===targetLayerId&&savedTask.entryPatch?.center.every((v,i)=>v===value.entryPatch.center[i]);
    const result=await p.request({action,start:action==='compareInputs'&&sameFrozenPatch?savedTask.entry:entry,end:target,radiusMm:radius,entryPatch:value.entryPatch,
      patchRadiusMm:patchRadius,constraints:value.constraints,targetLayerId,sensitivityConfig});
    if(action==='suggest'&&result?.selectedId){w.setCursor(result.approaches.find(a=>a.id===result.selectedId)?.target??target);w.setSelectedStructure(null)}
  }
  async function selectApproach(a:NativeApproach){if(await commitEdits()){p.select(a.id);w.setCursor(a.target)}}
  async function selectTarget(option:ApproachTargetOption|undefined){
    if(!await commitEdits())return;
    const result=await p.request({action:'list',targetLayerId:option?.layerId??null});
    if(!result)return;
    const preferred=p.selectedApproachIds?.[option?.layerId??''];
    const route=preferred===null?undefined:result.approaches.find(a=>a.id===preferred)??result.approaches.find(a=>a.targetLayerId===option?.layerId);
    w.setCursor(route?.target??option?.target??initialTarget);
    w.setSelectedStructure(null);
    if(option){const visibility={...w.state.visible};for(const row of p.targetOptions)visibility[row.layerId]=row.layerId===option.layerId;w.setVisible(visibility);}
  }
  const m=selected?.assessmentStatus==='current'?selected.measurements:undefined,status=selected?approachStatus(selected):'unassessed',assessment=selected?approachAssessment(selected):null;
  const referenceBound=m&&m.targets.length>0&&m.targets.every(row=>row.sourceSeries===b.document.reference_series)&&
    b.document.layers.some(row=>row.id===m.normalTraversal.layerId&&row.source_series===b.document.reference_series);
  const focus=(id:string,point?:unknown)=>{w.setVisible({...w.state.visible,[id]:true});w.focusStructure(id,Array.isArray(point)&&point.length===3?point as Point3:undefined)};
  const targetOption=p.targetOptions.find(option=>option.layerId===targetLayerId);
  const targetLabel=(id?:string|null)=>p.targetOptions.find(option=>option.layerId===id)?.label??id??undefined;
  const layerLabels=Object.fromEntries([...b.document.layers.map(row=>[row.id,row.label]),...(m?.structures??[]).map(row=>[row.layerId,row.label])]);
  return <div className="native-approaches unified-sidebar" data-approach-pending={p.pending} data-approach-dirty={dirty}>
    <div className="approach-intro"><h1>Compare approaches</h1><p>Static capsule measurements on source anatomy. EDEN cases use healthy acquired scans and separate synthetic targets.</p></div>
    {p.targetOptions.length>0&&<label className="approach-target-picker">Planning target<select aria-label="Planning target" value={targetLayerId??''} disabled={blocked||(dirty&&!valid)} onChange={event=>void selectTarget(p.targetOptions.find(option=>option.layerId===event.target.value))}>
      <option value="">Manual target point</option>{p.targetOptions.map(option=><option key={option.layerId} value={option.layerId} disabled={option.target===null}>{option.label}{option.target===null?' · unavailable':''}</option>)}</select></label>}
    {targetOption?.focusMeaning&&<p className="approach-caption">{targetOption.focusMeaning}. Coordinates are editable.</p>}
    <section className="approach-editor"><fieldset disabled={blocked}><div className="approach-section-heading"><h2>{selected?'Edit selected approach':'Set the entry patch'}</h2>{selected&&<button aria-label="Focus selected target" onClick={()=>w.setCursor(target)}><Focus size={13}/></button>}</div>
      <label className="approach-radius">Corridor radius <output>{mm(radius)} mm</output><input aria-label="Corridor radius" type="range" min=".5" max={Math.max(12,radius)} step=".5" disabled={blocked} value={radius} onChange={e=>edit({radius:+e.target.value})}/></label>
      <label>Entry patch radius · mm<input aria-label="Entry patch radius" type="number" min="0" max="40" value={patchRadius} onChange={e=>edit({patchRadius:+e.target.value})}/></label>
      <p className="approach-caption">Patch centre (RAS mm) {ras(fields.patchCenter??entry)}</p>
      <div className="approach-cursor-actions"><button disabled={blocked||!w.state.cursor} onClick={()=>{const point=w.state.cursor!;edit({entry:point,patchCenter:point,patchNormal:inward(point,target)})}}>Place patch at cursor</button><button disabled={blocked||!w.state.cursor} onClick={()=>edit({target:w.state.cursor!})}>Target at cursor</button></div>
      <details><summary>Coordinates & constraints</summary>
        <PointEditor label="Entry" value={entry} onChange={entry=>edit({entry})}/><button disabled={blocked||!w.state.cursor} onClick={()=>edit({entry:w.state.cursor!})}>Entry at cursor</button><PointEditor label="Target" value={target} onChange={target=>edit({target})}/>
        <label>Exact corridor radius · mm<input aria-label="Exact corridor radius" type="number" min="0" max="50" step="any" value={radius} onChange={e=>edit({radius:+e.target.value})}/></label>
        <label>Minimum conservative clearance · mm<input aria-label="Minimum clearance" type="number" min="0" max="50" step=".5" value={minimum} onChange={e=>edit({minimum:+e.target.value})}/></label>
        <label>Maximum corridor length · mm<input aria-label="Maximum corridor length" type="number" min=".1" max="1000" value={maximum} onChange={e=>edit({maximum:+e.target.value})}/></label>
        <p className="approach-caption">Saved-path edits recompute automatically. Moving an entry keeps the saved patch centre and plane fixed. An entry outside that disk fails a geometric constraint.</p>
      </details>
      <details className="approach-assumptions"><summary>Sensitivity settings · {mm(translationMm)} mm translation · {mm(boundaryMm)} mm boundary</summary>
        <label>Registration translation · mm<input aria-label="Registration translation" type="number" min="0" max="5" step=".1" value={translationMm} onChange={event=>edit({sensitivityConfig:{...sensitivityConfig,translationMm:+event.target.value}})}/></label>
        <label>Boundary sensitivity · mm<input aria-label="Boundary sensitivity" type="number" min="0" max="3" step=".1" value={boundaryMm} onChange={event=>edit({sensitivityConfig:{...sensitivityConfig,boundaryMm:+event.target.value}})}/></label>
        <p>Shared deterministic perturbations describe geometric sensitivity. Their magnitudes are user-declared and have no calibrated probabilities.</p>
      </details>
      {selected&&<div className="tool-row"><button disabled={blocked||!valid} onClick={()=>void (dirty?commitEdits():p.request({action:'assess',approachId:selected.id}))}>Reassess</button><button disabled={blocked} onClick={()=>void p.request({action:'delete',approachId:selected.id})}>Delete</button></div>}
    </fieldset></section>
    <div className="approach-primary-actions"><button className="approach-suggest" disabled={blocked||!valid} onClick={()=>void runComparison('suggest')}><Sparkles size={14}/>Suggest approaches</button><button aria-label="Create manual approach" disabled={blocked||!valid} onClick={async()=>{if(await commitEdits())void p.request({action:'upsert',approach:{...payload(),id:undefined,name:`Manual ${p.approaches.length+1}`}})}}><Plus size={14}/>Manual</button></div>
    <p className="approach-caption">Up to three distinct paths within the selected superior or lateral entry patch. Opening the skull and procedural effects are unassessed.</p>
    {p.error&&<p className="approach-error" role="alert">{p.error}{dirty?' Edits remain unsaved. Use Save to retry.':''}</p>}
    {!valid&&<p className="approach-error" role="alert">Use distinct finite endpoints. Supported ranges are corridor radius above 0 to 50 mm, patch radius 0–40 mm, clearance 0–50 mm, length 0.1–1000 mm, translation 0–5 mm and boundary sensitivity 0–3 mm.</p>}
    {p.loading&&<p role="status" className="approach-computing">Loading source-bound comparisons…</p>}
    {p.pending&&<p role="status" className="approach-computing">Measuring source anatomy…</p>}
    <div className="approach-cards" aria-label="Saved approach comparison">{p.approaches.map((a,index)=>{const summary=approachAssessment(a),measured=a.assessmentStatus==='current'?a.measurements:undefined;return <button className={`approach-card ${a.id===p.selectedId?'selected':''}`} key={a.id} aria-pressed={a.id===p.selectedId} data-approach-id={a.id} disabled={blocked||(dirty&&!valid)} onClick={()=>void selectApproach(a)}>
      <span className="approach-card-heading"><b>{String.fromCharCode(65+index)} · {a.name}</b><span>{mm(measured?.lengthMm)} mm</span></span>
      {a.targetLayerId&&<span>{targetLabel(a.targetLayerId)}</span>}
      <span className={`approach-observed ${summary.observed}`} data-observed-result={summary.observed}>{summary.observedLabel}</span>
      <span className="approach-evidence" data-assessment-completeness={summary.completeness}>{summary.completenessLabel} · {summary.assessedCount}/{summary.totalCount} structures measured{summary.limitations.length?` · ${summary.limitations.length} limitations`:''}</span>
      {summary.intersections.length>0&&<span className="approach-card-concern">Intersects: {summary.intersections.map(s=>s.label).join(', ')}</span>}
      {summary.unavailable.length>0&&<span>Unavailable: {summary.unavailable.map(s=>s.label).join(', ')}</span>}
      {summary.unresolved.length>0&&<span>Unresolved contact: {summary.unresolved.map(s=>s.label).join(', ')}</span>}
      {summary.constraintFailures.length>0&&<span className="approach-constraint-failure">{summary.constraintFailures.length} geometric constraint {summary.constraintFailures.length===1?'failure':'failures'}: {summary.constraintFailures[0]}</span>}
      <span className="approach-card-metric">Radius <b>{mm(a.radiusMm)} mm</b> · minimum supplied clearance bound <b>{bound(summary.minimumBoundMm)} mm</b></span>
      <span className="approach-card-metric">Target voxel centres reached <b>{mm(summary.targetReachedMm3)} mm³</b> · brain traversal upper bound <b>{mm(summary.traversalMm3)} mm³</b></span>
    </button>})}</div>
    {p.report&&<ApproachProposalDetails report={p.report} targetLabel={targetLabel(p.report.task?.targetLayerId??p.report.targetLayerId)} labels={layerLabels}/>}
    {assessment&&<section className="approach-measurements" data-assessment-status={status}><h2>Source-bound assessment</h2>
      <p className={`approach-observed ${assessment.observed}`} data-observed-result={assessment.observed}>{assessment.observedLabel}</p>
      <p className="approach-evidence" data-assessment-completeness={assessment.completeness}>{assessment.completenessLabel} · {assessment.assessedCount}/{assessment.totalCount} structures measured.</p>
      {assessment.constraintFailures.length>0&&<div className="approach-constraint-failures" role="status"><h3>Geometric constraint failures</h3><ul>{assessment.constraintFailures.map((failure,i)=><li key={i}>{failure}</li>)}</ul></div>}
      {m&&<>
        <div className="approach-metrics"><span>Length<b>{mm(m.lengthMm)} mm</b></span><span>Azimuth / elevation<b>{mm(m.orientation.azimuthDeg)}° / {mm(m.orientation.elevationDeg)}°</b></span><span>Whole capsule inside reference field<b>{m.coverage.withinReferenceEnvelope===true?'Yes':m.coverage.withinReferenceEnvelope===false?'No · unassessed outside':'Unknown'}</b></span></div>
        {selected&&<details><summary>Capsule geometry & source identity</summary><dl className="approach-value-list">
          <div><dt>Entry (RAS mm)</dt><dd>{ras(selected.entry)}</dd></div><div><dt>Target (RAS mm)</dt><dd>{ras(selected.target)}</dd></div>
          <div><dt>Radius, including both spherical caps</dt><dd>{mm(selected.radiusMm)} mm</dd></div>
          {selected.entryPatch&&<><div><dt>Patch centre (RAS mm)</dt><dd>{ras(selected.entryPatch.center)}</dd></div><div><dt>Patch normal (RAS)</dt><dd>{ras(selected.entryPatch.normal)}</dd></div><div><dt>Patch radius</dt><dd>{mm(selected.entryPatch.radiusMm)} mm</dd></div></>}
        </dl><p className="approach-identity">Source revision {selected.sourceSnapshotId}</p></details>}
        <div className="approach-concerns">{[...m.structures].sort((a,b)=>Number(!!b.confirmedIntersection)-Number(!!a.confirmedIntersection)||(a.clearanceLowerBoundMm??Infinity)-(b.clearanceLowerBoundMm??Infinity)).map(s=><button key={s.layerId} disabled={!s.available} onClick={()=>focus(s.layerId,s.closestPointRasMm??s.nearestPoint)}>
          <span>{s.label}<small>{s.kind==='tractography'?'Supplied streamlines':s.kind}{s.excluded?' · excluded':''}</small><small>{s.coverage?.wholeEnvelopeWithinField===false||s.acquisitionCoverage?.wholeEnvelopeWithinField===false?'Partial field · outside unassessed':!s.available?s.reason:undefined}</small></span>
          <span className={s.confirmedIntersection?'is-conflict':''}>{!s.available?'Unassessed':s.confirmedIntersection?'Intersects':s.intersectsOrUnresolved?'Unresolved contact':'No detected contact'}<small>{s.available&&`${bound(s.clearanceLowerBoundMm)} mm lower bound`}</small></span>
        </button>)}</div>
        <details><summary>Source coverage and provenance</summary>{m.structures.map(s=><div key={s.layerId} className="approach-source-row"><h3>{s.label}</h3><p>{s.sourceSeries} · {s.reviewStatus??'Review status unknown'}</p><p>{s.coverageMeaning??s.reason??'Coverage unassessed'}</p>
          <dl className="approach-value-list"><div><dt>Label or streamline grid</dt><dd>{fieldCoverage(s.coverage)}</dd></div><div><dt>Acquired field of view</dt><dd>{fieldCoverage(s.acquisitionCoverage)}</dd></div></dl>
          {s.sourceIdentity?.dataSha256&&<p className="approach-identity">Source SHA-256 {s.sourceIdentity.dataSha256}</p>}</div>)}</details>
        <details><summary>Target reach & tissue traversal</summary>
          {m.targets.map((t,i)=><p key={i}>{t.label??'Target'} · {t.available&&t.coverage?.wholeEnvelopeWithinField&&t.acquisitionCoverage?.wholeEnvelopeWithinField
            ?`${mm(Number(t.coveredVoxelCenterFraction)*100)}% of labelled voxel centres approached, ${mm(t.coveredVoxelCenterVolumeMm3)} mm³.`
            :t.available?'Target reach unassessed over part or all of the capsule.':'Target reach unassessed.'}
            {t.available&&` ${t.targetPointInsideLabel?'Endpoint inside the supported target label.':'Endpoint outside or unassessed in the target label.'}`}</p>)}
          <p>Supplied brain mask traversal {m.normalTraversal.available?`${mm(m.normalTraversal.conservativeVolumeUpperBoundMm3)} mm³ whole-cell non-target upper bound`:'unassessed'}.</p>
          <dl className="approach-value-list"><div><dt>Supported non-target voxel centres</dt><dd>{typeof m.normalTraversal.supportedNonTargetVoxelCenterVolumeMm3==='number'?`${mm(m.normalTraversal.supportedNonTargetVoxelCenterVolumeMm3)} mm³`:'Unknown'}</dd></div>
            <div><dt>Unclassified brain voxel centres</dt><dd>{typeof m.normalTraversal.unclassifiedBrainVoxelCenterCount==='number'?m.normalTraversal.unclassifiedBrainVoxelCenterCount.toLocaleString():'Unknown'}</dd></div>
            <div><dt>Brain mask field</dt><dd>{fieldCoverage(m.normalTraversal.coverage)}</dd></div><div><dt>Acquired brain scan field</dt><dd>{fieldCoverage(m.normalTraversal.acquisitionCoverage)}</dd></div></dl>
          <p>Simulated treatment, tissue removal, brain shift and functional outcome are unassessed.</p>
        </details>
        {m.sensitivity&&<ApproachSensitivityDetails sensitivity={m.sensitivity} scenarioSet={p.report?.scenarioSet} labels={layerLabels} referenceBound={!!referenceBound}/>}
      </>}
      {assessment.limitations.length>0&&<div className="approach-limitations"><h3>Evidence limitations</h3><p>{assessment.limitations[0]}</p><details><summary>All {assessment.limitations.length} limitations & assumptions</summary>{assessment.limitations.map((text,i)=><p key={i}>{text}</p>)}{(m?.assumptions??[]).map((text,i)=><p key={`assumption-${i}`}>{text}</p>)}</details></div>}
      {!m&&<p className="approach-caption">Reassess this path against the current source anatomy to display geometric measurements.</p>}
    </section>}
    <section className="approach-input-comparison"><h2>Development input comparison</h2>
      <p className="approach-caption">Run the same target, patch and candidate budget with rich and restricted inputs, then inspect both through shared source-geometry checks.</p>
      <button disabled={blocked||!valid||!targetLayerId} onClick={()=>void runComparison('compareInputs')}>Compare input sets</button>
      {p.inputComparison&&<ApproachInputComparison report={p.inputComparison} labels={layerLabels} targetLabel={targetLabel(p.inputComparison.task?.targetLayerId)}/>}
    </section>
    <button className="approach-save" disabled={blocked||(!selected&&!p.report&&!p.inputComparison)||!valid} onClick={async()=>{if(await commitEdits())await w.save()}}><ArrowDownToLine size={14}/>Save study & comparison</button>
    <details className="approach-structure-disclosure"><summary>Structures & named tract legend</summary><NativeStructureControls volume={w.volume} visibleLayers={w.state.visible} onVisibilityChange={w.setVisible} settings={w.state.settings} onSettingsChange={w.setSettings} selectedStructure={w.selectedStructure} onSelectStructure={w.setSelectedStructure} onFocusStructure={w.focusStructure}/></details>
  </div>;
}
