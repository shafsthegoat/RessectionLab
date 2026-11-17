import {useEffect,useState} from 'react';
import {ArrowDownToLine,Focus,Plus,Sparkles} from 'lucide-react';
import type {NativeWorkspace} from './use-native-workspace';
import type {Point3} from './viewer/coordinates';
import type {NativeApproach} from './native-approaches';
import {approachAssessment,approachStatus,lowerBoundText} from './native-approaches';
import {NativeStructureControls} from './NativeStructureControls';
import './native-approaches.css';
const mm=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(1):'—';
const bound=lowerBoundText;
type EditorFields={entry:Point3;target:Point3;radius:number;patchRadius:number;minimum:number;maximum:number};
function editorFields(selected:NativeApproach|null,entry:Point3,target:Point3):EditorFields{
  return {entry:[...(selected?.entry??entry)],target:[...(selected?.target??target)],radius:selected?.radiusMm??2,
    patchRadius:selected?.entryPatch?.radiusMm??15,minimum:selected?.constraints.minClearanceMm??0,maximum:selected?.constraints.maxLengthMm??150};
}
function PointEditor({label,value,onChange}:{label:string;value:Point3;onChange:(p:Point3)=>void}){return <div className="approach-point"><span>{label} · RAS mm</span><div>{['R','A','S'].map((axis,i)=><label key={axis}>{axis}<input aria-label={`${label} ${axis}`} type="number" step="1" value={Number(value[i].toFixed(2))} onChange={e=>onChange(value.map((v,j)=>j===i?+e.target.value:v) as Point3)}/></label>)}</div></div>}
export function NativeApproachPanel({workspace:w}:{workspace:NativeWorkspace}){
  const p=w.planning,selected=p.selected,b=w.binding;
  const editorKey=`${b?.collectionId}/${b?.document.snapshot_id}/${selected?.id}/${selected?.parameterHash}`;
  const initialTarget=b?.document.display?.cursor_ras_mm??w.state.cursor??[0,0,40];
  const initialEntry=b?.document.display?.entry_ras_mm??[initialTarget[0],initialTarget[1],initialTarget[2]+40];
  const initial=()=>({key:editorKey,fields:editorFields(selected,initialEntry,initialTarget),dirty:false,attempted:false});
  const [editor,setEditor]=useState(initial);
  // A newly selected/source-revised object is also used on its first render,
  // before passive effects run, so an old draft cannot enter its request.
  const current=editor.key===editorKey?editor:initial(),fields=current.fields;
  const {entry,target,radius,patchRadius,minimum,maximum}=fields,dirty=!!selected&&current.dirty;
  useEffect(()=>{setEditor(initial())},[editorKey]);
  const blocked=w.busy||!!w.draft||p.pending;
  const valid=[...entry,...target,radius,patchRadius,minimum,maximum].every(Number.isFinite)&&radius>0&&radius<=50&&patchRadius>=0&&patchRadius<=40&&minimum>=0&&minimum<=50&&maximum>=.1&&maximum<=1000&&Math.hypot(...target.map((v,i)=>v-entry[i]))>1e-6;
  function edit(change:Partial<EditorFields>){setEditor({...current,fields:{...fields,...change},dirty:true,attempted:false})}
  const payload=()=>{
    const length=Math.hypot(...target.map((v,i)=>v-entry[i]));
    return {...(selected??{}),name:selected?.name??'Manual approach',entry,target,radiusMm:radius,
      entryPatch:{...selected?.entryPatch,center:selected?.entryPatch?.center??entry,normal:selected?.entryPatch?.normal??target.map((v,i)=>(v-entry[i])/length) as Point3,radiusMm:patchRadius},
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
  async function suggest(){
    if(!await commitEdits())return;
    const value=payload();
    const result=await p.request({action:'suggest',start:entry,end:target,radiusMm:radius,entryPatch:value.entryPatch,patchRadiusMm:patchRadius,constraints:value.constraints});
    if(result?.approaches.length){w.setCursor(result.approaches.find(a=>a.id===result.selectedId)?.target??result.approaches[0].target);w.setSelectedStructure(null)}
  }
  async function selectApproach(a:NativeApproach){if(await commitEdits()){p.select(a.id);w.setCursor(a.target)}}
  const m=selected?.assessmentStatus==='current'?selected.measurements:undefined,status=selected?approachStatus(selected):'unassessed',assessment=selected?approachAssessment(selected):null;
  const focus=(id:string,point?:unknown)=>{w.setVisible({...w.state.visible,[id]:true});w.focusStructure(id,Array.isArray(point)&&point.length===3?point as Point3:undefined)};
  const reasons=p.report?.reasons;
  return <div className="native-approaches unified-sidebar" data-approach-pending={p.pending} data-approach-dirty={dirty}>
    <div className="approach-intro"><span className="eyebrow">ANATOMICAL PLANNING</span><h1>Compare approaches</h1><p>Healthy acquired anatomy · synthetic target. Static geometric inspection; clinical outcomes unassessed.</p></div>
    <div className="approach-primary-actions"><button className="approach-suggest" disabled={blocked||!valid} onClick={()=>void suggest()}><Sparkles size={14}/>Suggest approaches</button><button aria-label="Create manual approach" disabled={blocked||!valid} onClick={async()=>{if(await commitEdits())void p.request({action:'upsert',approach:{...payload(),id:undefined,name:`Manual ${p.approaches.length+1}`}})}}><Plus size={14}/>Manual</button></div>
    <p className="approach-caption">Up to three distinct paths within the selected supratentorial entry patch. Compare observations, evidence coverage and declared constraints separately.</p>
    {p.error&&<p className="approach-error" role="alert">{p.error}{dirty?' Edits remain unsaved; use Save to retry.':''}</p>}
    {!valid&&<p className="approach-error" role="alert">Use distinct finite entry and target coordinates, a positive corridor radius ≤50 mm, an entry patch from 0–40 mm, minimum clearance from 0–50 mm and maximum length from 0.1–1000 mm.</p>}
    {p.pending&&<p role="status" className="approach-computing">Measuring source anatomy…</p>}
    <div className="approach-cards" aria-label="Saved approach comparison">{p.approaches.map((a,index)=>{const summary=approachAssessment(a),measured=a.assessmentStatus==='current'?a.measurements:undefined;return <button className={`approach-card ${a.id===p.selectedId?'selected':''}`} key={a.id} aria-pressed={a.id===p.selectedId} data-approach-id={a.id} disabled={blocked||(dirty&&!valid)} onClick={()=>void selectApproach(a)}>
      <span className="approach-card-heading"><b>{String.fromCharCode(65+index)} · {a.name}</b><span>{mm(measured?.lengthMm)} mm</span></span>
      <span className={`approach-observed ${summary.observed}`} data-observed-result={summary.observed}>{summary.observedLabel}</span>
      <span className="approach-evidence" data-assessment-completeness={summary.completeness}>{summary.completenessLabel} · {summary.assessedCount}/{summary.totalCount} structures measured{summary.limitations.length?` · ${summary.limitations.length} limitations`:''}</span>
      {summary.intersections.length>0&&<span className="approach-card-concern">Intersects: {summary.intersections.map(s=>s.label).join(', ')}</span>}
      {summary.unavailable.length>0&&<span>Unavailable: {summary.unavailable.map(s=>s.label).join(', ')}</span>}
      {summary.unresolved.length>0&&<span>Unresolved contact: {summary.unresolved.map(s=>s.label).join(', ')}</span>}
      {summary.constraintFailures.length>0&&<span className="approach-constraint-failure">{summary.constraintFailures.length} geometric constraint {summary.constraintFailures.length===1?'failure':'failures'}: {summary.constraintFailures[0]}</span>}
      <span className="approach-card-metric">Radius <b>{mm(a.radiusMm)} mm</b> · minimum clearance bound <b>{bound(summary.minimumBoundMm)} mm</b></span>
      <span className="approach-card-metric">Target voxel centres reached <b>{mm(summary.targetReachedMm3)} mm³</b> · brain traversal upper bound <b>{mm(summary.traversalMm3)} mm³</b></span>
    </button>})}</div>
    {p.report&&<details className="approach-proposal-report"><summary>{p.report.accepted} alternatives retained · proposal details</summary><p>{p.report.selectionRule}</p>{p.report.whyFewer&&<p>{p.report.whyFewer}</p>}<p>{p.report.evaluated} candidates measured; {p.report.rejected} rejected.</p>{typeof reasons==='string'?<p>{reasons}</p>:reasons&&typeof reasons==='object'?<ul>{Object.entries(reasons).map(([reason,count])=><li key={reason}>{reason}: {String(count)}</li>)}</ul>:null}</details>}
    <section className="approach-editor"><fieldset disabled={blocked}><div className="approach-section-heading"><h2>{selected?'Edit selected approach':'Set an entry and target'}</h2>{selected&&<button aria-label="Focus selected target" onClick={()=>w.setCursor(target)}><Focus size={13}/></button>}</div>
      <label className="approach-radius">Corridor radius <output>{mm(radius)} mm</output><input aria-label="Corridor radius" type="range" min=".5" max={Math.max(12,radius)} step=".5" disabled={blocked} value={radius} onChange={e=>edit({radius:+e.target.value})}/></label>
      <div className="approach-cursor-actions"><button disabled={blocked||!w.state.cursor} onClick={()=>edit({entry:w.state.cursor!})}>Entry at cursor</button><button disabled={blocked||!w.state.cursor} onClick={()=>edit({target:w.state.cursor!})}>Target at cursor</button></div>
      <details><summary>Coordinates, entry patch & constraints</summary><PointEditor label="Entry" value={entry} onChange={entry=>edit({entry})}/><PointEditor label="Target" value={target} onChange={target=>edit({target})}/><label>Allowed entry patch · mm<input aria-label="Entry patch radius" type="number" min="0" max="40" value={patchRadius} onChange={e=>edit({patchRadius:+e.target.value})}/></label><label>Minimum conservative clearance · mm<input aria-label="Minimum clearance" type="number" min="0" max="50" step=".5" value={minimum} onChange={e=>edit({minimum:+e.target.value})}/></label><label>Maximum corridor length · mm<input aria-label="Maximum corridor length" type="number" min=".1" max="1000" value={maximum} onChange={e=>edit({maximum:+e.target.value})}/></label><p className="approach-caption">Saved-path edits recompute automatically. The patch centre and plane stay fixed when the entry moves; an entry outside that disk is a geometric constraint failure.</p></details>
      {selected&&<div className="tool-row"><button disabled={blocked||!valid} onClick={async()=>{if(await commitEdits())void p.request({action:'assess',approachId:selected.id})}}>Reassess</button><button disabled={blocked} onClick={()=>void p.request({action:'delete',approachId:selected.id})}>Delete</button></div>}
    </fieldset></section>
    {assessment&&<section className="approach-measurements" data-assessment-status={status}><h2>Source-bound assessment</h2>
      <p className={`approach-observed ${assessment.observed}`} data-observed-result={assessment.observed}>{assessment.observedLabel}</p>
      <p className="approach-evidence" data-assessment-completeness={assessment.completeness}>{assessment.completenessLabel} · {assessment.assessedCount}/{assessment.totalCount} structures measured.</p>
      {assessment.constraintFailures.length>0&&<div className="approach-constraint-failures" role="status"><h3>Geometric constraint failures</h3><ul>{assessment.constraintFailures.map((failure,i)=><li key={i}>{failure}</li>)}</ul></div>}
      {m&&<><div className="approach-metrics"><span>Length<b>{mm(m.lengthMm)} mm</b></span><span>Azimuth / elevation<b>{mm(m.orientation.azimuthDeg)}° / {mm(m.orientation.elevationDeg)}°</b></span><span>Whole envelope in reference<b>{m.coverage.withinReferenceEnvelope?'Yes':'No · unassessed outside'}</b></span></div>
      <div className="approach-concerns">{[...m.structures].sort((a,b)=>Number(!!b.confirmedIntersection)-Number(!!a.confirmedIntersection)||(a.clearanceLowerBoundMm??Infinity)-(b.clearanceLowerBoundMm??Infinity)).map(s=><button key={s.layerId} disabled={!s.available} onClick={()=>focus(s.layerId,s.nearestPoint)}><span>{s.label}<small>{s.kind==='tractography'?'Supplied streamlines':s.kind}</small></span><span className={s.confirmedIntersection?'is-conflict':''}>{!s.available?'Unassessed':s.confirmedIntersection?'Intersects':`${bound(s.clearanceLowerBoundMm)} mm`}<small>{s.available?s.intersectsOrUnresolved&&!s.confirmedIntersection?'Unresolved contact · lower bound':'Clearance lower bound':''}</small></span></button>)}</div>
      <details><summary>Target reach & tissue traversal</summary>{m.targets.map((t,i)=><p key={i}>{String(t.label??'Target')}: {t.available?`${mm(Number(t.coveredVoxelCenterFraction)*100)}% of voxel centres in capsule; target endpoint ${t.targetPointInsideLabel?'inside':'outside'} label.`:'Unassessed'}</p>)}<p>Brain traversal: {m.normalTraversal.available?`${mm(m.normalTraversal.conservativeVolumeUpperBoundMm3)} mm³ whole-cell upper bound`:'Unassessed'}.</p><p>Geometric reach estimates do not predict removal, brain shift, or preserved function.</p></details></>}
      {assessment.limitations.length>0&&<div className="approach-limitations"><h3>Evidence limitations</h3><p>{assessment.limitations.slice(0,2).join('; ')}.</p><details><summary>All {assessment.limitations.length} limitations & assumptions</summary>{assessment.limitations.map((text,i)=><p key={i}>{text}</p>)}{(m?.assumptions??[]).map((text,i)=><p key={`assumption-${i}`}>{text}</p>)}</details></div>}
      {!m&&<p className="approach-caption">Reassess this path against the current source anatomy to display geometric measurements.</p>}
    </section>}
    <button className="approach-save" disabled={blocked||!selected||!valid} onClick={async()=>{if(await commitEdits())await w.save()}}><ArrowDownToLine size={14}/>Save study & comparison</button>
    <details className="approach-structure-disclosure"><summary>Structures & named tract legend</summary><NativeStructureControls volume={w.volume} visibleLayers={w.state.visible} onVisibilityChange={w.setVisible} settings={w.state.settings} onSettingsChange={w.setSettings} selectedStructure={w.selectedStructure} onSelectStructure={w.setSelectedStructure} onFocusStructure={w.focusStructure}/></details>
  </div>;
}
