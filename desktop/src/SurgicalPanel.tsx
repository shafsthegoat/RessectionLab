import {useEffect,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {Euler,Quaternion} from 'three';
import type {SurgicalController} from './use-surgical';
import type {SurgicalRegion,SurgicalImportOptions,SurgicalOutcome,SurgicalView} from './surgical-types';
import './SurgicalPanel.css';
import {ResearchPlanningPanel} from './ResearchPlanningPanel';
import {summarizeSurgicalEvents} from './surgical-types.ts';
const materials=['unknown','bone','soft_tissue','tumor','membrane','vessel','fluid','functional_region'];
function Numeric({value,onChange,label,min,max,step=.1}:{value:number;onChange:(n:number)=>void;label:string;min?:number;max?:number;step?:number}){
  const [draft,setDraft]=useState(String(+value.toFixed(3))),focused=useRef(false);
  useEffect(()=>{if(!focused.current)setDraft(String(+value.toFixed(3)));},[value]);
  return <input aria-label={label} type="number" min={min} max={max} step={step} value={draft} onFocus={()=>{focused.current=true;}} onChange={e=>{setDraft(e.target.value);const n=e.target.valueAsNumber;if(Number.isFinite(n)&&(min===undefined||n>=min)&&(max===undefined||n<=max))onChange(n);}} onBlur={()=>{focused.current=false;setDraft(String(+value.toFixed(3)));}}/>;
}
const parameterSpecs:Record<string,{label:string;min:number;max:number;default:number}>={rpm:{label:'Drill speed · rpm',min:0,max:80000,default:40000},forceN:{label:'Contact force · N',min:0,max:20,default:1},duty:{label:'Drill duty fraction',min:0,max:1,default:1},amplitudeMa:{label:'Stimulus amplitude · mA',min:0,max:20,default:1},frequencyHz:{label:'Frequency · Hz',min:0,max:500,default:60},pulseWidthUs:{label:'Pulse width · µs',min:0,max:1000,default:100},radiusMm:{label:'Tip radius · mm',min:.1,max:12,default:2},depthMm:{label:'Working depth · mm',min:.1,max:20,default:2},flowMm3PerSecond:{label:'Flow · mm³/s',min:0,max:1000,default:20},power:{label:'Power · model units',min:0,max:10,default:1},jawOpeningMm:{label:'Jaw opening · mm',min:0,max:12,default:2}};

function ActionResult({outcome,title}:{outcome:Pick<SurgicalOutcome,'reason'|'events'>&{status:string;completedFraction?:number};title:string}){
  const summaries=summarizeSurgicalEvents(outcome.events??[]);
  return <section aria-label={title} className="surgical-preview" role="status"><strong>{title} · {outcome.status}</strong>
    {Number.isFinite(outcome.completedFraction)&&<p>{+(100*outcome.completedFraction!).toFixed(1)}% of requested action completed</p>}
    {outcome.reason&&<p>{outcome.reason}</p>}
    {summaries.length?<ul>{summaries.map(s=><li key={s.kind}>{s.label}{s.count>1?` (${s.count} samples)`:''}{s.details.length?`: ${s.details.join(' · ')}`:''}</li>)}</ul>:<p>{outcome.reason?'No effect events were recorded.':'No effect events were reported; the engine supplied no further effect reason.'}</p>}
  </section>;
}
const displayValue=(value:unknown):string=>value===undefined||value===null?'Not supplied':typeof value==='object'?JSON.stringify(value):String(value).replaceAll('_',' ');
function StimulationObservation({observation,preview}:{observation:Record<string,unknown>;preview:boolean}){
  const evidence=Array.isArray(observation.evidence)?observation.evidence:[];
  return <section aria-label="Latest stimulation observation" className="surgical-preview"><strong>{preview?'Preview':'Latest recorded'} stimulation observation</strong>
    <dl>{[['Probe',observation.probe],['Result',observation.result??'unknown'],['Coverage',observation.coverage??'unknown'],['Uncertainty',observation.uncertainty??'not supplied'],['Position · RAS mm',observation.position],['Contact cells',Array.isArray(observation.cells)?observation.cells.length:undefined]].map(([label,value])=><div key={String(label)}><dt>{String(label)}</dt><dd>{displayValue(value)}</dd></div>)}</dl>
    <p>Stimulus: {Object.entries((observation.parameters??{}) as Record<string,unknown>).map(([key,value])=>`${parameterSpecs[key]?.label??key}: ${displayValue(value)}`).join(' · ')||'Parameters not supplied'}</p>
    {evidence.length?<details><summary>Supplied evidence · {evidence.length} records</summary><ol>{evidence.map((record,i)=><li key={i}>{record&&typeof record==='object'?<dl>{Object.entries(record).map(([key,value])=><div key={key}><dt>{key.replaceAll('_',' ')}</dt><dd>{displayValue(value)}</dd></div>)}</dl>:displayValue(record)}</li>)}</ol></details>:<p>No matching supplied functional evidence. Unknown does not establish a negative stimulation response.</p>}
  </section>;
}

function DeclaredFields({data}:{data:Record<string,unknown>}){
  return <dl className="surgical-declared-fields">{Object.entries(data).map(([key,value])=><div key={key}><dt>{key.replace(/([a-z])([A-Z])/g,'$1 $2').replaceAll('_',' ')}</dt><dd>{value&&typeof value==='object'&&!Array.isArray(value)?<details><summary>Declared fields</summary><DeclaredFields data={value as Record<string,unknown>}/></details>:displayValue(value)}</dd></div>)}</dl>;
}
function SceneResearchDetails({view,preview}:{view:SurgicalView;preview:boolean}){
  const source=view.scene.source,adapter=source?.adapter,physics=view.physics??view.scene.physics;
  const v2=physics?.profile?.schema==='resectionlab.tissue-physics.v2';
  const patientDerivative=adapter?.type==='native-patient-research-derivative';
  const mapped=new Set(view.scene.regions.map(r=>r.material));
  const metrics:Record<string,unknown>={...view.metrics,...physics?.metrics};
  const metricLabels:Record<string,string>={severedVesselConnections:'Severed local vessel adjacencies',activeVesselWounds:'Active modeled wound outlets',modeledVesselFluidGeneratedMm3:'Modeled vessel fluid generated · mm³',modeledVesselReservoirMm3:'Modeled vessel fluid retained · mm³'};
  const wounds=physics?.wounds??view.sources?.filter(s=>s.kind==='modeled-vessel-wound');
  return <section aria-label="Scene provenance and research behavior" className="surgical-research-context">
    {patientDerivative&&<div className="surgical-warning"><strong>Patient-derived research estimate</strong><p>This scene is derived from patient anatomy. Material properties and flow behavior are declared research assumptions, not measured patient physiology. Derived geometry is separate from source annotations.</p></div>}
    {adapter?.warnings?.length? <ul className="surgical-warning">{adapter.warnings.map((warning,i)=><li key={i}>{warning}</li>)}</ul>:null}
    <p className="surgical-note">Scene grid: {view.scene.resolutionMm.map(n=>+n.toFixed(3)).join(' × ')} mm · {view.scene.shape.join(' × ')} cells.</p>
    {v2&&<p className="surgical-note">V2 research mechanics: coupled brain relaxation, work and heat based drilling, and pressure-driven clustered bleeding. Parameters are uncalibrated assumptions; the volume penalty is approximate and global perfusion is unknown.</p>}
    <details><summary>Provenance, material behavior, and assumptions</summary>
      {source&&<DeclaredFields data={Object.fromEntries(Object.entries(source).filter(([key])=>key!=='adapter'))}/>}
      {adapter&&<details><summary>Source adapter and declared patient source</summary><DeclaredFields data={adapter}/></details>}
      <ul className="surgical-note">
        {mapped.has('bone')&&<li>Bone: rigid cells; {v2?'drilling accumulates work before removal and models heat generation and cooling.':'drilling and kerf cutting remove cells and produce modeled debris.'}</li>}
        {(mapped.has('soft_tissue')||mapped.has('tumor'))&&<li>Brain / soft tissue and tumor: {v2?'coupled local tissue deformation with timed relaxation after release, distinct assumed stiffness, and an approximate volume penalty.':'lattice patch displacement and compatible-tool removal; no calibrated deformation law.'}</li>}
        {mapped.has('vessel')&&<li>Vessels: local adjacency cutting retains pieces. {v2?'Clustered wound outlets use assumed pressure, finite vascular reserves, and refill; the fluid budget is tracked.':physics?.capabilities?.damageTriggeredVesselWounds===true?'Declared damage-triggered wound outlets use an assumed local flow proxy.':'Damage-triggered vessel wound behavior is not declared as enabled.'} Global perfusion is unknown.</li>}
        {mapped.has('membrane')&&<li>Membranes: retained cut seams and graph separation; no fabricated physical aperture.</li>}
        {mapped.has('unknown')&&<li>Unknown material remains explicitly unmapped; no tissue behavior is inferred from its label.</li>}
      </ul>
      {view.scene.approximations.length>0&&<ul className="surgical-note">{view.scene.approximations.map((a,i)=><li key={i}>{a}</li>)}</ul>}
      {physics?<><p>Declared research physics: {physics.enabled===true?'enabled':physics.enabled===false?'disabled':'enablement not supplied'}.</p><DeclaredFields data={physics.profile??Object.fromEntries(Object.entries(physics).filter(([key])=>!['wounds','metrics','limits','capabilities'].includes(key)))}/>{physics.limits?.length?<ul className="surgical-note">{physics.limits.map((limit,i)=><li key={i}>{limit}</li>)}</ul>:null}{physics.capabilities&&<details><summary>Declared capabilities</summary><DeclaredFields data={physics.capabilities}/></details>}</>:<p className="surgical-note">No material physics profile was supplied. No patient physiology is inferred.</p>}
    </details>
    {(physics?.capabilities?.damageTriggeredVesselWounds===true||!!wounds?.length)&&<details><summary>{preview?'Preview':'Recorded'} modeled vessel wounds / fluid</summary><p className="surgical-note">Equivalent wound outlets describe the declared local model. These volumes are not estimates of patient blood loss; global perfusion remains unknown.</p><dl className="surgical-declared-fields">{Object.entries(metricLabels).map(([key,label])=><div key={key}><dt>{label}</dt><dd>{displayValue(metrics[key])}</dd></div>)}</dl>{wounds?.length?<details><summary>Wound outlet records · {wounds.length}</summary>{wounds.map((wound,i)=><DeclaredFields key={String(wound.id??i)} data={wound}/>)}</details>:<p>No modeled wound records supplied for this state.</p>}{view.vesselConnections&&Object.keys(view.vesselConnections).length>0&&<details><summary>Local severed connections · {Object.keys(view.vesselConnections).length}</summary><DeclaredFields data={view.vesselConnections}/></details>}{view.vesselWounds&&Object.keys(view.vesselWounds).length>0&&<details><summary>Declared wound interpretations</summary><DeclaredFields data={view.vesselWounds}/></details>}</details>}
  </section>;
}

function MechanicsV2Details({view,preview}:{view:SurgicalView;preview:boolean}){
  const drill=view.drillingV2,brain=view.deformationModel,vessels=view.vesselsV2;
  const sum=(values:Record<string,number>|undefined)=>values?Object.values(values).reduce((a,b)=>a+b,0):undefined;
  const displacement=view.deformation?Math.max(0,...Object.values(view.deformation).map(v=>Math.hypot(...v))):undefined;
  const temperature=drill&&Object.keys(drill.temperatureC).length?Object.values(drill.temperatureC).reduce((max,value)=>Math.max(max,value),-Infinity):view.metrics.drillMaxTemperatureC;
  const rows:Array<[string,unknown]>=[
    ['Drill input energy · J',drill?.inputEnergyJ??view.metrics.drillInputEnergyJ],
    ['Pending cutting work · J',sum(drill?.workJ)??view.metrics.drillPendingWorkJ],
    ['Generated heat · J',drill?.heatGeneratedJ??view.metrics.drillHeatGeneratedJ],
    ['Maximum modeled temperature · °C',temperature],
    ['Brain elastic energy · N·mm',brain?.elasticEnergyNmm],
    ['Active deformation cells',brain?.activeNodes??brain?.active?.length],
    ['Maximum displacement · mm',displacement],
    ['Maximum neighbor displacement / spacing',brain?.maxNeighborDisplacementRatio],
    ['Vascular pressure · Pa',vessels?.pressurePa],
    ['Vascular reserve remaining · mm³',vessels?.remainingMm3],
    ['Vascular refill · mm³',vessels?.refilledMm3],
    ['Generated wound fluid · mm³',vessels?.generatedMm3],
    ['Vascular budget residual · mm³',vessels?vessels.initialMm3+vessels.refilledMm3-vessels.remainingMm3-vessels.generatedMm3:undefined],
  ];
  return <details><summary>{preview?'Preview':'Recorded'} V2 mechanics and fluid budget</summary><dl className="surgical-declared-fields">{rows.map(([label,value])=><div key={label}><dt>{label}</dt><dd>{typeof value==='number'&&Number.isFinite(value)?+value.toFixed(5):displayValue(value)}</dd></div>)}</dl><p className="surgical-note">The vascular budget compares initial reserve plus refill against remaining reserve and generated fluid. These modeled quantities do not estimate patient blood loss or predict tissue injury.</p></details>;
}

export function SurgicalPanel({c,onFixture,onImport,available,anatomyControls}:{anatomyControls?:ReactNode;c:SurgicalController;onFixture:()=>void;onImport?:(options:SurgicalImportOptions)=>Promise<void>|void;available:boolean}){
  const [importOptions,setImportOptions]=useState<SurgicalImportOptions>({frame:'RAS+',units:'mm',material:'unknown',resolution:1,topology:'solid',thicknessMm:1});
  const [importing,setImporting]=useState(false),[importError,setImportError]=useState<string|null>(null);
  const [search,setSearch]=useState(''),[duration,setDuration]=useState(1),[operation,setOperation]=useState('apply'),[playing,setPlaying]=useState(false);
  const [mapping,setMapping]=useState<SurgicalRegion[]|null>(null);
  const v=c.view,selected=v?.slots[c.slot],tool=v?.catalog.find(t=>t.id===selected?.instrument),r=c.result;
  const displayed=c.preview??v;
  const v2=v?.physics?.profile?.schema==='resectionlab.tissue-physics.v2';
  const latestOutcome=c.preview?c.previewOutcome:r?.outcome??v?.history.at(-1);
  const observation=displayed?.observations?.at(-1);
  const q=new Euler().setFromQuaternion(new Quaternion(...c.pose.quaternion));
  const steps=r?.branches.find(b=>b.id===r.branchId)?.steps??0;
  useEffect(()=>{setMapping(null);setPlaying(false);setImportError(null);setOperation('apply');setDuration(1);},[v?.scene.sceneId]);
  useEffect(()=>{if(!playing||c.pending)return;if(!r||r.position>=steps){setPlaying(false);return;}const timer=setTimeout(()=>{void c.run('seek',{position:r.position+1}).then(ok=>{if(!ok)setPlaying(false);});},600);return()=>clearTimeout(timer);},[playing,c.pending,r?.position,r?.branchId,steps]);
  const keys=['radiusMm',...(v2&&tool?.id==='cranial_drill'?['rpm','forceN','duty']:v2&&tool?.materials.includes('vessel')?['forceN']:[]),...(tool?.effect==='probe'?['amplitudeMa','frequencyHz','pulseWidthUs']:[]),...(['incise','kerf','sheet_cut','transect','irrigate'].includes(tool?.effect??'')?['depthMm']:[]),...(['suction','irrigate'].includes(tool?.effect??'')?['flowMm3PerSecond']:[]),...(tool?.effect==='coagulate'?['power']:[]),...(['forceps','scissors'].includes(tool?.geometry??'')?['jawOpeningMm']:[])];
  const params=Object.fromEntries(keys.map(k=>[k,c.parameters[k]??(k==='radiusMm'?tool?.radiusMm:undefined)??parameterSpecs[k].default]));
  const compatible=tool?.materials.length===0||v?.scene.regions.some(row=>tool?.materials.includes(row.material));
  const changeMap=(fn:(rows:SurgicalRegion[])=>void)=>{const rows=structuredClone(mapping??v?.scene.regions??[]);fn(rows);setMapping(rows);};
  const stopRun=(command:string,extra:Record<string,unknown>={})=>{setPlaying(false);void c.run(command,extra);};
  return <section className="surgical-panel" aria-label="Surgical instrument toolbox"><span className="eyebrow">SURGICAL WORKBENCH</span><h2>Instrument toolbox</h2>
    <p className="surgical-note">Research rehearsal · source anatomy preserved. Mechanics remain uncalibrated.</p>
    <button disabled={c.pending||importing} onClick={onFixture}>Open engineering test scene</button>
    {onImport&&<details><summary>Import segmentation / geometry</summary><p className="surgical-note">Choose NIfTI labels (.nii/.nii.gz), OBJ, or STL. Declare the source coordinates and units before selecting a file. Imported segmentation has no acquired MRI intensity image.</p><fieldset disabled={c.pending||importing} className="surgical-controls">
      <label>Source frame<select aria-label="Import source frame" value={importOptions.frame} onChange={e=>setImportOptions({...importOptions,frame:e.target.value as SurgicalImportOptions['frame']})}><option value="RAS+">RAS+</option><option value="LPS+">LPS+</option></select></label>
      <label>Source units<select aria-label="Import source units" value={importOptions.units} onChange={e=>setImportOptions({...importOptions,units:e.target.value as SurgicalImportOptions['units']})}><option value="mm">Millimeters</option><option value="meter">Meters</option><option value="micron">Microns</option></select></label>
      <label>Initial material<select aria-label="Import material" value={importOptions.material} onChange={e=>setImportOptions({...importOptions,material:e.target.value})}>{materials.map(m=><option key={m}>{m}</option>)}</select></label>
      <p className="surgical-note">NIfTI nonzero labels start with this material; edit individual regions after preparation. Mesh settings below do not resample a NIfTI grid.</p>
      <label>Mesh topology<select aria-label="Import mesh topology" value={importOptions.topology} onChange={e=>setImportOptions({...importOptions,topology:e.target.value as SurgicalImportOptions['topology']})}><option value="solid">Closed solid</option><option value="sheet">Open sheet</option><option value="strand">OBJ line strand</option></select></label>
      <label>Mesh grid · mm<Numeric label="Import mesh resolution" value={importOptions.resolution} min={.01} max={100} onChange={n=>setImportOptions({...importOptions,resolution:n})}/></label>
      {importOptions.topology!=='solid'&&<label>{importOptions.topology==='strand'?'Strand diameter':'Sheet thickness'} · mm<Numeric label="Import thickness" value={importOptions.thicknessMm??1} min={.01} max={100} onChange={n=>setImportOptions({...importOptions,thicknessMm:n})}/></label>}
      <p className="surgical-note">Mesh occupancy uses voxel centers; features smaller than the grid may disappear. Original sheet and strand topology is retained; cuts show graph seams, not a physical opening.</p>
      <button onClick={()=>{setImporting(true);setImportError(null);Promise.resolve().then(()=>onImport({...importOptions,...(importOptions.topology==='solid'?{thicknessMm:undefined}:{})})).catch(e=>setImportError(String(e))).finally(()=>setImporting(false));}}>{importing?'Importing…':'Choose segmentation file…'}</button>
      </fieldset>{importError&&<p role="alert" className="surgical-error">{importError}</p>}</details>}
    {!v?<button className="primary-button" disabled={!available||c.pending||importing} onClick={()=>void c.create()}>Prepare instrument scene</button>:<>
    <section className="environment-summary" aria-label="Environment readiness">
      <strong>{v.scene.environment?.standardized?'Standard V2 research environment':'Saved research environment'}</strong>
      <p>{v.scene.resolutionMm.map(n=>+n.toFixed(3)).join(' × ')} mm · {v2?'V2 mechanics':'Historical mechanics profile'} · RAS+ mm</p>
      <p>{Object.entries(v.scene.environment?.anatomy??{}).map(([role,status])=>`${role}: ${status}`).join(' · ')}</p>
      <p className="surgical-note">Supplied masks do not establish completeness. Missing anatomy and function remain unknown.</p>
      <p className={v.protection?.mode==='strict-research'?'surgical-note':'surgical-warning'} role="status">Protection: {v.protection?.mode??'unconfigured-research'}. {v.protection?.configured?`${v.protection.structures.length} declared exclusion masks; clearance ${v.protection.clearanceMm??0} mm.`:'No critical-structure exclusion contract is bound to this episode.'} Anatomical completeness remains unknown.</p>
      {!!v.protection?.missingRequired.length&&<p className="surgical-warning">Required evidence unavailable: {v.protection.missingRequired.join(', ')}.</p>}
      {!v.scene.environment?.standardized&&<button disabled={c.pending} onClick={()=>void c.create(v.scene.regions)}>Create standard V2 environment</button>}
    </section>
    {anatomyControls}
    <ResearchPlanningPanel c={c} pause={()=>setPlaying(false)}/>
    <details><summary>Source provenance and research assumptions</summary><SceneResearchDetails view={displayed??v} preview={!!c.preview}/></details>
    <div className="surgical-slots">{Object.entries(v.slots).map(([s,t])=><button key={s} disabled={c.pending||playing} aria-pressed={c.slot===s} onClick={()=>c.selectSlot(s)}><strong>{s} hand</strong><span>{v.catalog.find(x=>x.id===t.instrument)?.name}</span><small>{t.inserted?'Inserted':'Withdrawn'} · {t.active?'Active':'Inactive'}</small>{v.holds?.[s]&&<small>{v.holds[s].cells.length?`Holding ${v.holds[s].cells.length} source cell${v.holds[s].cells.length===1?'':'s'}`:'Hold empty · no retained tissue'}</small>}</button>)}</div>
    <details><summary>Exchange instrument</summary><input aria-label="Search instruments" placeholder="Search instruments…" value={search} onChange={e=>setSearch(e.target.value)}/>
    <div className="surgical-toolbox">{v.catalog.filter(t=>(t.name+' '+t.category).toLowerCase().includes(search.toLowerCase())).map(t=><button key={t.id} aria-pressed={selected?.instrument===t.id} disabled={c.pending||playing||selected?.instrument===t.id} onClick={()=>{if(selected?.instrument!==t.id)stopRun('step',{action:{slot:c.slot,operation:'exchange',instrument:t.id,duration:1}})}}><small>{t.category}</small>{t.name}</button>)}</div>
    <p className="surgical-note">Choosing a different instrument commits an exchange. The selected instrument does not advance time.</p></details><h3>{tool?.name}</h3><p className="surgical-note">{tool?.materials.length?`Compatible: ${tool.materials.join(', ')}`:'Acts on modeled fluid and debris.'}</p>
    {!compatible&&<p className="surgical-warning">No compatible material is mapped in this scene. Placement and inspection remain available; tissue effects require compatible material.</p>}
    <fieldset disabled={c.pending||playing} className="surgical-controls">
    <label>Viewport<select aria-label="Instrument interaction" value={c.mode} onChange={e=>c.setMode(e.target.value as typeof c.mode)}><option value="navigate">Navigate anatomy</option><option value="draw">Draw stroke on anatomy</option><option value="translate">Translate tool</option><option value="rotate">Rotate tool</option></select></label>
    <label>Action<select aria-label="Instrument action" value={operation} onChange={e=>{setOperation(e.target.value);c.clearPreview();}}>{['apply','place','move','hold','grasp','extract','activate','deactivate','release','withdraw','inspect'].map(op=><option key={op}>{op}</option>)}</select></label>
    <div className="surgical-vector">{c.pose.position.map((n,i)=><label key={i}>{['R','A','S'][i]} mm<Numeric label={`Tool position ${['R','A','S'][i]}`} step={.5} value={n} onChange={value=>{const p=[...c.pose.position] as [number,number,number];p[i]=value;c.setPose({...c.pose,position:p});}}/></label>)}</div>
    <div className="surgical-vector">{[q.x,q.y,q.z].map((n,i)=><label key={i}>{['Pitch','Yaw','Roll'][i]} °<Numeric label={`Tool ${['pitch','yaw','roll'][i]}`} value={n*180/Math.PI} onChange={value=>{const a=[q.x,q.y,q.z];a[i]=value*Math.PI/180;const b=new Quaternion().setFromEuler(new Euler(a[0],a[1],a[2]));c.setPose({...c.pose,quaternion:b.toArray()});}}/></label>)}</div>
    {keys.map(k=><label key={k}>{parameterSpecs[k].label}<Numeric label={parameterSpecs[k].label} min={parameterSpecs[k].min} max={parameterSpecs[k].max} value={params[k]} onChange={n=>c.setParameters({...params,[k]:n})}/></label>)}
    {['scissors','forceps'].includes(tool?.geometry??'')&&<div className="surgical-actions"><button onClick={()=>c.setParameters({...params,jawOpeningMm:0})}>Close jaws</button><button onClick={()=>c.setParameters({...params,jawOpeningMm:2})}>Open jaws</button><small>Cutting and grasping require a jaw opening of 0.5 mm or less.</small></div>}
    <label>Duration · s<Numeric label="Action duration" min={.01} max={30} value={duration} onChange={n=>{setDuration(n);c.clearPreview();}}/></label>
    <p>{c.path.length} stroke points{c.path.length?' · editing pose replaces this stroke':''} · {Number(v.metrics.elapsedSeconds).toFixed(2)} simulated s</p>
    <div className="surgical-actions"><button onClick={()=>stopRun('preview',{action:c.action(operation,params,duration)})}>Preview</button><button className="primary-button" onClick={()=>stopRun('step',{action:c.action(operation,params,duration)})}>Apply</button><button onClick={c.cancel}>Cancel</button></div>
    </fieldset>
    {c.preview&&<div className="surgical-preview" role="status"><strong>PREVIEW · uncommitted outcome</strong><p>Target removal +{(Number(c.preview.metrics.targetRemovedMm3)-Number(v.metrics.targetRemovedMm3)).toFixed(2)} mm³ · other removal +{(Number(c.preview.metrics.nonTargetRemovedMm3)-Number(v.metrics.nonTargetRemovedMm3)).toFixed(2)} mm³. Apply commits; Cancel returns to recorded anatomy.</p></div>}
    {latestOutcome&&<ActionResult outcome={latestOutcome} title={c.preview?'Preview result':'Last action result'}/>}
    {displayed&&v2&&<MechanicsV2Details view={displayed} preview={!!c.preview}/>}
    {observation&&<StimulationObservation observation={observation} preview={!!c.preview}/>}
    <div className="surgical-actions"><button disabled={c.pending||!r?.position} onClick={()=>stopRun('undo')}>Undo</button><button disabled={c.pending||r?.position===steps} onClick={()=>stopRun('redo')}>Redo</button><button disabled={c.pending} onClick={()=>stopRun('reset')}>Reset</button><button disabled={c.pending} onClick={()=>stopRun('branch')}>Branch</button></div>
    {!!r?.previousScenes?.length&&<label>Previous environment<select aria-label="Previous environment" value="" disabled={c.pending} onChange={e=>{if(e.target.value!=='')stopRun('selectScene',{sceneIndex:+e.target.value});}}><option value="">Reopen a previous environment…</option>{r.previousScenes.map(s=><option key={s.index} value={s.index}>{s.label} · {s.branches} branches</option>)}</select></label>}
    <label>Branch<select disabled={c.pending} value={r?.branchId} onChange={e=>stopRun('selectBranch',{branchId:e.target.value})}>{r?.branches.map(b=><option key={b.id} value={b.id}>{b.id} · {b.steps} actions</option>)}</select></label>
    <div className="surgical-actions"><button disabled={!steps||(!playing&&c.pending)} aria-pressed={playing} onClick={()=>{c.cancel();if(!playing&&r?.position===steps){void c.run('seek',{position:0}).then(ok=>setPlaying(ok));}else setPlaying(!playing);}}>{playing?'Pause replay':'Play replay'}</button><span>Recorded state {r?.position??0} / {steps}</span></div>
    <input aria-label="Surgical timeline" type="range" min="0" max={steps} value={r?.position??0} disabled={c.pending} onChange={e=>stopRun('seek',{position:+e.target.value})}/>
    <p className="surgical-note">Replay seeks recorded states. It does not advance simulation physics.</p>
    <div className="surgical-metrics">{[['targetRemovedMm3','Target removed · mm³'],['nonTargetRemovedMm3','Other removed · mm³'],['contactCellSeconds','Contact · cell·s'],['fluidPresentMm3','Fluid present · mm³']].map(([k,label])=><div key={k}><small>{label}</small><strong>{Number(v.metrics[k]??0).toFixed(2)}</strong></div>)}</div>
    {v.scene.environment&&<p>Target remaining: {Math.max(0,v.scene.environment.target.initialVolumeMm3-Number(v.metrics.targetRemovedMm3??0)).toFixed(2)} mm³ / {v.scene.environment.target.initialVolumeMm3.toFixed(2)} mm³ initial. Target assignment is a research task declaration.</p>}
    <details><summary>{c.preview?'Preview':'Recorded'} effect details</summary><dl>
      {[['Thermal exposure · model units',displayed?.metrics.thermalExposure??Object.values(displayed?.thermal??{}).reduce((a,b)=>a+b,0)],['Fluid added · mm³',displayed?.metrics.fluidAddedMm3??0],['Fluid removed · mm³',displayed?.metrics.fluidRemovedMm3??0],['Fluid escaped · mm³',displayed?.metrics.fluidEscapedMm3??0],['Debris present · mm³',displayed?.metrics.debrisMm3??0],['Grid cut edges',displayed?.cutEdges?.length??0],['Transected grid edges',displayed?.transectedEdges?.length??0],['Mesh cut edges',displayed?.meshCutEdges?.length??0],['Retained mesh seams',displayed?.meshSeams?.length??0],['Separated attachments',displayed?.separatedAttachments?.length??0],['Retained flap components',displayed?.flapComponents?.length??0],['Separable flap components',displayed?.flapComponents?.filter(f=>f.separable===true).length??0],['Flaps with unknown separability',displayed?.flapComponents?.filter(f=>f.separable===null).length??0],...Object.entries(displayed?.slots??{}).map(([slot,state])=>[`${slot} hand held cells`,state.grip.length])].map(([label,value])=><div key={String(label)}><dt>{String(label)}</dt><dd>{typeof value==='number'?+value.toFixed(3):displayValue(value)}</dd></div>)}
    </dl></details>
    <p className="surgical-note">Effect markers: cyan contact · orange heat · blue fluid · gray-brown debris · red first obstruction · gold cut seams / transections. Mesh seams represent retained topology graph cuts, not a physical opening or removed tissue.</p>
    <details><summary>Action timeline</summary><ol>{(r?.timeline??v.history).map((h,i)=><li key={i}><button disabled={c.pending} onClick={()=>stopRun('seek',{position:i+1})}>{i+1}. {h.action.operation} · {h.status}</button>{h.reason&&<small>{h.reason}</small>}</li>)}</ol></details>
    <details><summary>Material mapping · later rows own overlaps</summary><p className="surgical-note">Target assignment changes removal metrics. Unknown regions stay visible. Mapping creates a new scene.</p>{(mapping??v.scene.regions).map((row,i,rows)=><div className="surgical-map-row" key={row.id}><strong>{row.name}</strong><label>Material<select aria-label={`${row.name} material`} value={row.material} onChange={e=>changeMap(next=>{next[i].material=e.target.value;})}>{[...new Set([...materials,row.material])].map(m=><option key={m}>{m}</option>)}</select></label><label><input type="checkbox" checked={row.target} onChange={e=>changeMap(next=>{next[i].target=e.target.checked;})}/>Target region</label><div className="surgical-actions"><button disabled={i===0} aria-label={`Lower ${row.name} precedence`} onClick={()=>changeMap(next=>{[next[i-1],next[i]]=[next[i],next[i-1]];})}>↑ Lower priority</button><button disabled={i===rows.length-1} aria-label={`Raise ${row.name} precedence`} onClick={()=>changeMap(next=>{[next[i+1],next[i]]=[next[i],next[i+1]];})}>↓ Higher priority</button></div></div>)}<button disabled={c.pending||!mapping||JSON.stringify(mapping)===JSON.stringify(v.scene.regions)} onClick={()=>{setPlaying(false);void c.create(mapping??v.scene.regions);}}>Apply mapping / new scene</button></details>
    </>}{c.pending&&<p role="status">Computing instrument action…</p>}{c.error&&<p role="alert" className="surgical-error">{c.error}</p>}
  </section>;
}
