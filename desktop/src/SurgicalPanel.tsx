import {PatientPositionControls} from './PatientPositionControls';
import {useEffect,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {Euler,Quaternion} from 'three';
import type {SurgicalController} from './use-surgical';
import type {SurgicalRegion,SurgicalImportOptions,SurgicalOutcome,SurgicalView} from './surgical-types';
import './SurgicalPanel.css';
import {reconstructedPrefix,surgicalReplayRoutes} from './surgical-authority.ts';
import {nextSurgicalPlaybackFrame,surgicalPlaybackPosition,SurgicalPlaybackClock} from './surgical-playback.ts';
import {ResearchPlanningPanel} from './ResearchPlanningPanel';
import {SectorStrategiesPanel} from './SectorStrategiesPanel';
import {JointStrategiesPanel} from './JointStrategiesPanel';
import {StagedStrategiesPanel} from './StagedStrategiesPanel';
import {PrecisionContinuationPanel} from './PrecisionContinuationPanel';
import {RetractionStudyPanel} from './RetractionStudyPanel';
import {ConsequenceComparisonPanel} from './ConsequenceComparisonPanel';
import {LearnedSectorExecutionPanel} from './LearnedSectorExecutionPanel';
import {RescueLearningPanel} from './RescueLearningPanel';
import {LearnedExecutionPanel} from './LearnedExecutionPanel';
import {PatientCusaLearningPanel} from './PatientCusaLearningPanel';
import {FrozenCusaExecutionPanel} from './FrozenCusaExecutionPanel';
import {cusaContactModel,cusaParameterSpecs} from './cusa-research';
import {FullCaseRehearsalPanel} from './FullCaseRehearsalPanel';
import {PlanarBladeManeuverPanel} from './PlanarBladeManeuverPanel';
import {SurgicalOutcomeSummary,amount} from './SurgicalOutcomeSummary';
import {summarizeSurgicalEvents} from './surgical-types.ts';
const materials=['unknown','bone','soft_tissue','tumor','membrane','vessel','fluid','functional_region'];
/** Closed native details still mount their children. Defer large read-only
 * records until opened so replay does not reconcile thousands of hidden rows. */
function RecordedDetails({summary,children}:{summary:ReactNode;children:()=>ReactNode}){
  const [open,setOpen]=useState(false);
  return <details onToggle={event=>setOpen(event.currentTarget.open)}><summary>{summary}</summary>{open?children():null}</details>;
}
function Numeric({value,onChange,label,min,max,step=.1}:{value:number;onChange:(n:number)=>void;label:string;min?:number;max?:number;step?:number}){
  const [draft,setDraft]=useState(String(+value.toFixed(3))),focused=useRef(false);
  useEffect(()=>{if(!focused.current)setDraft(String(+value.toFixed(3)));},[value]);
  return <input aria-label={label} type="number" min={min} max={max} step={step} value={draft} onFocus={()=>{focused.current=true;}} onChange={e=>{setDraft(e.target.value);const n=e.target.valueAsNumber;if(Number.isFinite(n)&&(min===undefined||n>=min)&&(max===undefined||n<=max))onChange(n);}} onBlur={()=>{focused.current=false;}}/>;
}
const parameterSpecs:Record<string,{label:string;min:number;max:number;default:number}>={rpm:{label:'Drill speed · rpm',min:0,max:80000,default:40000},forceN:{label:'Contact force · N',min:0,max:20,default:1},duty:{label:'Drill duty fraction',min:0,max:1,default:1},amplitudeMa:{label:'Stimulus amplitude · mA',min:0,max:20,default:1},frequencyHz:{label:'Frequency · Hz',min:0,max:500,default:60},pulseWidthUs:{label:'Pulse width · µs',min:0,max:1000,default:100},radiusMm:{label:'Tip radius · mm',min:.1,max:12,default:2},depthMm:{label:'Working depth · mm',min:.1,max:20,default:2},flowMm3PerSecond:{label:'Flow · mm³/s',min:0,max:1000,default:20},power:{label:'Power · model units',min:0,max:10,default:1},jawOpeningMm:{label:'Jaw opening · mm',min:0,max:12,default:2}};

function ActionResult({outcome,title,onWithdraw}:{outcome:Pick<SurgicalOutcome,'reason'|'events'>&{status:string;completedFraction?:number};title:string;onWithdraw?:()=>void}){
  const summaries=summarizeSurgicalEvents(outcome.events??[]);
  const explanation:Record<string,string>={
    TISSUE_ASPIRATION_IN_PROGRESS:'The tip is held at retained tissue while finite aspiration work accumulates.',
    WITHDRAW_BEFORE_EXCHANGE:'Withdraw this instrument before exchanging it.',
    WITHDRAW_BOTH_TOOLS_BEFORE_SETUP:'Withdraw and deactivate both instruments before changing the setup.',
    SHAFT_COLLISION:'Retained tissue blocks the shaft. Preview a shorter stroke or another approach.',
    TOOL_COLLISION:'The instruments would collide. Reposition one hand before retrying.',
    RETAINED_TISSUE_COLLISION:'Retained tissue blocks this movement. Preview a shorter stroke or another approach.',
    CONTINUUM_SMALL_GRADIENT_DOMAIN:'The requested deformation exceeds this research model’s supported range. The accepted state is unchanged.',
    CONTINUUM_UNSUPPORTED_RIGID_COMPONENT:'The local tissue component lacks declared mechanical support.',
  };
  const why=outcome.reason?explanation[outcome.reason.split(':')[0]]:undefined;
  return <section aria-label={title} className="surgical-preview" role="status"><strong>{title} · {outcome.status}</strong>
    {outcome.status==='partial'&&Number.isFinite(outcome.completedFraction)&&<p>{+(100*outcome.completedFraction!).toFixed(1)}% completed · accepted partial state is displayed.</p>}
    {outcome.reason&&<p>{why??outcome.reason.replaceAll('_',' ')}</p>}
    {onWithdraw&&outcome.reason?.startsWith('WITHDRAW_BEFORE_EXCHANGE')&&<button onClick={onWithdraw}>Set action to withdraw</button>}
    <details><summary>Action events and technical reason</summary>
      {outcome.reason&&<p><code>{outcome.reason}</code></p>}
      {summaries.length?<ul>{summaries.map(s=><li key={s.kind}>{s.label}{s.count>1?` (${s.count} samples)`:''}{s.details.length?`: ${s.details.join(' · ')}`:''}</li>)}</ul>:<p>No effect events were reported.</p>}
    </details>
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
  const continuum=physics?.brain?.model==='resectionlab.voxel-continuum.v1';
  const vascular=physics?.vessel?.model==='resectionlab.vascular-graph.v1'?physics.vessel:undefined;
  const patientDerivative=adapter?.type==='native-patient-research-derivative';
  const refinement=source?.workingGridRefinement as {schema?:unknown;factor?:unknown;noNewAnatomy?:unknown}|undefined;
  const mapped=new Set(view.scene.regions.map(r=>r.material));
  const metrics:Record<string,unknown>={...view.metrics,...physics?.metrics};
  const metricLabels:Record<string,string>={severedVesselConnections:'Severed local vessel adjacencies',activeVesselWounds:'Active modeled wound outlets',modeledVesselFluidGeneratedMm3:'Modeled vessel fluid generated · mm³',modeledVesselReservoirMm3:'Modeled vessel fluid retained · mm³'};
  const wounds=physics?.wounds??view.sources?.filter(s=>s.kind==='modeled-vessel-wound');
  return <section aria-label="Scene provenance and research behavior" className="surgical-research-context">
    {patientDerivative&&<div className="surgical-warning"><strong>Patient-derived research estimate</strong><p>This scene is derived from patient anatomy. Material properties and flow behavior are declared research assumptions, not measured patient physiology. Derived geometry is separate from source annotations.</p></div>}
    {adapter?.warnings?.length? <ul className="surgical-warning">{adapter.warnings.map((warning,i)=><li key={i}>{warning}</li>)}</ul>:null}
    <p className="surgical-note">Scene grid: {view.scene.resolutionMm.map(n=>+n.toFixed(3)).join(' × ')} mm · {view.scene.shape.join(' × ')} cells.</p>
    {refinement?.schema==='resectionlab.surgical-working-grid-refinement.v1'&&refinement.factor===2&&refinement.noNewAnatomy===true&&<p className="surgical-warning">Refined computational grid: each parent material cell was subdivided into eight children with the same physical bounds. The image is derived interpolation. This adds no acquired image resolution or anatomical evidence; it is a discretization study.</p>}
    {v2&&<p className="surgical-note">{continuum?'Opted-in voxel continuum: small-gradient tissue deformation and declared stress relaxation.':'V2 research mechanics: coupled brain relaxation with an approximate volume penalty.'} {vascular?'The declared vascular graph separates compression, occlusion, transection and wound sealing.':'Work and heat based drilling and clustered bleeding retain their declared assumptions.'} Patient physiology and global perfusion are unvalidated.</p>}
    <details><summary>Provenance, material behavior, and assumptions</summary>
      {source&&<DeclaredFields data={Object.fromEntries(Object.entries(source).filter(([key])=>key!=='adapter'))}/>}
      {adapter&&<details><summary>Source adapter and declared patient source</summary><DeclaredFields data={adapter}/></details>}
      <ul className="surgical-note">
        {mapped.has('bone')&&<li>Bone: rigid cells; {v2?'drilling accumulates work before removal and models heat generation and cooling.':'drilling and kerf cutting remove cells and produce modeled debris.'}</li>}
        {(mapped.has('soft_tissue')||mapped.has('tumor'))&&<li>Brain / soft tissue and tumor: {continuum?'source-owned voxel elements with declared material properties, stress relaxation, cut-face separation and explicit local support. Large deformation and patient injury remain unsupported.':v2?'coupled local tissue deformation with timed relaxation after release, distinct assumed stiffness, and an approximate volume penalty.':'lattice patch displacement and compatible-tool removal; no calibrated deformation law.'}</li>}
        {mapped.has('vessel')&&<li>Vessels: local adjacency cutting retains pieces. {vascular?'Source-bound artery, vein, sinus or uncertain edges use declared conductance, compliance and boundaries. An occluded vessel may cause modeled flow deficits without bleeding. Sealing a cut does not reconnect it.':v2?'Clustered wound outlets use assumed pressure, finite vascular reserves, and refill; the fluid budget is tracked.':physics?.capabilities?.damageTriggeredVesselWounds===true?'Declared damage-triggered wound outlets use an assumed local flow proxy.':'Damage-triggered vessel wound behavior is not declared as enabled.'} Global perfusion is unknown.</li>}
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
  const continuum=view.physics?.brain?.model==='resectionlab.voxel-continuum.v1'?view.physics.brain.summary:undefined;
  const vascular=view.physics?.vessel?.model==='resectionlab.vascular-graph.v1'?view.physics.vessel:undefined;
  const bipolar=view.physics?.bipolarThermal?.model==='resectionlab.bipolar-thermal.v1'?view.physics.bipolarThermal:undefined;
  const sum=(values:Record<string,number>|undefined)=>values?Object.values(values).reduce((a,b)=>a+b,0):undefined;
  const displacement=view.deformation?Math.max(0,...Object.values(view.deformation).map(v=>Math.hypot(...v))):undefined;
  const temperature=drill&&Object.keys(drill.temperatureC).length?Object.values(drill.temperatureC).reduce((max,value)=>Math.max(max,value),-Infinity):view.metrics.drillMaxTemperatureC;
  const rows:Array<[string,unknown]>=[
    ['Drill input energy · J',drill?.inputEnergyJ??view.metrics.drillInputEnergyJ],
    ['Pending cutting work · J',sum(drill?.workJ)??view.metrics.drillPendingWorkJ],
    ['Generated heat · J',drill?.heatGeneratedJ??view.metrics.drillHeatGeneratedJ],
    ['Maximum modeled drill temperature · °C',temperature],
    ...(bipolar?[
      ['Bipolar deposited heat · J',view.metrics.bipolarHeatInputJ],
      ['Bipolar stored / bath / removed heat · J',[view.metrics.bipolarHeatStoredJ,view.metrics.bipolarHeatBathExportJ,view.metrics.bipolarHeatRemovedJ]],
      ['Bipolar current / peak modeled temperature · °C',[view.metrics.bipolarMaxTemperatureC,view.metrics.bipolarPeakTemperatureC]],
      ['Bipolar heat budget residual · J',view.metrics.bipolarHeatResidualJ],
      ['Bipolar excess temperature exposure · mm³·K·s · no injury calibration',view.metrics.bipolarExcessTemperatureVolumeTimeMm3KSeconds],
    ] as Array<[string,unknown]>:[]),
    ['Brain elastic energy · N·mm',brain?.elasticEnergyNmm],
    ['Active deformation cells',brain?.activeNodes??brain?.active?.length],
    ['Maximum displacement · mm',displacement],
    ['Maximum neighbor displacement / spacing',brain?.maxNeighborDisplacementRatio],
    ...(continuum?[
      ['Continuum active elements / DOFs',`${amount(continuum.activeElements,0)} / ${amount(continuum.activeDofs,0)}`],
      ['Modeled viscous dissipation · N·mm',continuum.dissipatedEnergyNmm],
      ['Stored energy exported with removed tissue · N·mm',continuum.removedStoredEnergyNmm],
      ['Modeled mechanical exposure by original label · no injury calibration',continuum.exposureBySourceLabel],
    ] as Array<[string,unknown]>:[]),
    ...(vascular?[
      ['Declared vascular graph coverage',vascular.coverage],
      ['Pressure-supporting vascular storage · mm³',vascular.summary?.storageMm3],
      ['Boundary inflow / outflow · mm³',[vascular.summary?.boundaryInMm3,vascular.summary?.boundaryOutMm3]],
      ['Cut / occluded / compressed graph edges',[vascular.summary?.cutEdgeCount,vascular.summary?.occludedEdgeCount,vascular.summary?.compressedEdgeCount]],
      ['Generated wound fluid · mm³',vascular.summary?.generatedMm3],
      ['Vascular budget residual · mm³',vascular.summary?.conservationResidualMm3],
      ['Declared supply / drainage flow deficits · no injury calibration',vascular.summary?.regions],
    ] as Array<[string,unknown]>:[
    ['Vascular pressure · Pa',vessels?.pressurePa],
    ['Vascular reserve remaining · mm³',vessels?.remainingMm3],
    ['Vascular refill · mm³',vessels?.refilledMm3],
    ['Generated wound fluid · mm³',vessels?.generatedMm3],
    ['Vascular budget residual · mm³',vessels?vessels.initialMm3+vessels.refilledMm3-vessels.remainingMm3-vessels.generatedMm3:undefined],
    ] as Array<[string,unknown]>),
  ];
  return <details><summary>{preview?'Preview':'Recorded'} V2 mechanics and fluid budget</summary><dl className="surgical-declared-fields">{rows.map(([label,value])=><div key={label}><dt>{label}</dt><dd>{typeof value==='number'&&Number.isFinite(value)?+value.toFixed(5):displayValue(value)}</dd></div>)}</dl>{bipolar&&<details><summary>Declared bipolar heat model and assumptions</summary><DeclaredFields data={bipolar}/><p>Delivered heat and temperature are modeled quantities. The coagulation proxy and vessel patency remain separate; temperature does not predict injury or successful hemostasis.</p></details>}{vascular&&<details><summary>Declared vascular graph and assumptions</summary><DeclaredFields data={vascular}/></details>}<p className="surgical-note">{vascular?'The graph budget compares initial pressure-supporting storage plus boundary inflow against remaining storage, boundary outflow and generated wound fluid. Zero unstressed volume and all flow coefficients are declared assumptions.':'The vascular budget compares initial reserve plus refill against remaining reserve and generated fluid.'} These modeled quantities do not estimate patient blood loss or predict tissue injury.</p></details>;
}

export function SurgicalPanel({c,onFixture,onImport,available,anatomyControls,onFrameField,fieldDisplayControls,microscopeControls,learningReport,executionReport,cusaLearning,onRunFrozenCusa,onRunFrozenSector,frozenCusaPending=false,inspectionNotice}:{inspectionNotice?:string|null;onRunFrozenCusa?:(branchId:string)=>Promise<void>;onRunFrozenSector?:(method:string)=>Promise<void>;frozenCusaPending?:boolean;cusaLearning?:unknown;executionReport?:unknown;learningReport?:unknown;anatomyControls?:ReactNode;onFrameField?:()=>void;fieldDisplayControls?:ReactNode;microscopeControls?:ReactNode;c:SurgicalController;onFixture:()=>void;onImport?:(options:SurgicalImportOptions)=>Promise<void>|void;available:boolean}){
  const [importOptions,setImportOptions]=useState<SurgicalImportOptions>({frame:'RAS+',units:'mm',material:'unknown',resolution:1,topology:'solid',thicknessMm:1});
  const [importing,setImporting]=useState(false),[importError,setImportError]=useState<string|null>(null);
  const [search,setSearch]=useState(''),[duration,setDuration]=useState(1),[operation,setOperation]=useState('apply'),[playing,setPlaying]=useState(false),[playbackSpeed,setPlaybackSpeed]=useState(1);
  const [mapping,setMapping]=useState<SurgicalRegion[]|null>(null),[seekTarget,setSeekTarget]=useState<number|null>(null);
  const playbackClock=useRef(new SurgicalPlaybackClock());
  useEffect(()=>{c.setPlaybackDisplay({playing,speed:playbackSpeed});},[playing,playbackSpeed]);
  useEffect(()=>()=>c.stopPlayback(),[]);
  const v=c.view,selected=v?.slots[c.slot],tool=v?.catalog.find(t=>t.id===selected?.instrument),r=c.result;
  const displayed=c.preview??v;
  const routes=surgicalReplayRoutes(r),prefix=reconstructedPrefix(r),observed=r?.playback,imported=routes.imported,unverifiedImport=imported&&!prefix,readOnly=!!observed||imported||r?.readOnly===true,position=surgicalPlaybackPosition(observed,r?.position??0);
  const freshOrigin=!!prefix&&!!r?.prefixOrigin,preparedOnly=prefix?.preparedCache?.mode==='only'&&!freshOrigin;
  const v2=displayed?.physics?.profile?.schema==='resectionlab.tissue-physics.v2';
  const cusaModel=cusaContactModel(v),cusaSelected=!!cusaModel&&tool?.id==='cusa';
  const specs:typeof parameterSpecs=cusaSelected?{...parameterSpecs,...cusaParameterSpecs}:parameterSpecs;
  const latestOutcome=c.preview?c.previewOutcome:observed?(observed.frameIndex===observed.frameCount-1?r?.timeline?.[observed.actionIndex]:null):r?.outcome??(prefix?r?.timeline?.[(r?.position??0)-1]:v?.history.at(-1));
  const outcomeError=!!latestOutcome?.reason&&(c.error===latestOutcome.reason||c.error===`${latestOutcome.status}: ${latestOutcome.reason}`);
  const importantOutcome=!!latestOutcome&&(['partial','rejected'].includes(latestOutcome.status)||latestOutcome.reason?.startsWith('TISSUE_ASPIRATION_IN_PROGRESS'));
  const observation=displayed?.observations?.at(-1);
  const q=new Euler().setFromQuaternion(new Quaternion(...c.pose.quaternion));
  const steps=r?.branches.find(b=>b.id===r.branchId)?.steps??0;
  useEffect(()=>{setMapping(null);setPlaying(false);setImportError(null);setOperation('apply');setDuration(1);},[v?.scene.sceneId]);
  useEffect(()=>setPlaying(false),[c.playbackVersion]);
  useEffect(()=>{if(!c.pending)setSeekTarget(null);},[c.pending,r?.position,r?.branchId]);
  const playback=r?nextSurgicalPlaybackFrame(r,steps,playbackSpeed):null;
  useEffect(()=>{if(!playing){playbackClock.current.reset();c.stopPlayback();return;}if(c.pending)return;if(!playback){setPlaying(false);return;}
    if(!c.startPlayback()){setPlaying(false);return;}
    const delay=playbackClock.current.delay(observed?.observedTimeSeconds??Number(v?.metrics.elapsedSeconds),playback.milliseconds,playbackSpeed,performance.now());
    const timer=setTimeout(()=>{void c.runPlayback({branchId:playback.branchId,position:playback.position,frameIndex:playback.frameIndex}).then(ok=>{if(!ok)setPlaying(false);});},delay);return()=>clearTimeout(timer);},[playing,c.pending,r,steps,playbackSpeed]);
  const keys=['radiusMm',...(cusaSelected?['power','duty']:[]),...(v2&&tool?.id==='cranial_drill'?['rpm','forceN','duty']:v2&&tool?.materials.includes('vessel')?['forceN']:[]),...(tool?.effect==='probe'?['amplitudeMa','frequencyHz','pulseWidthUs']:[]),...(['incise','kerf','sheet_cut','transect','irrigate'].includes(tool?.effect??'')?['depthMm']:[]),...(['suction','irrigate'].includes(tool?.effect??'')?['flowMm3PerSecond']:[]),...(tool?.effect==='coagulate'?['power']:[]),...(['forceps','scissors'].includes(tool?.geometry??'')?['jawOpeningMm']:[])];
  const params=Object.fromEntries(keys.map(k=>[k,cusaSelected&&k==='radiusMm'?1.5:c.parameters[k]??(k==='radiusMm'?tool?.radiusMm:undefined)??specs[k].default]));
  const compatible=tool?.materials.length===0||v?.scene.regions.some(row=>tool?.materials.includes(row.material));
  const changeMap=(fn:(rows:SurgicalRegion[])=>void)=>{const rows=structuredClone(mapping??v?.scene.regions??[]);fn(rows);setMapping(rows);};
  const stopRun=(command:string,extra:Record<string,unknown>={})=>{setPlaying(false);void c.run(command,extra);};
  return <section className="surgical-panel" aria-label="Surgical instrument toolbox" data-state-id={displayed?.stateId} data-recorded-position={r?.position}>
    <header className="rehearsal-heading"><h2>Rehearse</h2><span className="surgical-note">Research model · uncalibrated</span></header>
    {v&&<section className="rehearsal-state" aria-label="Current rehearsal state" role="status">
      <strong>{c.preview?'Preview · uncommitted':observed?'Read-only replay':readOnly?'Read-only recorded state':'Committed state · editable'}</strong>
      <p>Action {position} / {steps} · {r?.branchId}</p>
      {inspectionNotice&&<p className="surgical-warning">{inspectionNotice}</p>}
      {c.preview&&<><p>Apply commits; Cancel returns to committed anatomy.</p><button onClick={c.cancel}>Cancel preview</button></>}
      {c.pending&&<><button onClick={c.cancel}>Cancel</button><p>Cancel clears the draft. The current request continues.</p></>}
      {observed&&<button disabled={c.pending} onClick={()=>stopRun(routes.boundary,{branchId:r?.branchId,position})}>Return to recorded boundary</button>}
      {imported&&!observed&&<button disabled={c.pending} onClick={()=>stopRun('inspect',{branchId:r?.branchId})}>Verify saved replay</button>}
    </section>}
    {c.pending&&<p role="status">{c.pendingCommand==='inspect'?'Verifying saved source-bound rehearsal… First opening can take several minutes.':playing||observed||['seek','seekFrame','seekPrefix','seekPrefixFrame','selectBranch','selectScene'].includes(c.pendingCommand??'')?'Preparing an exact recorded state… Cold verification can take several minutes; the previous accepted display remains visible.':c.simulationPending?'Computing instrument action…':'Preparing accepted display… Previous anatomy and quantities remain visible.'}</p>}{c.error&&!outcomeError&&<p role="alert" className="surgical-error">{c.error}</p>}
    {!v?<button className="primary-button" disabled={!available||c.pending||importing} onClick={()=>void c.create()}>Prepare instrument scene</button>:<>

    {latestOutcome&&importantOutcome&&<ActionResult outcome={latestOutcome} title={c.preview?'Preview result':observed?'Recorded action endpoint result':'Last action result'} onWithdraw={!readOnly?()=>{setOperation('withdraw');c.clearPreview();}:undefined}/>}
    <div className="surgical-slots">{Object.entries(v.slots).map(([s,t])=><button key={s} disabled={c.pending||playing||readOnly} aria-pressed={c.slot===s} onClick={()=>c.selectSlot(s)}><strong>{s} hand</strong><span>{t.instrument==='cusa'?'CUSA':v.catalog.find(x=>x.id===t.instrument)?.name}</span><small>{t.inserted?'Inserted':'Withdrawn'}{(observed?.effectiveActivation?.[s]??t.active)?' · active':t.inserted?' · inactive':''}{observed?' · observed':''}</small>{v.holds?.[s]&&<small>{v.holds[s].cells.length?`Holding ${v.holds[s].cells.length} source cell${v.holds[s].cells.length===1?'':'s'}`:'Hold empty · no retained tissue'}</small>}</button>)}</div>
    <details><summary>Exchange instrument</summary><input aria-label="Search instruments" placeholder="Search instruments…" value={search} onChange={e=>setSearch(e.target.value)}/>
    <div className="surgical-toolbox">{v.catalog.filter(t=>(t.name+' '+t.category).toLowerCase().includes(search.toLowerCase())).map(t=><button key={t.id} aria-pressed={selected?.instrument===t.id} disabled={c.pending||playing||readOnly||!!v.planarBladeManeuver||selected?.instrument===t.id} onClick={()=>{if(selected?.instrument!==t.id)stopRun('step',{action:{slot:c.slot,operation:'exchange',instrument:t.id,duration:1}})}}><small>{t.category}</small>{t.name}</button>)}</div>
    <p className="surgical-note">Choosing a different instrument commits an exchange. The selected instrument does not advance time.</p></details>

    {!v.planarBladeManeuver&&<FullCaseRehearsalPanel c={c} pause={()=>setPlaying(false)}/>}
    {v.planarBladeManeuver&&<PlanarBladeManeuverPanel c={c} disabled={playing||readOnly} pause={()=>{setPlaying(false);c.pauseFullCase();}}/>}
    <section className="manual-actions" aria-label="Manual instrument actions"><h3>Manual action</h3>
    {v.planarBladeManeuver&&<p className="surgical-note">Use the declared fixed blade maneuver above. Other instrument actions are unavailable in this scene.</p>}
    {!compatible&&<p className="surgical-warning">No compatible material is mapped in this scene. Placement and inspection remain available; tissue effects require compatible material.</p>}
    <fieldset disabled={c.pending||playing||readOnly||!!v.planarBladeManeuver} className="surgical-controls">
    <label>Action<select aria-label="Instrument action" value={operation} onChange={e=>{setOperation(e.target.value);c.clearPreview();}}>{['apply','place','move','hold','grasp','extract','activate','deactivate','release','withdraw','inspect'].map(op=><option key={op}>{op}</option>)}</select></label>
    </fieldset>
    <div className="surgical-actions"><button disabled={c.pending||playing||readOnly||!!v.planarBladeManeuver} onClick={()=>stopRun('preview',{action:c.action(operation,params,duration)})}>Preview</button><button className="primary-button" disabled={c.pending||playing||readOnly||!!v.planarBladeManeuver} onClick={()=>stopRun('step',{action:c.action(operation,params,duration)})}>Apply</button>{!c.preview&&!c.pending&&<button onClick={c.cancel}>Cancel</button>}</div>
    <details><summary>Pose, stroke and tool settings</summary>
    <fieldset disabled={c.pending||playing||readOnly||!!v.planarBladeManeuver} className="surgical-controls">
    <p className="surgical-note">{tool?.name} · {tool?.materials.length?`Compatible: ${tool.materials.join(', ')}`:'Modeled fluid and debris'}</p>
    <label>Viewport<select aria-label="Instrument interaction" value={c.mode} onChange={e=>c.setMode(e.target.value as typeof c.mode)}><option value="navigate">Navigate anatomy</option><option value="draw">Draw stroke on anatomy</option><option value="translate">Translate tool</option><option value="rotate">Rotate tool</option></select></label>
    <div className="surgical-vector">{c.pose.position.map((n,i)=><label key={i}>{['R','A','S'][i]} mm<Numeric label={`Tool position ${['R','A','S'][i]}`} step={.5} value={n} onChange={value=>{const p=[...c.pose.position] as [number,number,number];p[i]=value;c.setPose({...c.pose,position:p});}}/></label>)}</div>
    <div className="surgical-vector">{[q.x,q.y,q.z].map((n,i)=><label key={i}>{['Pitch','Yaw','Roll'][i]} °<Numeric label={`Tool ${['pitch','yaw','roll'][i]}`} value={n*180/Math.PI} onChange={value=>{const a=[q.x,q.y,q.z];a[i]=value*Math.PI/180;const b=new Quaternion().setFromEuler(new Euler(a[0],a[1],a[2]));c.setPose({...c.pose,quaternion:b.toArray()});}}/></label>)}</div>
    {keys.map(k=><label key={k}>{specs[k].label}<Numeric label={specs[k].label} min={specs[k].min} max={specs[k].max} value={params[k]} onChange={n=>c.setParameters({...params,[k]:n})}/></label>)}
    {['scissors','forceps'].includes(tool?.geometry??'')&&<div className="surgical-actions"><button onClick={()=>c.setParameters({...params,jawOpeningMm:0})}>Close jaws</button><button onClick={()=>c.setParameters({...params,jawOpeningMm:2})}>Open jaws</button><small>Cutting and grasping require a jaw opening of 0.5 mm or less.</small></div>}
    <label>Duration · s<Numeric label="Action duration" min={.01} max={30} value={duration} onChange={n=>{setDuration(n);c.clearPreview();}}/></label>
    <p>{c.path.length} stroke points{c.path.length?' · editing pose replaces this stroke':''} · {amount(v.metrics.elapsedSeconds)} simulated s</p>
    </fieldset></details>
    <fieldset disabled={c.pending||playing||readOnly||!!v.planarBladeManeuver} className="surgical-controls">
    {!cusaContactModel(v)&&Object.values(v.slots).filter(slot=>slot.instrument==='cusa'&&slot.inserted).length===1&&<button disabled={Object.values(v.slots).some(slot=>slot.grip.length||(slot.instrument!=='cusa'&&slot.inserted))||Object.keys(v.holds??{}).length>0||Number(v.metrics.nonTargetRemovedMm3)>200||(r?.position??0)>180||r?.exitRecovery?.sourceStateId===v.stateId}
      onClick={()=>{c.pauseFullCase();stopRun('recoverExit');}}>Checked withdrawal</button>}
    {!cusaContactModel(v)&&Object.values(v.slots).some(slot=>slot.instrument==='cusa'&&slot.inserted)&&<details><summary>Checked withdrawal limits</summary><p className="surgical-note">Checked withdrawal uses the recorded outside placement and native collision checks, with zero additional non-target removal. A latest partial shaft-collision action must match its observed tip pose before its accepted steps can be reversed. Development research sources only.</p></details>}
    </fieldset></section>
    <section className="rehearsal-replay" aria-label="Replay controls"><h3>Replay</h3>
    {unverifiedImport&&<section className="surgical-warning" aria-label="Unverified imported checkpoint" role="status"><strong>Imported endpoint · unverified · read only</strong><p>The saved checkpoint matches its source and state hash. Its actions have not been replay-verified in this session. Reconstruct a read-only prefix from the beginning, or verify the full archive before planning and instrument edits.</p><button disabled={c.pending} onClick={()=>{setPlaying(false);void c.run('seekPrefix',{position:0,branchId:r?.branchId}).then(ok=>{if(ok)setPlaying(true);});}}>Replay from start · read only</button><button disabled={c.pending} onClick={()=>{setPlaying(false);void c.run('seekPrefix',{position:0,branchId:r?.branchId,cacheOnly:true}).then(ok=>{if(ok)setPlaying(true);});}}>Replay prepared observations · read only</button><button disabled={c.pending} onClick={()=>stopRun('inspect',{branchId:r?.branchId})}>Verify replay for planning</button></section>}
    {prefix&&<section className="surgical-warning" aria-label="Reconstructed prefix" role="status"><strong>{preparedOnly?'Prepared observations · read only':'Reconstructed prefix · read only'}</strong><p>{preparedOnly?'Prepared':'Reconstructed'} {prefix.verifiedThrough} / {prefix.totalActions} recorded actions. Full archive planning verification is still required; prefix reconstruction does not authorize instrument edits or planning.</p>{preparedOnly&&<p>Replay uses previously prepared observations. Missing observations stop replay without reconstructing actions.</p>}{freshOrigin&&<p>Reconstructs the saved actions during playback.</p>}{(r?.budgetStopped??prefix.budgetStopped)&&<p>Reconstruction stopped at boundary {r?.position}{(r?.requestedPosition??prefix.requestedPosition)!==undefined?` while requesting ${r?.requestedPosition??prefix.requestedPosition}`:''}. {prefix.failure?(preparedOnly?'Prepared observations are unavailable at this boundary. Start a fresh read-only reconstruction to prepare missing actions.':'The failed action was not admitted.'):'The request budget was reached; seek again to continue from this boundary.'}</p>}{prefix.failure?.reason&&<p>Action {(prefix.failure.actionIndex??0)+1}: {prefix.failure.reason}</p>}{prefix.endpointValidationError&&<p>Saved endpoint check: {prefix.endpointValidationError}</p>}{preparedOnly&&<button disabled={c.pending} onClick={()=>{setPlaying(false);void c.run('seekPrefix',{position:0,branchId:r?.branchId}).then(ok=>{if(ok)setPlaying(true);});}}>Replay from start · read only</button>}<button disabled={c.pending} onClick={()=>stopRun('inspect',{branchId:r?.branchId})}>Verify replay for planning</button></section>}
    <label>Branch<select aria-label="Surgical branch" disabled={c.pending} value={r?.branchId} onChange={e=>stopRun(routes.branch,{branchId:e.target.value})}>{r?.branches.map(b=><option key={b.id} value={b.id}>{b.id} · {b.steps} actions</option>)}</select></label>
    <div className="surgical-actions"><button disabled={unverifiedImport||!steps||(!playing&&c.pending)} aria-pressed={playing} onClick={()=>{c.cancel();c.pauseFullCase();if(playing){c.stopPlayback();setPlaying(false);return;}if(position>=steps){void c.run(routes.boundary,{position:0,branchId:r?.branchId}).then(ok=>{if(ok)setPlaying(true);});}else setPlaying(true);}}>{playing?'Pause replay':'Play replay'}</button><span>Recorded action boundary {position} / {steps}</span></div>
    <div className="surgical-actions"><button aria-label="Previous recorded state" disabled={unverifiedImport||c.pending||position===0} onClick={()=>stopRun(routes.boundary,{branchId:r?.branchId,position:Math.max(0,position-1)})}>Previous action</button><button aria-label="Next recorded state" disabled={unverifiedImport||c.pending||!r||position>=steps} onClick={()=>stopRun(routes.boundary,{branchId:r?.branchId,position:position+1})}>Next action</button></div>
    <label>Replay speed<select aria-label="Replay speed" value={playbackSpeed} onChange={e=>setPlaybackSpeed(+e.target.value)}>{[.25,.5,1,2,4].map(speed=><option key={speed} value={speed}>{speed}×</option>)}</select></label>
    <input aria-label="Surgical timeline" type="range" min="0" max={steps} value={seekTarget??position} disabled={!r||unverifiedImport} onChange={e=>{const position=+e.target.value;setSeekTarget(position);stopRun(routes.boundary,{branchId:r?.branchId,position});}}/>
    {seekTarget!==null&&c.pending&&<p role="status">Preparing recorded state {seekTarget}… Displaying the previous accepted state.</p>}
    {observed&&<section className="surgical-preview" aria-label="Observed microstep replay"><strong>Observed microstep · read only</strong><p>Action {observed.actionIndex+1} · frame {observed.frameIndex} / {observed.frameCount-1} · {observed.observedTimeSeconds.toFixed(3)} simulated s</p><p>{observed.actionElapsedSeconds.toFixed(3)} / {observed.actionDurationSeconds.toFixed(3)} s within action</p>
      <input aria-label="Observed action frames" type="range" min="0" max={observed.frameCount-1} value={observed.frameIndex} onChange={e=>stopRun(routes.frame,{branchId:observed.branchId,position:observed.position,frameIndex:+e.target.value})}/>
      <div className="surgical-actions"><button disabled={c.pending||observed.frameIndex===0} onClick={()=>stopRun(routes.frame,{branchId:observed.branchId,position:observed.position,frameIndex:observed.frameIndex-1})}>Previous microstep</button><button disabled={c.pending||observed.frameIndex===observed.frameCount-1} onClick={()=>stopRun(routes.frame,{branchId:observed.branchId,position:observed.position,frameIndex:observed.frameIndex+1})}>Next microstep</button><button disabled={c.pending} onClick={()=>stopRun(routes.boundary,{branchId:r?.branchId,position})}>Return to recorded state {position}</button></div>
    </section>}
    <div className="surgical-actions"><button disabled={c.pending||readOnly||!r?.position} onClick={()=>stopRun('undo')}>Undo</button><button disabled={c.pending||readOnly||r?.position===steps} onClick={()=>stopRun('redo')}>Redo</button><button disabled={c.pending||readOnly} onClick={()=>stopRun('reset')}>Reset</button><button disabled={c.pending||readOnly} onClick={()=>stopRun('branch')}>Branch</button></div>
    {playing&&<p role="status">{c.pending?'Buffering an exact recorded microstep…':`Playing observed engine states at ${playbackSpeed}×.`}</p>}
    <RecordedDetails summary="Action timeline">{()=><ol>{(r?.timeline??v.history).map((h,i)=><li key={i}><button disabled={c.pending||unverifiedImport} onClick={()=>stopRun(routes.boundary,{branchId:r?.branchId,position:i+1})}>{i+1}. {h.action.operation} · {h.status}</button>{h.reason&&<small>{h.reason}</small>}</li>)}</ol>}</RecordedDetails>
    <details><summary>Replay display and field transfers</summary>
    <p className="surgical-note">{playing?(c.pending?'Buffering an exact recorded microstep…':`Playing observed engine states at ${playbackSpeed}×. Buffering can slow wall-clock playback.`):((v?.toolMechanicsRevision??0)>=1?'Replay reconstructs accepted native microsteps. Instrument capsules and static jaw opening match collision geometry; recorded RPM drives drill phase.':'Replay reconstructs successful engine microsteps and finalized action boundaries. Anatomy and tool poses stay exact. Active jaw close/release and recorded-RPM drill rotation are illustrative working-part animation during playback; they imply no additional cutting events.')}</p>
    {observed&&<section className="surgical-preview" aria-label="Recorded field transfers"><strong>Recorded field transfers</strong>
      {observed.flowDisplay?<><p>{observed.flowDisplay.startSeconds.toFixed(3)}–{observed.flowDisplay.endSeconds.toFixed(3)} simulated s · anatomy shown at interval end.</p>
        <label><input type="checkbox" checked={c.showFlowTransfers} onChange={e=>c.setShowFlowTransfers(e.target.checked)}/>Show recorded flow effects</label>
        <p className="surgical-note">During playback, droplets replay the preceding recorded interval once as an illustrative overlay. Their radius uses the operation volume with a 3× display gain; this is not a fluid surface or measured velocity. Deposits grow at the recorded outlet cell, uptake shrinks locally, and only recorded neighboring-cell transfers move. Pause and scrub show static evidence. Arrows show recorded neighboring-cell transfers. Diamonds mark local deposit, uptake or escape; they show no route to a tool. Off-grid escapes have no glyph. Arrow and diamond sizes are fixed. Blue is fluid and gray-brown is debris. Red and cyan deposits identify modeled wound and irrigation releases; transported fluid is mixed.</p>
        {observed.flowDisplay.truncated?<p role="status">Some transfer indicators were omitted. Quantitative field ledgers remain complete.</p>:observed.flowDisplay.operations.length===0?<p>No field operations were recorded in this interval.</p>:null}
        <RecordedDetails summary={`Sampled field operations · ${observed.flowDisplay.operations.length}`}>{()=><ol>{observed.flowDisplay!.operations.map((op,i)=><li key={i}>{op.kind} · {op.field} · {op.provenance} · {op.volumeMm3.toPrecision(4)} mm³ · {op.fromCell??(op.kind==='escape'?'off-grid outlet':'source')} → {op.toCell??'sink'} · {op.startSeconds.toFixed(3)}–{op.endSeconds.toFixed(3)} s</li>)}</ol>}</RecordedDetails>
      </>:<p>Transfer records are unavailable for this observation. Static field markers show the recorded endpoint.</p>}
    </section>}
    </details></section>
    <SurgicalOutcomeSummary c={c}/>
    {latestOutcome&&!importantOutcome&&<details><summary>{c.preview?'Preview action events':'Last action result'} · {latestOutcome.status}</summary><ActionResult outcome={latestOutcome} title={c.preview?'Preview result':observed?'Recorded action endpoint result':'Last action result'}/></details>}
    {observation&&<StimulationObservation observation={observation} preview={!!c.preview}/>}

    <section className="rehearsal-view-controls" aria-label="View and inspection controls">
      {microscopeControls}
      <details><summary>Field and anatomy display</summary>{onFrameField&&<button onClick={onFrameField}>Frame operative field</button>}{fieldDisplayControls}{anatomyControls}</details>
    </section>
    <details className="rehearsal-secondary"><summary>Advanced research workflows</summary>
    <RescueLearningPanel c={c} report={learningReport} pause={()=>setPlaying(false)}/>
    <ConsequenceComparisonPanel c={c} pause={()=>setPlaying(false)}/>
    <LearnedExecutionPanel c={c} report={executionReport} pause={()=>setPlaying(false)}/>
    <LearnedSectorExecutionPanel c={c} artifact={cusaLearning} pause={()=>{setPlaying(false);c.pausePlayback();}}/>
    <PatientCusaLearningPanel c={c} artifact={cusaLearning} pause={()=>setPlaying(false)}/>
    <FrozenCusaExecutionPanel c={c} pending={frozenCusaPending} onRun={onRunFrozenCusa} onRunSector={onRunFrozenSector} pause={()=>{setPlaying(false);c.pausePlayback();}}/>
    <JointStrategiesPanel c={c} artifact={cusaLearning} pause={()=>{setPlaying(false);c.pausePlayback();}}/>
    <RetractionStudyPanel c={c} pause={()=>{setPlaying(false);c.pausePlayback();}}/>
    <StagedStrategiesPanel c={c} artifact={cusaLearning}
      pause={()=>{setPlaying(false);c.pausePlayback();}}
      play={branchId=>{setPlaying(false);c.pausePlayback();void c.run('selectBranch',{branchId,position:0}).then(ok=>{if(ok)setPlaying(true);});}}/>
    <PrecisionContinuationPanel c={c} artifact={cusaLearning}
      pause={()=>{setPlaying(false);c.pausePlayback();}}
      play={branchId=>{setPlaying(false);c.pausePlayback();void c.run('selectBranch',{branchId,position:0}).then(ok=>{if(ok)setPlaying(true);});}}/>
    <fieldset disabled={readOnly} className="surgical-controls"><ResearchPlanningPanel c={c} pause={()=>setPlaying(false)}/></fieldset>
    <fieldset disabled={readOnly} className="surgical-controls"><SectorStrategiesPanel c={c} pause={()=>{setPlaying(false);c.pausePlayback();}}/></fieldset>
    </details>
    <RecordedDetails summary="Source provenance and research assumptions">{()=><SceneResearchDetails view={displayed??v} preview={!!c.preview}/>}</RecordedDetails>
    <details><summary>Mechanics, fluid ledgers and effect records</summary>
    {displayed&&v2&&<MechanicsV2Details view={displayed} preview={!!c.preview}/>}
    {displayed&&cusaContactModel(displayed)&&<details><summary>CUSA contact and aspiration</summary><p className="surgical-note">Active contact supplies finite capacity; whole voxels remain until complete. Activation and duty are fractions, with fixed 1.5 mm tip radius. Rates and complete fragment capture are uncalibrated research assumptions.</p><dl>{[['cusaAspiratedSolidMm3','Aspirated solid · mm³'],['cusaPendingCapacityMm3','Pending contact capacity · mm³'],['cusaInputCapacityMm3','Supplied capacity · mm³'],['cusaUnusedCapacityMm3','Unused capacity · mm³'],['cusaDiscardedCapacityMm3','Discarded capacity · mm³']].map(([key,label])=><div key={key}><dt>{label}</dt><dd>{amount(displayed.metrics[key],3)}</dd></div>)}</dl><p className="surgical-note">Pending capacity is work toward voxel clearance, not removed tissue. Liquid aspiration and field fluid use separate ledgers.</p><DeclaredFields data={cusaContactModel(displayed)!}/></details>}
    <details><summary>{c.preview?'Preview':'Recorded'} effect details</summary><dl>
      {[['Thermal exposure · model units',displayed?.metrics.thermalExposure??(displayed?.thermal?Object.values(displayed.thermal).reduce((a,b)=>a+b,0):undefined)],['Fluid added · mm³',displayed?.metrics.fluidAddedMm3],['Fluid removed · mm³',displayed?.metrics.fluidRemovedMm3],['Fluid escaped · mm³',displayed?.metrics.fluidEscapedMm3],['Debris present · mm³',displayed?.metrics.debrisMm3],['Grid cut edges',displayed?.cutEdges?.length],['Transected grid edges',displayed?.transectedEdges?.length],['Mesh cut edges',displayed?.meshCutEdges?.length],['Retained mesh seams',displayed?.meshSeams?.length],['Separated attachments',displayed?.separatedAttachments?.length],['Retained flap components',displayed?.flapComponents?.length],['Separable flap components',displayed?.flapComponents?.filter(f=>f.separable===true).length],['Flaps with unknown separability',displayed?.flapComponents?.filter(f=>f.separable===null).length],...Object.entries(displayed?.slots??{}).map(([slot,state])=>[`${slot} hand held cells`,state.grip.length])].map(([label,value])=><div key={String(label)}><dt>{String(label)}</dt><dd>{typeof value==='number'?+value.toFixed(3):displayValue(value)}</dd></div>)}
    </dl></details>
    <p className="surgical-note">Effect markers: cyan contact · orange heat · blue fluid · gray-brown debris · red first obstruction · gold cut seams / transections. Mesh seams represent retained topology graph cuts, not a physical opening or removed tissue.</p>
    {r?.exitRecovery?.sourceSceneId===v.scene.sceneId&&<section className="checked-exit-recovery" aria-label="Checked withdrawal evidence">
      <h4>Recorded checked withdrawal · {r.exitRecovery.status==='accepted'?'complete':'refused'}</h4>
      <p>{r.exitRecovery.sourceNativeActions} → {r.exitRecovery.finalNativeActions} native actions · {r.exitRecovery.mode?.replaceAll('_',' ')??r.exitRecovery.reason}. Independent native replay exact.</p>
      <p>Target {r.exitRecovery.sourceMetrics.targetRemovedMm3.toFixed(3)} → {r.exitRecovery.finalMetrics.targetRemovedMm3.toFixed(3)} mm³ · non-target {r.exitRecovery.sourceMetrics.nonTargetRemovedMm3.toFixed(3)} → {r.exitRecovery.finalMetrics.nonTargetRemovedMm3.toFixed(3)} mm³.</p>
      <p>{r.exitRecovery.cost.physicalCalls} / 240 public native calls including final replay · {r.exitRecovery.wallSeconds.toFixed(1)} / 45 wall seconds. This records instrument recovery; resection task and clinical outcomes remain separate.</p>
      <button disabled={c.pending} data-exit-recovery-branch={r.exitRecovery.finalBranchId} data-exit-recovery-position={r.exitRecovery.finalNativeActions} onClick={()=>stopRun('selectBranch',{branchId:r.exitRecovery!.finalBranchId,position:r.exitRecovery!.finalNativeActions})}>Inspect withdrawal result</button>
    </section>}
      <details><summary>Accepted instrument poses and holds</summary><DeclaredFields data={v.slots}/></details>
    </details>
    <details><summary>Environment and source setup</summary>
    {r?.researchSetup&&<p role="note" className="surgical-warning" aria-label="Exposure assumptions">{r.researchSetup.description}</p>}
    <section className="environment-summary" aria-label="Environment readiness">
      <strong>{v.scene.environment?.standardized?'Standard V2 research environment':'Saved research environment'}</strong>
      <p>{v.scene.resolutionMm.map(n=>+n.toFixed(3)).join(' × ')} mm · {v2?'V2 mechanics':'Historical mechanics profile'} · RAS+ mm</p>
      {cusaModel&&<p>CUSA v3 contact model enabled · finite, uncalibrated aspiration capacity.</p>}
      <p>{Object.entries(v.scene.environment?.anatomy??{}).map(([role,status])=>`${role}: ${status==='supplied'?'mapped':status}`).join(' · ')}</p>
      <p className="surgical-note">Mapped masks may be source annotations or unreviewed estimates. See source provenance; missing anatomy and function remain unknown.</p>
      <p className={v.protection?.mode==='strict-research'?'surgical-note':'surgical-warning'} role="status">Protection: {v.protection?.mode??'unconfigured-research'}. {v.protection?.configured?`${v.protection.structures.length} declared exclusion masks; clearance ${amount(v.protection.clearanceMm)} mm.`:'No critical-structure exclusion contract is bound to this episode.'} Anatomical completeness remains unknown.</p>
      {!!v.protection?.missingRequired.length&&<p className="surgical-warning">Required evidence unavailable: {v.protection.missingRequired.join(', ')}.</p>}
      {!v.scene.environment?.standardized&&<button disabled={c.pending||readOnly} onClick={()=>void c.create(v.scene.regions)}>Create standard V2 environment</button>}
    </section>
    <section aria-label="Instrument mechanics revision"><p className="surgical-note">{(v.toolMechanicsRevision??0)>=1?'Revised contact mechanics · finite CUSA, closed bipolar contact, checked fragment exit.':'Historical instrument behavior is preserved for this episode.'}</p>{(v.toolMechanicsRevision??0)<2&&<button disabled={c.pending||readOnly||!!v.planarBladeManeuver||Object.values(v.slots).some(s=>s.inserted||s.active)||Object.keys(v.holds??{}).length>0} onClick={()=>stopRun('step',{action:{slot:c.slot,operation:'configure',mechanicsRevision:2,duration:.1}})}>{v.toolMechanicsRevision===1?'Upgrade operative tool geometry':'Enable revised instrument mechanics'}</button>}</section>
    <PatientPositionControls c={c} readOnly={readOnly||!!v.planarBladeManeuver}/>
    {!!r?.previousScenes?.length&&<label>Previous environment<select aria-label="Previous environment" value="" disabled={c.pending||readOnly} onChange={e=>{if(e.target.value!=='')stopRun('selectScene',{sceneIndex:+e.target.value});}}><option value="">Reopen a previous environment…</option>{r.previousScenes.map(s=><option key={s.index} value={s.index}>{s.label} · {s.branches} branches</option>)}</select></label>}
    <fieldset disabled={readOnly} className="surgical-controls"><details><summary>Material mapping · later rows own overlaps</summary><p className="surgical-note">Target assignment changes removal metrics. Unknown regions stay visible. Mapping creates a new scene.</p>{(mapping??v.scene.regions).map((row,i,rows)=><div className="surgical-map-row" key={row.id}><strong>{row.name}</strong><label>Material<select aria-label={`${row.name} material`} value={row.material} onChange={e=>changeMap(next=>{next[i].material=e.target.value;})}>{[...new Set([...materials,row.material])].map(m=><option key={m}>{m}</option>)}</select></label><label><input type="checkbox" checked={row.target} onChange={e=>changeMap(next=>{next[i].target=e.target.checked;})}/>Target region</label><div className="surgical-actions"><button disabled={i===0} aria-label={`Lower ${row.name} precedence`} onClick={()=>changeMap(next=>{[next[i-1],next[i]]=[next[i],next[i-1]];})}>↑ Lower priority</button><button disabled={i===rows.length-1} aria-label={`Raise ${row.name} precedence`} onClick={()=>changeMap(next=>{[next[i+1],next[i]]=[next[i],next[i+1]];})}>↓ Higher priority</button></div></div>)}<button disabled={c.pending||!mapping||JSON.stringify(mapping)===JSON.stringify(v.scene.regions)} onClick={()=>{setPlaying(false);void c.create(mapping??v.scene.regions);}}>Apply mapping / new scene</button></details></fieldset>
    </details>
    </>}
    <details><summary>Import geometry and engineering test scenes</summary>
    <button disabled={c.pending||importing||readOnly} onClick={onFixture}>Open engineering test scene</button>
    {onImport&&<details><summary>Import segmentation / geometry</summary><p className="surgical-note">Choose NIfTI labels (.nii/.nii.gz), OBJ, or STL. Declare the source coordinates and units before selecting a file. Imported segmentation has no acquired MRI intensity image.</p><fieldset disabled={c.pending||importing||readOnly} className="surgical-controls">
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
    </details>
  </section>;
}
