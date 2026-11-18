import {useEffect,useRef,useState} from 'react';
import {ChevronLeft,ChevronRight,Pause,Play,RotateCcw} from 'lucide-react';
import type {NativeAnatomyApi} from './native-anatomy-data';
import type {NativeApproach} from './native-approaches';
import type {ResectionApi} from './types';
import {operativeSteps,playableSteps,readOperativeField,timelinePose} from './operative-field';
import type {Detail,OperativeField,OperativeIntensity,OperativePhase} from './operative-field';
import {OperativeFieldCanvas} from './viewer/OperativeFieldCanvas';
import {tissueColors} from './viewer/operativeFieldAppearance';
import './operative-steps.css';

const descriptions:Record<OperativePhase,string>={
  overview:'Intact labels along the selected approach',
  bone:'Cut the modeled outline and lift the bone flap',
  dura:'Open and reflect the dural flap',
  brain:'Displace the labeled brain and hold the corridor open',
  tumor:'Advance the ultrasonic aspirator and inspect the modeled cavity',
};
const detailLabels:Record<Detail,string>={grayMatter:'Gray matter',whiteMatter:'White matter',csf:'CSF',ventricles:'Ventricular CSF',
  artery:'Artery',vein:'Vein',sinus:'Sinus',enhancing:'Enhancing compartment',nonenhancingCore:'Non-enhancing core',
  perilesionalSignal:'Perilesional signal',tumorCore:'Tumor core',wholeAbnormality:'Whole abnormality'};
const pitch=(value:number[])=>value.map(n=>Number(n.toFixed(2))).join(' × ')+' mm';

function SourceOverviewInset({image}:{image:OperativeIntensity}){
  const ref=useRef<HTMLCanvasElement>(null),view=image.overview;
  useEffect(()=>{
    if(!view||!ref.current)return;
    const ctx=ref.current.getContext('2d');
    if(!ctx)return;
    const [height,width]=view.shape,pixels=ctx.createImageData(width,height);
    for(let i=0;i<view.voxels.length;i++){
      const value=view.voxels[i];
      pixels.data.set([value,value,value,255],4*i);
    }
    ctx.putImageData(pixels,0,0);
    ctx.strokeStyle='#92e6c2';ctx.lineWidth=1.4;ctx.beginPath();
    ctx.moveTo(...view.entryPx);ctx.lineTo(...view.targetPx);ctx.stroke();
    for(const [point,color] of [[view.entryPx,'#92e6c2'],[view.targetPx,'#ff9ec8']] as const){
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(...point,2.2,0,Math.PI*2);ctx.fill();
    }
  },[view]);
  if(!view)return null;
  const orientation=(axis:number[])=>{
    const i=axis.map(Math.abs).indexOf(Math.max(...axis.map(Math.abs)));
    return axis[i]>=0?['R','A','S'][i]:['L','P','I'][i];
  };
  return <figure className="operative-overview" title={image.sourceSeries+' · full source plane through entry and target'}>
    <canvas ref={ref} width={view.shape[1]} height={view.shape[0]} aria-label="Source scan showing the saved entry and target"/>
    <span className="operative-up">{orientation(view.upRas)}</span><span className="operative-right">{orientation(view.rightRas)}</span>
    <figcaption>{image.modality} · route overview</figcaption>
  </figure>;
}
const legend={bone:'Skull',dura:'Dura',brain:'Brain',tumor:'Lesion',vessels:'Vessels'};

export function OperativeSteps({api,collectionId,route}:{
  api:NativeAnatomyApi&ResectionApi;collectionId:string;route:NativeApproach;
}){
  const [field,setField]=useState<OperativeField|null>(null);
  const [error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  useEffect(()=>{
    let live=true;
    setField(null);setError('');
    void (async()=>{
      const result=await api.anatomyPlan({collectionId,snapshotId:route.sourceSnapshotId,action:'operativeField',
        approachId:route.id,parameterHash:route.parameterHash,targetLayerId:route.targetLayerId});
      if(!live)return;
      const loaded=await readOperativeField(result,route,id=>api.readAsset(id));
      if(live)setField(loaded);
    })().catch(e=>{if(live)setError(e instanceof Error?e.message:String(e))});
    return ()=>{live=false};
  },[api,collectionId,route.sourceSnapshotId,route.id,route.parameterHash,route.targetLayerId,attempt]);
  if(!field)return <div className="operative-placeholder" role={error?'alert':'status'}>
    <strong>{error?'Operative field unavailable':'Preparing the operative field…'}</strong>
    <p>{error||'Sampling the selected approach in aligned patient anatomy.'}</p>
    {error&&<button onClick={()=>setAttempt(n=>n+1)}>Retry preview</button>}
  </div>;
  return <OperativePlayback field={field}/>;
}

function OperativePlayback({field}:{field:OperativeField}){
  const [retract,setRetract]=useState(true),[cutaway,setCutaway]=useState(true);
  const [value,setValue]=useState(0),[playing,setPlaying]=useState(false);
  const [superficialVessels,setSuperficialVessels]=useState(0);
  const clock=useRef(0),steps=playableSteps(field,retract),end=steps.length;
  const pose=timelinePose(steps,value);
  const proxies=field.layers.filter(l=>l.role==='dura'&&l.sampledVoxels>0);
  const duralProxy=proxies.length>0&&proxies.every(l=>l.proxy);
  const tissues=(Object.keys(legend) as Array<keyof typeof legend>).filter(role=>field.layers.some(l=>l.role===role&&l.sampledVoxels>0));
  const details=Array.from(new Set(field.layers.filter(l=>l.detail&&l.sampledVoxels>0&&l.role!=='tumor').map(l=>l.detail!)));
  const contacts=field.encounters.filter(s=>s.status==='contact');
  function seek(next:number){
    setPlaying(false);
    clock.current=Math.max(0,Math.min(next,end));
    setValue(clock.current);
  }
  useEffect(()=>{
    if(!playing)return;
    let frame:number,previous:number|undefined;
    function tick(now:number){
      if(previous!==undefined)clock.current=Math.min(end,clock.current+Math.min(150,now-previous)/4200);
      previous=now;setValue(clock.current);
      if(clock.current<end)frame=requestAnimationFrame(tick);
      else setPlaying(false);
    }
    frame=requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(frame);
  },[playing,end]);
  function togglePlay(){
    if(value>=end){clock.current=0;setValue(0);}
    setPlaying(on=>!on);
  }
  function toggleRetractor(){
    const next=!retract,available=playableSteps(field,next);
    const index=available.findIndex(step=>step.id===pose.phase);
    const nextValue=index>=0?index+pose.progress:pose.phase==='brain'
      ?available.filter(step=>step.id==='bone'||step.id==='dura').length:0;
    setPlaying(false);clock.current=nextValue;setValue(nextValue);setRetract(next);
  }
  return <section className="operative-playback" aria-label="Operative steps" data-operative-phase={pose.phase} data-operative-progress={pose.progress.toFixed(3)}>
    <div className="operative-topline"><div><strong>Operative field</strong><small>{field.spacingMm.toFixed(2)} mm display grid · Geometric rehearsal</small></div>
      <details className="operative-sources"><summary>Field evidence</summary><div>
        {field.intensity&&<p><b>Surface contrast</b><br/>{field.intensity.modality} · {field.intensity.sourceSeries} · acquired scan intensities</p>}
        {superficialVessels>0&&<p><b>Exposed skull</b><br/>{superficialVessels.toLocaleString()} superficial vessel samples are hidden with the overlying tissue. Anatomy shows the source estimates.</p>}
        <h3>Layers in this field</h3>
        {field.layers.map(layer=><p key={layer.id}><b>{layer.label}</b><br/>{layer.sourceSeries} · {layer.evidence}{layer.reviewStatus?' · '+layer.reviewStatus.replaceAll('_',' '):''}{layer.sourceSpacingMm&&<><br/>Scan {pitch(layer.sourceSpacingMm)} · label {pitch(layer.labelSpacingMm??layer.sourceSpacingMm)}</>}
          {Array.isArray(layer.inferenceSpacingMm)&&<> · inference {pitch(layer.inferenceSpacingMm)}</>}
          {layer.adjudication&&<><br/>CT preferred within coverage · {layer.adjudication.disagreementVoxels.toLocaleString()} differing display cells</>}
          {layer.sampledVoxels===0?' · No labeled cells in this field':''}</p>)}
        {field.excluded.length>0&&<><h3>Excluded layers</h3>{field.excluded.map(l=><p key={l.id}>{l.label} · {l.reason}</p>)}</>}
        {[...new Set([...field.blockers,...field.stationIssues,...field.unknowns])].map(reason=><p key={reason}>{reason}</p>)}
      </div></details>
    </div>
    <div className="operative-image"><OperativeFieldCanvas field={field} phase={pose.phase} progress={pose.progress} withRetraction={retract} cutaway={cutaway} onSuperficialVessels={setSuperficialVessels}/>
      {field.intensity?.overview&&<SourceOverviewInset image={field.intensity}/>}
      <div className="operative-legend" aria-label="Visible tissues">{tissues.map(role=><span key={role} data-tissue={role}>{role==='dura'&&duralProxy?'Dural proxy':legend[role]}</span>)}
        {details.map(key=><span key={key} style={{'--tissue':tissueColors[key]} as React.CSSProperties}>{detailLabels[key]}</span>)}
        {!tissues.includes('vessels')&&<small>Vessels unmapped here</small>}
      </div>
    </div>
    <div className="operative-transport">
      {(contacts.length>0||field.blockers.length>0)&&<div className="operative-conflict">{contacts.length?'Intersection with '+contacts.map(c=>c.label).join(', '):field.blockers[0]}</div>}
      <div className="operative-stage-title"><strong>{descriptions[pose.phase]}</strong><span>{pose.phase==='overview'?'Start':Math.round(pose.progress*100)+'% of step'}</span></div>
      <div className="operative-scrub"><button aria-label={playing?'Pause steps':'Play steps'} disabled={!end} onClick={togglePlay}>{playing?<Pause size={16}/>:<Play size={16}/>}</button>
        <input aria-label="Operative timeline" aria-valuetext={descriptions[pose.phase]+' · '+Math.round(pose.progress*100)+'% of step'} type="range" min={0} max={end||1} step={.01} value={value} disabled={!end} onChange={e=>seek(+e.target.value)}/>
        <button aria-label="Previous step" disabled={value<=0} onClick={()=>seek(Math.max(0,Math.ceil(value)-1))}><ChevronLeft size={16}/></button>
        <button aria-label="Next step" disabled={value>=end} onClick={()=>seek(Math.floor(value)+1)}><ChevronRight size={16}/></button>
        <button aria-label="Reset operative steps" disabled={!value&&!playing} onClick={()=>seek(0)}><RotateCcw size={15}/></button>
      </div>
      <div className="operative-chapters">{operativeSteps.map(step=>{
        const index=steps.findIndex(s=>s.id===step.id);
        const absent=field.stations[step.id]===null;
        const detail=absent?'No label on path':step.id==='brain'&&!retract?'Omitted':
          step.id==='tumor'&&index<0?'No eligible tip position':field.stationEvidence[step.id]==='proxy'?'Geometric proxy':null;
        return <button key={step.id} aria-label={step.label} aria-pressed={pose.phase===step.id} disabled={index<0}
          onClick={()=>seek(index+1)}>{step.label}{detail&&<small>{detail}</small>}</button>;
      })}</div>
      <div className="operative-options">
        <label><input type="checkbox" checked={cutaway} onChange={e=>setCutaway(e.target.checked)}/>Inspection cutaway</label>
        <label><input type="checkbox" checked={retract&&field.stations.brain!==null} disabled={field.stations.brain===null} onChange={toggleRetractor}/>Include retraction</label>
      </div>
    </div>
  </section>;
}
