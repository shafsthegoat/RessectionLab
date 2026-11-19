import {useLayoutEffect,useMemo,useRef,useState} from 'react';
import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import {modeledAnatomy} from './modeledAnatomy';
import {inspectionSource,inspectionSection,sectionClasses} from './engineInspection';
import type {InspectionSection} from './engineInspection';
import './engineInspection.css';

const axisNames=['I','J','K'];
const volumeLabel=(n:unknown)=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:2}):'—';

function paintSection(canvas:HTMLCanvasElement,section:InspectionSection,labelsOnly:boolean) {
  const {width,height,membership,signal}=section;
  const sx=Math.max(2,Math.min(24,Math.round(480/width))),sy=Math.max(2,Math.min(24,Math.round(480/height)));
  canvas.width=width*sx;canvas.height=height*sy;
  const ctx=canvas.getContext('2d');if(!ctx)return false;
  const values=Array.from(signal).filter(Number.isFinite).sort((a,b)=>a-b);
  const low=values[Math.floor(values.length*.01)]??0,high=values[Math.floor(values.length*.99)]??1;
  const grey=(s:number)=>Number.isFinite(s)?Math.round(18+172*Math.min(1,Math.max(0,(s-low)/Math.max(1e-6,high-low)))):18;
  const image=document.createElement('canvas');image.width=width;image.height=height;
  const scan=image.getContext('2d');if(!scan)return false;
  const pixels=scan.createImageData(width,height);
  signal.forEach((s,i)=>{const g=labelsOnly?31:grey(s);pixels.data.set([g,g,g,255],i*4);});
  scan.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(image,0,0,canvas.width,canvas.height);
  // Every colored rectangle represents one exact engine-grid cell. Luminance
  // within retained labels follows the corresponding source image sample.
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=y*width+x,kind=membership[i];if(!kind)continue;
    const g=labelsOnly?80:grey(signal[i]);
    if(kind===sectionClasses.target)ctx.fillStyle=`rgb(${28+g*.16},${102+g*.36},${115+g*.38})`;
    else if(kind===sectionClasses.cavity)ctx.fillStyle='#553916';
    else if(kind===sectionClasses.vessel)ctx.fillStyle='#df6175';
    else ctx.fillStyle='#b995ec';
    ctx.fillRect(x*sx,y*sy,sx,sy);
  }
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=y*width+x,kind=membership[i];
    if(kind!==sectionClasses.target&&kind!==sectionClasses.cavity)continue;
    ctx.fillStyle=kind===sectionClasses.cavity?'#ffc36b':'#82dbde';
    const w=kind===sectionClasses.cavity?Math.max(1,Math.min(3,sx/5)):1;
    if(!x||membership[i-1]!==kind)ctx.fillRect(x*sx,y*sy,w,sy);
    if(x===width-1||membership[i+1]!==kind)ctx.fillRect((x+1)*sx-w,y*sy,w,sy);
    if(!y||membership[i-width]!==kind)ctx.fillRect(x*sx,y*sy,sx,w);
    if(y===height-1||membership[i+width]!==kind)ctx.fillRect(x*sx,(y+1)*sy-w,sx,w);
  }
  return true;
}

export function EngineOperatedSection({volume,view,labelsOnly,sourceLabel}:{
  volume:ViewerVolume;view:SurgicalView;labelsOnly:boolean;sourceLabel:string;
}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const owner=modeledAnatomy(volume)?.surfaceReference?.owner;
  const [axis,setAxis]=useState<0|1|2>(2),[depth,setDepth]=useState<number|null>(null);
  const base=useMemo(()=>owner?inspectionSource(owner,view):null,[owner,view.scene.sceneId]);
  const section=useMemo(()=>base?inspectionSection(volume,view,base,axis,depth):null,[volume,view,base,axis,depth]);
  useLayoutEffect(()=>{
    const node=canvas.current;if(!node||!section)return;
    delete node.dataset.engineInspectionState;
    if(paintSection(node,section,labelsOnly))node.dataset.engineInspectionState=view.stateId;
    return ()=>{delete node.dataset.engineInspectionState;};
  },[view.stateId,section,labelsOnly]);
  if(!section)return <section className="engine-section" role="status">Operated section unavailable for this source mapping.</section>;
  const visibleTips=section.tips.filter(p=>p.inPlane&&p.x>=0&&p.y>=0&&p.x<=section.width&&p.y<=section.height);
  const outside=section.tips.length-visibleTips.length;
  return <section className="engine-section" aria-label="Operated anatomy section">
    <header><h3>Operated anatomy</h3><span>{labelsOnly?'Engine labels':sourceLabel} · {section.widthMm.toFixed(0)} mm field</span></header>
    <div className="engine-section-image">
      <div className="engine-section-frame" style={{aspectRatio:`${section.widthMm} / ${section.heightMm}`}}>
        <canvas ref={canvas} aria-label="Source section with residual tissue and modeled cavity"
          data-slice-axis={axis} data-slice-index={section.depth} data-slice-origin={JSON.stringify(section.origin)}
          data-slice-size={JSON.stringify([section.width,section.height])} data-source-case={volume.caseHash}
          data-scene-id={view.scene.sceneId} data-cavity-cells={section.cavityPixels} data-residual-cells={section.targetPixels}/>
        <svg viewBox={`0 0 ${section.width} ${section.height}`} preserveAspectRatio="none" aria-hidden="true">
          {visibleTips.map(p=><g key={p.hand} transform={`translate(${p.x} ${p.y})`}>
            <circle r={section.width*.029}/><path d={`M ${-section.width*.046} 0 h ${section.width*.022} M ${section.width*.024} 0 h ${section.width*.022} M 0 ${-section.height*.046} v ${section.height*.022} M 0 ${section.height*.024} v ${section.height*.022}`}/>
          </g>)}
        </svg>
        {section.labels.map((label,i)=><b key={i} className={`edge edge-${i}`}>{label}</b>)}
      </div>
    </div>
    <div className="engine-section-key"><span className="residual">Residual target</span><span className="cavity">Removed space</span>
      {section.hasVessels&&<span className="vessels">Mapped vessels</span>}{section.hasProtected&&<span className="protected">Protected label</span>}
      {!!visibleTips.length&&<span className="tool">Tool tip</span>}</div>
    <div className="engine-section-metrics">
      <div><span>Target removed</span><strong>{volumeLabel(view.metrics.targetRemovedMm3)} <small>mm³</small></strong></div>
      <div><span>Whole target left</span><strong>{volumeLabel(view.originalTargetResidualMm3)} <small>mm³</small></strong></div>
      <p>Non-target removed <b>{volumeLabel(view.metrics.nonTargetRemovedMm3)} mm³</b></p>
    </div>
    <div className="engine-section-position">
      <label>Grid <select value={axis} aria-label="Operated section axis" onChange={e=>{setAxis(Number(e.target.value) as 0|1|2);setDepth(null);}}>
        {axisNames.map((a,i)=><option key={a} value={i}>{a}</option>)}</select></label>
      <input aria-label="Operated section slice" type="range" min={0} max={view.scene.shape[axis]-1} value={section.depth} onChange={e=>setDepth(Number(e.target.value))}/>
      <output>{section.depth+1}/{view.scene.shape[axis]}</output>
    </div>
    <div className="engine-section-status"><button onClick={()=>setDepth(null)} disabled={depth===null}>{depth===null?'Following effect':'Follow effect'}</button>
      <span>{outside?`${outside===1?'Tip':'Tips'} outside section`:!section.tips.length?'No tool inserted':visibleTips.map(p=>p.instrument.replaceAll('_',' ')).join(' · ')}</span></div>
  </section>;
}
