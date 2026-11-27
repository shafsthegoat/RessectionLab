import {useEffect, useMemo, useRef} from 'react';
import type {Affine, Point3} from './coordinates';
import type {ViewerVolume} from './contracts';
import {microscopeCamera} from './microscope';
import type {MicroscopeSettings} from './workspaceSettings';
import {depthRuns, exteriorAbove, paintSection, sampleSection, sectionWindow} from './operativeSection';

const STEP_MM=.5, WIDTH_MM=60, ABOVE_MM=6, DEPTH_MM=70;
const sub=(a:Point3,b:Point3):Point3=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const unit=(a:Point3):Point3=>{const l=Math.hypot(...a)||1;return [a[0]/l,a[1]/l,a[2]/l];};
const cross=(a:Point3,b:Point3):Point3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const along=(p:Point3,d:Point3,s:number):Point3=>[p[0]+d[0]*s,p[1]+d[1]*s,p[2]+d[2]*s];

/** Inspection cutaway companion: a longitudinal section along the microscope
 * axis through the focus, from the skin down, with the case scan on the cut
 * and the layers met along the axis listed by depth. Display only. */
export function OperativeDepthSection({source, inverse, microscope, stateId}:{
  source:ViewerVolume; inverse:Affine; microscope:MicroscopeSettings; stateId:string;
}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const camera=microscopeCamera(microscope);
  const key=JSON.stringify(camera);
  const section=useMemo(()=>{
    const down=unit(sub(camera.target,camera.position)),across=unit(cross(down,camera.up));
    const layers=source.compartments.filter(layer=>layer.mask.length===source.mri.length);
    // Find the skin on the axis: march in from outside the working grid.
    const probe=sampleSection(layers,source.mri,source.shape,inverse,{origin:along(camera.target,down,-120),across,down,width:1,height:480,stepMm:STEP_MM});
    let surface=-1;for(let y=0;y<probe.height;y++)if(probe.label[y]){surface=y;break;}
    if(surface<0)return null;
    const skin=along(camera.target,down,-120+(surface+.5)*STEP_MM);
    const width=Math.round(WIDTH_MM/STEP_MM),height=Math.round((ABOVE_MM+DEPTH_MM)/STEP_MM);
    const origin=along(along(skin,down,-ABOVE_MM),across,-WIDTH_MM/2);
    const sample=sampleSection(layers,source.mri,source.shape,inverse,{origin,across,down,width,height,stepMm:STEP_MM});
    const exterior=exteriorAbove(sample);
    const pixels=paintSection(sample,layers,sectionWindow(sample),exterior);
    const {runs}=depthRuns(sample,layers,Math.floor(width/2));
    const focusMm=(sub(camera.target,skin)).reduce((s,v,a)=>s+v*down[a],0);
    return {width,height,pixels,runs,focusMm};
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[source,inverse,key]);
  useEffect(()=>{
    const node=canvas.current,ctx=node?.getContext('2d');
    if(!node||!ctx||!section)return;
    node.width=section.width;node.height=section.height;
    ctx.putImageData(new ImageData(section.pixels,section.width,section.height),0,0);
  },[section]);
  if(!section)return null;
  const {width,height,runs,focusMm}=section,scale=1/STEP_MM,top=ABOVE_MM*scale;
  const ticks=Array.from({length:Math.floor(DEPTH_MM/10)+1},(_,i)=>i*10);
  return <figure className="operative-depth-section" data-operative-depth-state={stateId} aria-label="Longitudinal section along the microscope axis">
    <figcaption>Section along the microscope axis · {WIDTH_MM} mm wide</figcaption>
    <div className="operative-depth-body">
      <div className="operative-depth-image" style={{aspectRatio:`${width} / ${height}`}}>
        <canvas ref={canvas}/>
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          <line className="axis" x1={width/2} x2={width/2} y1={top} y2={height}/>
          {focusMm>-ABOVE_MM&&focusMm<DEPTH_MM&&<line className="focus" x1={width/2-14} x2={width/2+14} y1={top+focusMm*scale} y2={top+focusMm*scale}/>}
          {ticks.map(t=><line key={t} className="tick" x1={0} x2={t%20?4:8} y1={top+t*scale} y2={top+t*scale}/>)}
        </svg>
        {ticks.filter(t=>t%20===0).map(t=><b key={t} className="depth-label" style={{top:`${100*(top+t*scale)/height}%`}}>{t}</b>)}
      </div>
      <ol className="operative-depth-key" aria-label="Layers along the microscope axis from the skin, in millimetres">
        {runs.slice(0,12).map((run,i)=><li key={i}><i style={{background:run.colour}}/>
          <span>{run.name}</span><small>{run.fromMm.toFixed(0)}–{run.toMm.toFixed(0)}</small></li>)}
      </ol>
    </div>
    <span className="operative-depth-note">Depth from the skin on the axis (mm) · scan intensity on the cut · focus {focusMm.toFixed(0)} mm</span>
  </figure>;
}
