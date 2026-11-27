import {useEffect, useMemo, useRef} from 'react';
import {cIndex, sampleTrilinear, transformPoint} from './coordinates';
import type {Affine, Bounds3, Point3} from './coordinates';
import type {ViewerVolume} from './contracts';
import {microscopeCamera} from './microscope';
import {mprPixel, mprPoint, mprView} from './mprFraming';
import type {MicroscopeSettings} from './workspaceSettings';

const width=144, height=112;

export function OperativeSourceLocator({source, bounds, inverse, microscope, labelsOnly, stateId}:{
  source:ViewerVolume; bounds:Bounds3; inverse:Affine; microscope:MicroscopeSettings;
  labelsOnly:boolean; stateId:string;
}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const focus=microscopeCamera(microscope).target;
  const inside=transformPoint(inverse,focus).every((v,a)=>v>=0&&v<=source.shape[a]-1);
  const planeX=inside?focus[0]:(bounds[0][0]+bounds[1][0])/2;
  const plane=useMemo(()=>mprView(bounds,bounds,'sagittal',width,height,{fit:'image'}),[bounds]);
  const [cx,cy]=mprPixel(plane,'sagittal',focus);
  useEffect(()=>{
    const ctx=canvas.current?.getContext('2d');
    if(!ctx)return;
    const cursor:Point3=[planeX,0,0], image=ctx.createImageData(width,height);
    const values=new Float32Array(width*height).fill(NaN), samples:number[]=[];
    const labelSets=new Map(source.labelSets?.map(set=>[set.id,set.data]));
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const point=mprPoint(plane,'sagittal',cursor,x+.5,y+.5,inverse,source.shape);
      if(!point)continue;
      const voxel=transformPoint(inverse,point);
      let value:number;
      if(labelsOnly){
        const index=cIndex(source.shape,voxel.map(Math.round) as Point3);
        value=source.compartments.some(layer=>layer.mask[index]||
          (layer.labelSet&&layer.labelValue!==undefined&&labelSets.get(layer.labelSet)?.[index]===layer.labelValue))?1:0;
      }else value=sampleTrilinear(source.mri,source.shape,voxel);
      values[y*width+x]=value;
      if(value!==0&&Number.isFinite(value))samples.push(value);
    }
    samples.sort((a,b)=>a-b);
    const low=labelsOnly?0:Math.min(0,samples[Math.floor(samples.length*.01)]??0);
    const high=labelsOnly?1:samples[Math.floor(samples.length*.99)]??1;
    values.forEach((value,index)=>{
      if(!Number.isFinite(value))return;
      const grey=Math.round(210*Math.max(0,Math.min(1,(value-low)/Math.max(1e-8,high-low))));
      image.data.set([grey,grey,grey,255],4*index);
    });
    ctx.putImageData(image,0,0);
  },[plane,planeX,inverse,source,labelsOnly]);
  // Folded to a small chip by default so the operative image keeps the corner.
  return <details className="operative-source-locator-fold" onDoubleClick={event=>event.stopPropagation()}>
    <summary title="Show where the microscope focus lies in the whole source">Source extent</summary>
    <figure className="operative-source-locator" data-operative-locator-state={stateId}
    data-source-case={source.caseHash} data-focus-inside={inside} aria-label="Whole-source operative locator">
    <figcaption>Source extent · sagittal</figcaption>
    <div><canvas ref={canvas} width={width} height={height}/>
      {inside&&<svg viewBox={`0 0 ${width} ${height}`} aria-hidden="true"><circle cx={cx} cy={cy} r="5"/><path d={`M${cx-8},${cy}h5m6,0h5M${cx},${cy-8}v5m0,6v5`}/></svg>}
    </div>
    <span>{inside?'Microscope focus':'Focus outside source'}{labelsOnly?' · labels':''}</span>
  </figure></details>;
}
