import {useLayoutEffect,useMemo,useRef} from 'react';
import {Vector3} from 'three';
import {cIndex,inverseAffine,rasAffine,sampleTrilinear,transformPoint} from './viewer/coordinates';
import type {Point3} from './viewer/coordinates';
import type {VerifiedStudyReplay} from './types';
import type {SurgicalView} from './surgical-types';
import type {ViewerVolume,ViewerWorkspaceProps} from './viewer/contracts';
import {ViewerWorkspace} from './viewer/ViewerWorkspace';
import {recordedOverviewRoute} from './study-engine-presentation';

type ReferenceProps=Pick<ViewerWorkspaceProps,'caseData'|'visibleLayers'|'workspaceSettings'|'cursor'|'overlayOpacity'>;
const noCursorChange=()=>{};

function RecordedRoutePlane({source,route,targetId}:{source:ViewerVolume;targetId?:string;route:NonNullable<VerifiedStudyReplay['recordedApproach']>}){
  const canvas=useRef<HTMLCanvasElement>(null);
  const width=198,height=134;
  const frame=useMemo(()=>{
    const entry=new Vector3(...route.entry),target=new Vector3(...route.target),center=entry.clone().lerp(target,.5);
    const along=target.clone().sub(entry),length=along.length();
    if(!length)return null;
    along.normalize();
    const up=Math.abs(along.z)<.9?new Vector3(0,0,1):new Vector3(0,1,0);
    up.addScaledVector(along,-up.dot(along)).normalize();
    const spacing=Math.max(32,length*1.85)/width;
    return {center,along,up,spacing,half:length/(2*spacing)};
  },[route]);
  useLayoutEffect(()=>{
    const ctx=canvas.current?.getContext('2d');
    if(!frame||!ctx)return;
    const inverse=inverseAffine(rasAffine(source.affine,source.frame));
    const {center,along,up,spacing}=frame;
    const samples=new Float32Array(width*height),membership=new Uint8Array(width*height);
    const layer=source.compartments.find(c=>c.name===targetId);
    const labels=layer?.labelSet?source.labelSets?.find(set=>set.id===layer.labelSet)?.data:undefined;
    const point=new Vector3();
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      point.copy(center).addScaledVector(along,(x+.5-width/2)*spacing).addScaledVector(up,(height/2-y-.5)*spacing);
      const voxel=transformPoint(inverse,point.toArray() as Point3),i=y*width+x;
      samples[i]=sampleTrilinear(source.mri,source.shape,voxel);
      const cell=voxel.map(Math.round) as Point3;
      if(layer&&cell.every((v,k)=>v>=0&&v<source.shape[k])){
        const j=cIndex(source.shape,cell);
        membership[i]=Number(!!layer.mask[j]||(!!labels&&layer.labelValue===labels[j]));
      }
    }
    const sorted=samples.slice().sort(),low=sorted[Math.floor(.02*sorted.length)],high=sorted[Math.floor(.98*sorted.length)];
    const pixels=ctx.createImageData(width,height);
    samples.forEach((v,i)=>{
      const gray=Math.max(0,Math.min(255,(v-low)*235/Math.max(1e-8,high-low)));
      const edge=membership[i]&&(i%width===0||i%width===width-1||!membership[i-1]||!membership[i+1]||!membership[i-width]||!membership[i+width]);
      pixels.data.set(edge?[98,191,212,255]:[gray,gray,gray,255],i*4);
    });
    ctx.putImageData(pixels,0,0);
  },[source,frame,targetId]);
  if(!frame)return null;
  const a=width/2-frame.half,b=width/2+frame.half,h=height/2;
  return <figure className="engine-route-plane" aria-label="Recorded entry and target on intact source">
    <div><canvas ref={canvas} width={width} height={height}/>
      <svg viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
        <path d={`M ${a} ${h} H ${b}`} stroke="#15251d" strokeWidth="3.5"/>
        <path d={`M ${a} ${h} H ${b}`} stroke="#a3f1cb" strokeWidth="1.5"/>
        <circle cx={a} cy={h} r="3" fill="#a3f1cb" stroke="#15251d"/>
        <circle cx={b} cy={h} r="3" fill="#62bfd4" stroke="#15251d"/>
        <text x={a} y={h-8}>Entry</text><text x={b} y={h-8}>Target</text>
      </svg>
    </div><figcaption>Source plane through approach</figcaption>
  </figure>;
}

export function EngineStudyReference({receipt,nativeSource,reference,targetLabel,engineScene}:{
  receipt:VerifiedStudyReplay;nativeSource:boolean;reference:ReferenceProps;targetLabel:string;engineScene?:SurgicalView['scene'];
}){
  const route=receipt.recordedApproach;
  const engineTargets=engineScene?.sceneId===receipt.sceneId?engineScene.regions.filter(r=>r.target):[];
  const referenceTarget=nativeSource?receipt.targetLayerId:engineTargets.length===1?engineTargets[0].name:undefined;
  const routes=useMemo(()=>route?[recordedOverviewRoute(route)]:[],[route]);
  const settings=useMemo(()=>({...reference.workspaceSettings,operativeView:'external' as const,
    layout:'3d-focus' as const,camera:null,externalCamera:null,showPlane:false}),[reference.workspaceSettings]);
  return <aside className="engine-intact-reference" aria-label="Intact patient and recorded approach"
    data-study-snapshot={receipt.sourceSnapshotId} data-recorded-approach={route?.id} data-target-layer={receipt.targetLayerId}>
    <header><strong>Intact anatomy</strong><span>{receipt.patientId} · {nativeSource?'Native study':'Recorded source grid'}</span></header>
    <div className="engine-reference-image">
      <ViewerWorkspace {...reference} workspaceSettings={settings} routes={routes}
        onCursorChange={noCursorChange} cameraMode="anatomy"/>
    </div>
    {route&&reference.caseData&&<RecordedRoutePlane source={reference.caseData} route={route} targetId={referenceTarget}/>}
    <footer>{route&&<><strong>Recorded approach</strong><span>{route.name}</span></>}
      <strong>Target</strong><span>{targetLabel}</span>
      {receipt.actionGenerator&&<><strong>Action generator</strong><span>{receipt.actionGenerator.kind} · {receipt.actionGenerator.name}</span></>}
    </footer>
  </aside>;
}
