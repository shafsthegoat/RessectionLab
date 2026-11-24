import {useEffect,useRef,useState} from 'react';
import type {NativeWorkspace} from './use-native-workspace';
import type {LocalAlignmentEvidence,RegistrationSection} from './native-anatomy-data';
import {transformPoint,type Point3} from './viewer/coordinates';
import {scanLabel} from './native-study-state';

type Side='moving'|'reference';
type Pick={source:Point3;patient:Point3};
type Pair=Record<Side,Pick>;
type ImagePlane={geometry:RegistrationSection;image:ImageData};
const normals:Record<string,Point3>={Axial:[0,0,1],Coronal:[0,1,0],Sagittal:[1,0,0]};
const product=(a:Point3,b:Point3)=>a.reduce((sum,n,i)=>sum+n*b[i],0);
const probeLabels={insufficient_source_coverage:'Incomplete source coverage',weak_image_structure:'Weak image structure',
  search_boundary:'Peak at search limit',competing_matches:'Competing matches',distinguishable_peak:'Distinct local peak'};

function LocalEvidence({value,blocked,onInspect}:{value:LocalAlignmentEvidence;blocked:boolean;onInspect:(point:Point3)=>void}){
  const overlap=(100*value.commonCoverageCount/value.sampleCount).toFixed(1);
  const distinct=value.patches.filter(p=>p.status==='distinguishable_peak').length;
  return <details open>
    <summary>{value.fieldWidthMm} mm alignment screen</summary>
    <p>Common acquired coverage {overlap}% · {distinct} of {value.patches.length} distinct local peaks · {value.transformStatus}</p>
    <p>Image comparisons suggest where to check paired landmarks. Review status is unchanged.</p>
    <details><summary>Local measurements</summary>
      {value.patches.map(p=><div key={p.centerRasMm.join(',')}>
        <strong>[{p.centerRasMm.map(n=>n.toFixed(1)).join(', ')}] RAS mm</strong> · {probeLabels[p.status]}
        {p.movingSamplingShiftMm&&<span> · moving sample shift [{p.movingSamplingShiftMm.join(', ')}] mm</span>}
        {p.candidateScore!==undefined&&p.bestScore!==undefined&&<span> · edge agreement {p.candidateScore.toFixed(3)} → {p.bestScore.toFixed(3)}</span>}
        <button disabled={blocked} onClick={()=>onInspect(p.centerRasMm)}>Load paired slices here</button>
      </div>)}
    </details>
  </details>;
}

function Slice({plane,label,marks,onPick}:{plane:ImagePlane;label:string;marks:Array<{pick:Pick;name:string}>;onPick:(pick:Pick)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null);
  const g=plane.geometry,mid=(g.size-1)/2;
  useEffect(()=>{canvas.current?.getContext('2d')?.putImageData(plane.image,0,0)},[plane]);
  return <figure style={{margin:'4px 0'}}><figcaption>{label}</figcaption><div style={{position:'relative',lineHeight:0}}>
    <canvas style={{width:'100%',display:'block',cursor:'crosshair'}} role="img" aria-label={label+' landmark slice'}
      ref={canvas} width={g.size} height={g.size} onClick={event=>{
        const rect=event.currentTarget.getBoundingClientRect();
        const x=Math.floor((event.clientX-rect.left)/rect.width*g.size);
        const y=Math.floor((event.clientY-rect.top)/rect.height*g.size);
        const patient=g.center.map((c,i)=>c+g.spacingMm*((x-mid)*g.u[i]+(y-mid)*g.v[i])) as Point3;
        const voxel=transformPoint(g.voxelFromSection!,patient);
        if(voxel.some((n,i)=>n<0||n>g.sourceShape![i]-1))return;
        onPick({patient,source:transformPoint(g.sourceFromSection!,patient)});
      }}/>
    {marks.map(({pick,name})=>{
      const delta=pick.patient.map((n,i)=>n-g.center[i]) as Point3;
      const x=product(delta,g.u),y=product(delta,g.v);
      if(Math.hypot(...delta.map((n,i)=>n-x*g.u[i]-y*g.v[i]))>g.spacingMm/2)return null;
      const left=(mid+x/g.spacingMm+.5)/g.size,top=(mid+y/g.spacingMm+.5)/g.size;
      if(left<0||top<0||left>=1||top>=1)return null;
      return <span key={name} style={{position:'absolute',left:(100*left)+'%',top:(100*top)+'%',color:'#ffdd49',
        textShadow:'0 0 3px black',pointerEvents:'none',lineHeight:1,transform:'translate(-5px, -50%)'}}>+{name}</span>;
    })}
  </div></figure>;
}

export function NativeRegistrationReview({workspace:w,blocked}:{workspace:NativeWorkspace;blocked:boolean}){
  const doc=w.binding!.document;
  const moving=doc.series.find(s=>s.id===w.series),reference=doc.series.find(s=>s.id===doc.reference_series);
  const [axis,setAxis]=useState('Axial'),[focus,setFocus]=useState('centre');
  const [shift,setShift]=useState({moving:0,reference:0});
  const [planes,setPlanes]=useState<Record<Side,ImagePlane>|null>(null);
  const [pairs,setPairs]=useState<Pair[]>([]),[pending,setPending]=useState<Partial<Pair>>({});
  const [manual,setManual]=useState(''),[error,setError]=useState('');
  const [local,setLocal]=useState<LocalAlignmentEvidence|null>(null);
  if(!moving||!reference||moving.id===reference.id)return null;
  const saved=doc.transforms[moving.id]?.landmarks;
  const route=w.planning.selected;
  const middle=reference.affine&&reference.data?transformPoint(reference.affine,reference.data.shape.map(n=>(n-1)/2) as Point3):null;
  const routeCurrent=route?.sourceSnapshotId===doc.snapshot_id&&route.assessmentStatus==='current';
  const center=focus==='cursor'?w.state.cursor:focus==='centre'?middle:
    route&&routeCurrent?(focus==='midpoint'?route.entry.map((n,i)=>(n+route.target[i])/2) as Point3:focus==='entry'?route.entry:route.target):null;
  function pick(side:Side,point:Pick){
    const next={...pending,[side]:point};
    if(next.moving&&next.reference){setPairs([...pairs,{moving:next.moving,reference:next.reference}]);setPending({});}
    else setPending(next);
  }
  function sample(point=center){
    if(!point)return;
    const normal=normals[axis],api=w.api!;
    setError('');setPlanes(null);
    void w.run(async()=>{
      const take=async(id:string,side:Side)=>{
        const g=await api.anatomySection({...w.args(),seriesId:id,center:point.map((n,i)=>n+shift[side]*normal[i]),
          normal,spacingMm:.45,registrationReview:true});
        if(!g.sourceFromSection||!g.voxelFromSection||!g.sourceShape)throw Error('Source section geometry unavailable');
        const bytes=await api.readAsset(g.pixels.assetId);
        return {geometry:g,image:new ImageData(new Uint8ClampedArray(bytes),g.size,g.size)};
      };
      try{
        const referencePlane=await take(reference!.id,'reference');
        const movingPlane=await take(moving!.id,'moving');
        setPlanes({reference:referencePlane,moving:movingPlane});
      }catch(e){setError(e instanceof Error?e.message:String(e));}
    });
  }
  function screenLocal(){
    if(!center)return;
    setError('');setLocal(null);
    void w.run(async()=>{
      try{
        setLocal(await w.api!.anatomyLocalAlignment({...w.args(),seriesId:w.series,center}));
      }catch(e){setError(e instanceof Error?e.message:String(e));}
    });
  }
  const visibleLocal=local&&center&&local.snapshotId===doc.snapshot_id&&local.centerRasMm.every((v,i)=>v===center[i])?local:null;
  const disabled=blocked||w.binding!.readOnlyStudy||!w.author.trim();
  function accept(movingPoints:Point3[],referencePoints:Point3[]){
    return w.change(()=>w.api!.anatomyRegister({...w.args(),seriesId:w.series,action:'review',movingPoints,referencePoints,reviewer:w.author}));
  }
  return <div className="registration-review">
    {saved&&!Array.isArray(saved)&&Array.isArray(saved.errors_mm)&&<details><summary>Saved landmark review · {saved.errors_mm.length} pairs</summary>
      <p>{saved.reviewer} · source coordinates in RAS mm</p>
      {saved.errors_mm.map((error,i)=><p key={i}>Pair {i+1} · {error.toFixed(2)} mm · moving [{saved.moving[i]?.map(n=>n.toFixed(1)).join(', ')}] · reference [{saved.reference[i]?.map(n=>n.toFixed(1)).join(', ')}]</p>)}
    </details>}
    <label>Review position<select value={focus} onChange={e=>{setPlanes(null);setFocus(e.target.value)}}>
    <option value="centre">Reference scan centre</option><option value="cursor" disabled={w.registration==='independent source frame'||!w.state.cursor}>Slice cursor</option>
    {routeCurrent&&['entry','midpoint','target'].map(p=><option value={p} key={p}>{route!.name} · {p}</option>)}
  </select></label><label>Plane<select value={axis} onChange={e=>{setPlanes(null);setAxis(e.target.value)}}>{Object.keys(normals).map(a=><option key={a}>{a}</option>)}</select></label>
    {(['reference','moving'] as const).map(side=><label key={side}>{side==='moving'?'Moving':'Reference'} slice offset · mm
      <input type="number" step="1" value={shift[side]} onChange={e=>{setPlanes(null);setShift({...shift,[side]:Number(e.target.value)})}}/></label>)}
    <button disabled={blocked||!center} onClick={()=>sample()}>Load paired source slices</button>
    <button disabled={blocked||!center} onClick={screenLocal}>Check local alignment evidence</button>
    {error&&<p role="alert">{error}</p>}
    {visibleLocal&&<LocalEvidence value={visibleLocal} blocked={blocked} onInspect={sample}/>}
    {planes&&<><p>Click the same anatomical feature in both slices. Purple marks space outside acquisition coverage.</p>
      {(['reference','moving'] as const).map(side=><Slice key={side} plane={planes[side]} label={side+' · '+scanLabel(side==='moving'?moving:reference)}
        marks={[...pairs.map((p,i)=>({pick:p[side],name:String(i+1)})),...(pending[side]?[{pick:pending[side]!,name:String(pairs.length+1)}]:[])]}
        onPick={point=>pick(side,point)}/>)}</>}
    {pairs.map((p,i)=><div key={i}>Pair {i+1} · {Math.hypot(...p.moving.patient.map((n,j)=>n-p.reference.patient[j])).toFixed(2)} mm
      <button aria-label={'Remove landmark pair '+(i+1)} onClick={()=>setPairs(pairs.filter((_,j)=>j!==i))}>Remove</button></div>)}
    {(pending.reference||pending.moving)&&<p>Choose the corresponding {pending.reference?'moving':'reference'} landmark. <button onClick={()=>setPending({})}>Clear pick</button></p>}
    <label>Reviewer<input value={w.author} onChange={e=>w.setAuthor(e.target.value)}/></label>
    <button disabled={disabled||pairs.length<3||!!pending.reference||!!pending.moving} onClick={()=>accept(pairs.map(p=>p.moving.source),pairs.map(p=>p.reference.source))}>Accept landmark review</button>
    <p>Review requires at least three non-collinear pairs. Residuals measure these landmarks only.</p>
    <details><summary>Enter paired RAS coordinates</summary><textarea aria-label="Paired RAS coordinates" value={manual} onChange={e=>setManual(e.target.value)} placeholder={'{"moving":[],"reference":[]}'}/>
      <button disabled={disabled||!manual.trim()} onClick={()=>{try{const p=JSON.parse(manual);void accept(p.moving,p.reference)}catch{setError('Enter valid JSON with moving and reference coordinate lists.')}}}>Review entered coordinates</button></details>
  </div>;
}
