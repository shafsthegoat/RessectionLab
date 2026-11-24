import {useEffect, useRef, useState} from 'react';
import type {AttachedVesselReview, AttachedVesselSection} from './native-anatomy-data';
import type {NativeWorkspace} from './use-native-workspace';
import type {Point3} from './viewer/coordinates';

export function NativeReviewSection({plane, label, showOverlay, pick, onPick}:{
  plane:Omit<AttachedVesselSection,'index'|'axis'>&{axis?:number};label:string;showOverlay:boolean;
  pick?:{point:Point3;radius:number};onPick?:(point:Point3)=>void;
}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const [height,width]=plane.shape;
  const [vertical,horizontal]=plane.spacingMm;
  const axis=plane.axis;
  const axes=axis===undefined?null:[0,1,2].filter(d=>d!==axis);
  useEffect(()=>{
    const context=canvas.current?.getContext('2d');
    if(!context)return;
    const gray=atob(plane.image),overlay=atob(plane.overlay);
    if(gray.length!==width*height||overlay.length!==gray.length)return;
    const pixels=context.createImageData(width,height);
    for(let cell=0;cell<gray.length;cell++){
      const intensity=gray.charCodeAt(cell);
      const kind=showOverlay?overlay.charCodeAt(cell):0;
      const offset=4*cell;
      pixels.data[offset]=kind===2?intensity*.45+140:kind===1?36:intensity;
      pixels.data[offset+1]=kind===2?intensity*.45+85:kind===1?212:intensity;
      pixels.data[offset+2]=kind===2?intensity*.45+15:kind===1?238:intensity;
      pixels.data[offset+3]=255;
    }
    context.putImageData(pixels,0,0);
    if(showOverlay&&pick&&axis!==undefined&&plane.startIjk){
      const [u,v]=[0,1,2].filter(d=>d!==axis);
      if(pick.point[axis]===plane.startIjk[axis]){
        const x=pick.point[u]-plane.startIjk[u]+.5,y=pick.point[v]-plane.startIjk[v]+.5;
        context.beginPath();context.ellipse(x,y,pick.radius/horizontal,pick.radius/vertical,0,0,2*Math.PI);
        context.strokeStyle='#ffffff';context.lineWidth=.65;context.stroke();
      }
    }
  },[plane,showOverlay,height,width,pick,axis,horizontal,vertical]);
  return <figure>
    <canvas ref={canvas} width={width} height={height} role="img" aria-label={label}
      style={{aspectRatio:(width*horizontal)/(height*vertical),cursor:onPick&&showOverlay?'crosshair':undefined}}
      onClick={onPick&&showOverlay&&axes&&plane.startIjk?event=>{
        const box=event.currentTarget.getBoundingClientRect();
        const x=Math.floor((event.clientX-box.left)*width/box.width),y=Math.floor((event.clientY-box.top)*height/box.height);
        if(x<0||x>=width||y<0||y>=height||atob(plane.overlay).charCodeAt(y*width+x)!==2)return;
        const point=[...plane.startIjk!] as Point3;
        point[axes[0]]+=x;point[axes[1]]+=y;onPick(point);
      }:undefined}/>
    <figcaption>{label}</figcaption>
  </figure>;
}

export function NativeAttachedVesselReview({workspace,layerId}:{workspace:NativeWorkspace;layerId:string}) {
  const [review,setReview]=useState<AttachedVesselReview|null>(null);
  const [showOverlay,setShowOverlay]=useState(true);
  const [pending,setPending]=useState(false);
  const [pick,setPick]=useState<Point3|null>(null);
  const [radius,setRadius]=useState(1);
  const active=useRef(false);
  useEffect(()=>{active.current=true;return()=>{active.current=false}},[]);
  const findings=review?.findings??[];
  const selected=findings.find(item=>item.id===review?.preview?.findingId);
  const editable=!workspace.binding?.readOnlyStudy&&(!workspace.draft||workspace.draft.mode==='erase-attached-tof');
  const validRadius=Number.isFinite(radius)&&radius>=.25&&radius<=4;

  async function inspect(findingId?:string,centerIjk?:Point3) {
    const {api,binding,series}=workspace;
    if(!api||!binding||workspace.busy)return;
    const args={...workspace.args(binding),layerId,seriesId:series};
    const matches=(result:AttachedVesselReview)=>active.current&&result.snapshotId===binding.document.snapshot_id&&
      (result.status==='unavailable'||(result.sourceLayer===layerId&&result.sourceSeries===series));
    await workspace.run(async()=>{
      setPending(true);
      try{
        const result=await api.anatomyAttachedVesselReview(
          findingId ? {...args,reviewId:review?.reviewId,findingId,...(centerIjk?{centerIjk}:{})} : args,
        );
        if(matches(result)){setReview(result);setPick(centerIjk??null);}
      }finally{if(active.current)setPending(false);}
    });
  }

  return <details className="unified-details native-attached-review">
    <summary>Attached vessel uncertainty</summary>
    <p>Look for thick regions with weak TOF signal beside brighter parts of the vessel estimate. A finding can include a real vessel with weak flow signal.</p>
    {!review&&<button disabled={workspace.busy} onClick={()=>void inspect()}>Find regions to review</button>}
    {pending&&<p role="status">{review?'Loading native sections…':'Reviewing native TOF signal…'}</p>}
    {review?.status==='unavailable'&&<p role="status">{review.reason}</p>}
    {review?.status==='partial'&&<p role="status">The review reached its sampling or workspace limits.</p>}
    {review?.status==='complete'&&!findings.length&&<p role="status">No region met the screen criteria.</p>}
    {!!findings.length&&<label>Region
      <select aria-label="Attached vessel region" disabled={workspace.busy} value={selected?.id??''}
        onChange={event=>void inspect(event.target.value)}>
        {!selected&&<option value="" disabled>Select a region</option>}
        {findings.map(finding=><option key={finding.id} value={finding.id}>
          {finding.volumeMm3.toFixed(1)} mm³ · IJK {finding.centerIjk.join(', ')}
        </option>)}
      </select>
    </label>}
    {selected&&review?.preview&&<>
      <p>{selected.voxels.toLocaleString()} native voxels · median model score {selected.uncalibratedModelScoreMedian.toFixed(2)} (uncalibrated)</p>
      <label><input type="checkbox" checked={showOverlay} onChange={event=>setShowOverlay(event.target.checked)}/> Show review overlay</label>
      {showOverlay&&<p><span className="attached-source-label">Cyan original estimate</span> · <span className="attached-finding-label">amber review region</span></p>}
      <div className="native-attached-sections">{review.preview.sections.map(plane=><NativeReviewSection key={plane.axis} plane={plane}
        label={`Native ${['I','J','K'][plane.axis]} slice ${plane.index}`} showOverlay={showOverlay}
        pick={pick&&validRadius?{point:pick,radius}:undefined} onPick={editable&&!workspace.busy?setPick:undefined}/>)}</div>
      {editable&&<>
        <p>Select an amber voxel on an acquired section to preview a local erasure.</p>
        <button disabled={workspace.busy||!showOverlay} onClick={()=>setPick(selected.centerIjk)}>Select region center</button>
        {pick&&<><p>Selected IJK {pick.join(', ')}</p>
          <button disabled={workspace.busy} onClick={()=>void inspect(selected.id,pick)}>Center sections at selection</button></>}
        <label>Erasure radius · mm <input aria-label="Native TOF erasure radius" type="number" min=".25" max="4" step=".25"
          value={radius} onChange={event=>setRadius(+event.target.value)}/></label>
        <button disabled={workspace.busy||!showOverlay||!pick||!validRadius}
          onClick={()=>pick&&void workspace.edit('erase-attached-tof',[pick],{seriesId:workspace.series,
            reviewId:review.reviewId,findingId:selected.id,radiusMm:radius},layerId)}>Preview local erasure</button>
      </>}
    </>}
  </details>;
}
