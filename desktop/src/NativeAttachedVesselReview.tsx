import {useEffect, useRef, useState} from 'react';
import type {AttachedVesselReview, AttachedVesselSection} from './native-anatomy-data';
import type {NativeWorkspace} from './use-native-workspace';

function NativeReviewSection({plane, showOverlay}:{plane:AttachedVesselSection;showOverlay:boolean}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const [height,width]=plane.shape;
  const [vertical,horizontal]=plane.spacingMm;
  const label=`Native ${['I','J','K'][plane.axis]} slice ${plane.index}`;
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
  },[plane,showOverlay,height,width]);
  return <figure>
    <canvas ref={canvas} width={width} height={height} role="img" aria-label={label}
      style={{aspectRatio:(width*horizontal)/(height*vertical)}}/>
    <figcaption>{label}</figcaption>
  </figure>;
}

export function NativeAttachedVesselReview({workspace,layerId}:{workspace:NativeWorkspace;layerId:string}) {
  const [review,setReview]=useState<AttachedVesselReview|null>(null);
  const [showOverlay,setShowOverlay]=useState(true);
  const [pending,setPending]=useState(false);
  const active=useRef(false);
  useEffect(()=>{active.current=true;return()=>{active.current=false}},[]);
  const findings=review?.findings??[];
  const selected=findings.find(item=>item.id===review?.preview?.findingId);

  async function inspect(findingId?:string) {
    const {api,binding,series}=workspace;
    if(!api||!binding||workspace.busy)return;
    const args={...workspace.args(binding),layerId,seriesId:series};
    const matches=(result:AttachedVesselReview)=>active.current&&result.snapshotId===binding.document.snapshot_id&&
      (result.status==='unavailable'||(result.sourceLayer===layerId&&result.sourceSeries===series));
    await workspace.run(async()=>{
      setPending(true);
      try{
        const result=await api.anatomyAttachedVesselReview(
          findingId ? {...args,reviewId:review?.reviewId,findingId} : args,
        );
        if(matches(result))setReview(result);
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
      {showOverlay&&<p><span className="attached-source-label">Cyan estimate contour</span> · <span className="attached-finding-label">amber review region</span></p>}
      <div className="native-attached-sections">{review.preview.sections.map(plane=><NativeReviewSection key={plane.axis} plane={plane} showOverlay={showOverlay}/>)}</div>
    </>}
  </details>;
}
