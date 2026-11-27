import {useEffect,useRef,useState} from 'react';
import {NativeReviewSection} from './NativeAttachedVesselReview';
import type {VesselGapEvidence,VesselGapReview} from './native-anatomy-data';
import type {NativeWorkspace} from './use-native-workspace';

const evidenceLabel:Record<VesselGapEvidence,string>={
  'uncertain-image-supported':'Possible continuation',
  'weak-source-image-support':'Weak TOF signal',
  'source-image-unassessed':'Insufficient sampled signal',
};

export function NativeVesselGapReview({workspace,layerId}:{workspace:NativeWorkspace;layerId:string}) {
  const [review,setReview]=useState<VesselGapReview|null>(null);
  const [pending,setPending]=useState(false);
  const [showOverlay,setShowOverlay]=useState(true);
  const active=useRef(false);
  useEffect(()=>{active.current=true;return()=>{active.current=false;};},[]);

  const preview=review?.preview;
  const findings=review?.findings??[];
  const selected=findings.find(item=>item.id===preview?.findingId);

  const inspect=(findingId?:string)=>workspace.run(async()=>{
    const {api,binding,series}=workspace;
    if(!api||!binding)return;
    setPending(true);
    try{
      const result=await api.anatomyVesselGapReview({...workspace.args(binding),layerId,seriesId:series,
        ...(findingId?{findingId,reviewId:review?.reviewId}:{})});
      if(active.current&&result.snapshotId===binding.document.snapshot_id&&result.sourceLayer===layerId&&result.sourceSeries===series)
        setReview(result);
    }finally{if(active.current)setPending(false);}
  });

  return <details className="unified-details" data-tof-gap-review={review?.status}>
    <summary>TOF gap review</summary>
    {!review&&<button disabled={workspace.busy} onClick={()=>void inspect()}>Find nearby branch endpoints</button>}
    {pending&&<p role="status">{review?'Loading TOF sections…':'Tracing the vessel estimate and sampling TOF…'}</p>}
    {review?.status==='unavailable'&&<p role="status">{review.reason}</p>}
    {review?.status==='partial'&&<p className="subtle">Only endpoints with traceable source geometry were assessed. Coverage is partial.</p>}
    {review&&review.status!=='unavailable'&&!findings.length&&<p role="status">No branch pairs met the review criteria.</p>}
    {!!findings.length&&<label>Finding <select aria-label="TOF gap finding" disabled={workspace.busy} value={selected?.id??''}
      onChange={event=>void inspect(event.target.value)}>
      {findings.map(item=><option key={item.id} value={item.id}>{evidenceLabel[item.evidence]} · {item.distanceMm.toFixed(2)} mm</option>)}
    </select></label>}
    {preview&&selected&&<>
      <p>{selected.evidence==='source-image-unassessed'?'Source contrast could not be measured across this gap.':
        `${selected.supportedStations} of ${selected.stations} unlabelled path samples met the TOF contrast criteria.`} The proposed connection is uncertain.</p>
      <div className="tool-row">
        <label><input type="checkbox" checked={showOverlay} onChange={event=>setShowOverlay(event.target.checked)}/> Show gap overlay</label>
        <button disabled={workspace.busy} onClick={()=>workspace.setCursor(preview.centerRasMm)}>Center linked slices</button>
      </div>
      <p><span className="attached-source-label">Cyan estimate contour</span> · <span className="attached-finding-label">amber path samples</span></p>
      <div className="native-attached-sections native-gap-sections">{preview.sections.map(plane=><NativeReviewSection key={plane.title}
        plane={plane} label={plane.title} showOverlay={showOverlay}/>)}</div>
    </>}
  </details>;
}
