import {NativeCranialView} from './NativeCranialView';
import {NativeStructureControls} from './NativeStructureControls';
import {layerAppearance} from './viewer/workspaceSettings';
import {useEffect,useRef,useState} from 'react';
import {Check,ChevronRight,ScanLine} from 'lucide-react';
import {inverseAffine,rasAffine,sampleTrilinear,transformPoint,volumeBounds} from './viewer/coordinates';
import type {ViewerVolume} from './viewer/contracts';
import {scanLabel,studyTimepoint,studyLabel,sourceEvidenceLabel,defaultAnatomyPresentation,studyLayerDisplayKind} from './native-study-state';
import type {NativeVesselGraph,VesselScope} from './native-anatomy-data';
import type {NativeWorkspace} from './use-native-workspace';
import {AnatomyInspection} from './AnatomyInspection';
import {NativeImportPanel} from './NativeImportPanel';
import {NativePreparePanel} from './NativePreparePanel';
import {NativeRegistrationReview} from './NativeRegistrationReview';
import {AcquisitionContributions,NativeLayerEvidence,rehearsalAvailability} from './NativeStudyEvidence';
import './unified-anatomy.css';

function NativeVesselBranchPanel({workspace:w,layerId}:{workspace:NativeWorkspace;layerId:string}){
  const [result,setResult]=useState<NativeVesselGraph|null>(null);
  const [branchId,setBranchId]=useState<string|null>(null);
  const branches=result?.branches??[],branch=branches.find(item=>item.branch_id===branchId)??branches[0];
  const inspect=()=>w.run(async()=>{
    const graph=await w.api!.anatomyGraph({...w.args(),layerId,displaySeriesId:w.series,point:w.state.cursor});
    setResult(graph);setBranchId(null);w.setVesselTrace(null);
  });
  return <details className="unified-details" data-vessel-branches={branches.length}>
    <summary>Vessel branch inspection</summary>
    <button disabled={w.busy||!w.state.cursor} onClick={()=>void inspect()}>Inspect branches at cursor</button>
    {result&&!result.nearest&&<p role="status">No source centerline is available for this layer.</p>}
    {result?.nearest&&<p className="subtle">Nearest source cell {result.nearest.source_cell_id} · {result.nearest.distance_mm.toFixed(2)} mm from the query.{result.nearest.radius_mm!=null&&<> Distance to background cell centers {result.nearest.radius_mm.toFixed(2)} mm.</>}{result.nearest.radius_limited_by_image_extent&&<> Distance estimate limited by scan extent.</>}{result.nearest.radius_limited_by_workspace&&<> Distance estimate limited by sampling bounds.</>}</p>}
    {branch&&<>
      {branches.length>1&&<label>Branch<select aria-label="Vessel branch" value={branch.branch_id} onChange={event=>{setBranchId(event.target.value);w.setVesselTrace(null);}}>{branches.map(item=><option key={item.branch_id} value={item.branch_id}>{item.branch_id.slice(0,10)} · {item.length_mm.toFixed(1)} mm</option>)}</select></label>}
      <p className="subtle">Branch <code title={branch.branch_id}>{branch.branch_id.slice(0,12)}</code> · {branch.length_mm.toFixed(1)} mm · {branch.source_cells.length.toLocaleString()} source skeleton cells{branch.closed?' · closed path':''}.</p>
      {branch.connectivity==='source-h0-h1-matched-cavities-unrepresented'&&<p className="subtle">Source cavities are absent from the path graph.</p>}
      {!branch.connectivity.startsWith('source-h0-h1-matched')&&<p className="subtle">Source topology for this path is unverified.</p>}
      {branch.unsupported_diagonal_edges>0&&<p className="subtle">{branch.unsupported_diagonal_edges} diagonal steps lack support by adjoining source faces.</p>}
      <div className="tool-row"><button onClick={()=>w.focusStructure(layerId,branch.midpoint_ras_mm)}>Frame branch midpoint</button><button onClick={()=>w.setCursor(branch.start_ras_mm)}>Start slice</button><button onClick={()=>w.setCursor(branch.end_ras_mm)}>End slice</button></div>
      {branch.preview_ras_mm.length>1&&<button aria-pressed={!!w.vesselTrace} onClick={()=>w.setVesselTrace(w.vesselTrace?null:{layerId,points:branch.preview_ras_mm})}>{w.vesselTrace?'Hide branch trace':'Show branch trace'}</button>}
    </>}
  </details>;
}

/** A display-only, physical sagittal section sampled from the current source.
 * No source fetch, segmentation, or image substitution is involved. */
function NativeScanPreview({volume,label,modality}:{volume:ViewerVolume|null;label:string;modality:string}){
  const canvas=useRef<HTMLCanvasElement>(null);
  const sourceSignal=defaultAnatomyPresentation(modality)==='scan';
  useEffect(()=>{
    const context=canvas.current?.getContext('2d');if(!context||!volume||!sourceSignal)return;
    const width=240,height=148,affine=rasAffine(volume.affine,volume.frame),inverse=inverseAffine(affine),bounds=volumeBounds(volume.shape,affine);
    const sampled:number[]=[],stride=Math.max(1,Math.floor(volume.mri.length/16000));
    for(let i=0;i<volume.mri.length;i+=stride)if(Number.isFinite(volume.mri[i])&&volume.mri[i]!==0)sampled.push(volume.mri[i]);
    sampled.sort((a,b)=>a-b);
    const low=sampled[Math.floor(sampled.length*.005)]??0,high=Math.max(low+1e-6,sampled[Math.min(sampled.length-1,Math.floor(sampled.length*.995))]??1);
    const scale=Math.min((width-12)/Math.max(1,bounds[1][1]-bounds[0][1]),(height-12)/Math.max(1,bounds[1][2]-bounds[0][2]));
    const imageWidth=(bounds[1][1]-bounds[0][1])*scale,imageHeight=(bounds[1][2]-bounds[0][2])*scale,left=(width-imageWidth)/2,top=(height-imageHeight)/2;
    const pixels=context.createImageData(width,height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const index=(y*width+x)*4;pixels.data[index+3]=255;
      if(x<left||x>left+imageWidth||y<top||y>top+imageHeight)continue;
      const point=transformPoint(inverse,[(bounds[0][0]+bounds[1][0])/2,bounds[0][1]+(x-left)/scale,bounds[1][2]-(y-top)/scale]);
      if(point.some((value,axis)=>value<0||value>volume.shape[axis]-1))continue;
      const gray=Math.round(255*Math.min(1,Math.max(0,(sampleTrilinear(volume.mri,volume.shape,point)-low)/(high-low))));
      pixels.data[index]=gray;pixels.data[index+1]=gray;pixels.data[index+2]=gray;
    }
    context.putImageData(pixels,0,0);
  },[volume?.mri,volume?.affine,volume?.shape,sourceSignal]);
  return <span className="study-scan-preview">{volume&&sourceSignal?<><canvas ref={canvas} width="240" height="148" role="img" aria-label={`${label} · central sagittal section of the source scan`}/><span>Sagittal · source scan</span></>:<span className="study-scan-preview-empty">{sourceSignal?'Loading source preview…':'Source preview unavailable for this layer'}</span>}</span>;
}

export function NativeSidebar({workspace:w,tab,onOpenAnatomy,onOpenPlan,onOpenScans}:{
  workspace:NativeWorkspace;tab:'data'|'anatomy';onOpenAnatomy:()=>void;onOpenPlan:()=>void;onOpenScans:()=>void;
}){
  const [selected,setSelected]=useState<string|null>(w.draft?w.editLayer:w.selectedStructure),[editing,setEditing]=useState(!!w.draft),[size,setSize]=useState(0);
  const [componentPage,setComponentPage]=useState(0),[componentOrder,setComponentOrder]=useState('smallest');
  const [marginScore,setMarginScore]=useState(.35),[marginRadius,setMarginRadius]=useState(.8);
  useEffect(()=>setComponentPage(0),[w.review?.reviewId,componentOrder]);
  useEffect(()=>{setSelected(w.draft?w.editLayer:w.selectedStructure);if(!w.draft)setEditing(false)},[w.selectedStructure,w.draft?.layerId,w.binding?.collectionId]);
  const orderedComponents=componentOrder==='largest'?[...(w.review?.components??[])].reverse():(w.review?.components??[]);
  const b=w.binding;if(!b||!w.api)return null;const api=w.api;
  const timepoint=studyTimepoint(b);
  const source=b.document.series.find(s=>s.id===w.series),layer=b.document.layers.find(l=>l.id===selected);
  const model=w.volume?.patientContext,independent=w.registration==='independent source frame';
  const excludedSources=new Set(model?.sources.filter(s=>s.alignment==='unavailable').map(s=>s.seriesId));
  const unavailable=[...new Set([...w.unavailable,...b.document.layers.filter(l=>excludedSources.has(l.source_series)).map(l=>l.id)])];
  const selectedDisplay=w.volume?.compartments.find(l=>l.name===selected);
  const blocked=w.busy||!!w.draft||w.planning.pending;
  const choose=(id:string|null)=>{if(w.draft&&id!==w.editLayer)return;w.setSelectedStructure(id);setSelected(id);setEditing(false);w.setTool('navigate');w.setEditLayer(id)};
  const planning=w.planning;
  const usableTargets=planning.targetOptions.filter(option=>option.target);
  const activeTarget=usableTargets.some(option=>option.layerId===planning.activeTargetLayerId)&&!planning.targetSelectionIssue;
  const layerInspector=layer&&<section className="layer-inspector"><div className="inspector-heading"><strong>{layer.label}</strong><button aria-label="Close layer details" disabled={!!w.draft} onClick={()=>choose(null)}>×</button></div>
        <div className="tool-row"><input type="color" aria-label={`${layer.label} color`} value={w.state.settings.layerColor?.[layer.id]??layer.color} onChange={e=>w.setSettings({...w.state.settings,layerColor:{...w.state.settings.layerColor,[layer.id]:e.target.value}})}/><label>Opacity<input aria-label={`${layer.label} opacity`} type="range" min="0" max="1" step=".01" value={layerAppearance(layer.id,layer.color,w.state.settings,studyLayerDisplayKind(layer)).opacity} onChange={e=>w.setSettings({...w.state.settings,layerOpacity:{...w.state.settings.layerOpacity,[layer.id]:+e.target.value}})}/></label><button disabled={w.busy} onClick={()=>w.setVisible(Object.fromEntries(b.document.layers.map(l=>[l.id,l.id===layer.id])))}>Isolate</button></div>
        <p className="subtle layer-review-status">{layer.review_status.replaceAll('_',' ')}{w.volume?.compartments.find(l=>l.name===layer.id)?.volumeMm3!=null&&<> · source {(w.volume.compartments.find(l=>l.name===layer.id)!.volumeMm3!/1000).toFixed(2)} mL</>}</p>
        {w.volume?.compartments.find(l=>l.name===layer.id)?.qa&&(()=>{const q=w.volume!.compartments.find(l=>l.name===layer.id)!.qa!;return <details className="unified-details" open={q.voxelCount===0}><summary>Anatomy quality · {q.voxelCount===0?'empty segmentation':q.issues.length?'review findings':'source checks complete'}</summary><p>{q.componentCount.toLocaleString()} source components · anatomical accuracy unmeasured</p>{q.issues.map(issue=><p key={issue}>{issue}</p>)}<p>Findings do not change the source. Inspect linked slices before accepting corrections.</p></details>})()}
        {!w.draft&&w.vesselScope==='all'&&layer.source_series===w.series&&['vessel','vessels','artery','vein','sinus'].includes(layer.kind)&&
          (layer.provenance as {model?:string}|null)?.model!=='siam-v2'&&<NativeVesselBranchPanel key={`${b.document.snapshot_id}/${layer.id}`} workspace={w} layerId={layer.id}/>}
        {!b.readOnlyStudy&&<button className="edit-segmentation" disabled={w.busy||!w.canEditLayer(layer.id)} onClick={()=>{setEditing(!editing);w.setTool('navigate');w.setEditLayer(layer.id)}}>{editing?'Close tools':'Edit segmentation'}</button>}
        {editing&&['vessels','artery','vein','sinus'].includes(layer.kind)&&<div className="vessel-review-note"><p>Disconnected pieces are model estimates. Inspect them on the source slices before keeping or removing them.</p><button disabled={w.busy||!!w.draft||!w.canEditLayer(layer.id)} onClick={()=>{setEditing(true);w.setEditLayer(layer.id);void w.reviewFragments(layer.id)}}>Inspect vessel fragments</button></div>}
        {editing&&<div className="correction-tools"><label>Tool<select aria-label="Correction tool" value={w.tool} onChange={e=>w.setTool(e.target.value)}><option value="navigate">Navigate</option><option value="paint">Paint</option><option value="erase">Erase</option></select></label><label>Brush radius · mm<input type="number" min=".25" max="50" step=".25" value={w.radius} onChange={e=>w.setRadius(+e.target.value)}/></label>
          {layer.probability!=null&&<details><summary>Probability threshold</summary><input aria-label="Probability threshold" type="number" min=".01" max=".99" step=".01" value={w.threshold} onChange={e=>w.setThreshold(+e.target.value)}/><div className="tool-row"><button disabled={w.busy} onClick={()=>w.edit('threshold')}>Preview threshold</button><button disabled={w.busy||!w.state.cursor} onClick={()=>w.edit('seed',[w.state.cursor!])}>Grow from cursor</button></div></details>}
          {layer.probability!=null&&['vessels','artery','vein','sinus'].includes(layer.kind)&&<details><summary>Recover supported vessel margins</summary><p className="subtle">Preview additional cells supported by this scan’s saved model scores. Existing vessels stay; disconnected pieces cannot join. Scores are uncalibrated and anatomy needs review.</p><label>Minimum model score<input aria-label="Vessel margin score" type="number" min=".01" max=".49" step=".01" value={marginScore} onChange={e=>setMarginScore(+e.target.value)}/></label><label>Maximum margin · mm<input aria-label="Vessel margin radius" type="number" min=".1" max="2" step=".1" value={marginRadius} onChange={e=>setMarginRadius(+e.target.value)}/></label><button disabled={w.busy||!!w.draft||!Number.isFinite(marginScore)||marginScore<=0||marginScore>=.5||!Number.isFinite(marginRadius)||marginRadius<=0||marginRadius>2} onClick={()=>w.edit('refine-vessels',[],{threshold:marginScore,radiusMm:marginRadius})}>Preview supported margins</button><p className="subtle">Accepting creates a new anatomy revision used by planning and rehearsal. Smooth surfaces changes display only.</p></details>}
          <button disabled={w.busy||!!w.draft} onClick={()=>w.reviewFragments()}>Review fragments</button><details><summary>Seeded component selection</summary><button disabled={w.busy||!w.state.cursor} onClick={()=>w.edit('component',[w.state.cursor!])}>Preview keeping cursor component</button></details>
          {w.review&&w.review.layerId===layer.id&&<div className="fragment-review"><p>{w.review.components.length} disconnected components</p><p className="subtle">Select a piece to highlight it. Click its number to locate it on the slices. Removal stays a preview until accepted.</p><label>Select below · mm³<input aria-label="Fragment size threshold" type="number" min="0" value={size} onChange={e=>setSize(+e.target.value)}/></label><button disabled={w.busy||!!w.draft} onClick={()=>w.highlight(w.review!.components.filter(c=>c.volumeMm3<size).map(c=>c.id))}>Preview selection</button><label>Sort fragments<select aria-label="Fragment order" value={componentOrder} onChange={e=>setComponentOrder(e.target.value)}><option value="smallest">Smallest first</option><option value="largest">Largest first</option></select></label><div className="component-list">{orderedComponents.slice(componentPage*100,(componentPage+1)*100).map(c=><label key={c.id}><input type="checkbox" aria-label={`Select fragment ${c.id}`} checked={w.components.includes(c.id)} disabled={w.busy||!!w.draft} onChange={e=>w.highlight(e.target.checked?[...w.components,c.id]:w.components.filter(id=>id!==c.id))}/><button onClick={()=>w.setCursor(c.center)}>#{c.id}</button><span>{c.volumeMm3.toFixed(2)} mm³</span></label>)}</div><div className="tool-row"><button disabled={!componentPage} onClick={()=>setComponentPage(p=>p-1)}>Previous</button><span>{componentPage+1} / {Math.max(1,Math.ceil(orderedComponents.length/100))}</span><button disabled={(componentPage+1)*100>=orderedComponents.length} onClick={()=>setComponentPage(p=>p+1)}>Next</button></div><p className="subtle">Size selection includes every page.</p><button disabled={w.busy||!w.components.length||!!w.draft} onClick={()=>w.edit('remove-components',[],{componentIds:w.components,reviewId:w.review?.reviewId})}>Preview removal ({w.components.length})</button></div>}
          <label>Author / reviewer<input aria-label="Correction author" value={w.author} onChange={e=>w.setAuthor(e.target.value)}/></label>
        </div>}
        {selectedDisplay?.alignment&&<p className="subtle">Model placement · {selectedDisplay.alignment}</p>}
        {layer.source_series!==w.series&&<button disabled={blocked} onClick={()=>w.selectScan(layer.source_series)}>Review {scanLabel(b.document.series.find(s=>s.id===layer.source_series)!)} slices</button>}
        <NativeLayerEvidence layer={layer} document={b.document} workspace={w}/>
      </section>;
  return <div className={`unified-sidebar study-inspector study-inspector-${tab}`}>
    <div className="study-badge">{studyLabel(timepoint)} study <span>{b.document.series.length} {b.document.series.length===1?'scan':'scans'}</span></div>
    <h1 title={b.document.patient_id}>{b.document.patient_id.replace(/^ZENODO-17486320-/,'')}</h1>
    {b.readOnlyStudy&&<p className="subtle" role="status">Read-only vascular review</p>}
    {tab==='anatomy'&&!!b.document.series.length&&<label className="unified-source">{independent?'Independent scan':'Linked source slices'}<select aria-label="Source scan" value={w.series} disabled={blocked} onChange={e=>w.selectScan(e.target.value)}>{b.document.series.map(s=><option key={s.id} value={s.id}>{scanLabel(s)}</option>)}</select></label>}
    {tab==='data'?<>
      <div className="study-section-heading"><strong>Study scans</strong><span>{b.document.series.length}</span></div>
      <div className="study-scan-list" aria-label="Study scans">{b.document.series.map(s=><button key={s.id} data-scan-id={s.id} className={`study-scan-row${w.series===s.id?' is-selected':''}`} aria-pressed={w.series===s.id} disabled={blocked} onClick={()=>w.selectScan(s.id)}>
        <span className="study-scan-row-heading"><span className="study-modality">{s.modality}</span><strong>{scanLabel(s)}</strong>{w.series===s.id?<Check size={14}/>:<ChevronRight size={14}/>}</span>
        <span className="study-scan-row-meta">{s.grid_spacing_mm.join(' × ')} mm grid{s.id===b.document.reference_series?' · Reference':''}</span>
        {w.series===s.id&&<NativeScanPreview volume={w.volume} label={scanLabel(s)} modality={s.modality}/>}
      </button>)}</div>
      {source&&<div className="study-scan-status"><span className={w.registration==='alignment preview'?'needs-review':''}>{w.registration==='patient frame'?'Source / reviewed frame':w.registration==='alignment preview'?'Alignment preview · review required':'Alignment unavailable'}</span><span>{b.document.layers.filter(l=>l.source_series===source.id).length} source labels</span></div>}
      {source&&<details className="unified-details study-source-details"><summary>Source & acquisition details</summary><dl>
        <div><dt>Recorded modality</dt><dd>{source.modality}</dd></div>
        <div><dt>Source grid spacing</dt><dd>{source.grid_spacing_mm.join(' × ')} mm</dd></div>
        <div><dt>Acquisition resolution</dt><dd>{Array.isArray(source.acquisition_resolution)?source.acquisition_resolution.join(' × ')+' mm':source.acquisition_resolution??'Unreported'}</dd></div>
        {w.volume&&<div><dt>{w.volume.displayPreview?'Preview grid':'Source grid'}</dt><dd>{w.volume.shape.join(' × ')} voxels{w.volume.displayPreview?` · source sampled every ${w.volume.displayPreview.factor} voxels`:''}</dd></div>}
        <div><dt>Alignment</dt><dd>{b.document.transforms[source.id]?.status.replaceAll('_',' ')??'Unavailable'}</dd></div>
        <div><dt>Registration uncertainty</dt><dd>{typeof b.document.transforms[source.id]?.uncertainty_mm==='number'?b.document.transforms[source.id].uncertainty_mm+' mm reported':source.id===b.document.reference_series?'Reference frame':'Unmeasured'}</dd></div>
        {source.coverage?.status&&<div><dt>Recorded coverage</dt><dd>{source.coverage.status.replaceAll('_',' ')}</dd></div>}
        {!!b.document.source_assets?.length&&<div><dt>Archived source files</dt><dd>{b.document.source_assets.length}</dd></div>}
      </dl>{source.coverage?.limitations?.map((item,index)=><p key={index}>{item.kind.replaceAll('_',' ')}{item.basis?' · '+item.basis:''}</p>)}</details>}
      <AcquisitionContributions workspace={w} onInspect={layer=>{
        choose(layer.id);
        if(layer.source_series!==w.series)w.selectScan(layer.source_series,undefined,layer.id);
        else if(!w.state.visible[layer.id])w.setVisible({...w.state.visible,[layer.id]:true});
        onOpenAnatomy();
      }}/>
      {!b.readOnlyStudy&&<>
        <NativeImportPanel workspace={w} timepoint={timepoint}/>
        <NativePreparePanel workspace={w} onReviewAnatomy={onOpenAnatomy}/>
        <button className="wide-button" disabled={blocked} onClick={w.savePackage}>Save patient study file…</button>
      </>}
      {!!source&&<details className="unified-details"><summary>Alignment & registration</summary><p>{w.registration==='patient frame'?'Source or reviewed alignment.':w.registration==='alignment preview'?'An estimated transform is being used for display. Review landmarks before quantitative planning.':'Align this scan to show shared structures.'}</p>{Object.entries(w.volume?.alignmentQuality??{}).map(([sid,q])=><div key={sid} className="alignment-notice"><strong>{sid} · {q.status}</strong>{[...q.warnings,...(q.notes??[])].map((warning,i)=><span key={i}>{warning}</span>)}<details><summary>Measurements</summary><pre>{JSON.stringify(q.metrics,null,2)}</pre></details></div>)}<button disabled={blocked||!!b.readOnlyStudy||w.series===b.document.reference_series} onClick={()=>w.change(()=>api.anatomyRegister({...w.args(),seriesId:w.series,action:'estimate'}))}>Recalculate alignment</button><NativeRegistrationReview key={JSON.stringify([b.collectionId,b.document.snapshot_id,w.series])} workspace={w} blocked={blocked||!!b.readOnlyStudy}/></details>}
    </>:<>
      <div className="anatomy-presets" role="group" aria-label="Anatomy presentation"><button aria-pressed={w.state.settings.presentation==='scan'} disabled={blocked||!source||defaultAnatomyPresentation(source.modality)!=='scan'} onClick={()=>w.present('scan')}><ScanLine size={13}/>Scan</button><button aria-pressed={w.state.settings.presentation==='segmentation'} disabled={blocked} onClick={()=>w.present('segmentation')}>Structures</button><button aria-pressed={w.state.settings.presentation==='vascular'} disabled={blocked} onClick={()=>w.present('vascular')}>Vessels</button></div>
    <NativeCranialView workspace={w}/>
      {model&&<details className="unified-details patient-model-summary" data-frame="patient">
        <summary>Patient reconstruction · {model.sources.filter(s=>s.displayWindow).length} contributing scans</summary>
        <p>{w.state.settings.presentation==='scan'?'3D signal and linked slices show the selected scan.':'3D context shows the brightest windowed signal across aligned scans. Linked slices retain the selected scan’s intensity.'} Surfaces retain their source extent.</p>
        {model.field&&<p>3D context spacing · {Number(model.field.affine[0][0].toFixed(2))} mm</p>}
        {model.sources.map(s=><p key={s.seriesId}><strong>{scanLabel(b.document.series.find(scan=>scan.id===s.seriesId)!)}</strong> · {s.modality} · {s.alignment}{s.reason?' · '+s.reason:''}{s.displayWindow?' · display window '+s.displayWindow.map(n=>Number(n.toFixed(2))).join(' to ')+' '+(s.windowUnits??''):''}</p>)}
      </details>}
      {independent&&<div className="patient-model-summary" data-frame="independent"><p>Independent source coordinates. {w.volume?.sourceFrameIssue??'Study alignment is unavailable for this scan.'}</p>{w.series!==b.document.reference_series&&<button disabled={blocked} onClick={()=>w.selectScan(b.document.reference_series,'segmentation')}>Return to patient reconstruction</button>}</div>}
      {planning.selected&&!w.patientFrameReady&&!w.busy&&source&&source.id!==b.document.reference_series&&
        <div className="alignment-notice" aria-label="Plan on reference scan">
          <strong>Approach retained on {scanLabel(b.document.series.find(s=>s.id===b.document.reference_series)!)}</strong>
          <p>{scanLabel(source)} needs reviewed alignment before it can show the planned route.</p>
          <button disabled={blocked} onClick={()=>w.selectScan(b.document.reference_series)}>View plan on reference scan</button>
        </div>}
      <p className="presentation-source">{model&&w.state.settings.presentation!=='scan'?'Linked slices · ':''}{sourceEvidenceLabel(source,b.document.series)}</p>
      <NativeStructureControls volume={w.volume} layers={b.document.layers} tracts={b.document.tractography} visibleLayers={w.state.visible} onVisibilityChange={w.setVisible} settings={w.state.settings} onSettingsChange={w.setSettings} selectedStructure={w.selectedStructure} onSelectStructure={choose} onFocusStructure={w.focusStructure} unavailable={unavailable} pendingLayers={w.pendingLayers} layerErrors={w.layerErrors} onRetryLayer={w.retryLayer} selectionInspector={layerInspector} disabled={w.busy||!!w.draft} showSelectionControls={!!w.selectedStructure&&!layer}/>
      <details className="anatomy-display-tools"><summary>Display & fragments</summary>
        {b.document.layers.some(l=>(l.provenance as {model?:string}|null)?.model==='siam-v2')&&<>
          <label><input type="checkbox" aria-label="Unfiltered SIAM estimates (QC)" checked={w.original} disabled={blocked} onChange={w.compare}/>Unfiltered SIAM estimates (QC)</label>
          <span>{w.original?'Showing raw SIAM predictions, including estimates in background air.':'SIAM display uses MRI head support where available. Raw QC restores MRI bone alternatives in CT overlap.'}</span>
          {w.volume?.compartments.filter(l=>l.sourceSupport&&l.name===w.selectedStructure).map(l=><span key={l.name}>{l.sourceSupport?.status==='unavailable'?`${l.sourceSupport.reason}. Showing the unfiltered estimate.`:`${l.sourceSupport?.retainedVoxels?.toLocaleString()} of ${l.sourceSupport?.sourceVoxels?.toLocaleString()} source voxels in this SIAM view`}</span>)}
        </>}
        {!b.document.layers.some(l=>(l.provenance as {model?:string}|null)?.model==='siam-v2')&&
          b.document.layers.some(l=>(l.provenance as {model?:string}|null)?.model==='supersynth')&&
          b.document.series.some(s=>['CT','CTA'].includes(s.modality))&&
          <label><input type="checkbox" checked={w.original} disabled={blocked} onChange={w.compare}/>Raw source layers (QC)</label>}
        <label>Skull view<select aria-label="Skull view" disabled={blocked} value={w.state.settings.skullMode??'exterior'} onChange={e=>w.changeSkullMode(e.target.value as 'exterior'|'transparent'|'cutaway')}><option value="exterior">Opaque exterior</option><option value="transparent">Transparent</option><option value="cutaway">Cutaway</option></select></label><label>Vessel display<select aria-label="Vessel display" disabled={blocked} value={w.vesselScope} onChange={e=>w.changeVesselScope(e.target.value as VesselScope)}><option value="intracranial">Skull-supported only</option><option value="excluded">Excluded fragments (highlighted)</option><option value="brain">Within brain only</option><option value="all">Full source estimate</option></select></label><span>{w.vesselScope==='all'?'Vessel estimates are unreviewed. Completeness is unmeasured.':w.volume?.compartments.find(l=>l.displayFilter?.reason)?.displayFilter?.reason??'Preparing filtered vessel display…'}</span>{w.vesselScope!=='all'&&<><span>Filters hide source estimates and can introduce gaps. Source anatomy is preserved.</span><button disabled={blocked} onClick={()=>w.changeVesselScope('all')}>Show full source</button></>}</details>
      {b.document.series.length>0&&<details className="unified-details native-study-readiness"><summary>Planning & rehearsal availability</summary>
        {planning.loading?<p>Checking planning and rehearsal evidence…</p>:<>
          <p>{activeTarget?'The selected target supports geometric comparison.':usableTargets.length?
            'Select a supported lesion candidate for geometric comparison.':
            'Geometric comparison requires a nonempty lesion candidate with aligned source evidence.'}</p>
          <p>{rehearsalAvailability(planning.rehearsalOptions)}</p>
        </>}
        {!planning.loading&&planning.targetOptions.length>0&&<button className="wide-button" disabled={blocked} onClick={onOpenPlan}>{
          activeTarget?'Open Plan & compare':usableTargets.length?'Select target & compare':'Review target candidates'}</button>}
        {b.document.layers.length===0&&<button className="wide-button" disabled={blocked} onClick={onOpenScans}>Open Scans to prepare anatomy</button>}
      </details>}
      {!!unavailable.length&&<p className="subtle">{unavailable.length} structures need alignment. Open Scans → Alignment & registration.</p>}
      <details className="unified-details"><summary>Coverage & unavailable anatomy</summary>{b.document.missing.map(m=><p key={m}>{m}</p>)}</details>
    </>}
  </div>;
}
export function NativeCorrectionBar({workspace:w}:{workspace:NativeWorkspace}){
  if(!w.draft)return null;return <div className="unified-correction" role="toolbar" aria-label="Segmentation correction"><span>Preview · {w.draft.changedVoxels.toLocaleString()} changed voxels</span><button disabled={w.busy||!w.draft.undo} onClick={()=>w.edit('undo')}>Undo</button><button disabled={w.busy||!w.draft.redo} onClick={()=>w.edit('redo')}>Redo</button><button disabled={w.busy} onClick={w.compare}>{w.original?'Show preview':'Show original'}</button><div className="spacer"/><button disabled={w.busy} onClick={()=>w.edit('reject')}>Reject</button><button disabled={w.busy||!w.author.trim()} onClick={()=>w.edit('commit')}>Accept correction</button></div>;
}
export function NativeResearch({workspace:w}:{workspace:NativeWorkspace}){
  const [sample,setSample]=useState('');if(!w.binding||!w.api)return null;
  return <div className="unified-research"><h2>Inspect anatomy</h2><button disabled={w.busy||!w.state.cursor||w.registration!=='patient frame'} onClick={()=>w.run(async()=>setSample(JSON.stringify(await w.api!.anatomySample({...w.args(),point:w.state.cursor}),null,2)))}>Inspect at cursor</button>{sample&&<pre>{sample}</pre>}
  <AnatomyInspection original={w.original} api={w.api} binding={{collectionId:w.binding.collectionId,snapshotId:w.binding.document.snapshot_id}} patientFrame={w.registration==='patient frame'} series={w.series} layers={Object.keys(w.state.visible).filter(id=>w.state.visible[id])} cursor={w.state.cursor} settings={w.state.settings} onSettings={w.setSettings} onRefresh={w.refresh} busy={w.busy||!!w.draft} run={w.run}/>
  <button disabled={w.busy||!!w.draft} onClick={()=>w.run(async()=>{const result=await w.api!.anatomyExport(w.args());if(result)w.setMessage('Snapshot derivative exported')})}>Export snapshot</button></div>;
}
