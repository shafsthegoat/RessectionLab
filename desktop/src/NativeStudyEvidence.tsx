import {useEffect,useState} from 'react';
import type {NativeWorkspace} from './use-native-workspace';
import type {AcquisitionContribution,Document,EnvironmentOptions,Layer,PreparationStage} from './native-anatomy-data';
import {scanLabel} from './native-study-state';

export function rehearsalAvailability(options?:EnvironmentOptions){
  if(!options)return 'Rehearsal preparation eligibility is unavailable.';
  if(options.referenceIssue)return 'Limited rehearsal preparation unavailable. '+options.referenceIssue+'.';
  const ready=options.layers.some(layer=>layer.status==='ready');
  const estimated=options.layers.some(layer=>layer.status==='estimated_alignment');
  if(ready)return 'Source evidence supports preparation of a limited research rehearsal. Procedure admission is checked separately.';
  if(estimated)return 'Limited research rehearsal preparation requires an explicit choice to use estimated alignment.';
  return 'Limited rehearsal preparation needs an eligible label with source registration.';
}

const object=(value:unknown):Record<string,unknown>=>value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};
const readable=(value:string)=>value.replaceAll('_',' ').replaceAll('-',' ');
const text=(value:unknown)=>typeof value==='string'&&value.trim()?value:undefined;
const numericText=(value:number)=>value.toLocaleString('en-US',{maximumFractionDigits:3});
const spacing=(value:unknown)=>Array.isArray(value)&&value.length===3&&value.every(n=>typeof n==='number'&&Number.isFinite(n)&&n>0)?
  value.map(numericText).join(' × ')+' mm':undefined;

type CursorReading={sourceVoxel?:number[];sourceGridInspection?:boolean;known:boolean;occupied?:boolean|null;reason?:string;detail?:string;score?:number};

function SourceLocation({workspace:w,layer}:{workspace:NativeWorkspace;layer:Layer}){
  const {api,binding,state,registration,series,patientFrameReady}=w;
  const point=state.cursor;
  const independent=registration==='independent source frame';
  const frame=independent?'Source':'Patient';
  const canRead=!!binding&&!!api&&!!point&&(independent||patientFrameReady);
  const identity=JSON.stringify([binding?.collectionId,binding?.document.snapshot_id,series,layer.id,point,frame]);
  const [result,setResult]=useState<{identity:string;reading:CursorReading}|null>(null);
  useEffect(()=>{
    if(!canRead)return;
    let active=true;
    const timer=setTimeout(()=>{
      api!.anatomySample({collectionId:binding!.collectionId,snapshotId:binding!.document.snapshot_id,
        layerId:layer.id,point,...(independent?{independentSeries:series}:{})}).then(raw=>{
          const reading=(raw as {layers:Record<string,CursorReading>}).layers[layer.id];
          if(active)setResult({identity,reading});
        }).catch(error=>{if(active)setResult({identity,reading:{known:false,detail:String(error)}})});
    },180);
    return ()=>{active=false;clearTimeout(timer)};
  },[api,canRead,identity]);
  const reading=result?.identity===identity?result.reading:null;
  const status=reading?.detail??(reading?.reason?readable(reading.reason):reading?.occupied?'Positive source label':
    reading?.occupied===false?'Unlabelled at this voxel. Anatomical absence is unknown.':null);
  return <div className="alignment-notice" aria-label="Linked source location">
    <strong>Linked source location</strong>
    {canRead&&point&&<span>{frame} RAS · {point.map(numericText).join(', ')} mm</span>}
    {reading?.sourceVoxel&&<span>Native scan voxel (i, j, k) · {reading.sourceVoxel.join(', ')}</span>}
    <span>{!canRead?'A reviewed patient frame or the independent source scan is required.':status??'Sampling source label…'}</span>
    {reading?.sourceGridInspection&&<span>Indexed on its matching source scan. Annotation units were undeclared.</span>}
    {typeof reading?.score==='number'&&<span>Source score · {numericText(reading.score)}</span>}
  </div>;
}

export function NativeLayerEvidence({layer,document,workspace}:{layer:Layer;document:Document;workspace?:NativeWorkspace}){
  const [showProvenance,setShowProvenance]=useState(false);
  const source=document.series.find(row=>row.id===layer.source_series);
  const provenance=object(layer.provenance);
  const model=provenance.model,identity=object(model);
  const inference=object(provenance.inference),inferenceModel=object(inference.model);
  const modelName=text(model)??text(identity.name)??text(inferenceModel.name)??text(inference.model);
  const version=text(provenance.model_version)??text(provenance.version)??text(identity.version)??text(inferenceModel.version);
  const method=text(provenance.method)??text(provenance.kind);
  const acquisition=layer.acquisition_resolution??source?.acquisition_resolution;
  const grid=spacing(layer.acquisition_grid_spacing_mm)??spacing(source?.grid_spacing_mm);
  const output=spacing(layer.output_spacing_mm)??(layer.affine?spacing([0,1,2].map(column=>Math.hypot(
    ...layer.affine!.slice(0,3).map(row=>row[column]),
  ))):undefined);
  const ontology=layer.label_ontology??provenance.label_ontology??layer.ontology;
  const ontologyName=text(ontology)??text(object(ontology).meaning)??text(object(ontology).label);
  const field=layer.coverage_fields,bounds=field?.inference_bounds_voxel_inclusive;
  const limitedBounds=Array.isArray(bounds)&&bounds.length===2&&bounds.every(point=>Array.isArray(point)&&point.length===3);
  const coverageLimits=source?.coverage?.limitations;
  const evidenceUse=text(provenance.evidence_use);
  const slices=field?.supported_axial_indices;
  const alignment=document.transforms[layer.source_series],uncertainty=alignment?.uncertainty_mm;
  const bindingIssues=workspace?.planning.rehearsalOptions?.acquisitions?.flatMap(scan=>scan.structures??[]).find(link=>link.layerId===layer.id)?.staleInputs;
  const inputs=object(provenance.inputs);
  const channels=Array.isArray(inputs.channels)?inputs.channels.map(object):[];
  return <details className="unified-details native-evidence-detail" open={workspace?true:undefined}><summary>Source & coverage</summary>
    {!!bindingIssues?.length&&<p role="status">Stale model inputs · {bindingIssues.join(', ')}</p>}
    {workspace&&<SourceLocation workspace={workspace} layer={layer}/>}
    <dl>
      <div><dt>Source scan</dt><dd>{source?scanLabel(source):layer.source_series}</dd></div>
      {channels.length>0&&<div><dt>Model inputs</dt><dd>{channels.map(ch=>String(ch.modality??ch.series_id)).join(' + ')}</dd></div>}
      <div><dt>Role and review</dt><dd>{readable(layer.role)} · {layer.review_status}</dd></div>
      <div><dt>Method</dt><dd>{method?readable(method):'Unreported'}{modelName&&' · '+modelName}{version&&' · '+version}</dd></div>
      {evidenceUse&&<div><dt>Evidence use</dt><dd>{readable(evidenceUse)}</dd></div>}
      <div><dt>Acquisition resolution</dt><dd>{spacing(acquisition)??text(acquisition)??'Unreported'}</dd></div>
      {grid&&<div><dt>Source grid spacing</dt><dd>{grid}</dd></div>}
      {output&&<div><dt>Output grid spacing</dt><dd>{output}</dd></div>}
      {ontologyName&&<div><dt>Label meaning</dt><dd>{ontologyName}</dd></div>}
      <div><dt>Source alignment</dt><dd>{alignment?readable(alignment.status):'Unavailable'}{alignment?.method&&' · '+readable(alignment.method)}</dd></div>
      <div><dt>Registration uncertainty</dt><dd>{typeof uncertainty==='number'&&Number.isFinite(uncertainty)?numericText(uncertainty)+' mm reported':alignment?.status==='reference'?'Reference frame':'Unmeasured'}</dd></div>
      <div><dt>Anatomical uncertainty</dt><dd>{text(layer.uncertainty)??(layer.uncertainty?'See recorded evidence':'Unmeasured')}</dd></div>
    </dl>
    {layer.availability&&<p>Layer availability · {readable(layer.availability)}</p>}
    {layer.coverage&&<p>{layer.coverage}</p>}
    {Array.isArray(coverageLimits)&&coverageLimits.map((item,index)=><p key={index}>{readable(item.kind)}{item.basis&&' · '+item.basis}</p>)}
    {limitedBounds&&<p>Supported inference region · output voxel bounds [{bounds[0].join(', ')}] to [{bounds[1].join(', ')}], inclusive</p>}
    {Array.isArray(slices)&&<p>Supported axial slices · {slices.length.toLocaleString()}</p>}
    {field?.outside&&<p>Outside recorded coverage · {field.outside}</p>}
    {field?.zero_meaning&&<p>Unlabeled cells · {field.zero_meaning}</p>}
    {text(inputs.preprocessing)&&<p>Input preparation · {text(inputs.preprocessing)}</p>}
    {text(inputs.development_overlap)&&<p>Benchmark overlap · {text(inputs.development_overlap)}</p>}
    {text(provenance.coverage_limitation)&&provenance.coverage_limitation!==layer.coverage&&<p>{text(provenance.coverage_limitation)}</p>}
    {layer.score_meaning&&<p>Scores · {layer.score_meaning}</p>}
    <details onToggle={e=>setShowProvenance(e.currentTarget.open)}><summary>Recorded provenance and parameters</summary>{showProvenance&&<pre>{JSON.stringify({
      ontology,transformDependencies:layer.transform_dependencies,uncertainty:layer.uncertainty,provenance:layer.provenance,
    },null,2)}</pre>}</details>
  </details>;
}

type AcquisitionReading={covered:boolean|null;sourceVoxel?:number[];intensity?:number;reason?:string};
type BoneReading={layerId:string;seriesId:string;positiveLabel:boolean};
type StructureDifference={structure:string;sources:Array<BoneReading&{method:string}>;meaning:string};
type BoneDecision={preferred:BoneReading[];alternatives:BoneReading[];disagreement:boolean;meaning:string};
function AcquisitionChange({document:doc}:{document:Document}){
  const before=doc.before_acquisition;
  if(!before)return null;
  const old=new Map(before.layers.map(row=>[row.id,row]));
  const current=new Map(doc.layers.map(row=>[row.id,row]));
  const retained=doc.layers.filter(row=>old.get(row.id)?.arraySha256===row.data?.sha256&&old.get(row.id)?.labelValue===row.label_value);
  const changed=doc.layers.filter(row=>old.has(row.id)&&!retained.includes(row));
  const added=doc.layers.filter(row=>!old.has(row.id));
  const removed=before.layers.filter(row=>!current.has(row.id));
  const newSources=[...doc.series,...(doc.source_assets??[]).map(object).filter(r=>r.storage==='source-asset')].filter(r=>!before.sources.some(old=>old.id===r.id));
  return <details className="unified-details" aria-label="Last scan addition"><summary>Before and after the last scan addition</summary>
    <div className="alignment-notice"><strong>Revision {before.revision} → {doc.revision}</strong>
      <span>Before · {before.sources.map(r=>r.label+' ('+r.modality+')').join(', ')}</span>
      <span>Added · {newSources.map(r=>String(r.label??r.id)+' ('+r.modality+')').join(', ')}</span>
      <span>Label masks · {retained.length} identical, {added.length} added, {changed.length} changed, {removed.length} removed</span>
      {added.length>0&&<span>Added · {added.map(r=>r.label).join(', ')}</span>}
      {changed.length>0&&<span>Changed · {changed.map(r=>r.label).join(', ')}</span>}
      {removed.length>0&&<span>Removed · {removed.map(r=>r.label).join(', ')}</span>}
    </div>
    <details><summary>Inspect revision identities and label hashes</summary><pre>{JSON.stringify({before,
      current:{snapshotId:doc.snapshot_id,layers:doc.layers.map(row=>({id:row.id,sourceSeries:row.source_series,arraySha256:row.data?.sha256,labelValue:row.label_value,review:row.review_status}))}},null,2)}</pre></details>
  </details>;
}

function ContributingStructures({items,document,disabled,onInspect}:{items:NonNullable<AcquisitionContribution['structures']>;
  document:Document;disabled:boolean;onInspect:(layer:Layer)=>void}){
  const [open,setOpen]=useState(false);
  return <details onToggle={e=>setOpen(e.currentTarget.open)}><summary>Structures and spatial coverage ({items.length})</summary>
    {open&&items.map(item=>{
      const layer=document.layers.find(r=>r.id===item.layerId);
      if(!layer)return null;
      return <div key={layer.id}><strong>{layer.label}</strong>
        <span>{[item.anchor?'Layer source grid':'',item.channel?'Model input':''].filter(Boolean).join(' · ')} · {readable(layer.kind)}</span>
        {item.staleInputs.length>0&&<span>Stale model inputs · {item.staleInputs.join(', ')}</span>}
        <NativeLayerEvidence layer={layer} document={document}/>
        <button disabled={disabled} onClick={()=>onInspect(layer)}>Inspect {layer.label} on source slices</button>
      </div>})}</details>;
}

export function AcquisitionContributions({workspace:w,onInspect}:{workspace:NativeWorkspace;onInspect:(layer:Layer)=>void}){
  const doc=w.binding!.document;
  const rows=w.planning.rehearsalOptions?.acquisitions;
  const rawOnly=doc.series.length===0?(doc.source_assets??[]).map(object).filter(s=>s.storage==='source-asset'):[];
  const [inspection,setInspection]=useState<{key:string;decision:BoneDecision|null;differences:StructureDifference[];scans:Record<string,AcquisitionReading>}|null>(null);
  const [stages,setStages]=useState<{snapshot:string;rows:PreparationStage[]}|null>(null);
  const snapshot=w.binding!.document.snapshot_id;
  async function reviewStages(){
    const plan=await w.api!.anatomyPrepare({...w.args(),action:'plan'});
    setStages({snapshot:plan.document.snapshot_id,rows:plan.plan.stages});
  }
  const cursor=w.state.cursor;
  const key=JSON.stringify([w.binding?.collectionId,w.binding?.document.snapshot_id,w.series,cursor]);
  const inspected=key===inspection?.key;
  const decision=inspected?inspection.decision:null;
  const differences=inspected?inspection.differences:[];
  const staleApproaches=w.planning.approaches.filter(r=>r.assessmentStatus==='stale');
  const scanReadings=inspected?inspection.scans:{};
  function inspect(){
    if(!cursor)return;
    void w.run(async()=>{
      const sample=await w.api!.anatomySample({...w.args(),point:cursor}) as {boneAdjudication:BoneDecision|null;structureDisagreements:StructureDifference[];acquisitions:Record<string,AcquisitionReading>};
      setInspection({key,decision:sample.boneAdjudication,differences:sample.structureDisagreements,scans:sample.acquisitions??{}});
    });
  }
  return <details className="unified-details" aria-label="Acquisition contributions"><summary>Acquisition contributions</summary>
    <AcquisitionChange document={doc}/>
    {staleApproaches.length>0&&<p role="status">{staleApproaches.length} saved approach assessments are stale for this revision. Reassess in Plan & compare.</p>}
    <button disabled={w.busy||!!w.draft||w.planning.pending} onClick={()=>w.run(reviewStages)}>Show default processing stages</button>
    {doc.series.length>0&&<button disabled={w.busy||!!w.draft||!cursor||!w.patientFrameReady} onClick={inspect}>Compare source labels at cursor</button>}
    {rows?rows.map(row=>{
      const source=doc.series.find(s=>s.id===row.seriesId);
      const sample=scanReadings[row.seriesId];
      return <div className="alignment-notice" key={row.seriesId} data-acquisition-id={row.seriesId}>
      <strong>{row.label} · {row.modality}</strong><span>{row.disposition} · {readable(row.registration)}{row.qc&&' · QC '+readable(row.qc)}</span>
      <span>{row.layerCount?row.layerCount+' source labels · '+row.anatomicalRoles.join(', '):row.asset?'Original acquisition archived':row.modelInputs?.length?'Input intensity used in multimodal estimation':'Source intensity only'}</span>
      {source?.data?.shape&&<span>Native grid · {source.data.shape.join(' × ')} voxels · {spacing(source.grid_spacing_mm)}</span>}
      {source?.coverage?.limitations?.map((issue,i)=><span key={i}>Coverage · {readable(issue.kind)}{issue.basis&&' · '+issue.basis}</span>)}
      {row.structures?.length? <ContributingStructures items={row.structures} document={doc} disabled={w.busy||!!w.draft} onInspect={onInspect}/>:null}
      {sample&&<span>Cursor · {sample.covered===null?sample.reason:sample.covered?
        'inside acquired grid at voxel '+sample.sourceVoxel?.join(', ')+' · source intensity '+numericText(sample.intensity!):
        'outside acquired grid'}</span>}
      {row.uses.map(use=><span key={use}>{use}</span>)}
      {stages?.snapshot===snapshot&&stages.rows.filter(stage=>stage.seriesId===row.seriesId).map(stage=>
        <span key={stage.id}>{stage.label} · {readable(stage.status)}{stage.reason?' · '+stage.reason:''}</span>)}
      {row.modelInputs?.length>0&&<span>Multimodal input · {row.modelInputs.join(', ')}</span>}
      {row.emptyLayers.length>0&&<span>Empty output · {row.emptyLayers.join(', ')}</span>}
      {row.suppliedTracts.length>0&&<span>Supplied tractography · {row.suppliedTracts.join(', ')}</span>}
      {decision?.disagreement&&[...decision.preferred,...decision.alternatives].some(r=>r.seriesId===row.seriesId)&&<span>Conflicting bone labels at cursor</span>}
      {differences.filter(d=>d.sources.some(s=>s.seriesId===row.seriesId)).map(d=><span key={d.structure}>Different {d.structure} labels at cursor</span>)}
      {row.reason&&<span>{row.reason}</span>}
      {row.warnings.map(w=><span key={w}>{w}</span>)}
    </div>}):rawOnly.length?rawOnly.map(s=><div key={String(s.id)} data-acquisition-id={String(s.id)} className="alignment-notice">
      <strong>{String(s.label??s.id)} · {String(s.modality)}</strong><span>Unsupported · archived source</span>
      <span>Scalar reconstruction unavailable. Original acquisition retained.</span>
    </div>):<p>{w.planning.error||(doc.series.length?'Loading source eligibility…':'No source acquisitions')}</p>}
    {inspected&&<div aria-live="polite">{decision?<><p>{decision.disagreement?'Bone labels disagree':'Bone labels agree'} at the sampled cursor.</p>
        {[...decision.preferred,...decision.alternatives].map(r=><p key={r.layerId}>{decision.preferred.includes(r)?'Preferred CT':'Retained MRI'} · {r.layerId} · {r.positiveLabel?'labelled bone':'unlabelled'}</p>)}
        <p>{decision.meaning}</p></>:<p>No overlapping eligible CT and MRI bone estimates at this point.</p>}
        {differences.map(d=><div key={d.structure}><strong>{d.structure} · different positive support</strong>
          {d.sources.map(r=><p key={r.layerId}>{r.layerId} · {r.method} · {r.positiveLabel?'labelled':'unlabelled'}</p>)}
          <p>{d.meaning}</p></div>)}
        {!differences.length&&<p>No further differences among comparable registered labels at this point.</p>}</div>}
  </details>;
}
