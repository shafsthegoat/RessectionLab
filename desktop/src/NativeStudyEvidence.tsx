import {useState} from 'react';
import type {NativeWorkspace} from './use-native-workspace';
import type {Document,EnvironmentOptions,Layer,PreparationStage} from './native-anatomy-data';
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

export function NativeLayerEvidence({layer,document}:{layer:Layer;document:Document}){
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
  return <details className="unified-details native-evidence-detail"><summary>Source & coverage</summary>
    <dl>
      <div><dt>Source scan</dt><dd>{source?scanLabel(source):layer.source_series}</dd></div>
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
    {layer.score_meaning&&<p>Scores · {layer.score_meaning}</p>}
    <details><summary>Recorded provenance and parameters</summary><pre>{JSON.stringify({
      ontology,transformDependencies:layer.transform_dependencies,uncertainty:layer.uncertainty,provenance:layer.provenance,
    },null,2)}</pre></details>
  </details>;
}

type BoneReading={layerId:string;seriesId:string;positiveLabel:boolean};
type StructureDifference={structure:string;sources:Array<BoneReading&{method:string}>;meaning:string};
type BoneDecision={preferred:BoneReading[];alternatives:BoneReading[];disagreement:boolean;meaning:string};
export function AcquisitionContributions({workspace:w}:{workspace:NativeWorkspace}){
  const doc=w.binding!.document;
  const rows=w.planning.rehearsalOptions?.acquisitions;
  const rawOnly=doc.series.length===0?(doc.source_assets??[]).map(object).filter(s=>s.storage==='source-asset'):[];
  const [inspection,setInspection]=useState<{key:string;decision:BoneDecision|null;differences:StructureDifference[]}|null>(null);
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
  const structures=(w.binding?.document.layers??[]).filter(r=>['bone','skull','brain','dura','vessel','vessels','artery','vein','sinus'].includes(r.kind)||['head-915','head-916'].includes(r.id));
  function inspect(){
    if(!cursor)return;
    void w.run(async()=>{
      const sample=await w.api!.anatomySample({...w.args(),point:cursor}) as {boneAdjudication:BoneDecision|null;structureDisagreements:StructureDifference[]};
      setInspection({key,decision:sample.boneAdjudication,differences:sample.structureDisagreements});
    });
  }
  return <details className="unified-details" aria-label="Acquisition contributions"><summary>Acquisition contributions</summary>
    <button disabled={w.busy||!!w.draft||w.planning.pending} onClick={()=>w.run(reviewStages)}>Show default processing stages</button>
    {rows?rows.map(row=><div className="alignment-notice" key={row.seriesId} data-acquisition-id={row.seriesId}>
      <strong>{row.label} · {row.modality}</strong><span>{row.disposition} · {readable(row.registration)}{row.qc&&' · QC '+readable(row.qc)}</span>
      <span>{row.layerCount?row.layerCount+' source labels · '+row.anatomicalRoles.join(', '):row.asset?'Original acquisition archived':'Source intensity only'}</span>
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
    </div>):rawOnly.length?rawOnly.map(s=><div key={String(s.id)} data-acquisition-id={String(s.id)} className="alignment-notice">
      <strong>{String(s.label??s.id)} · {String(s.modality)}</strong><span>Unsupported · archived source</span>
      <span>Scalar reconstruction unavailable. Original acquisition retained.</span>
    </div>):<p>{w.planning.error||(doc.series.length?'Loading source eligibility…':'No source acquisitions')}</p>}
    {structures.length>1&&<><button disabled={w.busy||!!w.draft||!cursor||!w.patientFrameReady} onClick={inspect}>Compare source labels at cursor</button>
      {inspected&&<div aria-live="polite">{decision?<><p>{decision.disagreement?'Bone labels disagree':'Bone labels agree'} at the sampled cursor.</p>
        {[...decision.preferred,...decision.alternatives].map(r=><p key={r.layerId}>{decision.preferred.includes(r)?'Preferred CT':'Retained MRI'} · {r.layerId} · {r.positiveLabel?'labelled bone':'unlabelled'}</p>)}
        <p>{decision.meaning}</p></>:<p>No overlapping eligible CT and MRI bone estimates at this point.</p>}
        {differences.map(d=><div key={d.structure}><strong>{d.structure} · different positive support</strong>
          {d.sources.map(r=><p key={r.layerId}>{r.layerId} · {r.method} · {r.positiveLabel?'labelled':'unlabelled'}</p>)}
          <p>{d.meaning}</p></div>)}
        {!differences.length&&<p>No further differences among comparable registered labels at this point.</p>}</div>}</>}
  </details>;
}
