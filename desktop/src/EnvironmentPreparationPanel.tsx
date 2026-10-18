import {useState} from 'react';
import type {Binding, Layer} from './native-anatomy-data';

const kinds:Record<string,string[]>={brain:['brain'],tumor:['tumor'],bone:['bone','skull'],vessels:['vessel','vessels','artery','vein']};
export interface EnvironmentPreparation {collectionId:string;snapshotId:string;layers:Record<string,string>;allowUnreviewedRegistration:boolean}
export function EnvironmentPreparationPanel({binding,busy,onPrepare,onAnatomy}:{binding:Binding;busy:boolean;onPrepare:(options:EnvironmentPreparation)=>Promise<void>;onAnatomy:()=>void}) {
  const candidates=(role:string)=>binding.document.layers.filter(layer=>kinds[role].includes(layer.kind));
  const [layers,setLayers]=useState<Record<string,string>>(()=>Object.fromEntries(Object.keys(kinds).map(role=>[role,candidates(role)[0]?.id??''])));
  const [allow,setAllow]=useState(false),[preparing,setPreparing]=useState(false),[error,setError]=useState<string|null>(null);
  const selected=Object.values(layers).filter(Boolean);
  const needsReview=(series:string)=>{
    const seen=new Set<string>();
    let current:string|undefined=series;
    while(current){
      if(seen.has(current))return true;
      seen.add(current);
      const record:{status:string;parent_series?:string|null}|undefined=binding.document.transforms[current];
      if(!record||!['reference','reviewed','source-provided'].includes(record.status))return true;
      current=record.parent_series??undefined;
    }
    return false;
  };
  const unreviewed=binding.document.layers.filter(layer=>selected.includes(layer.id)&&needsReview(layer.source_series));
  return <section className="surgical-panel" aria-label="Prepare standard rehearsal environment">
    <span className="eyebrow">STANDARD RESEARCH ENVIRONMENT</span><h2>Prepare this patient</h2>
    <p className="surgical-note">RAS+ millimeters · 1 mm simulation grid · V2 mechanics · two instruments. Source scans and segmentations stay unchanged.</p>
    <p className="surgical-note">Use the supplied anatomy below. Missing structures remain unknown; preparing a case does not infer vessels or establish surgical safety.</p>
    <fieldset className="surgical-controls" disabled={busy||preparing}>
      {Object.keys(kinds).map(role=><label key={role}>{role}<select aria-label={`Environment ${role} layer`} value={layers[role]} onChange={e=>setLayers({...layers,[role]:e.target.value})}>
        <option value="">Unknown / not supplied</option>{candidates(role).map((layer:Layer)=><option key={layer.id} value={layer.id}>{layer.label||layer.id} · {layer.review_status}</option>)}
      </select></label>)}
      <p>Reference scan: {binding.document.series.find(s=>s.id===binding.document.reference_series)?.label||binding.document.reference_series}. The displayed CT/MRI choice does not change the preparation grid.</p>
      {!!unreviewed.length&&<p className="surgical-warning">Estimated alignment needs review for: {unreviewed.map(l=>l.label||l.id).join(', ')}.</p>}
      <label><input aria-label="Allow estimated alignment for research" type="checkbox" checked={allow} onChange={e=>setAllow(e.target.checked)}/>Use estimated alignment for this research derivative. Review status stays unchanged.</label>
      <button className="primary-button" disabled={!selected.length||(!allow&&unreviewed.length>0)} onClick={async()=>{setPreparing(true);setError(null);try{await onPrepare({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,layers:Object.fromEntries(Object.entries(layers).filter(([,id])=>id)),allowUnreviewedRegistration:allow});}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setPreparing(false);}}}>{preparing?'Preparing environment…':'Prepare standard environment'}</button>
      <button onClick={onAnatomy}>Review anatomy and alignment</button>
    </fieldset>{error&&<p role="alert" className="surgical-error">{error}</p>}
  </section>;
}
