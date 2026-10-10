import {useState} from 'react';
import type {AnnotationKind,DisplayModality,DisplaySeriesView,ImportDisplaySeriesRequest} from './workspace-imaging-types';
export function ImagingWorkspacePanel(props:{caseHash:string;series:DisplaySeriesView[];selectedId:string|null;busy:boolean;importAvailable:boolean;onSelect:(id:string|null)=>void;onImport:(request:ImportDisplaySeriesRequest)=>void}) {
  const [modality,setModality]=useState<DisplayModality>('T2');
  const [annotationKind,setAnnotationKind]=useState<AnnotationKind>('none');
  return <section className="case-section" aria-label="Imaging workspace">
    <h2>Imaging workspace <span>{1+props.series.length}</span></h2>
    <label>Inspect scan<select aria-label="Inspect scan" value={props.selectedId??''} onChange={e=>props.onSelect(e.target.value||null)} disabled={props.busy}>
      <option value="">Primary planning image</option>
      {props.series.map(s=><option key={s.descriptor.seriesId} value={s.descriptor.seriesId}>{s.descriptor.modality} · separate image grid</option>)}
    </select></label>
    {props.selectedId && <p className="muted-note">Separate native frame. Registration and same-person association are unverified. Planning overlays and episode replay are hidden.</p>}
    <label>Additional image modality<select aria-label="Additional image modality" value={modality} onChange={e=>setModality(e.target.value as DisplayModality)}>
      {(['T1','T1CE','T2','FLAIR','CT','CTA','TOF-MRA','MRA','SWI','other-3D-scalar'] as const).map(m=><option key={m}>{m}</option>)}
    </select></label>
    <label>Optional aligned labels<select aria-label="Annotation provenance" value={annotationKind} onChange={e=>setAnnotationKind(e.target.value as AnnotationKind)}>
      <option value="none">No annotation</option><option value="source-provided">Source-provided annotation</option><option value="estimated">Model estimate · provenance unreviewed</option>
    </select></label>
    <button disabled={props.busy||!props.importAvailable||props.series.length>=4} onClick={()=>props.onImport({caseHash:props.caseHash,modality,annotationKind})}>Add image for inspection</button>
    {!props.importAvailable && <p className="muted-note">This engine does not provide additional-image import.</p>}
    <p className="muted-note">Save preserves attached images and their view settings. Sources keep separate native frames; saving does not establish registration or planning suitability.</p>
    {props.series.map(s=><details key={s.descriptor.seriesId}><summary>{s.descriptor.modality} · {s.descriptor.annotationKind==='none'?'imported image':s.descriptor.annotationKind+' labels'}</summary>
      <p>{s.volume.shape.join(' × ')} · {s.volume.frame} · mm. {s.descriptor.registration.reason}</p>
      <p>{s.descriptor.annotationCoverage}. Acquisition time unknown. Display only; not a planning or evaluation input.</p>
    </details>)}
  </section>;
}
