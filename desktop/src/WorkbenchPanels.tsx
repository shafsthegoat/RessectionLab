import {useState} from 'react';
import {studyLabel} from './native-study-state';
import type {CasePayload,ResectionApi} from './types';
import {ImagingWorkspacePanel} from './ImagingWorkspacePanel';
import type {DisplaySeriesView,ImportDisplaySeriesRequest} from './workspace-imaging-types';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
export type LibraryCase={id:string;title:string;recent:boolean;bytes:number;native?:boolean;portable?:boolean;timepoint?:string;scanCount?:number;qualifiedCatalog?:boolean;source?:string;anatomy?:string[];missingAnatomy?:string[]};
export type AnatomyRecord={label?:string;provenance?:string;kind?:string;coverage?:string;review_status?:string;runtime_seconds?:number;
  source_sha256?:string;cache_key?:string;model?:{name?:string;version?:string;license?:string;requirements?:string;weights_path?:string;weights_sha256?:string;device?:string;peak_memory_bytes?:number}};
export function anatomyDescription(record:AnatomyRecord|undefined,fallback:string):string {
  if(!record)return fallback;
  if(record.provenance==='simulated'||record.provenance==='generated')return `Generated research geometry · Unvalidated${record.coverage?'. '+record.coverage:''}`;
  if(record.provenance==='estimated')return `${record.model?.name??'Model'} estimate · ${record.review_status??'Unreviewed'}${record.coverage?'. '+record.coverage:''}`;
  if(record.provenance==='source-provided'||record.provenance==='observed')return `Source-provided annotation${record.coverage?'. '+record.coverage:''}`;
  return `${fallback}${record.coverage?'. '+record.coverage:''}`;
}
/** Search only this row's own displayed identity and present anatomy. Catalog
 * receipts contain evidence about other people and must not enter search text. */
export function filterLibraryCases(cases:LibraryCase[],query:string):LibraryCase[] {
  const normalize=(value:string)=>value.normalize('NFKD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
  const terms=normalize(query.trim()).split(/\s+/).filter(Boolean);
  if(!terms.length)return cases;
  return cases.filter(row=>{const text=normalize([row.title,row.source??'',...(row.anatomy??[])].join(' '));return terms.every(term=>text.includes(term));});
}
export function CaseLibraryPanel({cases,onOpen,onBrowse,busy}:{cases:LibraryCase[];onOpen:(id:string)=>void;onBrowse:()=>void;busy:boolean}) {
  const [query,setQuery]=useState('');
  const shown=filterLibraryCases(cases,query),hasCatalog=cases.some(row=>row.qualifiedCatalog);
  return <section className="poc-case-library" aria-label="Saved cases">
    <span className="eyebrow">CASES ON THIS MAC</span><h2>Find a case</h2>
    <p className="muted-note">Open a saved study or explore source-checked research cases.</p>
    <button className="outline-button" onClick={onBrowse} disabled={busy}>Open a saved case…</button>
    <label>Search cases<input type="search" aria-label="Search cases by name, source or anatomy" placeholder="Name, source or anatomy" value={query} maxLength={100} onChange={event=>setQuery(event.target.value)} style={{width:'100%',marginTop:7}}/></label>
    <p className="muted-note" role="status">{shown.length} {shown.length===1?'case':'cases'}{query.trim()?` matching “${query.trim()}”`:''}</p>
    {shown.map(row=><button className="poc-case-card" key={row.id} onClick={()=>onOpen(row.id)} disabled={busy}>
      <strong>{row.title}</strong><span>{row.recent?'Recently opened':row.qualifiedCatalog?`${row.source} catalog`:'Prepared public case'} · {row.native?`${studyLabel(row.timepoint??'preoperative')} · ${row.scanCount??0} scans`:row.portable?'Portable patient study':`${(row.bytes/1048576).toFixed(0)} MB`}</span>
      {row.qualifiedCatalog&&<><span>{row.anatomy?.length?row.anatomy.map(name=>name[0].toUpperCase()+name.slice(1)).join(' + '):'Anatomy coverage unknown'}</span><span>Source checked · anatomy unreviewed</span></>}
    </button>)}
    {!shown.length&&<p className="muted-note">{cases.length?'No matching cases. Try a different name, source or anatomy.':'Open or import imaging to start a case.'}</p>}
    {hasCatalog&&<p className="muted-note">Catalog entries are existing source cases. Anatomical accuracy and independence across datasets remain unverified.</p>}
  </section>;
}
export function WorkbenchDataPanel(props:{source:CasePayload;displayed:CasePayload;series:DisplaySeriesView[];selectedId:string|null;busy:boolean;
  api:ResectionApi|null;onSelect:(id:string|null)=>void;onImport:(request:ImportDisplaySeriesRequest)=>void;
  settings:ViewerWorkspaceSettings;onSettingsChange:(settings:ViewerWorkspaceSettings)=>void}) {
  const {source,displayed}=props;
  const registered=props.series.filter(s=>s.descriptor.registration.overlayPermitted);
  const originals=source.workspaceSession?.originalFiles??[];
  const patient=source.metadata.patient_identity as Record<string,unknown>|undefined;
  const anatomy=displayed.metadata.anatomy_layers as Record<string,AnatomyRecord>|undefined;
  const models=Object.entries(anatomy??{}).filter(([,record])=>record.provenance==='estimated'&&record.model);
  return <div className="poc-data-panel">
    <ImagingWorkspacePanel caseHash={source.caseHash} series={props.series} selectedId={props.selectedId}
      busy={props.busy} importAvailable={!!props.api?.importDisplaySeries&&!props.api.readOnly} onSelect={props.onSelect} onImport={props.onImport}/>
    <section className="case-section">
      <h2>Registered image blend</h2>
      {props.selectedId?<p className="muted-note">Select the reference scan to blend a registered modality.</p>:registered.length?<>
        <label>Blend with<select aria-label="Blend registered modality" value={props.settings.fusion?.seriesId??''}
          onChange={e=>props.onSettingsChange({...props.settings,fusion:{seriesId:e.target.value||null,opacity:props.settings.fusion?.opacity??.4}})}>
          <option value="">None</option>{registered.map(s=><option key={s.descriptor.seriesId} value={s.descriptor.seriesId}>{s.descriptor.modality} · source registered</option>)}
        </select></label>
        {!!props.settings.fusion?.seriesId&&<label>Blend {Math.round(props.settings.fusion.opacity*100)}%
          <input type="range" aria-label="Registered blend opacity" min="0" max="1" step="0.01" value={props.settings.fusion.opacity}
            onChange={e=>props.onSettingsChange({...props.settings,fusion:{seriesId:props.settings.fusion!.seriesId,opacity:Number(e.target.value)}})}/></label>}
        <p className="muted-note">Source dataset registration. Independent anatomical review pending.</p>
      </>:<p className="muted-note">No registered second modality. Unverified images remain in their own frames.</p>}
    </section>
    <section className="case-section"><h2>Source record</h2>
      <dl className="poc-source-facts"><dt>Patient / case</dt><dd>{String(patient?.patient_id??source.caseId)}</dd>
        <dt>Displayed scan</dt><dd>{String(displayed.metadata.selected_modality??'Imported scan')}</dd>
        <dt>Acquired</dt><dd>{String(displayed.metadata.acquisition_datetime??'Unknown in downloaded source')}</dd>
        <dt>Physical frame</dt><dd>{displayed.frame} · millimeters</dd>
        <dt>Grid</dt><dd>{displayed.shape.join(' × ')} · {displayed.spacingMm.map(v=>v.toFixed(2)).join(' × ')} mm</dd>
      </dl>
      {String(source.metadata.primary_source_equivalence??'').startsWith('unverified')&&<p className="poc-caution">Public mirror. Byte equivalence to the primary release is unverified.</p>}
      {!!source.metadata.original_source_status&&<p className="muted-note">{String(source.metadata.original_source_status)}</p>}
      <details><summary>Original files · {originals.length}</summary>{originals.length?originals.map(file=><div className="poc-original" key={file.sha256+file.name}>
        <strong>{file.name}</strong><span>{file.role} · {(file.bytes/1048576).toFixed(1)} MB</span><code>{file.sha256}</code>
      </div>):<p className="muted-note">This older bundle stores image arrays. Original file bytes are not archived.</p>}</details>
      <details><summary>Provenance and licenses</summary>{displayed.sourceRefs?.map(ref=><div className="poc-original" key={ref.source_id}>
        <strong>{ref.source_id}</strong><span>{ref.provenance??'Unknown'} · {ref.license??'License not recorded'}</span><p>{ref.uri}</p><code>{ref.sha256}</code>
      </div>)}</details>
    </section>
    {!!models.length&&<section className="case-section"><h2>Saved model outputs</h2>{models.map(([name,record])=><details key={name}>
      <summary>{record.label??name} · {record.model?.name}</summary><dl className="poc-source-facts">
        <dt>Method</dt><dd>{record.model?.name} {record.model?.version}</dd><dt>Input requirements</dt><dd>{record.model?.requirements??'See saved model record'}</dd>
        <dt>Run</dt><dd>{record.runtime_seconds?.toFixed(2)??'Recorded'} s · {record.model?.device??'Local'} · cached</dd>
        <dt>Weights</dt><dd>{record.model?.weights_path??'See saved model record'}</dd><dt>Weights SHA256</dt><dd>{record.model?.weights_sha256??'See saved model record'}</dd>
        <dt>Saved mask SHA256</dt><dd>{record.source_sha256}</dd><dt>License</dt><dd>{record.model?.license??'See saved model record'}</dd>
      </dl><p className="muted-note">{record.coverage} {record.review_status}</p>
    </details>)}</section>}
  </div>;
}
