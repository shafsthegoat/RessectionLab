import {useDeferredValue,useEffect,useId,useMemo,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {ArrowUpRight,ChevronDown,ChevronLeft,ChevronRight,FolderOpen,Info,Layers3,Search,X} from 'lucide-react';
import {studyLabel} from './native-study-state';
import type {CasePayload,ResectionApi,CaseLibraryFolder} from './types';
import {ImagingWorkspacePanel} from './ImagingWorkspacePanel';
import type {DisplaySeriesView,ImportDisplaySeriesRequest} from './workspace-imaging-types';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
export type LibraryCase={id:string;title:string;recent:boolean;bytes:number;native?:boolean;portable?:boolean;timepoint?:string;scanCount?:number;qualifiedCatalog?:boolean;reviewedExternal?:boolean;source?:string;anatomy?:string[];capabilities?:string[];limitations?:string[];missingAnatomy?:string[];sourceLocation?:{label:string;status:'source-reported-unreviewed'}};
export type AnatomyRecord={label?:string;provenance?:string;kind?:string;coverage?:string;review_status?:string;runtime_seconds?:number;
  source_sha256?:string;cache_key?:string;model?:{name?:string;version?:string;license?:string;requirements?:string;weights_path?:string;weights_sha256?:string;device?:string;peak_memory_bytes?:number}};
export function anatomyDescription(record:AnatomyRecord|undefined,fallback:string):string {
  if(!record)return fallback;
  if(record.provenance==='simulated'||record.provenance==='generated')return `Generated research geometry · Unvalidated${record.coverage?'. '+record.coverage:''}`;
  if(record.provenance==='estimated')return `${record.model?.name??'Model'} estimate · ${record.review_status??'Unreviewed'}${record.coverage?'. '+record.coverage:''}`;
  if(record.provenance==='source-provided'||record.provenance==='observed')return `Source-provided annotation${record.coverage?'. '+record.coverage:''}`;
  return `${fallback}${record.coverage?'. '+record.coverage:''}`;
}
/** Search only this row's own displayed identity, present anatomy and source report. Catalog
 * receipts contain evidence about other people and must not enter search text. */
function normalizeLibraryText(value:string):string {
  return value.normalize('NFKD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
}
function librarySearchText(row:LibraryCase):string {
  const location=row.qualifiedCatalog&&row.sourceLocation?.status==='source-reported-unreviewed'?row.sourceLocation.label:'';
  return normalizeLibraryText([row.title,row.source??'',...(row.anatomy??[]),...(row.capabilities??[]).map(name=>name.replaceAll('_',' ')),location].join(' '));
}
export function filterLibraryCases(cases:LibraryCase[],query:string):LibraryCase[] {
  const terms=normalizeLibraryText(query.trim()).split(/\s+/).filter(Boolean);
  if(!terms.length)return cases;
  return cases.filter(row=>{const text=librarySearchText(row);return terms.every(term=>text.includes(term));});
}
const libraryPageSize=50;
type LibraryScope='all'|'recent'|'studies'|'catalog';
function libraryStudyLabel(row:LibraryCase):string {
  return row.native?`${studyLabel(row.timepoint??'preoperative')} · ${row.scanCount??0} scans`:row.portable?'Portable patient study':`${(row.bytes/1048576).toFixed(0)} MB`;
}
function libraryAnatomyLabel(row:LibraryCase):string {
  return row.anatomy?.length?row.anatomy.map(name=>name?name[0].toUpperCase()+name.slice(1):'').join(' + '):'Anatomy coverage unknown';
}
function libraryGateLabel(row:LibraryCase):string {
  return row.reviewedExternal?'Engineering review · limited capabilities':row.qualifiedCatalog?'Source checked · anatomy unreviewed':row.native||row.portable?'Saved patient study':'Prepared public case';
}
export function CaseLibraryPanel({cases,onOpen,onBrowse,busy,loading=false,libraryFolder,onChooseLibrary,libraryError,newStudyControl}:{cases:LibraryCase[];onOpen:(id:string)=>void;onBrowse:()=>void;busy:boolean;loading?:boolean;libraryFolder?:CaseLibraryFolder|null;onChooseLibrary?:()=>void;libraryError?:string|null;newStudyControl?:ReactNode}) {
  const [query,setQuery]=useState('');
  const [scope,setScope]=useState<LibraryScope>('all');
  const [page,setPage]=useState(0);
  const [selectedId,setSelectedId]=useState<string|null>(null);
  const searchRef=useRef<HTMLInputElement>(null),detailRef=useRef<HTMLElement>(null),detailTriggerRef=useRef<HTMLButtonElement|null>(null),listRef=useRef<HTMLUListElement>(null);
  const detailId=useId(),resultId=useId(),deferredQuery=useDeferredValue(query);
  // Index only the allowed row evidence once per listing, then keep rendering bounded.
  const indexed=useMemo(()=>cases.map(row=>({row,text:librarySearchText(row)})),[cases]);
  const shown=useMemo(()=>{
    const terms=normalizeLibraryText(deferredQuery.trim()).split(/\s+/).filter(Boolean);
    return indexed.filter(({row,text})=>(scope==='all'||scope==='recent'&&row.recent||scope==='studies'&&(row.native||row.portable)||scope==='catalog'&&(row.qualifiedCatalog||row.reviewedExternal))&&terms.every(term=>text.includes(term))).map(({row})=>row);
  },[indexed,deferredQuery,scope]);
  const pageCount=Math.max(1,Math.ceil(shown.length/libraryPageSize)),currentPage=Math.min(page,pageCount-1);
  const visible=shown.slice(currentPage*libraryPageSize,(currentPage+1)*libraryPageSize);
  const selected=shown.find(row=>row.id===selectedId);
  const hasCatalog=useMemo(()=>cases.some(row=>row.qualifiedCatalog||row.reviewedExternal),[cases]);
  const searching=query!==deferredQuery,error=libraryError||libraryFolder?.error;
  useEffect(()=>{if(selectedId)detailRef.current?.focus();},[selectedId]);
  const clearFilters=()=>{setQuery('');setScope('all');setPage(0);setSelectedId(null);searchRef.current?.focus();};
  const closeDetails=()=>{setSelectedId(null);requestAnimationFrame(()=>detailTriggerRef.current?.focus());};
  const changePage=(next:number)=>{setPage(next);setSelectedId(null);requestAnimationFrame(()=>{listRef.current?.scrollTo({top:0});listRef.current?.querySelector<HTMLButtonElement>('.poc-case-card:not(:disabled),.case-library-detail-toggle')?.focus();});};
  const folderControl=onChooseLibrary&&<div className="poc-library-folder case-library-location">
    <FolderOpen size={15} aria-hidden="true"/><div><span>Case library folder</span><p title={libraryFolder?.folder??undefined}>{libraryFolder?.folder??'No library folder selected'}</p>
    {!libraryFolder?.folder&&<small>Choose the folder containing your source-checked case catalog.</small>}</div>
    <button className="outline-button" onClick={onChooseLibrary} disabled={busy}>Choose case library…</button>
  </div>;
  return <section className="poc-case-library case-library-workspace" aria-label="Saved cases">
    <header className="case-library-heading">
      <h1>Case library</h1>
      <div className="case-library-actions">{!!cases.length&&!error&&folderControl&&<details className="case-library-config" onKeyDown={event=>{if(event.key==='Escape'){event.currentTarget.open=false;event.currentTarget.querySelector('summary')?.focus();}}}><summary><FolderOpen size={14}/><span>Library location</span><ChevronDown size={13}/></summary>{folderControl}</details>}<button className="outline-button" onClick={onBrowse} disabled={busy}><FolderOpen size={15}/><span>Open a saved case…</span></button>{newStudyControl}</div>
    </header>
    {(!cases.length||!!error)&&folderControl}
    {error&&<div className="case-library-error" role="alert"><Info size={17} aria-hidden="true"/><div><strong>Case library needs attention</strong><p>{error}</p><small>Open a saved case directly{onChooseLibrary?' or choose the library folder again':''}.</small></div></div>}
    <div className="case-library-toolbar">
      <label className="case-library-search"><Search size={17} aria-hidden="true"/><span className="case-library-sr-only">Search cases</span><input ref={searchRef} type="search" aria-label="Search cases by name, source, anatomy or reported site" aria-describedby={resultId} placeholder="Search name, source, anatomy or reported site" value={query} maxLength={100} onChange={event=>{setQuery(event.target.value);setPage(0);setSelectedId(null);}}/>
        {!!query&&<button type="button" aria-label="Clear case search" onClick={()=>{setQuery('');setPage(0);setSelectedId(null);searchRef.current?.focus();}}><X size={15}/></button>}</label>
      <label className="case-library-scope"><span>Show</span><select aria-label="Filter case library" value={scope} onChange={event=>{setScope(event.target.value as LibraryScope);setPage(0);setSelectedId(null);}}><option value="all">All cases</option><option value="recent">Recently opened</option><option value="studies">Patient studies</option><option value="catalog">Research catalog</option></select></label>
      <p id={resultId} className="case-library-result-count" role="status" aria-live="polite">{loading?'Loading case library…':searching?'Searching…':`${shown.length} ${shown.length===1?'case':'cases'}${query.trim()?` matching “${query.trim()}”`:''}`}</p>
    </div>
    <div className={`case-library-content ${selected?'has-preview':''}`}>
      <div className="case-library-results" aria-busy={loading||searching}>
        {!!shown.length&&<>
          <div className="case-library-column-headings" aria-hidden="true"><span>Patient / case</span><span>Source & anatomy</span><span>Evidence status</span><span/></div>
          <ul ref={listRef} tabIndex={-1} className="case-library-rows" aria-label="Available cases">
            {visible.map(row=><li className={`case-library-row ${selectedId===row.id?'is-selected':''}`} key={row.id}>
              <button className="poc-case-card" onClick={()=>onOpen(row.id)} disabled={busy} aria-label={`Open ${row.title}`}>
                <span className="case-library-identity"><span className="case-library-case-icon" aria-hidden="true">{row.native||row.portable?<FolderOpen size={21}/>:<Layers3 size={21}/>}</span><span><strong title={row.title}>{row.title}</strong><span className="case-library-study">{libraryStudyLabel(row)}{row.recent&&<em>Recently opened</em>}</span></span></span>
                <span className="case-library-source"><span>{row.source?`${row.source}${row.qualifiedCatalog?' catalog':''}`:row.native||row.portable?'Local patient study':'Prepared public case'}</span>{(row.qualifiedCatalog||row.reviewedExternal)&&<span>{row.anatomy?.length?libraryAnatomyLabel(row):row.reviewedExternal?row.capabilities?.slice(0,2).map(name=>name.replaceAll('_',' ')).join(' · ')||'See case capabilities':libraryAnatomyLabel(row)}</span>}</span>
                <span className="case-library-evidence"><span className={`case-library-gate ${row.reviewedExternal||row.qualifiedCatalog?'is-research':''}`}>{libraryGateLabel(row)}</span>{row.reviewedExternal&&!!row.limitations?.length&&<span className="case-library-limitation" title={row.limitations[0]}>{row.limitations[0]}</span>}
                {row.qualifiedCatalog&&row.sourceLocation?.status==='source-reported-unreviewed'&&<span className="poc-source-location">Reported site: {row.sourceLocation.label}<small>Dataset / paper report · review pending</small></span>}</span>
              </button>
              <button className="case-library-detail-toggle" aria-label={`Details for ${row.title}`} aria-expanded={selectedId===row.id} aria-controls={selectedId===row.id?detailId:undefined} onClick={event=>{detailTriggerRef.current=event.currentTarget;setSelectedId(selectedId===row.id?null:row.id);}}><Info size={15}/><span>Details</span></button>
            </li>)}
          </ul>
          {pageCount>1&&<nav className="case-library-pagination" aria-label="Case library pages"><span>{currentPage*libraryPageSize+1}–{Math.min((currentPage+1)*libraryPageSize,shown.length)} of {shown.length}</span><div><button aria-label="Previous case page" disabled={currentPage===0} onClick={()=>changePage(currentPage-1)}><ChevronLeft size={15}/></button><span>Page {currentPage+1} of {pageCount}</span><button aria-label="Next case page" disabled={currentPage===pageCount-1} onClick={()=>changePage(currentPage+1)}><ChevronRight size={15}/></button></div></nav>}
        </>}
        {!shown.length&&<div className="case-library-empty"><FolderOpen size={28} strokeWidth={1}/><h2>{loading?'Loading case library…':cases.length?'No matching cases':'No cases yet'}</h2><p>{loading?'Reading local cases…':cases.length?'Try a different name, source, anatomy or reported site.':'Open or import imaging to start a case.'}</p>{cases.length>0&&!loading&&<button className="outline-button" onClick={clearFilters}>Clear search and filters</button>}{!cases.length&&!loading&&onChooseLibrary&&!libraryFolder?.folder&&<small>Choose a case library folder to browse a local research catalog.</small>}</div>}
      </div>
      {selected&&<aside ref={detailRef} id={detailId} tabIndex={-1} className="case-library-preview" aria-label={`Case details: ${selected.title}`} onKeyDown={event=>{if(event.key==='Escape'){event.preventDefault();closeDetails();}}}>
        <div className="case-library-preview-heading"><span className="eyebrow">CASE DETAILS</span><button className="icon-button" aria-label="Close case details" onClick={closeDetails}><X size={17}/></button></div>
        <div className="case-library-preview-symbol"><Layers3 size={37} strokeWidth={1}/><span>Saved case record</span></div>
        <h2>{selected.title}</h2><p className="case-library-preview-study">{libraryStudyLabel(selected)}</p>
        <p className="case-library-preview-gate">{libraryGateLabel(selected)}</p>
        <dl><dt>Source</dt><dd>{selected.source??(selected.native||selected.portable?'Local patient study':'Prepared public case')}</dd>{(selected.qualifiedCatalog||selected.reviewedExternal)&&<><dt>Present anatomy</dt><dd>{libraryAnatomyLabel(selected)}</dd></>}
        {selected.qualifiedCatalog&&selected.sourceLocation?.status==='source-reported-unreviewed'&&<><dt>Reported site</dt><dd>{selected.sourceLocation.label}<small>Dataset / paper report · review pending</small></dd></>}
        {!!selected.capabilities?.length&&<><dt>Available capabilities</dt><dd>{selected.capabilities.map(name=>name.replaceAll('_',' ')).join(' · ')}</dd></>}
        {!!selected.missingAnatomy?.length&&<><dt>Missing anatomy · unassessed</dt><dd>{selected.missingAnatomy.map(name=>name.replaceAll('_',' ')).join(' · ')}</dd></>}</dl>
        {!!selected.limitations?.length&&<div className="case-library-preview-limits"><h3>Scope & limitations</h3><ul>{selected.limitations.map((limit,index)=><li key={index}>{limit}</li>)}</ul></div>}
        <button className="outline-button case-library-open" disabled={busy} onClick={()=>onOpen(selected.id)}><span>Open case</span><ArrowUpRight size={16}/></button>
      </aside>}
    </div>
    {hasCatalog&&<p className="case-library-catalog-note">Catalog entries are existing source cases. Anatomical accuracy and independence across datasets remain unverified.</p>}
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
          <option value="">None</option>{registered.map(s=><option key={s.descriptor.seriesId} value={s.descriptor.seriesId}>{s.descriptor.modality} · {s.descriptor.registration.status==='verified'?'reviewed derived alignment':'source registered'}</option>)}
        </select></label>
        {!!props.settings.fusion?.seriesId&&<label>Blend {Math.round(props.settings.fusion.opacity*100)}%
          <input type="range" aria-label="Registered blend opacity" min="0" max="1" step="0.01" value={props.settings.fusion.opacity}
            onChange={e=>props.onSettingsChange({...props.settings,fusion:{seriesId:props.settings.fusion!.seriesId,opacity:Number(e.target.value)}})}/></label>}
        <p className="muted-note">{registered.some(s=>s.descriptor.registration.status==='verified')?'Derived alignment reviewed for its stated research scope. Original observations and alignment uncertainty remain in the evidence record.':'Source dataset registration. Independent anatomical review pending.'}</p>
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
