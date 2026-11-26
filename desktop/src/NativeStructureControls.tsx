import {Fragment,useEffect,useId,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {ChevronRight,Eye,EyeOff,Focus,RotateCcw,Search} from 'lucide-react';
import type {ViewerVolume} from './viewer/contracts';
import type {Layer} from './native-anatomy-data';
import {studyLayerDisplayKind} from './native-study-state';
import {structureDisplayName} from './structure-display-name';
import {tractColor} from './viewer/tractDisplay';
import {layerAppearance,withLayerColor} from './viewer/workspaceSettings';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import './native-structure-controls.css';

export interface NativeStructureControlsProps {
  volume:ViewerVolume|null;
  layers?:ReadonlyArray<Layer>;
  tracts?:ReadonlyArray<{id:string;label:string;source_series:string}>;
  visibleLayers:Record<string,boolean>;
  onVisibilityChange:(visible:Record<string,boolean>)=>void;
  settings:ViewerWorkspaceSettings;
  onSettingsChange:(settings:ViewerWorkspaceSettings)=>void;
  selectedStructure:string|null;
  onSelectStructure:(id:string|null)=>void;
  onFocusStructure:(id:string)=>void;
  unavailable?:ReadonlyArray<string>;
  pendingLayers?:ReadonlyArray<string>;
  layerErrors?:Record<string,string>;
  onRetryLayer?:(id:string)=>void;
  disabled?:boolean;
  showSelectionControls?:boolean;
  selectionInspector?:ReactNode;
}

const vascular=(kind:string)=>['vessel','vessels','artery','vein','sinus'].includes(kind);
const lesionContext=(kind:string)=>['enhancing-tumor','nonenhancing-tumor','tumor-core','tumor-whole','edema','lesion-context'].includes(kind);

export function NativeStructureControls({volume,layers,tracts,visibleLayers,onVisibilityChange,settings,onSettingsChange,selectedStructure,onSelectStructure,onFocusStructure,unavailable=[],pendingLayers=[],layerErrors={},onRetryLayer,disabled=false,showSelectionControls=true,selectionInspector}:NativeStructureControlsProps){
  const previousVisibility=useRef<Record<string,boolean>|null>(null);
  const [isolated,setIsolated]=useState<string|null>(null),[search,setSearch]=useState('');
  const list=useRef<HTMLElement>(null),lastSelection=useRef<string|null>(null),rowId=useId();
  const resident=volume?.compartments??[];
  const structures=[...(layers?layers.map(layer=>{const row=resident.find(item=>item.name===layer.id);return {id:layer.id,label:layer.label,kind:layer.kind,sourceRole:layer.role,displayKind:studyLayerDisplayKind(layer),color:layer.color,source:layer.source_series,qa:row?.qa,provenance:layer.provenance,volumeMm3:row?.volumeMm3};}):
    resident.map(layer=>({id:layer.name,label:layer.label??layer.name.replaceAll('_',' '),kind:layer.kind??'label',sourceRole:undefined,displayKind:layer.displayKind??layer.kind,color:layer.color,source:layer.sourceIdentity,qa:layer.qa,provenance:undefined,volumeMm3:layer.volumeMm3}))),...(tracts??(volume?.tractography??[]).map(t=>({...t,source_series:t.sourceSeries}))).map((tract,index)=>({id:tract.id,label:tract.label,kind:'tract',sourceRole:undefined,displayKind:'tract',color:tractColor(index),source:tract.source_series,qa:undefined,provenance:undefined,volumeMm3:undefined}))]
    .map(structure=>({...structure,display:structureDisplayName({label:structure.label,kind:structure.kind,sourceRole:structure.sourceRole,provenance:structure.provenance,volumeMm3:structure.volumeMm3,voxelCount:structure.qa?.voxelCount})}));
  const query=search.trim().toLowerCase(),matches=(structure:typeof structures[number])=>[structure.label,structure.display.name,structure.kind].join(' ').toLowerCase().includes(query);
  useEffect(()=>{
    if(lastSelection.current===selectedStructure)return;
    if(!selectedStructure){lastSelection.current=null;return;}
    const target=structures.find(structure=>structure.id===selectedStructure);
    if(query&&target&&!matches(target)){
      setSearch('');
      return;
    }
    const row=list.current?.querySelector<HTMLElement>(`[data-structure-id="${CSS.escape(selectedStructure)}"]`);
    if(!row)return;
    const group=row.closest('details');
    if(group)group.open=true;
    row.scrollIntoView({block:'nearest'});
    lastSelection.current=selectedStructure;
  },[selectedStructure,layers,volume,tracts,search]);
  if(!volume&&!layers)return null;
  const residentIds=new Set([...resident.map(layer=>layer.name),...(volume?.tractography??[]).map(tract=>tract.id)]);
  const selected=structures.find(structure=>structure.id===selectedStructure);
  const select=(id:string)=>{if(visibleLayers[id]===false)onVisibilityChange({...visibleLayers,[id]:true});onSelectStructure(id);};
  const isolate=(id:string)=>{
    if(isolated===id&&previousVisibility.current){onVisibilityChange(previousVisibility.current);previousVisibility.current=null;setIsolated(null);return;}
    if(!previousVisibility.current)previousVisibility.current={...visibleLayers};
    onVisibilityChange({...visibleLayers,...Object.fromEntries(structures.map(s=>[s.id,s.id===id]))});setIsolated(id);onSelectStructure(id);
  };
  const reset=()=>{onVisibilityChange({...visibleLayers,...Object.fromEntries(structures.map(s=>[s.id,true]))});onSettingsChange({...settings,layerOpacity:{},layerColor:{}});onSelectStructure(null);previousVisibility.current=null;setIsolated(null);};
  // `noun` completes the group's "Show all …" / "Hide all …" control name.
  const groups=[
    {id:'targets',label:'Lesion',noun:'lesion layers',items:structures.filter(s=>s.kind==='tumor')},
    {id:'lesion-context',label:'Lesion compartments',noun:'lesion compartments',items:structures.filter(s=>lesionContext(s.kind))},
    {id:'anatomy',label:'Anatomy',noun:'anatomy',items:structures.filter(s=>!vascular(s.kind)&&!lesionContext(s.kind)&&!['tumor','tract','brain-region','review-region','head-envelope'].includes(s.kind))},
    {id:'vessels',label:'Vessels',noun:'vessels',items:structures.filter(s=>vascular(s.kind))},
    {id:'regions',label:'Brain regions',noun:'brain regions',items:structures.filter(s=>s.kind==='brain-region')},
    {id:'tracts',label:'Tracts',noun:'tracts',items:structures.filter(s=>s.kind==='tract')},
    {id:'review',label:'Review layers',noun:'review layers',items:structures.filter(s=>['review-region','head-envelope'].includes(s.kind))},
  ];
  return <section ref={list} className="native-structures" aria-label="Anatomical structures">
    <div className="native-structure-heading"><div><strong>Structures</strong><span>{structures.filter(s=>residentIds.has(s.id)&&visibleLayers[s.id]!==false&&!unavailable.includes(s.id)&&!layerErrors[s.id]&&!pendingLayers.includes(s.id)).length} / {structures.length} shown</span></div><button disabled={disabled} title="Restore all structure visibility, opacity and colors" aria-label="Reset structure display" onClick={reset}><RotateCcw size={13}/></button></div>
    {layers&&<label className="native-surface-choice">3D surface<select aria-label="Native surface representation" disabled={disabled}
      value={settings.smoothSurfaces===false?'source':'conditioned'} onChange={e=>onSettingsChange({...settings,smoothSurfaces:e.target.value==='conditioned'})}>
      <option value="conditioned">Conditioned contours</option><option value="source">Source isosurface</option></select></label>}
    {structures.length>7&&<label className="native-structure-search"><Search size={14}/><input aria-label="Search anatomy" placeholder="Find a structure…" value={search} onChange={event=>setSearch(event.target.value)}/></label>}
    {!structures.length&&<p className="native-structure-empty">No structures prepared. Open Scans to add supplied labels or review preparation.</p>}
    <div className="native-structure-list">{groups.filter(group=>group.items.length>0).map(group=>{const items=group.items.filter(matches);if(!items.length)return null;const allVisible=group.items.every(s=>visibleLayers[s.id]!==false);return <details className="native-structure-group" key={group.id+(query?'search':'browse')} open={!!query||items.some(item=>item.id===selectedStructure)||!['regions','review'].includes(group.id)}>
      <summary className="native-structure-group-title"><ChevronRight className="native-structure-group-chevron" size={13} aria-hidden="true"/><span>{group.label}</span><small>{group.items.length}</small><button disabled={disabled} aria-label={`${allVisible?'Hide':'Show'} all ${group.noun}`} title={`${allVisible?'Hide':'Show'} all ${group.noun}`} onClick={event=>{event.preventDefault();onVisibilityChange({...visibleLayers,...Object.fromEntries(group.items.map(s=>[s.id,!allVisible]))});}}>{allVisible?<Eye size={14}/>:<EyeOff size={14}/>}</button></summary>
      {items.map(structure=>{
        const appearance=layerAppearance(structure.id,structure.color,settings,structure.displayKind),visible=visibleLayers[structure.id]!==false,needsAlignment=unavailable.includes(structure.id),pending=pendingLayers.includes(structure.id),failure=layerErrors[structure.id];
        // A load or alignment state replaces the quantity; provenance stays on the line.
        const state=needsAlignment?'Needs alignment':pending?'Loading…':failure?'Not loaded':undefined,{name,detail,qualifiers,role}=structure.display;
        const detailId=`${rowId}-detail-${structures.indexOf(structure)}`,roleId=`${rowId}-role-${structures.indexOf(structure)}`;
        return <Fragment key={structure.id}><div data-structure-id={structure.id} data-structure-label={structure.label} className={`native-structure-row${selectedStructure===structure.id?' is-selected':''}${visible?'':' is-hidden'}`}>
          <button className="native-structure-name" disabled={disabled} title={structure.label} aria-label={structure.label} aria-describedby={role?`${roleId} ${detailId}`:detailId} aria-pressed={selectedStructure===structure.id} onClick={()=>select(structure.id)}>
            <i className="native-structure-swatch" style={{background:appearance.color}} aria-hidden="true"/>
            <span className="native-structure-text">
              <span className="native-structure-title">{name||structure.label}{role&&<span className="native-structure-role" id={roleId}> · {role}</span>}</span>
              <span className="native-structure-detail" id={detailId}>{state?<><span className={`native-structure-state${needsAlignment?' needs-alignment':''}`} {...(pending?{role:'status','data-layer-pending':true}:{})}>{state}</span>{qualifiers&&' · '+qualifiers}</>:detail}</span>
            </span>
          </button>
          {failure&&onRetryLayer?<button className="native-structure-retry" disabled={disabled} data-layer-error="true" title={failure} onClick={()=>onRetryLayer(structure.id)}>Retry</button>:<button className="native-structure-focus" disabled={disabled||needsAlignment||!resident.some(row=>row.name===structure.id)&&structure.kind!=='tract'} title={`Center ${structure.label} in linked views`} aria-label={`Focus ${structure.label}`} onClick={()=>{select(structure.id);onFocusStructure(structure.id);}}><Focus size={14}/></button>}
          <button className="native-structure-visibility" disabled={disabled} aria-label={`${visible?'Hide':'Show'} ${structure.label}`} title={visible?'Hide':'Show'} aria-pressed={visible} onClick={()=>onVisibilityChange({...visibleLayers,[structure.id]:!visible})}>{visible?<Eye size={15}/>:<EyeOff size={15}/>}</button>
        </div>{selectedStructure===structure.id&&selectionInspector}</Fragment>;
      })}
    </details>})}</div>
    {query&&!structures.some(matches)&&<p className="native-structure-empty">No matching structures.</p>}
    {selected&&showSelectionControls&&<div className="native-structure-selection">
      <div><strong title={selected.label}>{selected.label}</strong><button aria-pressed={isolated===selected.id} onClick={()=>isolate(selected.id)}>{isolated===selected.id?'Restore others':'Isolate'}</button></div>
      {selected.display.detail&&<p className="native-structure-selection-detail">{selected.display.detail}</p>}
      <label>Opacity<input aria-label={`${selected.label} opacity`} type="range" min="0" max="1" step=".01" value={layerAppearance(selected.id,selected.color,settings,selected.displayKind).opacity} onChange={event=>onSettingsChange({...settings,layerOpacity:{...settings.layerOpacity,[selected.id]:+event.target.value}})}/><output>{Math.round(layerAppearance(selected.id,selected.color,settings,selected.displayKind).opacity*100)}%</output></label>
      <label>Color<input aria-label={`${selected.label} color`} type="color" value={layerAppearance(selected.id,selected.color,settings,selected.displayKind).color} onChange={event=>onSettingsChange(withLayerColor(settings,selected.id,selected.kind,event.target.value))}/></label>
    </div>}
    {showSelectionControls&&<details className="native-structure-details"><summary>Source & display details</summary><p>Visibility and opacity change the display only. Measurements retain all available source labels.</p><p>Tracts keep supplied bundle names and registration. Streamline previews reduce display load; source data remains unchanged. Slice fibers lie within ±0.65 mm of each source plane.</p>{selected&&<p>{selected.kind==='tract'?'Supplied tractography · independent registration accuracy unmeasured.':`${selected.kind.replaceAll('-',' ')} · inspect source evidence for provenance`}{selected.qa?` · ${selected.qa.reviewStatus}`:''}</p>}{selected?.source&&<p className="native-structure-source">{selected.source}</p>}</details>}
  </section>;
}
