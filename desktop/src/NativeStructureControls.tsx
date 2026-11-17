import {useRef,useState} from 'react';
import {Eye,EyeOff,Focus,RotateCcw} from 'lucide-react';
import type {ViewerVolume} from './viewer/contracts';
import {tractColor} from './viewer/tractDisplay';
import {layerAppearance} from './viewer/workspaceSettings';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import './native-structure-controls.css';

export interface NativeStructureControlsProps {
  volume:ViewerVolume|null;
  visibleLayers:Record<string,boolean>;
  onVisibilityChange:(visible:Record<string,boolean>)=>void;
  settings:ViewerWorkspaceSettings;
  onSettingsChange:(settings:ViewerWorkspaceSettings)=>void;
  selectedStructure:string|null;
  onSelectStructure:(id:string|null)=>void;
  onFocusStructure:(id:string)=>void;
}

export function NativeStructureControls({volume,visibleLayers,onVisibilityChange,settings,onSettingsChange,selectedStructure,onSelectStructure,onFocusStructure}:NativeStructureControlsProps){
  const previousVisibility=useRef<Record<string,boolean>|null>(null);
  const [isolated,setIsolated]=useState<string|null>(null);
  if(!volume)return null;
  const structures=[...volume.compartments.map(layer=>({id:layer.name,label:layer.label??layer.name.replaceAll('_',' '),kind:layer.kind??'label',color:layer.color,source:layer.sourceIdentity,qa:layer.qa})),...(volume.tractography??[]).map((tract,index)=>({id:tract.id,label:tract.label,kind:'tract',color:tractColor(index),source:undefined,qa:undefined}))];
  const selected=structures.find(structure=>structure.id===selectedStructure);
  const select=(id:string)=>{if(visibleLayers[id]===false)onVisibilityChange({...visibleLayers,[id]:true});onSelectStructure(id);};
  const isolate=(id:string)=>{
    if(isolated===id&&previousVisibility.current){onVisibilityChange(previousVisibility.current);previousVisibility.current=null;setIsolated(null);return;}
    if(!previousVisibility.current)previousVisibility.current={...visibleLayers};
    onVisibilityChange(Object.fromEntries(structures.map(s=>[s.id,s.id===id])));setIsolated(id);onSelectStructure(id);
  };
  const reset=()=>{onVisibilityChange(Object.fromEntries(structures.map(s=>[s.id,true])));onSettingsChange({...settings,layerOpacity:{},layerColor:{}});onSelectStructure(null);previousVisibility.current=null;setIsolated(null);};
  const groups=[{id:'anatomy',label:'Anatomical context',items:structures.filter(s=>s.kind!=='tract'&&!['vessel','vessels','artery','vein','sinus'].includes(s.kind))},{id:'vessels',label:'Vessel estimates',items:structures.filter(s=>['vessel','vessels','artery','vein','sinus'].includes(s.kind))},{id:'tracts',label:'Supplied tract bundles',items:structures.filter(s=>s.kind==='tract')}];
  return <section className="native-structures" aria-label="Anatomical structures">
    <div className="native-structure-heading"><strong>Structures</strong><button title="Restore all structure visibility, opacity and colors" aria-label="Reset structure display" onClick={reset}><RotateCcw size={12}/>Reset</button></div>
    <div className="native-structure-list">{groups.map(group=><div className="native-structure-group" key={group.id}>
      <div className="native-structure-group-title">{group.label}<span>{group.items.length}</span></div>
      {group.items.length===0?<p className="native-structure-empty">{group.id==='tracts'?'No supplied tract bundles loaded.':group.id==='vessels'?'No labeled vessel estimate loaded.':'No context labels loaded.'}</p>:group.items.map(structure=>{
        const appearance=layerAppearance(structure.id,structure.color,settings,structure.kind),visible=visibleLayers[structure.id]!==false;
        return <div key={structure.id} className={`native-structure-row${selectedStructure===structure.id?' is-selected':''}${visible?'':' is-hidden'}`}>
          <button className="native-structure-visibility" aria-label={`${visible?'Hide':'Show'} ${structure.label}`} aria-pressed={visible} onClick={()=>onVisibilityChange({...visibleLayers,[structure.id]:!visible})}>{visible?<Eye size={14}/>:<EyeOff size={14}/>}</button>
          <button className="native-structure-name" title={structure.label} aria-pressed={selectedStructure===structure.id} onClick={()=>select(structure.id)}><i style={{background:appearance.color}}/><span>{structure.label}</span></button>
          <button className="native-structure-focus" title={`Center ${structure.label} in linked views`} aria-label={`Focus ${structure.label}`} onClick={()=>{select(structure.id);onFocusStructure(structure.id);}}><Focus size={13}/></button>
        </div>;
      })}
    </div>)}</div>
    {selected&&<div className="native-structure-selection">
      <div><strong>{selected.label}</strong><button aria-pressed={isolated===selected.id} onClick={()=>isolate(selected.id)}>{isolated===selected.id?'Restore others':'Isolate'}</button></div>
      <label>Opacity<input aria-label={`${selected.label} opacity`} type="range" min="0" max="1" step=".01" value={layerAppearance(selected.id,selected.color,settings,selected.kind).opacity} onChange={event=>onSettingsChange({...settings,layerOpacity:{...settings.layerOpacity,[selected.id]:+event.target.value}})}/><output>{Math.round(layerAppearance(selected.id,selected.color,settings,selected.kind).opacity*100)}%</output></label>
      <label>Color<input aria-label={`${selected.label} color`} type="color" value={layerAppearance(selected.id,selected.color,settings,selected.kind).color} onChange={event=>onSettingsChange({...settings,layerColor:{...settings.layerColor,[selected.id]:event.target.value}})}/></label>
    </div>}
    <details className="native-structure-details"><summary>Source & display details</summary><p>Visibility and opacity change the display only. Measurements retain all available source labels.</p><p>Tracts keep supplied bundle names and registration. Streamline previews reduce display load; source data remains unchanged. Slice fibers lie within ±0.65 mm of each source plane.</p>{selected&&<p>{selected.kind==='tract'?'Supplied tractography · independent registration accuracy unmeasured.':selected.kind==='tumor'?'Target layer · inspect source details for synthetic demonstration provenance.':`${selected.kind.replaceAll('-',' ')} · source label`}{selected.qa?` · ${selected.qa.reviewStatus}`:''}</p>}{selected?.source&&<p className="native-structure-source">{selected.source}</p>}</details>
  </section>;
}
