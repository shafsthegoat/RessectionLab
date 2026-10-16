import { layerAppearance } from './viewer/workspaceSettings';
import type { ViewerWorkspaceSettings } from './viewer/workspaceSettings';
import './AnatomyPanel.css';

function missingStructureLabel(name:string):string {
  const labels:Record<string,string>={tissue_classes:'Brain tissue types',fresh_vessel_inference:'Vessels on this scan'};
  const text=Object.hasOwn(labels,name)?labels[name]:name.replaceAll('_',' ');
  return text.charAt(0).toUpperCase()+text.slice(1);
}

export interface AnatomyPanelProps {
  layers: Array<{name:string;label?:string;color:string;volumeMm3?:number;provenance?:string}>;
  visibleLayers: Record<string,boolean>;
  onVisibilityChange: (layers:Record<string,boolean>)=>void;
  settings: ViewerWorkspaceSettings;
  onSettingsChange: (settings:ViewerWorkspaceSettings)=>void;
  missingStructures?: string[];
  provenanceLabel?: string;
}
export function AnatomyPanel(props: AnatomyPanelProps) {
  const visible=props.layers.filter(layer=>props.visibleLayers[layer.name]!==false).length;
  return <section className="anatomy-layer-panel" aria-label="Anatomical layers">
    <div className="anatomy-panel-heading"><h2>Anatomy <span>{visible}/{props.layers.length}</span></h2>
      {!!props.layers.length && <div><button onClick={()=>props.onVisibilityChange(Object.fromEntries(props.layers.map(layer=>[layer.name,true])))}>Show all</button>
        <button onClick={()=>props.onVisibilityChange(Object.fromEntries(props.layers.map(layer=>[layer.name,false])))}>Hide all</button></div>}
    </div>
    {props.layers.map(layer=>{
      const appearance=layerAppearance(layer.name,layer.color,props.settings);
      const name=layer.label??layer.name.replaceAll('_',' ');
      const provenance=layer.provenance??props.provenanceLabel??'Provenance not supplied';
      return <div className="anatomy-layer" key={layer.name} data-visible={props.visibleLayers[layer.name]!==false}>
        <div className="anatomy-layer-main">
          <input type="checkbox" aria-label={`Show ${name}`} checked={props.visibleLayers[layer.name]!==false}
            onChange={event=>props.onVisibilityChange({...props.visibleLayers,[layer.name]:event.target.checked})}/>
          <input className="anatomy-color" type="color" aria-label={`${name} color`} value={appearance.color}
            onChange={event=>props.onSettingsChange({...props.settings,layerColor:{...props.settings.layerColor,[layer.name]:event.target.value}})}/>
          <div className="anatomy-layer-identity"><strong>{name}</strong><span className="anatomy-layer-origin" title={provenance}>{provenance}</span></div>
          {Number.isFinite(layer.volumeMm3)&&<span>{(layer.volumeMm3!/1000).toFixed(1)} mL</span>}
        </div>
        <label className="anatomy-opacity">Opacity
          <input type="range" min="0" max="1" step=".01" aria-label={`${name} opacity`} value={appearance.opacity}
            onChange={event=>props.onSettingsChange({...props.settings,layerOpacity:{...props.settings.layerOpacity,[layer.name]:Number(event.target.value)}})}/>
          <output>{Math.round(appearance.opacity*100)}%</output>
        </label>
        <details className="anatomy-provenance"><summary>Provenance details</summary><p>{provenance}</p></details>
      </div>;
    })}
    {!props.layers.length&&<p className="anatomy-empty">No segmentation layers are attached to this image. Original imaging remains available.</p>}
    {!!props.missingStructures?.length&&<div className="anatomy-missing"><h3>Missing anatomy</h3>{props.missingStructures.map(name=><span key={name}>{missingStructureLabel(name)}<b>Not available</b></span>)}</div>}
  </section>;
}
