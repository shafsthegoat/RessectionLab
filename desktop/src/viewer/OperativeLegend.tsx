import {useMemo} from 'react';
import type {SurgicalView} from '../surgical-types';
import type {Point3} from './coordinates';
import {vesselSourceBoundaryColor} from './anatomySurfaceMaterial';
import {operativeLegend,originalLabelLocation} from './operativeLegendData';
import type {ViewerWorkspaceSettings} from './workspaceSettings';

/** Layer keys and contour comparison. In the microscope view they fold into
 * one "Layers" disclosure so the operative image keeps the space; a picked
 * layer's inspection stays visible outside the fold. */
export function OperativeLegend({scene,settings,cursor,picked,onPick,onSmooth,visible,sourceVolumeBoundary,collapsible,unknownVesselColor}:{
  scene:SurgicalView['scene'];settings:ViewerWorkspaceSettings;cursor:Point3;
  picked:string|null;onPick:(name:string|null)=>void;onSmooth:(smooth:boolean)=>void;visible:Record<string,boolean>;
  sourceVolumeBoundary?:boolean;collapsible?:boolean;
  /** Source-surface vessel tubes without an artery or vein class: their drawn colour (the kind stays UNKNOWN). */
  unknownVesselColor?:string;
}){
  const rows=useMemo(()=>operativeLegend(scene,settings),[scene.sceneId,settings.layerColor,settings.operativeView,settings.microscope?.tissueStyle]);
  const selected=rows.find(r=>r.name===picked);
  const primary=rows.filter(r=>r.target||r.isVessel||r===selected);
  const smooth=settings.smoothSurfaces!==false;
  const point=selected?originalLabelLocation(selected,cursor):null;
  const keys=<>
    <div className="engine-anatomy-key-keys">{primary.map(row=><button key={row.name} aria-pressed={row===selected}
      title={row.target?'Selected treatment target':row.method??row.title} data-region={row.name}
      data-source-color={row.sourceColor} data-field-color={row.fieldColor}
      onClick={()=>onPick(row===selected?null:row.name)}>
      <i aria-hidden="true" style={{background:row.fieldColor}}/>{row.title}{row.fieldColor!==row.sourceColor&&<small className="engine-source-swatch"><i aria-hidden="true" style={{background:row.sourceColor}}/>slices</small>}{row.target&&!row.title.toLowerCase().includes('target')?' · target':''}{visible[row.name]===false?' · hidden':''}
    </button>)}</div>
    <div role="group" aria-label="Compare surface contours" className="operative-contour-controls">
      <button aria-pressed={!smooth} onClick={()=>onSmooth(false)}>Source edges</button>
      <button aria-pressed={smooth} onClick={()=>onSmooth(true)}>Conditioned edges</button>
    </div>
    {unknownVesselColor&&<span className="operative-source-boundary-key" title="Vessel model tubes the head model does not class as artery or vein">
      <i aria-hidden="true" style={{background:unknownVesselColor}}/>Vessel · kind UNKNOWN
    </span>}
    {sourceVolumeBoundary&&<span className="operative-source-boundary-key">
      <i aria-hidden="true" style={{background:vesselSourceBoundaryColor}}/>Source volume boundary
    </span>}
  </>;
  return <div className="engine-anatomy-key" aria-label="Operative anatomy legend" onDoubleClick={e=>e.stopPropagation()}>
    {collapsible?<details className="engine-anatomy-fold">
      <summary title="Show layer keys and contour comparison">Layers{selected?` · ${selected.title}`:''}</summary>
      {keys}
    </details>:keys}
    {selected&&<div className="operative-layer-inspection" role="status">
      <strong>{selected.layerId??selected.name}</strong>
      {selected.method&&<span>{selected.method}{selected.review?' · '+selected.review:''}</span>}
      {selected.series&&<span>{selected.series}{selected.sampling?' · '+selected.sampling:''}</span>}
      <span>Cursor RAS {cursor.map(v=>v.toFixed(1)).join(', ')} mm</span>
      {point&&<span>{point.within?'Nearest source voxel '+point.point.join(', '):'Outside the registered source label grid'}</span>}
      {selected.isVessel&&selected.model&&<span>Positive estimates only. Unlabelled vessels remain unassessed.</span>}
      {selected.sourceColor!==selected.fieldColor&&<span><i aria-hidden="true" style={{background:selected.sourceColor}}/>Source slices</span>}
    </div>}
  </div>;
}
