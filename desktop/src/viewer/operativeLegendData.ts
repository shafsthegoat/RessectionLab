import type {SurgicalView} from '../surgical-types';
import {inverseAffine, transformPoint} from './coordinates.ts';
import type {Point3} from './coordinates.ts';
import {layerAppearance} from './workspaceSettings.ts';
import type {ViewerWorkspaceSettings} from './workspaceSettings.ts';
import {operativeSurfaceStyle} from './operativeMaterial.ts';

function object(value:unknown):Record<string,unknown> {
  return value && typeof value==='object' && !Array.isArray(value) ? value as Record<string,unknown> : {};
}
const text=(v:unknown)=>typeof v==='string'&&v.length ? v : undefined;
const words=(v:string)=>v.replaceAll('_',' ');

export function operativeLegend(scene:SurgicalView['scene'],settings:ViewerWorkspaceSettings) {
  const source=scene.source?.adapter;
  const records=object(source?.sourceLayers);
  const manifest=object(source?.sourceManifest);
  const layers=Array.isArray(manifest.layers)?manifest.layers.map(object):[];
  const acquisitions=Array.isArray(manifest.series)?manifest.series.map(object):[];
  return scene.regions.map(region=>{
    const record=object(records[region.name]);
    const layer=layers.find(row=>row.id===record.layerId);
    const acquisition=acquisitions.find(row=>row.id===layer?.source_series);
    const p=object(record.provenance);
    const model=text(p.model), supplied=p.kind==='observed_source_annotation';
    const sourceLabel=text(layer?.label)??text(record.layerId)??region.name;
    const isVessel=['vessel','vessels','artery','vein','sinus'].includes(region.material);
    const title=region.target
      ? (supplied?'Supplied ':'')+words(sourceLabel)
      : isVessel && model ? words(region.material)+' estimate' : words(sourceLabel);
    const color=layerAppearance(region.name,region.color,settings,region.material).color;
    return {name:region.name,title,target:region.target,isVessel,layerId:text(record.layerId),
      model,method:text(p.method)??model,review:text(record.reviewStatus)??text(layer?.review_status),
      series:text(acquisition?.modality)??text(layer?.source_series),sampling:text(record.sampling),
      sourceColor:color,fieldColor:operativeSurfaceStyle(region.material,settings)?.color??color,
      registeredAffine:record.registeredAffine,shape:object(record.sourceArray).shape};
  });
}

export function originalLabelLocation(row:ReturnType<typeof operativeLegend>[number],ras:Point3) {
  const a=row.registeredAffine, shape=row.shape;
  if(!Array.isArray(a)||a.length!==4||!a.every(r=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite))||
    !Array.isArray(shape)||shape.length!==3||!shape.every(n=>Number.isSafeInteger(n)&&n>0))return null;
  try {
    const point=transformPoint(inverseAffine(a),ras).map(Math.round);
    if(!point.every(Number.isFinite))return null;
    return {point,within:point.every((n,i)=>n>=0&&n<shape[i])};
  } catch { return null; }
}
