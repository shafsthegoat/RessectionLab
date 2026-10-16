import type {Binding, NativeScanState, NativeViewState, VesselScope} from './native-anatomy-data';

export function studyTimepoint(b:Binding):string {
  return b.document.study_timepoint??b.document.series[0]?.timepoint??'preoperative';
}
export function studyLabel(value:string):string {
  return value==='postoperative'?'Post-op':value==='preoperative'?'Pre-op':value;
}
export function scanLabel(series:{id:string;modality:string;label?:string}):string {
  return series.label??(/^[a-f\d-]{30,}$/i.test(series.id)?series.modality:series.id.toUpperCase().replaceAll('-',' '));
}
/** Scan contrast/camera stays per scan; selected anatomy belongs to the study. */
export function nextScanState(current:NativeScanState, saved:NativeScanState):NativeScanState {
  return {...saved,visible:{...current.visible},settings:{...saved.settings,
    skullMode:current.settings.skullMode,clipping:current.settings.clipping,clipBox:current.settings.clipBox,obliqueClip:current.settings.obliqueClip,
    layerColor:current.settings.layerColor,layerOpacity:current.settings.layerOpacity}};
}

/** Legacy transparency was an implicit default. Migrate only absent mode. */
export function migrateSkullView(view:NativeScanState,layers:Binding['document']['layers']):NativeScanState {
  if(view.settings.skullMode)return view;
  return {...view,settings:{...view.settings,skullMode:'exterior',
    layerOpacity:{...view.settings.layerOpacity,...Object.fromEntries(layers.filter(l=>['bone','skull'].includes(l.kind)).map(l=>[l.id,1]))},
    clipping:view.settings.clipping?{...view.settings.clipping,enabled:false}:undefined,
    clipBox:view.settings.clipBox?{...view.settings.clipBox,enabled:false}:undefined,
    obliqueClip:view.settings.obliqueClip?{...view.settings.obliqueClip,enabled:false}:undefined}};
}

/** Legacy scopes were implicit filters; migrate once without changing anatomy. */
export function vesselScopeForView(view?:NativeViewState|null):VesselScope {
  return view?.vesselDisplayVersion===2 ? view.vesselScope??'all' : 'all';
}
