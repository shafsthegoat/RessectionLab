import type {Binding, NativeScanState, NativeViewState, VesselScope} from './native-anatomy-data';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';

export type AnatomyPresentation = 'segmentation'|'vascular';
/** Presentation derivatives only; all selected structures and source masks stay intact. */
export function anatomyPresentation(settings:ViewerWorkspaceSettings,layers:Binding['document']['layers'],mode:AnatomyPresentation,modality:string):ViewerWorkspaceSettings {
  const vascular=mode==='vascular',ct=/^\s*CT(?:A)?(?:\b|[-_])/i.test(modality);
  const layerOpacity={...settings.layerOpacity},layerColor={...settings.layerColor};
  for(const layer of layers){
    if(['vessels','vessel','artery','vein','sinus'].includes(layer.kind)){
      layerColor[layer.id]='#ad4730';layerOpacity[layer.id]=vascular?1:.22;
      if(['vein','sinus'].includes(layer.kind)){layerColor[layer.id]='#6188a8';layerOpacity[layer.id]=vascular?.65:.18;}
    }else if(layer.kind==='tumor'){
      layerColor[layer.id]='#42bbdf';layerOpacity[layer.id]=.96;
    }else if(['bone','skull'].includes(layer.kind)){
      layerColor[layer.id]='#ece2ca';layerOpacity[layer.id]=.035;
    }else if(['brain','tissue','soft_tissue'].includes(layer.kind)){
      const brainRim=!vascular&&layer.kind==='brain';
      layerColor[layer.id]=brainRim?'#b64bb9':'#c8d0d5';layerOpacity[layer.id]=brainRim?.12:vascular?.035:.055;
    }else if(layer.kind==='head-envelope')layerOpacity[layer.id]=.035;
  }
  return {...settings,presentation:mode,layout:'3d-focus',sliceOverlay:'contour',sliceFraming:{fit:'anatomy',zoom:1},contextStyle:'shell',smoothSurfaces:true,showPlane:false,
    skullMode:'transparent',layerColor,layerOpacity,
    volumeRendering:{enabled:true,preset:'auto',density:ct?(vascular?.14:.075):(vascular?.035:.025),shadows:true},
    clipping:settings.clipping?{...settings.clipping,enabled:false}:undefined,
    clipBox:settings.clipBox?{...settings.clipBox,enabled:false}:undefined,
    obliqueClip:settings.obliqueClip?{...settings.obliqueClip,enabled:false}:undefined};
}

export function studyTimepoint(b:Binding):string {
  return b.document.study_timepoint??b.document.series[0]?.timepoint??'preoperative';
}
export function studyLabel(value:string):string {
  return value==='postoperative'?'Post-op':value==='preoperative'?'Pre-op':value==='source-session-unknown'?'Unspecified':value;
}
export function scanLabel(series:{id:string;modality:string;label?:string}):string {
  return series.label??(/^[a-f\d-]{30,}$/i.test(series.id)?series.modality:series.id.toUpperCase().replaceAll('-',' '));
}
/** Scan contrast/camera stays per scan; selected anatomy belongs to the study. */
export function nextScanState(current:NativeScanState, saved:NativeScanState):NativeScanState {
  return {...saved,visible:{...current.visible},settings:{...saved.settings,
    presentation:current.settings.presentation,sliceOverlay:current.settings.sliceOverlay,contextStyle:current.settings.contextStyle,
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
