import type {Binding, NativeScanState, NativeViewState, VesselScope} from './native-anatomy-data';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';

export type AnatomyPresentation = 'scan'|'segmentation'|'vascular';
/** Only an acquired scalar source can be the initial Scan presentation. */
export function defaultAnatomyPresentation(modality:string):AnatomyPresentation {
  return /segmentation only|analytic signal|generated image signal|diagnostic/i.test(modality)?'segmentation':'scan';
}
export function studyLayerDisplayKind(layer:Binding['document']['layers'][number]):string {
  const provenance=layer.provenance as {model?:string;label_ontology?:{value?:number}}|null;
  const ontology=layer.label_ontology as {value?:number}|undefined;
  return layer.kind==='context'&&provenance?.model==='siam-v2'&&(ontology??provenance.label_ontology)?.value===16?'outer-head-tissue':layer.kind;
}
/** Presentation derivatives only; all selected structures and source masks stay intact. */
export function anatomyPresentation(settings:ViewerWorkspaceSettings,layers:Binding['document']['layers'],mode:AnatomyPresentation,modality:string):ViewerWorkspaceSettings {
  if(mode==='scan'&&defaultAnatomyPresentation(modality)!=='scan')mode='segmentation';
  const scan=mode==='scan',vascular=mode==='vascular',ct=/^\s*CT(?:A)?(?:\b|[-_])/i.test(modality);
  const layerOpacity={...settings.layerOpacity},layerColor={...settings.layerColor};
  for(const layer of layers){
    if(['vessels','vessel','artery','vein','sinus'].includes(layer.kind)){
      layerColor[layer.id]='#ad4730';layerOpacity[layer.id]=scan?.16:vascular?1:.28;
      if(['vein','sinus'].includes(layer.kind)){layerColor[layer.id]='#6188a8';layerOpacity[layer.id]=scan?.08:vascular?.65:.18;}
    }else if(layer.kind==='tumor'){
      layerColor[layer.id]='#65bfd3';layerOpacity[layer.id]=scan?.32:1;
    }else if(['bone','skull'].includes(layer.kind)){
      layerColor[layer.id]='#ece2ca';layerOpacity[layer.id]=scan?.015:vascular?.06:.52;
    }else if(['brain','tissue','soft_tissue'].includes(layer.kind)){
      const brainRim=!vascular&&layer.kind==='brain';
      layerColor[layer.id]=brainRim?'#b8a8b6':'#c8d0d5';layerOpacity[layer.id]=scan?.012:brainRim?.12:vascular?.035:.055;
    }else if(layer.kind==='head-envelope')layerOpacity[layer.id]=scan?.008:.035;
    else if(studyLayerDisplayKind(layer)==='outer-head-tissue')layerOpacity[layer.id]=scan?.035:vascular?.065:.14;
    else if(layer.kind==='dura')layerOpacity[layer.id]=scan?.04:vascular?.06:.16;
    else if(layer.kind==='csf')layerOpacity[layer.id]=scan?.025:vascular?.035:.07;
    else if(layer.kind==='ventricles')layerOpacity[layer.id]=scan?.055:vascular?.08:.26;
    else if(['brain-region','cerebellar-white-matter','cerebellar-cortex','brainstem'].includes(layer.kind))layerOpacity[layer.id]=scan?.035:vascular?.055:.42;
  }
  return {...settings,presentation:mode,layout:'3d-focus',sliceOverlay:'contour',sliceFraming:{fit:'anatomy',zoom:1},contextStyle:scan?'uniform':'shell',smoothSurfaces:scan?(settings.smoothSurfaces??true):true,showPlane:false,
    skullMode:'transparent',layerColor,layerOpacity,
    // Explicit presentation action only. Saved contrast, opacity and transfer
    // choices are otherwise restored verbatim; source signal is never altered.
    volumeRendering:{enabled:true,preset:'auto',density:scan?(ct?.26:.28):ct?(vascular?.14:.075):(vascular?.035:.025),shadows:true},
    clipping:settings.clipping?{...settings.clipping,enabled:false}:undefined,
    clipBox:settings.clipBox?{...settings.clipBox,enabled:false}:undefined,
    obliqueClip:settings.obliqueClip?{...settings.obliqueClip,enabled:false}:undefined};
}

export function studyTimepoint(b:Binding):string {
  return b.document.study_timepoint??b.document.series[0]?.timepoint??'source-session-unknown';
}
export function studyLabel(value:string):string {
  return value==='postoperative'?'Post-op':value==='preoperative'?'Pre-op':value==='source-session-unknown'?'Unspecified':value;
}
export function scanLabel(series:{id:string;modality:string;label?:string}):string {
  return series.label??(/^[a-f\d-]{30,}$/i.test(series.id)?series.modality:series.id.toUpperCase().replaceAll('-',' '));
}
/** Describe selected contrast without denying angiography in another study scan. */
export function sourceEvidenceLabel(source:{modality:string}|undefined,series:ReadonlyArray<{modality:string}>):string {
  const modality=source?.modality??'Source';
  if (/^CTA\b|CT.*angiogra/i.test(modality)) return 'CTA signal + sourced structures';
  if (/MRA|TOF|angiogra/i.test(modality)) return 'MR angiography · bright signal, no CT bone';
  if (/diagnostic/i.test(modality)) return `${modality} · derived map; planning use excluded`;
  const angiography=series.some(s=>/^CTA\b|MRA|TOF|angiogra/i.test(s.modality));
  return `${modality} + estimated structures · ${angiography?'angiography available in Scans':'no acquired angiography in this study'}`;
}
/** Scan contrast/camera stays per scan; selected anatomy belongs to the study. */
export function nextScanState(current:NativeScanState, saved:NativeScanState,linked=true):NativeScanState {
  return {...saved,cursor:linked&&current.cursor?[...current.cursor]:saved.cursor?[...saved.cursor]:null,visible:{...current.visible},settings:{...saved.settings,
    presentation:current.settings.presentation,
    ...(linked&&current.settings.presentation&&current.settings.presentation!=='scan'?{camera:current.settings.camera,volumeRendering:current.settings.volumeRendering}:{}),
    sliceOverlay:current.settings.sliceOverlay,contextStyle:current.settings.contextStyle,
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
