import type {Binding, CranialView, Layer, NativeScanState, NativeViewState, VesselScope} from './native-anatomy-data';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import {isCtAngiography,isCtModality,isMrAngiography} from './viewer/volumeTransfer.ts';

export type AnatomyPresentation = 'scan'|'segmentation'|'vascular';
type PresentationEvidence = {
  visible: Readonly<Record<string, boolean>>;
  series: ReadonlyArray<Pick<Binding['document']['series'][number],'id'|'modality'>>;
};
const vesselKinds=['artery','vein','sinus','vessel','vessels'];
const mrModality=(modality:string)=>defaultAnatomyPresentation(modality)==='scan'&&
  /\b(?:MR|MRI|MRA|TOF|T1(?:CE|W)?|T2W?|FLAIR|SWI|MPRAGE|SPGR)\b/i.test(modality.replaceAll('_',' '));
const angiographicModality=(modality:string)=>/\b(?:CTA|MRA|TOF)\b|\b(?:CT|MR|MRI)[- _]+angiogra/i.test(modality);
export const isStructuralMrModality=(modality:string)=>mrModality(modality)&&!angiographicModality(modality.replaceAll('_',' '));
/** Only an acquired scalar source can be the initial Scan presentation. */
export function defaultAnatomyPresentation(modality:string):AnatomyPresentation {
  return /segmentation only|analytic signal|generated image signal|diagnostic/i.test(modality)?'segmentation':'scan';
}
export function studyLayerDisplayKind(layer:Layer):string {
  const provenance=layer.provenance as {model?:string;label_ontology?:{value?:number}}|null;
  const ontology=(layer.label_ontology??provenance?.label_ontology) as {value?:number}|undefined;
  const model=typeof provenance?.model==='string'?provenance.model.toLowerCase():undefined;
  if(model==='siam-v2'&&layer.kind==='context'&&ontology?.value===16)return 'outer-head-tissue';
  if(layer.kind==='brain-region'){
    if(model==='siam-v2'&&ontology?.value===1)return 'gray-matter';
    if(['supersynth','synthseg','fastsurfer'].includes(model??'')&&[3,42].includes(ontology?.value??layer.label_value??-1))return 'gray-matter';
  }
  return layer.kind;
}

export function corticalLayers(layers:Layer[]):Layer[] {
  return layers.filter(layer=>['gray-matter','cerebral-cortex'].includes(studyLayerDisplayKind(layer)));
}

/** Decide which anatomy can drive the automatic appearance of this scan. */
export function presentationLayers(study:Binding['document'],seriesId:string,displaySources?:ReadonlyArray<{seriesId:string;alignment:string}>):Layer[] {
  if(displaySources){
    // A source-bound estimate may already be visible in the shared scene.
    // Style its surfaces with the rest of the patient, including an imported CT skull.
    const admitted=new Set(displaySources.filter(s=>s.alignment!=='unavailable').map(s=>s.seriesId));
    if(admitted.has(seriesId))return study.layers.filter(layer=>admitted.has(layer.source_series));
  }
  const shared=(id:string)=>id===study.reference_series||['reference','reviewed','source-provided'].includes(study.transforms?.[id]?.status);
  return study.layers.filter(layer=>layer.source_series===seriesId||(shared(seriesId)&&shared(layer.source_series)));
}

export function hasNativeAngiographicVessels(layers:Layer[],modality:string,seriesId?:string):boolean {
  return !!seriesId&&angiographicModality(modality)&&
    layers.some(layer=>vesselKinds.includes(layer.kind)&&layer.source_series===seriesId);
}

export function initialNativeLayers(study:Binding['document'],seriesId:string,independent:boolean){
  const owned=study.layers.filter(layer=>layer.source_series===seriesId);
  const declared=study.display?.initial_layers;
  // The reference frame's list can omit every native layer of a separately acquired scan.
  const initial=independent&&declared?.length&&!owned.some(layer=>declared.includes(layer.id))?undefined:declared;
  const admitted=independent?owned:presentationLayers(study,seriesId);
  const available=admitted.filter(layer=>!initial||initial.includes(layer.id));
  const cortex=corticalLayers(available);
  const visible=Object.fromEntries(study.layers.map(layer=>{
    const selected=initial?initial.includes(layer.id):cortex.includes(layer)||!['brain-region','review-region','head-envelope'].includes(layer.kind);
    return [layer.id,selected&&(!independent||layer.source_series===seriesId)];
  }));
  const option=ctaCoverageDeclarations(study).find(option=>option.series_id===seriesId);
  const combined=ctaCoverageOptions(study,seriesId);
  if(option&&!initial){
    if(combined)applyCtaCoverageVisibility(visible,combined,true);
    else if(option.retired){
      for(const id of option.wider_layers??[])visible[id]=false;
      for(const id of option.focus_layers)if(owned.some(layer=>layer.id===id))visible[id]=true;
    }
  }
  const resolved=resolveCtaCoverageVisibility(study,visible);
  return {available:option?admitted.filter(layer=>resolved[layer.id]):available,cortex,visible:resolved};
}

function ctaCoverageDeclarations(study:Binding['document']){
  const stored=study.display?.cta_arterial_coverage;
  return Array.isArray(stored)?stored.filter(option=>option&&Array.isArray(option.focus_layers)):
    stored&&Array.isArray(stored.focus_layers)?[stored]:[];
}

export function isCtaCoverageLayer(layer:Layer|undefined){
  return (layer?.provenance as {kind?:string}|null)?.kind==='derived_annotation_union';
}

export function ctaCoverageInputs(study:Binding['document'],seriesId:string){
  const series=study.series.find(item=>item.id===seriesId);
  if(!series?.data?.shape||!series.affine||!isCtAngiography(series.modality))return [];
  return study.layers.filter(layer=>{
    const evidence=layer.provenance as {kind?:string;source?:{name?:string;path?:string}}|null;
    return layer.source_series===seriesId&&layer.kind==='artery'&&layer.availability!=='empty_output'&&!layer.edits?.length&&
      ['supplied_annotation','observed_source_annotation'].includes(evidence?.kind??'')&&typeof evidence?.source?.path==='string'&&
      JSON.stringify(layer.data?.shape)===JSON.stringify(series.data?.shape)&&JSON.stringify(layer.affine)===JSON.stringify(series.affine);
  });
}

export function ctaCoverageOptions(study:Binding['document'],seriesId:string){
  const option=ctaCoverageDeclarations(study).find(row=>row.series_id===seriesId&&!row.retired);
  if(!option)return;
  const series=study.series.find(row=>row.id===seriesId);
  if(!series?.original?.sha256||!isCtAngiography(series.modality)||
    !Array.isArray(option.focus_layers)||!option.focus_layers.length||new Set(option.focus_layers).size!==option.focus_layers.length)return;
  const layers=new Map(study.layers.filter(row=>row.source_series===seriesId&&row.kind==='artery').map(layer=>[layer.id,layer]));
  const combined=layers.get(option.layer_id);
  const record=combined?.provenance as {kind?:string;source_scan_sha256?:string;
    input_layers?:Array<{id:string;data_sha256:string;revision:number;label_value:number}>}|undefined;
  if(record?.kind!=='derived_annotation_union'||record.source_scan_sha256!==series.original.sha256||
    combined?.revision!==0||!Array.isArray(record.input_layers))return;
  const inputs=record.input_layers;
  const unchanged=inputs.every(input=>{
    const layer=input&&layers.get(input.id);
    return !!layer&&!layer.edits?.length&&layer.data?.sha256===input.data_sha256&&layer.revision===input.revision&&layer.label_value===input.label_value;
  });
  if(!unchanged)return;
  const ids=inputs.map(input=>input.id);
  const wider=ids.filter(id=>!option.focus_layers.includes(id));
  if(!wider.length||new Set(ids).size!==ids.length||ids.includes(option.layer_id)||!option.focus_layers.every(id=>ids.includes(id))||
    (option.wider_layers&&JSON.stringify(wider)!==JSON.stringify(option.wider_layers)))return;
  return {combinedId:option.layer_id,focusIds:option.focus_layers,widerIds:wider,memberIds:[option.layer_id,...ids]};
}

export function applyCtaCoverageVisibility(visible:Record<string,boolean>, coverage:NonNullable<ReturnType<typeof ctaCoverageOptions>>,combined:boolean){
  for(const id of coverage.memberIds)visible[id]=combined?id===coverage.combinedId:coverage.focusIds.includes(id);
  return visible;
}

export function resolveCtaCoverageVisibility(study:Binding['document'],visible:Record<string,boolean>){
  const known=new Set([...study.layers,...(study.tractography??[])].map(layer=>layer.id));
  const proposed={...visible};
  for(const option of ctaCoverageDeclarations(study)){
    if(!proposed[option.layer_id]||ctaCoverageOptions(study,option.series_id))continue;
    delete proposed[option.layer_id];
    for(const id of option.wider_layers??[])if(known.has(id))proposed[id]=false;
    for(const id of option.focus_layers)if(known.has(id))proposed[id]=true;
  }
  return Object.fromEntries(Object.entries(proposed).filter(([id])=>known.has(id)));
}

function peripheralReviewVessels(layer:Layer):boolean {
  const source=layer.provenance as {method?:string;evidence_use?:string}|null;
  return ['vesselfm-review-posterior-superficial','vesselfm-review-field-edge','vesselfm-review-skull-exterior','vesselfm-review-source-disagreement'].includes(layer.id)&&
    source?.evidence_use==='context_only'&&source.method==='Review display group of complete native TOF components';
}

/** Presentation derivatives only; all selected structures and source masks stay intact. */
export function anatomyPresentation(settings:ViewerWorkspaceSettings,layers:Layer[],mode:AnatomyPresentation,modality:string,seriesId?:string,evidence?:PresentationEvidence):ViewerWorkspaceSettings {
  if(mode==='scan'&&defaultAnatomyPresentation(modality)!=='scan')mode='segmentation';
  const scan=mode==='scan',vascular=mode==='vascular',ct=isCtModality(modality);
  const cranial=mode==='segmentation'&&corticalLayers(layers).length>0;
  const nativeVessels=!scan&&hasNativeAngiographicVessels(layers,modality,seriesId);
  const vesselFocus=vascular||nativeVessels;
  const ctaVessels=nativeVessels&&isCtAngiography(modality);
  const nativeTofVessels=vascular&&nativeVessels&&isMrAngiography(modality);
  const sourceModality=new Map(evidence?.series.map(series=>[series.id,series.modality]));
  const visibleLayers=evidence?layers.filter(layer=>evidence.visible[layer.id]):[];
  const sourcedVesselContext=vascular&&!ct&&mrModality(modality)&&
    visibleLayers.some(layer=>vesselKinds.includes(layer.kind)&&sourceModality.has(layer.source_series)&&!peripheralReviewVessels(layer))&&
    corticalLayers(visibleLayers).some(layer=>{
      const acquisition=sourceModality.get(layer.source_series);
      return !!acquisition&&isStructuralMrModality(acquisition);
    });
  const boneLayers=ct&&vesselFocus?layers.filter(layer=>layer.kind==='bone'||layer.kind==='skull'):[];
  const sameSourceBone=boneLayers.length===1&&seriesId!==undefined&&boneLayers[0].source_series===seriesId;
  const layerOpacity={...settings.layerOpacity},layerColor={...settings.layerColor};
  for(const layer of layers){
    const displayKind=studyLayerDisplayKind(layer);
    if((cranial||vascular)&&['gray-matter','cerebral-cortex','cerebellar-cortex'].includes(displayKind)){
      layerColor[layer.id]=sourcedVesselContext?'#afa6b0':vesselFocus?'#ad9d9e':'#c69891';
      layerOpacity[layer.id]=sourcedVesselContext?.22:vascular?.11:nativeVessels?.18:.66;
    }else if(cranial&&['head-envelope','outer-head-tissue'].includes(displayKind)){
      layerColor[layer.id]='#d3dce5';layerOpacity[layer.id]=nativeVessels?.055:.78;
    }else if(cranial&&['bone','skull','brain','dura','csf'].includes(displayKind)){
      layerOpacity[layer.id]=displayKind==='csf'?.012:.018;
    }else if(vesselFocus&&['bone','skull','brain'].includes(displayKind)){
      const acquisition=sourceModality.get(layer.source_series);
      const estimatedSkull=sourcedVesselContext&&displayKind!=='brain'&&!!acquisition&&mrModality(acquisition)&&
        (layer.provenance as {kind?:string}|null)?.kind==='image_derived_estimate';
      layerColor[layer.id]=displayKind==='brain'?'#a399ad':estimatedSkull?'#8b99a7':ct?'#ded0b6':'#b8af9d';
      layerOpacity[layer.id]=displayKind==='brain'?(sourcedVesselContext?.035:.07):estimatedSkull?.035:ct?.028:.065;
    }else if(vesselKinds.includes(layer.kind)){
      const source=layer.provenance as {kind?:string;publisher_class_id?:number}|null;
      const reviewedEvidence=peripheralReviewVessels(layer);
      const namedArtery=layer.kind==='artery'&&source?.kind==='supplied_annotation'&&
        Number.isInteger(layer.label_value)&&(layer.label_value??0)>0&&source.publisher_class_id===layer.label_value;
      layerColor[layer.id]=(namedArtery||reviewedEvidence)&&/^#[0-9a-f]{6}$/i.test(layer.color)?layer.color:
        layer.kind==='artery'?'#aa352b':layer.kind==='vein'?'#42668f':layer.kind==='sinus'?'#54548b':nativeTofVessels?'#c16d72':'#a74b52';
      layerOpacity[layer.id]=reviewedEvidence?(scan?.08:.32):scan?(['vein','sinus'].includes(layer.kind)?.08:.16):
        cranial&&layer.kind==='vessels'&&!nativeVessels?.09:1;
    }else if(layer.kind==='tumor'){
      layerColor[layer.id]=sourcedVesselContext?'#9388a2':'#65bfd3';
      layerOpacity[layer.id]=sourcedVesselContext?.62:scan?.32:1;
    }else if(['enhancing-tumor','tumor-core','tumor-whole','edema','nonenhancing-tumor','lesion-context'].includes(layer.kind)){
      const annotation=layer.provenance as {kind?:string;method?:string;annotation_value?:number}|null;
      const value=annotation?.kind==='observed_source_annotation'&&annotation.method==='Pinned source BraTS compartment labels'?annotation.annotation_value:null;
      const signal=layer.kind==='edema'||value===2,core=layer.kind==='nonenhancing-tumor'||value===1||layer.kind==='tumor-core';
      layerColor[layer.id]=signal?'#cb9e60':core?'#9571b3':layer.kind==='enhancing-tumor'?'#ce77a5':'#a686ad';
      layerOpacity[layer.id]=scan?.06:vascular?.035:.13;
    }else if(['bone','skull'].includes(layer.kind)){
      layerColor[layer.id]='#ece2ca';layerOpacity[layer.id]=scan?.015:vascular?.06:1;
    }else if(['brain','tissue','soft_tissue'].includes(layer.kind)){
      const brainRim=!vascular&&layer.kind==='brain';
      layerColor[layer.id]=brainRim?'#b8a8b6':'#c8d0d5';layerOpacity[layer.id]=scan?.012:brainRim?1:vascular?.035:.055;
    }else if(layer.kind==='head-envelope')layerOpacity[layer.id]=scan?.008:.035;
    else if(studyLayerDisplayKind(layer)==='outer-head-tissue')layerOpacity[layer.id]=scan?.035:vascular?.065:.14;
    else if(layer.kind==='dura')layerOpacity[layer.id]=scan?.04:vascular?.06:.16;
    else if(layer.kind==='csf')layerOpacity[layer.id]=scan?.025:vascular?.035:.07;
    else if(layer.kind==='ventricles')layerOpacity[layer.id]=scan?.055:vascular?.08:.26;
    else if(['brain-region','cerebellar-white-matter','cerebellar-cortex','brainstem'].includes(layer.kind))layerOpacity[layer.id]=scan?.035:vascular?.055:.42;
  }
  return {...settings,presentation:mode,layout:'3d-focus',sliceOverlay:'contour',sliceFraming:{fit:'anatomy',zoom:1},contextStyle:scan?'uniform':'shell',smoothSurfaces:scan||sourcedVesselContext||nativeTofVessels?(settings.smoothSurfaces??true):true,showPlane:false,
    ...(ctaVessels&&settings.vesselShading===undefined?{vesselShading:'cta-contrast' as const}:{}),
    skullMode:mode==='segmentation'&&!cranial&&!nativeVessels?'exterior':'transparent',sourcedVesselContext:sourcedVesselContext||undefined,layerColor,layerOpacity,
    // Explicit presentation action only. Saved contrast, opacity and transfer
    // choices are otherwise restored verbatim; source signal is never altered.
    volumeRendering:{enabled:!sourcedVesselContext,preset:'auto',density:scan?(ct?.26:.28):ct?(vesselFocus?.12:.075):(nativeVessels?.018:vascular?.035:.025),
      ...(nativeTofVessels?{balanceTofContext:true}:{}),
      shadows:scan||!ct||sameSourceBone,
      ...(settings.volumeRendering?.showCtExteriorSignal!==undefined?{showCtExteriorSignal:settings.volumeRendering.showCtExteriorSignal}:{}),
      ...(settings.volumeRendering?.showCtBoneOverlay!==undefined||scan&&ct
        ?{showCtBoneOverlay:settings.volumeRendering?.showCtBoneOverlay??false}:{})},
    clipping:settings.clipping?{...settings.clipping,enabled:false}:undefined,
    clipBox:settings.clipBox?{...settings.clipBox,enabled:false}:undefined,
    obliqueClip:settings.obliqueClip?{...settings.obliqueClip,enabled:false}:undefined};
}

/** A sparse set of source-owned layers. Only the rendered composition changes. */
export function cranialComposition(view:NativeScanState,layers:Layer[],allIds:string[],mode:CranialView,
  targetId:string|null|undefined,selectedId:string|null):NativeScanState {
  const inspection=mode==='inspection';
  const head=layers.find(layer=>studyLayerDisplayKind(layer)==='outer-head-tissue')??layers.find(layer=>layer.kind==='head-envelope');
  const isCortex=(layer:Layer)=>['gray-matter','cerebral-cortex','cerebellar-cortex'].includes(studyLayerDisplayKind(layer));
  const internal=inspection?layers.find(layer=>layer.id===selectedId&&layer!==head&&layer.id!==targetId&&!isCortex(layer))
    ??layers.find(layer=>layer.kind==='ventricles'):undefined;
  const visible:Record<string,boolean>=Object.fromEntries(allIds.map(id=>[id,false]));
  const layerOpacity={...view.settings.layerOpacity},layerColor={...view.settings.layerColor};
  for(const layer of layers){
    if(layer.id===targetId){
      layerOpacity[layer.id]=1;layerColor[layer.id]='#65bfd3';
    }else if(layer===head){
      layerOpacity[layer.id]=inspection?.65:.23;layerColor[layer.id]='#d3dce5';
    }else if(layer===internal){
      layerOpacity[layer.id]=.88;layerColor[layer.id]=layer.color;
    }else if(isCortex(layer)){
      layerOpacity[layer.id]=inspection?.19:.71;layerColor[layer.id]='#c69891';
    }else continue;
    visible[layer.id]=true;
  }
  return {...view,cranialView:mode,visible,settings:{...view.settings,presentation:'segmentation',contextStyle:'shell',sourcedVesselContext:undefined,
    sliceOverlay:'contour',layerOpacity,layerColor,volumeRendering:{...view.settings.volumeRendering,enabled:false}}};
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
  if (modality==='DWI_B0') return 'Acquired diffusion baseline · native slice inspection';
  if (/^CTA\b|CT.*angiogra/i.test(modality)) return 'CTA signal + sourced structures';
  if (/MRA|TOF|angiogra/i.test(modality)) return 'MR angiography · bright signal, no CT bone';
  if (/diagnostic/i.test(modality)) return `${modality} · derived map; planning use excluded`;
  const angiography=series.some(s=>/^CTA\b|MRA|TOF|angiogra/i.test(s.modality));
  return `${modality} + estimated structures · ${angiography?'angiography available in Scans':'no acquired angiography in this study'}`;
}
/** Scan contrast/camera stays per scan; selected anatomy belongs to the study. */
export function nextScanState(current:NativeScanState, saved:NativeScanState,linked=true):NativeScanState {
  return {...saved,...(linked?{cranialView:current.cranialView}:{}),cursor:linked&&current.cursor?[...current.cursor]:saved.cursor?[...saved.cursor]:null,visible:{...current.visible},settings:{...saved.settings,
    ...(linked?{presentation:current.settings.presentation}:{}),
    ...(linked?{sourcedVesselContext:current.settings.sourcedVesselContext}:{}),
    ...(linked&&current.settings.presentation&&current.settings.presentation!=='scan'?{camera:current.settings.camera}:{}),
    sliceOverlay:current.settings.sliceOverlay,contextStyle:current.settings.contextStyle,
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
export function vesselScopeForView(view?:NativeViewState|null, useTofExterior=false):VesselScope {
  if(!view&&useTofExterior)return 'exterior-air';
  return view?.vesselDisplayVersion===2 ? view.vesselScope??'all' : 'all';
}
