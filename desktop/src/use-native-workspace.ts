import {useNativeApproaches} from './use-native-approaches';
import {nextScanState,migrateSkullView,vesselScopeForView,anatomyPresentation,defaultAnatomyPresentation} from './native-study-state';
import type {AnatomyPresentation} from './native-study-state';
import {drainLayerRequests} from './native-layer-queue';
import {mergeResidentLayers,prepareNativeLabels} from './native-layer-residency';
import {useCallback,useEffect,useRef,useState} from 'react';
import type {BridgeEvent,ResectionApi} from './types';
import {hydrateAnatomy} from './native-anatomy-data';
import type {Binding,Draft,NativeAnatomyApi,NativeScanState,NativeViewState,ComponentReview,Args,VesselScope,RetiredClipping} from './native-anatomy-data';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {rasAffine,transformPoint,volumeBounds} from './viewer/coordinates';
import {newClippingPosition,usesPatientContext,viewerCutaway} from './viewer/workspaceSettings';
import type {ViewerWorkspaceSettings,ViewerCameraState,ViewerCutaway} from './viewer/workspaceSettings';
export const nativeDefaults=():ViewerWorkspaceSettings=>({layout:'3d-focus',showPlane:false,sliceOverlay:'contour',layerOpacity:{brain:.06,tumor:1,vessels:1,bone:1,'scalp-envelope':.02,__fragment_review:1},volumeRendering:{enabled:true,density:.18}});
const retiredClippingMessage='Alignment updated · A cutaway outside the new field was cleared.';

function withoutRetiredClipping<T extends ViewerCutaway>(settings:T,cutaway:ViewerCutaway):T {
  let next=settings;
  for(const key of ['clipping','clipBox','obliqueClip'] as const){
    if(cutaway[key]&&JSON.stringify(settings[key])===JSON.stringify(cutaway[key])){
      if(next===settings)next={...settings};
      delete next[key];
    }
  }
  return next;
}

export function useNativeWorkspace(runtime:ResectionApi|null){
  const readCamera=useRef<(()=>ViewerCameraState|null)|null>(null);
  const api=runtime as (ResectionApi&NativeAnatomyApi)|null;
  const [binding,setBinding]=useState<Binding|null>(null),[volume,setVolume]=useState<ViewerVolume|null>(null);
  const [selectedStructure,setSelectedStructure]=useState<string|null>(null),[structureFocusRequest,setStructureFocusRequest]=useState(0);
  const [series,setSeries]=useState(''),[states,setStates]=useState<Record<string,NativeScanState>>({});
  const [patientCutaway,setPatientCutaway]=useState<ViewerCutaway>({});
  const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[registration,setRegistration]=useState('');
  const [scanFrameReviewed,setScanFrameReviewed]=useState(false);
  const patientFrameReady=!!volume&&scanFrameReviewed;
  const planning=useNativeApproaches(binding,api,!!volume&&!busy);
  const [progress,setProgress]=useState<BridgeEvent['progress']>();
  const [structureFocusPoint,setStructureFocusPoint]=useState<Point3|null>(null);
  const [vesselScope,setVesselScope]=useState<VesselScope>('all');
  const scopeRef=useRef<VesselScope>('all');
  const [unavailable,setUnavailable]=useState<string[]>([]),[draft,setDraft]=useState<Draft>(null),[original,setOriginal]=useState(false);
  const [editLayer,setEditLayer]=useState<string|null>(null),[tool,setTool]=useState('navigate'),[radius,setRadius]=useState(2),[threshold,setThreshold]=useState(.5),[author,setAuthor]=useState('');
  const [review,setReview]=useState<ComponentReview|null>(null),[components,setComponents]=useState<number[]>([]);
  const [pendingLayers,setPendingLayers]=useState<string[]>([]),[layerErrors,setLayerErrors]=useState<Record<string,string>>({});
  const residentVolume=useRef<ViewerVolume|null>(null),layerTask=useRef<Promise<void>|null>(null);
  const packing=useRef<AbortController|null>(null),failedLayers=useRef<Record<string,string>>({}),unavailableRef=useRef<string[]>([]);
  const contextCache=useRef<ViewerVolume['patientContext']>(undefined);
  const sourceImage=useRef<{key:string;data:Float32Array}|null>(null);
  const generation=useRef(0),active=useRef(false),requestId=useRef<string|null>(null),stroke=useRef<Point3[]>([]);
  const preparationRequest=useRef<string|null>(null);
  const live=useRef({binding,series,states});live.current={binding,series,states};
  useEffect(()=>()=>{generation.current++;packing.current?.abort()},[]);
  useEffect(()=>api?.onEvent(e=>{
    if(!e.op?.startsWith('anatomy')||(preparationRequest.current&&e.id!==preparationRequest.current))return;
    if(e.event==='started'){
      requestId.current=e.id;
      setProgress(undefined);
    }
    if(e.id!==requestId.current)return;
    if(e.event==='progress'&&e.progress){
      setProgress(e.progress);
      setMessage(e.progress.message);
    }
    if(['result','error','cancelled'].includes(e.event))requestId.current=null;
  }),[api]);
  const args=(b=live.current.binding)=>({collectionId:b!.collectionId,snapshotId:b!.document.snapshot_id});
  const initial=(b:Binding,s:string):NativeScanState=>{
    const modality=b.document.series.find(scan=>scan.id===s)?.modality??'MRI';
    const mode=b.document.series.length>1?'segmentation':defaultAnatomyPresentation(modality);
    return {cursor:s===b.document.reference_series?b.document.display?.cursor_ras_mm??null:null,settings:anatomyPresentation(nativeDefaults(),b.document.layers,mode,modality),visible:Object.fromEntries(b.document.layers.map(l=>[l.id,b.document.display?.initial_layers?b.document.display.initial_layers.includes(l.id):!['brain-region','review-region','head-envelope'].includes(l.kind)]))};
  };
  const scanState=states[series]??{cursor:null,visible:{},settings:nativeDefaults()};
  const patientField=volume?.patientContext?.field;
  const patientCutawayActive=usesPatientContext(scanState.settings,!!patientField);
  const state=patientCutawayActive
    ? {...scanState,settings:skullDisplay({...scanState.settings,...viewerCutaway(patientCutaway)})}
    : scanState;
  function skullDisplay(settings:ViewerWorkspaceSettings):ViewerWorkspaceSettings {
    const clipped=settings.clipping?.enabled||settings.clipBox?.enabled||settings.obliqueClip?.enabled;
    const transparent=binding?.document.layers.some(l=>['bone','skull'].includes(l.kind)&&(settings.layerOpacity?.[l.id]??1)<1);
    return {...settings,skullMode:clipped?'cutaway':transparent?'transparent':'exterior'};
  }
  function updateSettings(settings:ViewerWorkspaceSettings){
    if(patientCutawayActive){
      setPatientCutaway(viewerCutaway(settings));
      settings=skullDisplay({...settings,...viewerCutaway(scanState.settings)});
    }
    updateState({settings});
  }
  function retireClipping(retired:RetiredClipping[]|undefined,seriesId:string,view:NativeScanState):NativeScanState {
    if(!retired?.length)return view;
    const clearSource=(source:string,previous:NativeScanState):NativeScanState=>{
      let settings=previous.settings;
      for(const item of retired){
        if(item.seriesId===source)settings=withoutRetiredClipping(settings,item.cutaway);
      }
      return settings===previous.settings?previous:{...previous,settings:skullDisplay(settings)};
    };
    for(const item of retired){
      if(item.seriesId===null)setPatientCutaway(settings=>withoutRetiredClipping(settings,item.cutaway));
    }
    setStates(all=>Object.fromEntries(Object.entries(all).map(([id,previous])=>[id,clearSource(id,previous)])));
    return clearSource(seriesId,view);
  }
  function updateState(next:Partial<NativeScanState>){
    const s=live.current.series,current=live.current.states[s]??scanState;
    const views={...live.current.states,[s]:{...current,...next}};
    live.current={...live.current,states:views};setStates(views);
  }
  function installVolume(next:ViewerVolume|null){residentVolume.current=next;setVolume(next);}
  function resetLayers(){packing.current?.abort();packing.current=null;failedLayers.current={};setLayerErrors({});setPendingLayers([]);unavailableRef.current=[];}
  async function prepared(next:ViewerVolume){const controller=new AbortController();packing.current=controller;try{return await prepareNativeLabels(next,controller.signal)}finally{if(packing.current===controller)packing.current=null;}}
  const run=async(job:()=>Promise<void>)=>{if(active.current)return;active.current=true;setBusy(true);try{await layerTask.current;await job()}catch(e){setMessage(e instanceof Error?e.message:String(e))}finally{active.current=false;setBusy(false)}};
  async function loadView(b:Binding,s:string,view:NativeScanState,raw=false,extra:Args={},legacyCutaway=false){
    if(!api)return;setOriginal(raw);const seq=++generation.current;
    resetLayers();installVolume(null);setUnavailable([]);setMessage('Preparing anatomy…');
    const ids=Object.keys(view.visible).filter(id=>view.visible[id]);
    try{
      const result=await api.anatomyView({...args(b),seriesId:s,layers:ids,original:raw,patientModel:b.document.series.length>1,previewAlignment:true,vesselScope:scopeRef.current,...extra});
      if(seq!==generation.current)return;
      const imageKey=b.collectionId+"/"+s+"/"+result.mri.sha256;
      let hydrated=await hydrateAnatomy(result,api,sourceImage.current?.key===imageKey?sourceImage.current.data:undefined,{patientContext:contextCache.current});if(seq!==generation.current)return;
      contextCache.current=hydrated.patientContext;
      hydrated.displayIdentity=JSON.stringify([b.collectionId,s,result.mri.sha256,b.document.snapshot_id,raw,seq]);
      hydrated=await prepared(hydrated);if(seq!==generation.current)return;
      sourceImage.current={key:imageKey,data:hydrated.mri};
      const independent=result.registrationStatus==='independent source frame';
      const next={...view,settings:independent?{...view.settings,presentation:defaultAnatomyPresentation(result.modality)}:view.settings,cursor:view.cursor??transformPoint(hydrated.affine,hydrated.shape.map(n=>(n-1)/2) as Point3)};
      if(legacyCutaway&&usesPatientContext(next.settings,!!hydrated.patientContext?.field)){
        setPatientCutaway(viewerCutaway(next.settings));
        next.settings={...next.settings,...viewerCutaway({})};
      }
      setStates(all=>({...all,[s]:next}));installVolume(hydrated);setRegistration(result.registrationStatus);setScanFrameReviewed(result.patientFrameReady??result.registrationStatus==='patient frame');setUnavailable(result.unavailableLayers);unavailableRef.current=result.unavailableLayers;
      setMessage(result.unavailableLayers.length?'Some structures need scan alignment.':result.registrationStatus==='alignment preview'?'Alignment preview · review in Scans':'Anatomy ready');
    }catch(e){if(seq===generation.current){installVolume(null);throw e;}}
  }
  const install=useCallback(async(b:Binding)=>{
    generation.current++;resetLayers();installVolume(null);setBinding(b);setDraft(null);setOriginal(false);setReview(null);setComponents([]);setEditLayer(null);
    const saved=b.viewState?.snapshotId===b.document.snapshot_id?b.viewState:null;setSelectedStructure(saved?.selectedStructure??null);
    setPatientCutaway(saved?.patientCutaway??{});
    scopeRef.current=vesselScopeForView(saved);setVesselScope(scopeRef.current);
    const s=saved?.seriesId??b.document.reference_series??'';setSeries(s);
    const views=saved?.states??{};const v=migrateSkullView(views[s]??initial(b,s),b.document.layers);setStates(s?{...views,[s]:v}:{});
    // Recorded episodes and saved approaches belong to this exact source revision.
    // Registration changes are made through intake, preparation or explicit review.
    if(s)await run(()=>loadView(b,s,v,false,{},!!saved&&saved.patientCutaway===undefined));
    else setMessage('Add a source scan to begin.');
  },[api]);
  async function create(patientId:string,timepoint='preoperative'):Promise<Binding|null>{let result:Binding|null=null;await run(async()=>{result=await api!.anatomyCreate({patientId,timepoint})});if(result)await install(result);return result;}
  function clear(){contextCache.current=undefined;sourceImage.current=null;generation.current++;resetLayers();setBinding(null);installVolume(null);setPatientCutaway({});setDraft(null);setReview(null);setEditLayer(null);}
  function selectScan(s:string,mode?:AnatomyPresentation){if(!binding||busy||draft||s===series)return;
    const alignment=volume?.patientContext?.sources.find(source=>source.seriesId===s)?.alignment??'unavailable';
    const linked=patientFrameReady&&['reference','reviewed','source-provided'].includes(alignment);
    let v=migrateSkullView(nextScanState(scanState,states[s]??initial(binding,s),linked),binding.document.layers);
    if(patientCutawayActive&&alignment!=='unavailable')v={...v,settings:{...v.settings,presentation:scanState.settings.presentation}};
    if(mode){
      const modality=binding.document.series.find(scan=>scan.id===s)?.modality??'MRI';
      v={...v,settings:skullDisplay({...anatomyPresentation(v.settings,binding.document.layers,mode,modality),...viewerCutaway(v.settings)})};
    }
    generation.current++;resetLayers();installVolume(null);setSeries(s);setStates(all=>({...all,[s]:v}));setReview(null);setComponents([]);setEditLayer(null);
    void run(()=>loadView(binding,s,v));
  }
  function present(mode:AnatomyPresentation){
    if(!binding||busy||draft)return;
    const scan=binding.document.series.find(s=>s.id===series);
    const settings=anatomyPresentation(state.settings,binding.document.layers,mode,scan?.modality??'MRI');
    if(usesPatientContext(settings,!!patientField)===patientCutawayActive)updateSettings(settings);
    else updateState({settings:skullDisplay({...settings,...viewerCutaway(scanState.settings)})});
  }
  function queueLayers(){
    if(layerTask.current||active.current||!api)return;
    const b=live.current.binding,s=live.current.series,seq=generation.current;
    if(!b||!residentVolume.current)return;
    const task=Promise.resolve().then(()=>drainLayerRequests({
      isCurrent:()=>seq===generation.current,
      current:()=>({volume:residentVolume.current,order:b.document.layers.map(l=>l.id),visible:live.current.states[s]?.visible??{},unavailable:unavailableRef.current,failed:failedLayers.current}),
      pending:setPendingLayers,
      load:async(base,layers)=>{
        const result=await api.anatomyView({...args(b),seriesId:s,layers,original,patientModel:b.document.series.length>1,previewAlignment:true,vesselScope:scopeRef.current});
        if(seq!==generation.current)throw new DOMException('Cancelled','AbortError');
        const imageKey=b.collectionId+"/"+s+"/"+result.mri.sha256;
        if(sourceImage.current?.key!==imageKey)throw Error('Source image changed; reopen this scan');
        const incoming=await hydrateAnatomy(result,api,base.mri,base);
        if(seq!==generation.current)throw new DOMException('Cancelled','AbortError');
        incoming.displayIdentity=base.displayIdentity;
        if(result.alignmentPreviews?.length)setRegistration('alignment preview');
        const volume=await prepared(mergeResidentLayers(base,incoming,b.document.layers.map(l=>l.id)));
        return {volume,unavailable:[...new Set([...unavailableRef.current,...result.unavailableLayers])]};
      },
      install:({volume,unavailable})=>{installVolume(volume);unavailableRef.current=unavailable;setUnavailable(unavailable);setMessage(unavailable.length?'Some layers require reviewed alignment.':'Anatomy ready');},
      fail:(layers,message)=>{failedLayers.current={...failedLayers.current,...Object.fromEntries(layers.map(id=>[id,message]))};setLayerErrors(failedLayers.current);},
    })).finally(()=>{
      if(layerTask.current===task)layerTask.current=null;
      if(seq===generation.current)setPendingLayers([]);
    });
    layerTask.current=task;
  }
  function setVisible(visible:Record<string,boolean>){
    if(!binding||active.current)return;
    updateState({visible});queueLayers();
  }
  function retryLayer(id:string){
    const next={...failedLayers.current};delete next[id];failedLayers.current=next;setLayerErrors(next);queueLayers();
  }
  const refresh=()=>binding?loadView(binding,series,scanState,original):Promise.resolve();
  async function edit(action:string,points:Point3[]=[],extra:Args={}){if(!binding||!api||!editLayer)return;await run(async()=>{
    const result=await api.anatomyEdit({...args(),action,layerId:editLayer,points,radiusMm:radius,threshold,author,...extra});
    let b={...binding,document:result.document};
    let retired:RetiredClipping[]|undefined;
    setBinding(b);setDraft(result.draft);setOriginal(false);
    if(action==='commit'){setMessage('Checking alignment after correction…');const aligned=await api.anatomyRegister({...args(b),action:'ensure'});b={...b,document:aligned.document};retired=aligned.retiredClipping;}
    setBinding(b);setDraft(result.draft);setOriginal(false);
    if(['commit','reject'].includes(action)){setReview(null);setComponents([]);}
    await loadView(b,series,retireClipping(retired,series,scanState),false,action==='remove-components'?{highlightLayer:editLayer,highlightComponents:components,reviewId:review?.reviewId}:{});
    setMessage(retired?.length?retiredClippingMessage:action==='commit'?'Correction accepted as a new revision.':action==='reject'?'Correction rejected. Original preserved.':'Correction preview');
  });}
  const canEditLayer=(id:string)=>!!binding&&[series,binding.document.layers.find(l=>l.id===id)?.source_series].every(s=>!!s&&['reference','reviewed','source-provided'].includes(binding.document.transforms[s]?.status));
  function onStroke(point:Point3|null,phase:'start'|'move'|'end'){
    if(busy||!['paint','erase'].includes(tool)||!editLayer||!canEditLayer(editLayer))return;
    if(phase==='start')stroke.current=[];
    if(point){const prev=stroke.current.at(-1);if(prev){const distance=Math.hypot(...point.map((v,i)=>v-prev[i]));const steps=Math.min(200,Math.ceil(distance/Math.max(.2,radius*.4)));for(let i=1;i<=steps;i++)stroke.current.push(point.map((v,j)=>prev[j]+(v-prev[j])*i/steps) as Point3)}else stroke.current.push(point)}
    if(phase==='end'&&stroke.current.length)void edit(tool,stroke.current.splice(0,10000));
  }
  const currentViewState=():NativeViewState|null=>binding?{schema:'patient-anatomy-view-v1',patientId:binding.document.patient_id,snapshotId:binding.document.snapshot_id,seriesId:series,states,patientCutaway,vesselScope:scopeRef.current,vesselDisplayVersion:2,selectedApproachId:planning.selectedId,selectedPlanningTargetId:planning.activeTargetLayerId,selectedApproachIds:planning.selectedApproachIds,selectedStructure}:null;
  async function save(saveArchive=true){if(!binding||!api||draft)return;await run(async()=>{
    const viewState=currentViewState()!;
    if(series)await api.anatomySaveView({...args(),viewState,saveArchive});setMessage('Workspace saved · scans, layers and views preserved');
  });}
  async function reviewFragments(layerId=editLayer){
    if(!api||!layerId||!binding)return;setEditLayer(layerId);setTool('navigate');
    await run(async()=>{
      const next={...scanState,visible:{...scanState.visible,[layerId]:true}};
      const needsSource=scopeRef.current!=='all'||!state.visible[layerId];
      scopeRef.current='all';setVesselScope('all');updateState({visible:next.visible});
      if(needsSource)await loadView(binding,series,next);
      const r=await api.anatomyComponents({...args(),layerId});setReview(r);setComponents([]);
    });
  }
  function highlight(ids:number[]){if(!binding||!review||busy)return;setComponents(ids);const next={...scanState,cursor:ids.length===1?review.components.find(c=>c.id===ids[0])?.center??scanState.cursor:scanState.cursor};void run(()=>loadView(binding,series,next,original,{highlightLayer:editLayer,highlightComponents:ids,reviewId:review.reviewId}));}
  function compare(){
    if(!binding)return;
    const camera=readCamera.current?.()??scanState.settings.camera;
    const view={...scanState,settings:{...scanState.settings,camera}};
    void run(()=>loadView(binding,series,view,!original));
  }
  async function change(job:()=>Promise<{document:Binding['document'];retiredClipping?:RetiredClipping[]}|null>,{align=true}={}){
    if(!binding||!api)return;
    await run(async()=>{
      const result=await job();
      if(!result||(!align&&result.document.snapshot_id===binding.document.snapshot_id))return;
      let next={...binding,document:result.document};
      const retired=[...(result.retiredClipping??[])];
      setBinding(next);
      if(align){
        const aligned=await api.anatomyRegister({...args(next),action:'ensure'});
        next={...next,document:aligned.document};setBinding(next);
        retired.push(...(aligned.retiredClipping??[]));
      }
      const nextReference=next.document.reference_series,previousReference=binding.document.reference_series;
      if(nextReference!==previousReference)setPatientCutaway({});
      const formerReferenceSelected=series===previousReference&&nextReference!==previousReference;
      const retainSource=!formerReferenceSelected&&next.document.series.some(scan=>scan.id===series);
      const source=retainSource?series:nextReference??'';
      setSeries(source);
      if(!source)return;
      const previousLayers=new Set(binding.document.layers.map(layer=>layer.id));
      const added=Object.fromEntries(next.document.layers.filter(layer=>!previousLayers.has(layer.id)).map(layer=>[layer.id,!['brain-region','review-region','head-envelope'].includes(layer.kind)]));
      let current=states[source]??initial(next,source);
      if((binding.document.series.length<2&&next.document.series.length>1)||(!binding.document.layers.length&&next.document.layers.length)){
        const modality=next.document.series.find(scan=>scan.id===source)!.modality;
        current={...current,settings:anatomyPresentation(current.settings,next.document.layers,'segmentation',modality)};
      }
      current=retireClipping(retired,source,current);
      await loadView(next,source,migrateSkullView({...current,visible:{...current.visible,...added}},next.document.layers));
      if(retired.length)setMessage(retiredClippingMessage);
    });
  }
  async function runPreparation(input:Args){
    if(!api)throw Error('Anatomy preparation is unavailable');
    const requestId='prepare-'+crypto.randomUUID();
    preparationRequest.current=requestId;
    setProgress(undefined);
    try{return await api.anatomyPrepare({...input,action:'run',requestId});}
    finally{if(preparationRequest.current===requestId)preparationRequest.current=null;}
  }
  function cancel(){
    const preparationId=preparationRequest.current;
    if(preparationId){
      setMessage('Cancelling preparation…');
      void api?.cancel(preparationId).catch(error=>setMessage(error instanceof Error?error.message:String(error)));
      return;
    }
    generation.current++;resetLayers();setMessage('Operation cancelled');
    if(requestId.current)void api?.cancel(requestId.current);
  }
  function changeVesselScope(scope:VesselScope){
    if(active.current||busy||draft||!binding||scope===scopeRef.current)return;
    // Invalidate incremental requests before waiting for them; old filtered layers
    // must never be installed into the newly selected scope.
    generation.current++;resetLayers();installVolume(null);
    scopeRef.current=scope;setVesselScope(scope);
    void run(()=>loadView(binding,series,scanState,original));
  }
  async function savePackage(){if(!binding||!api||draft)return;await run(async()=>{
    const viewState=currentViewState()!;
    if(series)await api.anatomySaveView({...args(),viewState,saveArchive:false});
    const r=await api.anatomySavePackage(args());if(r)setMessage('Patient study file saved · '+r.path);
  });}
  function changeSkullMode(mode:NonNullable<ViewerWorkspaceSettings['skullMode']>){
    if(!binding||!volume)return;
    const bone=Object.fromEntries(binding.document.layers.filter(l=>['bone','skull'].includes(l.kind)).map(l=>[l.id,mode==='transparent'?.08:1]));
    const bounds=patientCutawayActive&&patientField
      ? volumeBounds(patientField.shape,patientField.affine,true)
      : volumeBounds(volume.shape,rasAffine(volume.affine,volume.frame),true);
    const value=newClippingPosition('z',bounds,state.cursor?.[2]);
    const prior=state.settings.clipping;
    updateSettings({...state.settings,skullMode:mode,layerOpacity:{...state.settings.layerOpacity,...bone},
      clipping:mode==='cutaway'?{axis:'z',value,enabled:true,flipped:true}:prior?{...prior,enabled:false}:undefined,
      clipBox:state.settings.clipBox?{...state.settings.clipBox,enabled:false}:undefined,
      obliqueClip:state.settings.obliqueClip?{...state.settings.obliqueClip,enabled:false}:undefined});
  }
  return {readCamera,viewState:currentViewState(),patientFrameReady,planning,selectedStructure,setSelectedStructure,structureFocusRequest,structureFocusPoint,focusStructure:(id:string,point?:Point3)=>{setStructureFocusPoint(point??null);setSelectedStructure(id);setStructureFocusRequest(n=>n+1);},present,changeSkullMode,canEditLayer,vesselScope,changeVesselScope,savePackage,api,binding,volume,series,state,busy,progress,pendingLayers,layerErrors,retryLayer,message,registration,unavailable,draft,original,editLayer,setEditLayer,tool,setTool,radius,setRadius,threshold,setThreshold,author,setAuthor,review,components,args,run,install,create,clear,selectScan,setVisible,refresh,edit,onStroke,save,reviewFragments,highlight,compare,change,runPreparation,setMessage,setCursor:(cursor:Point3)=>updateState({cursor}),setSettings:(settings:ViewerWorkspaceSettings)=>updateSettings(skullDisplay(settings)),cancel};
}
export type NativeWorkspace=ReturnType<typeof useNativeWorkspace>;
