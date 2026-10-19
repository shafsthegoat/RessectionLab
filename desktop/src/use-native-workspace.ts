import {nextScanState,migrateSkullView,vesselScopeForView,anatomyPresentation} from './native-study-state';
import type {AnatomyPresentation} from './native-study-state';
import {drainLayerRequests} from './native-layer-queue';
import {mergeResidentLayers,prepareNativeLabels} from './native-layer-residency';
import {useCallback,useEffect,useRef,useState} from 'react';
import type {ResectionApi} from './types';
import {hydrateAnatomy} from './native-anatomy-data';
import type {Binding,Draft,NativeAnatomyApi,NativeScanState,NativeViewState,ComponentReview,Args,VesselScope} from './native-anatomy-data';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {transformPoint} from './viewer/coordinates';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
export const nativeDefaults=():ViewerWorkspaceSettings=>({layout:'3d-focus',showPlane:false,sliceOverlay:'contour',layerOpacity:{brain:.06,tumor:1,vessels:1,bone:1,'scalp-envelope':.02,__fragment_review:1},volumeRendering:{enabled:true,density:.18}});
export function useNativeWorkspace(runtime:ResectionApi|null){
  const api=runtime as (ResectionApi&NativeAnatomyApi)|null;
  const [binding,setBinding]=useState<Binding|null>(null),[volume,setVolume]=useState<ViewerVolume|null>(null);
  const [series,setSeries]=useState(''),[states,setStates]=useState<Record<string,NativeScanState>>({});
  const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[registration,setRegistration]=useState('');
  const [vesselScope,setVesselScope]=useState<VesselScope>('all');
  const scopeRef=useRef<VesselScope>('all');
  const [unavailable,setUnavailable]=useState<string[]>([]),[draft,setDraft]=useState<Draft>(null),[original,setOriginal]=useState(false);
  const [editLayer,setEditLayer]=useState<string|null>(null),[tool,setTool]=useState('navigate'),[radius,setRadius]=useState(2),[threshold,setThreshold]=useState(.5),[author,setAuthor]=useState('');
  const [review,setReview]=useState<ComponentReview|null>(null),[components,setComponents]=useState<number[]>([]);
  const [pendingLayers,setPendingLayers]=useState<string[]>([]),[layerErrors,setLayerErrors]=useState<Record<string,string>>({});
  const residentVolume=useRef<ViewerVolume|null>(null),layerTask=useRef<Promise<void>|null>(null);
  const packing=useRef<AbortController|null>(null),failedLayers=useRef<Record<string,string>>({}),unavailableRef=useRef<string[]>([]);
  const sourceImage=useRef<{key:string;data:Float32Array}|null>(null);
  const generation=useRef(0),active=useRef(false),requestId=useRef<string|null>(null),stroke=useRef<Point3[]>([]);
  const live=useRef({binding,series,states});live.current={binding,series,states};
  useEffect(()=>()=>{generation.current++;packing.current?.abort()},[]);
  useEffect(()=>api?.onEvent(e=>{if(e.op?.startsWith('anatomy')){if(e.event==='started')requestId.current=e.id;if(e.event==='progress'&&e.progress)setMessage(e.progress.message);if(['result','error','cancelled'].includes(e.event))requestId.current=null;}}),[api]);
  const args=(b=live.current.binding)=>({collectionId:b!.collectionId,snapshotId:b!.document.snapshot_id});
  const initial=(b:Binding,s:string):NativeScanState=>{
    const modality=b.document.series.find(scan=>scan.id===s)?.modality??'MRI';
    const mode=/\b(CTA|MRA|TOF)\b|angiogra/i.test(modality)?'vascular':'segmentation';
    return {cursor:null,settings:anatomyPresentation(nativeDefaults(),b.document.layers,mode,modality),visible:Object.fromEntries(b.document.layers.map(l=>[l.id,!['brain-region','review-region','head-envelope'].includes(l.kind)]))};
  };
  const state=states[series]??{cursor:null,visible:{},settings:nativeDefaults()};
  function updateState(next:Partial<NativeScanState>){
    const s=live.current.series,current=live.current.states[s]??state;
    const views={...live.current.states,[s]:{...current,...next}};
    live.current={...live.current,states:views};setStates(views);
  }
  function installVolume(next:ViewerVolume|null){residentVolume.current=next;setVolume(next);}
  function resetLayers(){packing.current?.abort();packing.current=null;failedLayers.current={};setLayerErrors({});setPendingLayers([]);unavailableRef.current=[];}
  async function prepared(next:ViewerVolume){const controller=new AbortController();packing.current=controller;try{return await prepareNativeLabels(next,controller.signal)}finally{if(packing.current===controller)packing.current=null;}}
  const run=async(job:()=>Promise<void>)=>{if(active.current)return;active.current=true;setBusy(true);try{await layerTask.current;await job()}catch(e){setMessage(e instanceof Error?e.message:String(e))}finally{active.current=false;setBusy(false)}};
  async function loadView(b:Binding,s:string,view:NativeScanState,raw=false,extra:Args={}){
    if(!api)return;const seq=++generation.current;
    resetLayers();installVolume(null);setUnavailable([]);setMessage('Preparing anatomy…');
    const ids=Object.keys(view.visible).filter(id=>view.visible[id]);
    try{
      const result=await api.anatomyView({...args(b),seriesId:s,layers:ids,original:raw,previewAlignment:true,vesselScope:scopeRef.current,...extra});
      if(seq!==generation.current)return;
      const imageKey=b.collectionId+"/"+s+"/"+result.mri.sha256;
      let hydrated=await hydrateAnatomy(result,api,sourceImage.current?.key===imageKey?sourceImage.current.data:undefined);if(seq!==generation.current)return;
      hydrated.displayIdentity=JSON.stringify([b.collectionId,s,result.mri.sha256,b.document.snapshot_id,raw,seq]);
      hydrated=await prepared(hydrated);if(seq!==generation.current)return;
      sourceImage.current={key:imageKey,data:hydrated.mri};
      const next={...view,cursor:view.cursor??transformPoint(hydrated.affine,hydrated.shape.map(n=>(n-1)/2) as Point3)};
      setStates(all=>({...all,[s]:next}));installVolume(hydrated);setRegistration(result.registrationStatus);setUnavailable(result.unavailableLayers);unavailableRef.current=result.unavailableLayers;
      setMessage(result.unavailableLayers.length?'Some structures need scan alignment.':result.registrationStatus==='alignment preview'?'Alignment preview · review in Scans':'Anatomy ready');
    }catch(e){if(seq===generation.current){installVolume(null);throw e;}}
  }
  const install=useCallback(async(b:Binding)=>{
    generation.current++;resetLayers();installVolume(null);setBinding(b);setDraft(null);setOriginal(false);setReview(null);setComponents([]);setEditLayer(null);
    const saved=b.viewState?.snapshotId===b.document.snapshot_id?b.viewState:null;
    scopeRef.current=vesselScopeForView(saved);setVesselScope(scopeRef.current);
    const s=saved?.seriesId??b.document.reference_series??'';setSeries(s);
    const views=saved?.states??{};const v=migrateSkullView(views[s]??initial(b,s),b.document.layers);setStates({...views,[s]:v});
    if(s)await run(async()=>{setMessage('Checking study alignment…');const result=await api!.anatomyRegister({...args(b),action:'ensure'});const next={...b,document:result.document};setBinding(next);await loadView(next,s,v)});else setMessage('Add a source scan to begin.');
  },[api]);
  async function create(patientId:string,timepoint='preoperative'){let result:Binding|null=null;await run(async()=>{result=await api!.anatomyCreate({patientId,timepoint})});if(result)await install(result);return result;}
  function clear(){sourceImage.current=null;generation.current++;resetLayers();setBinding(null);installVolume(null);setDraft(null);setReview(null);setEditLayer(null);}
  function selectScan(s:string){if(!binding||busy||draft||s===series)return;
    const v=migrateSkullView(nextScanState(state,states[s]??initial(binding,s)),binding.document.layers);
    generation.current++;resetLayers();installVolume(null);setSeries(s);setStates(all=>({...all,[s]:v}));setReview(null);setComponents([]);setEditLayer(null);
    void run(async()=>{
      let b=binding;
      if(s!==b.document.reference_series){
        setMessage('Aligning scan to this study…');
        try{const r=await api!.anatomyRegister({...args(b),seriesId:s,action:'ensure'});b={...b,document:r.document};setBinding(b)}
        catch(e){throw Error(`Alignment unavailable: ${e instanceof Error?e.message:String(e)}`)}
      }
      await loadView(b,s,v);
    });
  }
  function present(mode:AnatomyPresentation){
    if(!binding||busy||draft)return;
    // Prefer an acquired angiographic scan for the vascular presentation, and
    // structural MRI for segmentation. Never manufacture a CTA from plain CT.
    const scans=binding.document.series;
    const candidate=mode==='vascular'
      ? scans.find(s=>/^(CTA|TOF|MRA)\b|angiogra/i.test(s.modality))
      : scans.find(s=>/T1CE|T1C|T1[-_ ]?POST/i.test(s.modality))??scans.find(s=>/MRI|^T1\b/i.test(s.modality));
    const target=candidate?.id??series,scan=scans.find(s=>s.id===target);
    const base=target===series?state:nextScanState(state,states[target]??initial(binding,target));
    const view={...base,settings:anatomyPresentation(base.settings,binding.document.layers,mode,scan?.modality??'MRI')};
    if(target===series){updateState({settings:view.settings});return;}
    generation.current++;resetLayers();installVolume(null);setSeries(target);setStates(all=>({...all,[target]:view}));setReview(null);setComponents([]);setEditLayer(null);
    void run(async()=>{
      let b=binding;
      if(target!==b.document.reference_series){const r=await api!.anatomyRegister({...args(b),seriesId:target,action:'ensure'});b={...b,document:r.document};setBinding(b)}
      await loadView(b,target,view);
    });
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
        const result=await api.anatomyView({...args(b),seriesId:s,layers,original,previewAlignment:true,vesselScope:scopeRef.current});
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
  const refresh=()=>binding?loadView(binding,series,state,original):Promise.resolve();
  async function edit(action:string,points:Point3[]=[],extra:Args={}){if(!binding||!api||!editLayer)return;await run(async()=>{
    const result=await api.anatomyEdit({...args(),action,layerId:editLayer,points,radiusMm:radius,threshold,author,...extra});
    let b={...binding,document:result.document};
    setBinding(b);setDraft(result.draft);setOriginal(false);
    if(action==='commit'){setMessage('Checking alignment after correction…');const aligned=await api.anatomyRegister({...args(b),action:'ensure'});b={...b,document:aligned.document};}
    setBinding(b);setDraft(result.draft);setOriginal(false);
    if(['commit','reject'].includes(action)){setReview(null);setComponents([]);}
    await loadView(b,series,state,false,action==='remove-components'?{highlightLayer:editLayer,highlightComponents:components,reviewId:review?.reviewId}:{});
    setMessage(action==='commit'?'Correction accepted as a new revision.':action==='reject'?'Correction rejected. Original preserved.':'Correction preview');
  });}
  const canEditLayer=(id:string)=>!!binding&&[series,binding.document.layers.find(l=>l.id===id)?.source_series].every(s=>!!s&&['reference','reviewed','source-provided'].includes(binding.document.transforms[s]?.status));
  function onStroke(point:Point3|null,phase:'start'|'move'|'end'){
    if(busy||!['paint','erase'].includes(tool)||!editLayer||!canEditLayer(editLayer))return;
    if(phase==='start')stroke.current=[];
    if(point){const prev=stroke.current.at(-1);if(prev){const distance=Math.hypot(...point.map((v,i)=>v-prev[i]));const steps=Math.min(200,Math.ceil(distance/Math.max(.2,radius*.4)));for(let i=1;i<=steps;i++)stroke.current.push(point.map((v,j)=>prev[j]+(v-prev[j])*i/steps) as Point3)}else stroke.current.push(point)}
    if(phase==='end'&&stroke.current.length)void edit(tool,stroke.current.splice(0,10000));
  }
  async function save(saveArchive=true){if(!binding||!api||draft)return;await run(async()=>{
    const viewState:NativeViewState={schema:'patient-anatomy-view-v1',patientId:binding.document.patient_id,snapshotId:binding.document.snapshot_id,seriesId:series,states,vesselScope:scopeRef.current,vesselDisplayVersion:2};
    await api.anatomySaveView({...args(),viewState,saveArchive});setMessage('Workspace saved · scans, layers and views preserved');
  });}
  async function reviewFragments(layerId=editLayer){
    if(!api||!layerId||!binding)return;setEditLayer(layerId);setTool('navigate');
    await run(async()=>{
      const next={...state,visible:{...state.visible,[layerId]:true}};
      const needsSource=scopeRef.current!=='all'||!state.visible[layerId];
      scopeRef.current='all';setVesselScope('all');updateState({visible:next.visible});
      if(needsSource)await loadView(binding,series,next);
      const r=await api.anatomyComponents({...args(),layerId});setReview(r);setComponents([]);
    });
  }
  function highlight(ids:number[]){if(!binding||!review||busy)return;setComponents(ids);const next={...state,cursor:ids.length===1?review.components.find(c=>c.id===ids[0])?.center??state.cursor:state.cursor};void run(()=>loadView(binding,series,next,original,{highlightLayer:editLayer,highlightComponents:ids,reviewId:review.reviewId}));}
  function compare(){if(!binding)return;setOriginal(!original);void run(()=>loadView(binding,series,state,!original));}
  async function change(job:()=>Promise<{document:Binding['document']}|null>){if(!binding)return;await run(async()=>{const r=await job();if(!r)return;const imported={...binding,document:r.document};setBinding(imported);const aligned=await api!.anatomyRegister({...args(imported),action:'ensure'});const b={...imported,document:aligned.document};setBinding(b);const s=series||b.document.reference_series||'';setSeries(s);if(s)await loadView(b,s,migrateSkullView(states[s]??initial(b,s),b.document.layers));});}
  function changeVesselScope(scope:VesselScope){
    if(active.current||busy||draft||!binding||scope===scopeRef.current)return;
    // Invalidate incremental requests before waiting for them; old filtered layers
    // must never be installed into the newly selected scope.
    generation.current++;resetLayers();installVolume(null);
    scopeRef.current=scope;setVesselScope(scope);
    void run(()=>loadView(binding,series,state,original));
  }
  async function savePackage(){if(!binding||!api||draft)return;await save(false);await run(async()=>{const r=await api.anatomySavePackage(args());if(r)setMessage('Patient study file saved · '+r.path)});}
  function changeSkullMode(mode:NonNullable<ViewerWorkspaceSettings['skullMode']>){
    if(!binding)return;
    const bone=Object.fromEntries(binding.document.layers.filter(l=>['bone','skull'].includes(l.kind)).map(l=>[l.id,mode==='transparent'?.08:1]));
    const value=state.cursor?.[2]??0;
    updateState({settings:{...state.settings,skullMode:mode,layerOpacity:{...state.settings.layerOpacity,...bone},clipping:{axis:'z',value,enabled:mode==='cutaway',flipped:true},clipBox:state.settings.clipBox?{...state.settings.clipBox,enabled:false}:undefined,obliqueClip:state.settings.obliqueClip?{...state.settings.obliqueClip,enabled:false}:undefined}});
  }
  return {present,changeSkullMode,canEditLayer,vesselScope,changeVesselScope,savePackage,api,binding,volume,series,state,busy,pendingLayers,layerErrors,retryLayer,message,registration,unavailable,draft,original,editLayer,setEditLayer,tool,setTool,radius,setRadius,threshold,setThreshold,author,setAuthor,review,components,args,run,install,create,clear,selectScan,setVisible,refresh,edit,onStroke,save,reviewFragments,highlight,compare,change,setMessage,setCursor:(cursor:Point3)=>updateState({cursor}),setSettings:(settings:ViewerWorkspaceSettings)=>{const clipped=settings.clipping?.enabled||settings.clipBox?.enabled||settings.obliqueClip?.enabled;const transparent=binding?.document.layers.some(l=>['bone','skull'].includes(l.kind)&&(settings.layerOpacity?.[l.id]??1)<1);updateState({settings:{...settings,skullMode:clipped?'cutaway':transparent?'transparent':'exterior'}})},cancel:()=>{generation.current++;resetLayers();setMessage('Operation cancelled');if(requestId.current)void api?.cancel(requestId.current)}};
}
export type NativeWorkspace=ReturnType<typeof useNativeWorkspace>;
