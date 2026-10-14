import {useCallback,useEffect,useRef,useState} from 'react';
import type {ResectionApi} from './types';
import {hydrateAnatomy} from './native-anatomy-data';
import type {Binding,Draft,NativeAnatomyApi,NativeScanState,NativeViewState,ComponentReview,Args} from './native-anatomy-data';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {transformPoint} from './viewer/coordinates';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
export const nativeDefaults=():ViewerWorkspaceSettings=>({layout:'3d-focus',showPlane:false,layerOpacity:{brain:.06,tumor:1,vessels:1,bone:.08,'scalp-envelope':.02,__fragment_review:1},volumeRendering:{enabled:true,density:.18}});
export function useNativeWorkspace(runtime:ResectionApi|null){
  const api=runtime as (ResectionApi&NativeAnatomyApi)|null;
  const [binding,setBinding]=useState<Binding|null>(null),[volume,setVolume]=useState<ViewerVolume|null>(null);
  const [series,setSeries]=useState(''),[states,setStates]=useState<Record<string,NativeScanState>>({});
  const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[registration,setRegistration]=useState('');
  const [unavailable,setUnavailable]=useState<string[]>([]),[draft,setDraft]=useState<Draft>(null),[original,setOriginal]=useState(false);
  const [editLayer,setEditLayer]=useState<string|null>(null),[tool,setTool]=useState('navigate'),[radius,setRadius]=useState(2),[threshold,setThreshold]=useState(.5),[author,setAuthor]=useState('');
  const [review,setReview]=useState<ComponentReview|null>(null),[components,setComponents]=useState<number[]>([]);
  const sourceImage=useRef<{key:string;data:Float32Array}|null>(null);
  const generation=useRef(0),active=useRef(false),requestId=useRef<string|null>(null),stroke=useRef<Point3[]>([]);
  const live=useRef({binding,series,states});live.current={binding,series,states};
  useEffect(()=>()=>{generation.current++},[]);
  useEffect(()=>api?.onEvent(e=>{if(e.op?.startsWith('anatomy')){if(e.event==='started')requestId.current=e.id;if(e.event==='progress'&&e.progress)setMessage(e.progress.message);if(['result','error','cancelled'].includes(e.event))requestId.current=null;}}),[api]);
  const args=(b=live.current.binding)=>({collectionId:b!.collectionId,snapshotId:b!.document.snapshot_id});
  const initial=(b:Binding,s:string):NativeScanState=>({cursor:null,settings:nativeDefaults(),visible:Object.fromEntries(b.document.layers.map(l=>[l.id,l.source_series===s&&!['brain-region','review-region','head-envelope'].includes(l.kind)]))});
  const state=states[series]??{cursor:null,visible:{},settings:nativeDefaults()};
  function updateState(next:Partial<NativeScanState>){setStates(all=>({...all,[series]:{...state,...next}}));}
  const run=async(job:()=>Promise<void>)=>{if(active.current)return;active.current=true;setBusy(true);try{await job()}catch(e){setMessage(e instanceof Error?e.message:String(e))}finally{active.current=false;setBusy(false)}};
  async function loadView(b:Binding,s:string,view:NativeScanState,raw=false,extra:Args={}){
    if(!api)return;const seq=++generation.current;
    setVolume(null);setUnavailable([]);setMessage('Preparing anatomy…');
    const ids=Object.keys(view.visible).filter(id=>view.visible[id]);
    try{
      const result=await api.anatomyView({...args(b),seriesId:s,layers:ids,original:raw,...extra});
      if(seq!==generation.current)return;
      const imageKey=b.collectionId+"/"+s+"/"+result.mri.sha256;
      const hydrated=await hydrateAnatomy(result,api,sourceImage.current?.key===imageKey?sourceImage.current.data:undefined);if(seq!==generation.current)return;
      sourceImage.current={key:imageKey,data:hydrated.mri};
      const next={...view,cursor:view.cursor??transformPoint(hydrated.affine,hydrated.shape.map(n=>(n-1)/2) as Point3)};
      setStates(all=>({...all,[s]:next}));setVolume(hydrated);setRegistration(result.registrationStatus);setUnavailable(result.unavailableLayers);
      setMessage(result.unavailableLayers.length?'Some layers require reviewed alignment.':'Anatomy ready');
    }catch(e){if(seq===generation.current){setVolume(null);throw e;}}
  }
  const install=useCallback(async(b:Binding)=>{
    generation.current++;setVolume(null);setBinding(b);setDraft(null);setOriginal(false);setReview(null);setComponents([]);setEditLayer(null);
    const saved=b.viewState?.snapshotId===b.document.snapshot_id?b.viewState:null;
    const s=saved?.seriesId??b.document.reference_series??'';setSeries(s);
    const views=saved?.states??{};const v=views[s]??initial(b,s);setStates({...views,[s]:v});
    if(s)await run(()=>loadView(b,s,v));else setMessage('Add a source scan to begin.');
  },[api]);
  async function create(patientId:string){let result:Binding|null=null;await run(async()=>{result=await api!.anatomyCreate({patientId})});if(result)await install(result);return result;}
  function clear(){sourceImage.current=null;generation.current++;setBinding(null);setVolume(null);setDraft(null);setReview(null);setEditLayer(null);}
  function selectScan(s:string){if(!binding||busy||draft||s===series)return;const v=states[s]??initial(binding,s);
    if(!v.cursor && registration==='patient frame' && ['reference','reviewed','source-provided'].includes(binding.document.transforms[s]?.status))v.cursor=state.cursor;
    setSeries(s);setReview(null);setComponents([]);setEditLayer(null);void run(()=>loadView(binding,s,v));}
  function setVisible(visible:Record<string,boolean>){
    if(!binding||busy)return;const next={...state,visible};updateState({visible});
    const resident=new Set(volume?.compartments.map(l=>l.name));
    if(volume&&Object.keys(visible).filter(id=>visible[id]).every(id=>resident.has(id)||unavailable.includes(id)))return;
    void run(()=>loadView(binding,series,next,original));
  }
  const refresh=()=>binding?loadView(binding,series,state,original):Promise.resolve();
  async function edit(action:string,points:Point3[]=[],extra:Args={}){if(!binding||!api||!editLayer)return;await run(async()=>{
    const result=await api.anatomyEdit({...args(),action,layerId:editLayer,points,radiusMm:radius,threshold,author,...extra});
    const b={...binding,document:result.document};setBinding(b);setDraft(result.draft);setOriginal(false);
    if(['commit','reject'].includes(action)){setReview(null);setComponents([]);}
    await loadView(b,series,state,false,action==='remove-components'?{highlightLayer:editLayer,highlightComponents:components,reviewId:review?.reviewId}:{});
    setMessage(action==='commit'?'Correction accepted as a new revision.':action==='reject'?'Correction rejected. Original preserved.':'Correction preview');
  });}
  function onStroke(point:Point3|null,phase:'start'|'move'|'end'){
    if(busy||!['paint','erase'].includes(tool)||registration!=='patient frame')return;
    if(phase==='start')stroke.current=[];
    if(point){const prev=stroke.current.at(-1);if(prev){const distance=Math.hypot(...point.map((v,i)=>v-prev[i]));const steps=Math.min(200,Math.ceil(distance/Math.max(.2,radius*.4)));for(let i=1;i<=steps;i++)stroke.current.push(point.map((v,j)=>prev[j]+(v-prev[j])*i/steps) as Point3)}else stroke.current.push(point)}
    if(phase==='end'&&stroke.current.length)void edit(tool,stroke.current.splice(0,10000));
  }
  async function save(){if(!binding||!api||draft)return;await run(async()=>{
    const viewState:NativeViewState={schema:'patient-anatomy-view-v1',patientId:binding.document.patient_id,snapshotId:binding.document.snapshot_id,seriesId:series,states};
    await api.anatomySaveView({...args(),viewState});setMessage('Workspace saved · scans, layers and views preserved');
  });}
  async function reviewFragments(){if(!api||!editLayer)return;await run(async()=>{const r=await api.anatomyComponents({...args(),layerId:editLayer});setReview(r);setComponents([])});}
  function highlight(ids:number[]){if(!binding||!review||busy)return;setComponents(ids);const next={...state,cursor:ids.length===1?review.components.find(c=>c.id===ids[0])?.center??state.cursor:state.cursor};void run(()=>loadView(binding,series,next,original,{highlightLayer:editLayer,highlightComponents:ids,reviewId:review.reviewId}));}
  function compare(){if(!binding)return;setOriginal(!original);void run(()=>loadView(binding,series,state,!original));}
  async function change(job:()=>Promise<{document:Binding['document']}|null>){if(!binding)return;await run(async()=>{const r=await job();if(!r)return;const b={...binding,document:r.document};setBinding(b);const s=series||b.document.reference_series||'';setSeries(s);if(s)await loadView(b,s,states[s]??initial(b,s));});}
  return {api,binding,volume,series,state,busy,message,registration,unavailable,draft,original,editLayer,setEditLayer,tool,setTool,radius,setRadius,threshold,setThreshold,author,setAuthor,review,components,args,run,install,create,clear,selectScan,setVisible,refresh,edit,onStroke,save,reviewFragments,highlight,compare,change,setMessage,setCursor:(cursor:Point3)=>updateState({cursor}),setSettings:(settings:ViewerWorkspaceSettings)=>updateState({settings}),cancel:()=>{generation.current++;setMessage('Operation cancelled');if(requestId.current)void api?.cancel(requestId.current)}};
}
export type NativeWorkspace=ReturnType<typeof useNativeWorkspace>;
