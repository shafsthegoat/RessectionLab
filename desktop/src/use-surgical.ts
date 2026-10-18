import {useEffect,useMemo,useRef,useState} from 'react';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {rasAffine} from './viewer/coordinates.ts';
import type {SurgicalAction,SurgicalOutcome,SurgicalApi,SurgicalPose,SurgicalRegion,SurgicalResponse,SurgicalView,SurgicalInteraction} from './surgical-types';

const geometryCache=new WeakMap<ViewerVolume,Map<string,ViewerVolume>>();
function geometryIdentity(key:string){let a=2166136261,b=5381;for(let i=0;i<key.length;i++){a=Math.imul(a^key.charCodeAt(i),16777619);b=Math.imul(b,33)^key.charCodeAt(i);}return `${key.length}:${a>>>0}:${b>>>0}`;}
export function validateMapping(regions:SurgicalRegion[]):string|null {
  if(!regions.length)return 'Map at least one source region.';
  if(new Set(regions.map(r=>r.id)).size!==regions.length||new Set(regions.map(r=>r.name)).size!==regions.length)return 'Region IDs and names must be unique.';
  if(regions.some(r=>!Number.isInteger(r.id)||r.id<1||r.id>65535||!r.name||!r.material||typeof r.target!=='boolean'))return 'Each region needs a positive ID, name, material, and target choice.';
  return null;
}
export function modeledVolume(source:ViewerVolume,view:SurgicalView):ViewerVolume {
  if(view.scene.shape.some((n,i)=>n!==source.shape[i]))throw Error('Simulation/source grid mismatch');
  const sourceRas=rasAffine(source.affine,source.frame),sceneRas=rasAffine(view.scene.affine,'RAS+');
  if(sourceRas.some((row,i)=>row.some((value,j)=>Math.abs(value-sceneRas[i][j])>1e-6)))throw Error('Simulation/source affine mismatch');
  const count=source.shape.reduce((a,b)=>a*b,1),regions=view.scene.regions;
  const coupled = view.physics?.profile?.schema === 'resectionlab.tissue-physics.v2';
  const key=JSON.stringify([regions,view.removed,view.moved,...(coupled?['coupled-v2']:[])]);
  const cached=geometryCache.get(source)??new Map<string,ViewerVolume>();
  const existing=cached.get(key);
  if(existing)return existing;
  let compartments:ViewerVolume['compartments'];
  {
    const owner=new Uint16Array(count);
    for(const r of regions){const layer=source.compartments.find(l=>l.name===r.name);if(!layer)continue;
      const labels=source.labelSets?.find(s=>s.id===layer.labelSet)?.data;
      for(let i=0;i<count;i++)if(labels?labels[i]===layer.labelValue:layer.mask[i])owner[i]=r.id;}
    const original=owner.slice(),valid=(i:number)=>Number.isInteger(i)&&i>=0&&i<count;
    for(const i of view.removed){if(!valid(i))throw Error('Invalid removed cell');owner[i]=0;}
    for(const [a,b] of view.moved){if(!valid(a)||!valid(b))throw Error('Invalid displacement');owner[a]=0;}
    const priority = new Map(regions.map((r,i)=>[r.id,i]));
    for(const [a,b] of view.moved){if(!coupled || (priority.get(original[a])??-1)>=(priority.get(owner[b])??-1))owner[b]=original[a];}
    compartments=regions.map(r=>{const mask=new Uint8Array(count);for(let i=0;i<count;i++)mask[i]=Number(owner[i]===r.id);return {name:r.name,color:r.color,mask};});
    // Regions absent from a mapping remain inspectable, with their original source masks.
    for(const layer of source.compartments)if(!regions.some(r=>r.name===layer.name)){
      const labels=source.labelSets?.find(s=>s.id===layer.labelSet)?.data;
      compartments.push({name:layer.name,color:layer.color,mask:labels?Uint8Array.from(labels,n=>Number(n===layer.labelValue)):layer.mask});
    }
  }
  const volume={...source,displayIdentity:`surgical:${source.caseHash}:${geometryIdentity(key)}`,packedLabels:undefined,labelSets:undefined,compartments};
  cached.set(key,volume);if(cached.size>3)cached.delete(cached.keys().next().value!);geometryCache.set(source,cached);
  return volume;
}

export function useSurgical(source:ViewerVolume|null,api:Partial<SurgicalApi>|null,enabled:boolean){
  const [result,setResult]=useState<SurgicalResponse|null>(null),[pending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
  const [slot,setSlot]=useState('left'),[pose,setRawPose]=useState<SurgicalPose>({position:[0,0,0],quaternion:[0,0,0,1]}),[path,setPath]=useState<SurgicalPose[]>([]);
  const [parameters,setParameters]=useState<Record<string,number>>({});
  const [mode,setMode]=useState<SurgicalInteraction['mode']>('navigate'),[preview,setPreview]=useState<SurgicalView|null>(null);
  const [previewOutcome,setPreviewOutcome]=useState<SurgicalOutcome|null>(null);
  const generation=useRef(0),busy=useRef(false),current=useRef({source,api,result,slot,enabled});current.current={source,api,result,slot,enabled};
  useEffect(()=>{generation.current++;busy.current=false;setResult(null);setPreview(null);setPreviewOutcome(null);setPath([]);setPending(false);setError(null);},[source?.caseHash]);
  useEffect(()=>()=>{generation.current++;},[]);
  async function run(command:string,extra:Record<string,unknown>={}):Promise<boolean>{
    const c=current.current;if(!c.enabled||!c.source||!c.api?.surgicalRehearsal||busy.current)return false;
    const token=++generation.current;busy.current=true;setPending(true);setError(null);
    try{const next=await c.api.surgicalRehearsal({caseHash:c.source.caseHash,command,expectedStateId:c.result?.view.stateId,...extra});
      if(token!==generation.current||!current.current.enabled||current.current.source?.caseHash!==c.source.caseHash)return false;
      if(command==='preview'){const response=next as unknown as (SurgicalOutcome&{view:SurgicalView})|null;setPreview(response?.view??null);setPreviewOutcome(response?{status:response.status,reason:response.reason,completedFraction:response.completedFraction,events:response.events}:null);if(response?.reason)setError(response.reason);return !!response;}
      setPreview(null);setPreviewOutcome(null);setResult(next);setPath([]);
      if(next){const t=next.view.slots[current.current.slot];if(t){setRawPose({position:[...t.position],quaternion:[...t.quaternion]});setParameters({...t.parameters});}if(next.outcome?.reason)setError(`${next.outcome.status}: ${next.outcome.reason}`);}
      return !!next;
    }catch(e){if(token===generation.current)setError(String(e));return false;}finally{if(token===generation.current){busy.current=false;setPending(false);}}
  }
  useEffect(()=>{if(enabled&&source&&api?.surgicalRehearsal)void run('inspect');else {generation.current++;busy.current=false;setPending(false);setPreview(null);setPreviewOutcome(null);}},[source?.caseHash,enabled,api?.surgicalRehearsal]);
  const view=enabled&&result?.caseHash===source?.caseHash?result?.view??null:null;
  const volume=useMemo(()=>source&&view?modeledVolume(source,view):null,[source,view]);
  const previewVolume=useMemo(()=>source&&enabled&&preview?modeledVolume(source,preview):null,[source,enabled,preview]);
  function setPose(p:SurgicalPose){setRawPose(p);setPreview(null);setPreviewOutcome(null);}
  function selectSlot(s:string){setSlot(s);setPreview(null);setPreviewOutcome(null);setPath([]);const t=result?.view.slots[s];if(t){setRawPose({position:[...t.position],quaternion:[...t.quaternion]});setParameters({...t.parameters});}}
  function onStroke(point:Point3,phase:'start'|'move'|'end'){
    if(phase==='start')setPath([{position:point,quaternion:pose.quaternion}]);
    else setPath(p=>p.length<512?[...p,{position:point,quaternion:pose.quaternion}]:p);
    setPose({...pose,position:point});
  }
  function action(operation:string,params:Record<string,number>=parameters,duration=1):SurgicalAction{return {slot,operation,parameters:params,duration,...(['apply','move','hold','grasp','place'].includes(operation)?{path:path.length?path:[pose]}:{})};}
  return {result,view,volume,previewVolume,pending,error,slot,pose,path,mode,preview,previewOutcome,parameters,setPose,setMode,selectSlot,run,action,
    setParameters:(p:Record<string,number>)=>{setParameters(p);setPreview(null);setPreviewOutcome(null);},
    clearPreview:()=>{setPreview(null);setPreviewOutcome(null);},
    create:(mapping?:SurgicalRegion[])=>{const issue=mapping?validateMapping(mapping):null;if(issue){setError(issue);return Promise.resolve(false);}return run('create',mapping?{mapping}:{});},
    cancel:()=>{setPreview(null);setPreviewOutcome(null);setPath([]);const t=result?.view.slots[slot];if(t)setRawPose({position:t.position,quaternion:t.quaternion});},
    interaction:{view,preview,outcome:preview?previewOutcome:result?.outcome??null,pose,path,parameters,selectedSlot:slot,mode,onPose:setPose,onStroke} as SurgicalInteraction};
}
export type SurgicalController=ReturnType<typeof useSurgical>;
