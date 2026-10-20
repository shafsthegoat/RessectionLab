import {useEffect,useMemo,useRef,useState} from 'react';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {SurgicalRequestQueue} from './surgical-request-queue.ts';
import {SurgicalDisplayAdmission} from './surgical-display-admission.ts';
import {FullCaseComputation} from './full-case-computation.ts';
import {checkedSurgicalPlayback,surgicalCursorCommand,updatesEditableSurgicalCursor} from './surgical-playback.ts';
import {modeledVolume} from './modeled-volume.ts';
export {modeledVolume} from './modeled-volume.ts';
import type {SurgicalAction,SurgicalOutcome,SurgicalApi,SurgicalPose,SurgicalRegion,SurgicalResponse,SurgicalView,SurgicalInteraction} from './surgical-types';

export function validateMapping(regions:SurgicalRegion[]):string|null {
  if(!regions.length)return 'Map at least one source region.';
  if(new Set(regions.map(r=>r.id)).size!==regions.length||new Set(regions.map(r=>r.name)).size!==regions.length)return 'Region IDs and names must be unique.';
  if(regions.some(r=>!Number.isInteger(r.id)||r.id<1||r.id>65535||!r.name||!r.material||typeof r.target!=='boolean'))return 'Each region needs a positive ID, name, material, and target choice.';
  return null;
}

export function useSurgical(source:ViewerVolume|null,api:Partial<SurgicalApi>|null,enabled:boolean){
  const [result,setResult]=useState<SurgicalResponse|null>(null),[requestPending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
  const [desiredResult,setDesiredResult]=useState<SurgicalResponse|null>(null),[desiredPreview,setDesiredPreview]=useState<SurgicalView|null>(null);
  const [desiredPreviewOutcome,setDesiredPreviewOutcome]=useState<SurgicalOutcome|null>(null),[displayFailed,setDisplayFailed]=useState(false);
  const [slot,setSlot]=useState('left'),[pose,setRawPose]=useState<SurgicalPose>({position:[0,0,0],quaternion:[0,0,0,1]}),[path,setPath]=useState<SurgicalPose[]>([]);
  const [parameters,setParameters]=useState<Record<string,number>>({});
  const [mode,setMode]=useState<SurgicalInteraction['mode']>('navigate'),[preview,setPreview]=useState<SurgicalView|null>(null);
  const [previewOutcome,setPreviewOutcome]=useState<SurgicalOutcome|null>(null);
  const [pendingCommand,setPendingCommand]=useState<string|null>(null);
  const backendResult=useRef<SurgicalResponse|null>(null);
  const responseSource=useRef<ViewerVolume|null>(null);
  const displayAdmission=useRef(new SurgicalDisplayAdmission<SurgicalView>());
  const pendingPoseView=useRef<SurgicalView|null>(null);
  const fullCaseComputation=useRef(new FullCaseComputation());
  const [fullCaseVersion,setFullCaseVersion]=useState(0);
  const [playbackVersion,setPlaybackVersion]=useState(0);
  function pauseFullCase(){if(fullCaseComputation.current.scenario){fullCaseComputation.current.pause();setFullCaseVersion(n=>n+1);}}
  const generation=useRef(0),busy=useRef(false),current=useRef({source,api,result,slot,enabled,displayFailed});current.current={source,api,result,slot,enabled,displayFailed};
  const desired=useRef({result:desiredResult,preview:desiredPreview,previewOutcome:desiredPreviewOutcome});desired.current={result:desiredResult,preview:desiredPreview,previewOutcome:desiredPreviewOutcome};
  const requestQueue=useRef<SurgicalRequestQueue<{command:string;extra:Record<string,unknown>;caseHash:string;source:ViewerVolume;generation:number}>|null>(null);
  useEffect(()=>{pauseFullCase();generation.current++;requestQueue.current?.invalidate();busy.current=false;backendResult.current=null;displayAdmission.current.invalidate();pendingPoseView.current=null;setResult(null);setDesiredResult(null);setDesiredPreview(null);setDesiredPreviewOutcome(null);setDisplayFailed(false);setPreview(null);setPreviewOutcome(null);setPath([]);setMode('navigate');setSlot('left');setPending(false);setError(null);},[source]);
  useEffect(()=>()=>{fullCaseComputation.current.pause();generation.current++;requestQueue.current?.invalidate();},[]);
  async function execute(command:string,extra:Record<string,unknown>,isCurrent:()=>boolean):Promise<boolean>{
    const c=current.current;if(!c.enabled||!c.source||!c.api?.surgicalRehearsal||busy.current)return false;
    if(c.result?.playback?.readOnly&&!surgicalCursorCommand(command)){setError('Return to a recorded action boundary before applying or branching instrument actions.');return false;}
    if(c.displayFailed&&['step','preview','plan','fullCase'].includes(command)){setError('The accepted anatomy could not be displayed. Seek to a recorded state or reopen the case before applying another instrument action.');return false;}
    const token=generation.current;busy.current=true;setPending(true);setPendingCommand(command);setError(null);
    try{const next=await c.api.surgicalRehearsal({caseHash:c.source.caseHash,command,expectedStateId:backendResult.current?.view.stateId,...extra});
      if(token!==generation.current||!current.current.enabled||current.current.source!==c.source)return false;
      // A superseded seek still moved the backend cursor. Preserve its state
      // identity for the next ordered request while withholding obsolete UI.
      if(command==='seekFrame'&&next){checkedSurgicalPlayback(next,extra);setMode('navigate');setPath([]);}
      if(updatesEditableSurgicalCursor(command))backendResult.current=next;
      if(!isCurrent())return false;
      setDisplayFailed(false);
      if(command==='preview'){const response=next as unknown as (SurgicalOutcome&{view:SurgicalView})|null;setDesiredPreview(response?.view??null);setDesiredPreviewOutcome(response?{status:response.status,reason:response.reason,completedFraction:response.completedFraction,events:response.events}:null);return !!response;}
      setDesiredPreview(null);setDesiredPreviewOutcome(null);pendingPoseView.current=next?.view??null;responseSource.current=c.source;setDesiredResult(next);
      return !!next;
    }catch(e){if(token===generation.current)setError(String(e));return false;}finally{if(token===generation.current){busy.current=false;setPending(false);setPendingCommand(null);}}
  }
  if(!requestQueue.current)requestQueue.current=new SurgicalRequestQueue(async(item,isCurrent)=>{
    if(item.generation!==generation.current||current.current.source!==item.source)return false;
    return execute(item.command,item.extra,isCurrent);
  },item=>item.command==='seek'||item.command==='seekFrame');
  async function run(command:string,extra:Record<string,unknown>={}):Promise<boolean>{
    if(command!=='fullCase')pauseFullCase();
    const c=current.current;if(!c.enabled||!c.source||!c.api?.surgicalRehearsal)return false;
    try{return await requestQueue.current!.request({command,extra:structuredClone(extra),caseHash:c.source.caseHash,source:c.source,generation:generation.current});}
    catch(cause){if(current.current.source===c.source)setError(String(cause));return false;}
  }
  useEffect(()=>{if(enabled&&source&&api?.surgicalRehearsal)void run('inspect');else {generation.current++;requestQueue.current?.invalidate();busy.current=false;displayAdmission.current.invalidate();setResult(null);setPending(false);setDesiredPreview(null);setDesiredPreviewOutcome(null);setPreview(null);setPreviewOutcome(null);}},[source,enabled,api?.surgicalRehearsal]);
  const view=enabled&&responseSource.current===source&&result?.caseHash===source?.caseHash?result?.view??null:null;
  const desiredView=enabled&&responseSource.current===source&&desiredResult&&desiredResult.caseHash===source?.caseHash?desiredResult.view:null;
  const activePreview=desiredView?desiredPreview:null,displayedPreview=view?preview:null;
  const requestedView=activePreview??desiredView,displayedView=displayedPreview??view;
  const displayPending=!displayFailed&&!!requestedView&&requestedView!==displayedView;
  const pending=requestPending||displayPending;
  useEffect(()=>{
    const computation=fullCaseComputation.current,before=computation.scenario;
    const slice=computation.claim(result,pending,enabled&&!displayFailed&&!result?.playback?.readOnly);
    if(before!==computation.scenario)setFullCaseVersion(n=>n+1);
    if(!slice)return;
    void run('fullCase',{scenario:slice.scenario,maxActions:slice.maxActions}).then(ok=>{computation.settled(slice.epoch,ok);setFullCaseVersion(n=>n+1);});
  },[fullCaseVersion,result,pending,enabled,displayFailed]);
  const volume=useMemo(()=>source&&desiredView?modeledVolume(source,desiredView):null,[source,desiredView]);
  const previewVolume=useMemo(()=>source&&activePreview?modeledVolume(source,activePreview):null,[source,activePreview]);
  const displayGeneration=generation.current,displaySource=source?.caseHash,displaySourceObject=source;
  const displayTicket=displayAdmission.current.expect(displaySource??'',displayGeneration,requestedView);
  const currentDisplay=(reported:SurgicalView|null)=>enabled&&generation.current===displayGeneration&&current.current.source===displaySourceObject&&
    desired.current.result?.caseHash===displaySource&&reported===(desired.current.preview??desired.current.result?.view);
  function onDisplayReady(reported:SurgicalView|null){
    if(!reported||!currentDisplay(reported))return;
    if(!displayAdmission.current.isCurrent(displayTicket,reported))return;
    current.current.displayFailed=false;setDisplayFailed(false);
    if(!displayAdmission.current.commit(displayTicket,reported))return;
    if(reported===desired.current.preview){setPreview(reported);setPreviewOutcome(desired.current.previewOutcome);if(desired.current.previewOutcome?.reason)setError(desired.current.previewOutcome.reason);return;}
    const next=desired.current.result!;setResult(next);setPreview(null);setPreviewOutcome(null);
    if(pendingPoseView.current===reported){
      pendingPoseView.current=null;setPath([]);
      const t=next.view.slots[current.current.slot];if(t){setRawPose({position:[...t.position],quaternion:[...t.quaternion]});setParameters({...t.parameters});}
    }
    if(!next.playback&&next.outcome?.reason)setError(`${next.outcome.status}: ${next.outcome.reason}`);
  }
  function onDisplayError(reported:SurgicalView|null,message:string){if(currentDisplay(reported)&&displayAdmission.current.isCurrent(displayTicket,reported)){current.current.displayFailed=true;setDisplayFailed(true);setError(message);}}
  function clearPreview(){setDesiredPreview(null);setDesiredPreviewOutcome(null);if(!current.current.displayFailed)setError(null);}
  function setPose(p:SurgicalPose){setRawPose(p);setPath([]);clearPreview();}
  function selectSlot(s:string){setSlot(s);clearPreview();setPath([]);const t=result?.view.slots[s];if(t){setRawPose({position:[...t.position],quaternion:[...t.quaternion]});setParameters({...t.parameters});}}
  function onStroke(point:Point3,phase:'start'|'move'|'end'){
    if(phase==='start')setPath([{position:point,quaternion:pose.quaternion}]);
    else setPath(p=>p.length<512?[...p,{position:point,quaternion:pose.quaternion}]:p);
    setRawPose({...pose,position:point});clearPreview();
  }
  function action(operation:string,params:Record<string,number>=parameters,duration=1):SurgicalAction{return {slot,operation,parameters:params,duration,...(['apply','move','hold','grasp','place'].includes(operation)?{path:path.length?path:[pose]}:{})};}
  return {result,view,volume,previewVolume,pending,pendingCommand,simulationPending:requestPending,displayPending,error,slot,pose,path,mode,preview:displayedPreview,previewOutcome,parameters,setPose,setMode,selectSlot,run,action,
    playbackVersion,pausePlayback:()=>setPlaybackVersion(n=>n+1),
    computingFullCase:fullCaseComputation.current.scenario,pauseFullCase,
    startFullCase:(scenario:'primary'|'vascular')=>{if(current.current.enabled&&!busy.current&&!current.current.result?.playback?.readOnly){fullCaseComputation.current.start(scenario);setFullCaseVersion(n=>n+1);}},
    setParameters:(p:Record<string,number>)=>{setParameters(p);clearPreview();},
    clearPreview,
    create:(mapping?:SurgicalRegion[])=>{const issue=mapping?validateMapping(mapping):null;if(issue){setError(issue);return Promise.resolve(false);}return run('create',mapping?{mapping}:{});},
    cancel:()=>{clearPreview();setPath([]);const t=result?.view.slots[slot];if(t){setRawPose({position:t.position,quaternion:t.quaternion});setParameters({...t.parameters});}},
    interaction:{view:desiredView,preview:activePreview,outcome:activePreview?desiredPreviewOutcome:desiredResult?.outcome??null,readOnly:!!desiredResult?.playback,effectiveActivation:desiredResult?.playback?.effectiveActivation,pose,path,parameters,selectedSlot:slot,mode:desiredResult?.playback?'navigate':mode,onPose:setPose,onStroke,onDisplayReady,onDisplayError} as SurgicalInteraction};
}
export type SurgicalController=ReturnType<typeof useSurgical>;
