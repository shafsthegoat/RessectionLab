import {useEffect,useMemo,useRef,useState} from 'react';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {SurgicalRequestQueue} from './surgical-request-queue.ts';
import {SurgicalDisplayAdmission} from './surgical-display-admission.ts';
import {FullCaseComputation} from './full-case-computation.ts';
import {checkedPrefixResponse,checkedPrefixWireResponse,frameCommand,prefixCommand,reconstructedPrefix,replayAuthorityKey,surgicalAuthorityIssue} from './surgical-authority.ts';
import {SurgicalTimelineStream} from './surgical-timeline-stream.ts';
import {SurgicalReplayPrefetch} from './surgical-replay-prefetch.ts';
import type {SurgicalReplaySelection} from './surgical-replay-prefetch';
import {checkedSurgicalPlayback,checkedSurgicalPlaybackWire,nextSurgicalPlaybackFrame,surgicalCursorCommand,updatesEditableSurgicalCursor} from './surgical-playback.ts';
import {modeledVolume} from './modeled-volume.ts';
export {modeledVolume} from './modeled-volume.ts';
import type {SurgicalAction,SurgicalOutcome,SurgicalApi,SurgicalPose,SurgicalRegion,SurgicalResponse,SurgicalView,SurgicalInteraction} from './surgical-types';

export function validateMapping(regions:SurgicalRegion[]):string|null {
  if(!regions.length)return 'Map at least one source region.';
  if(new Set(regions.map(r=>r.id)).size!==regions.length||new Set(regions.map(r=>r.name)).size!==regions.length)return 'Region IDs and names must be unique.';
  if(regions.some(r=>!Number.isInteger(r.id)||r.id<1||r.id>65535||!r.name||!r.material||typeof r.target!=='boolean'))return 'Each region needs a positive ID, name, material, and target choice.';
  return null;
}

export function useSurgical(source:ViewerVolume|null,api:Partial<SurgicalApi>|null,enabled:boolean,initialArchivePreview=false){
  const [result,setResult]=useState<SurgicalResponse|null>(null),[requestPending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
  const [replayRequestPending,setReplayPending]=useState(false);
  const replayPendingLease=useRef<number|null>(null);
  const [desiredResult,setDesiredResult]=useState<SurgicalResponse|null>(null),[desiredPreview,setDesiredPreview]=useState<SurgicalView|null>(null);
  const [desiredPreviewOutcome,setDesiredPreviewOutcome]=useState<SurgicalOutcome|null>(null),[displayFailed,setDisplayFailed]=useState(false);
  const [slot,setSlot]=useState('left'),[pose,setRawPose]=useState<SurgicalPose>({position:[0,0,0],quaternion:[0,0,0,1]}),[path,setPath]=useState<SurgicalPose[]>([]);
  const [parameters,setParameters]=useState<Record<string,number>>({});
  const [showFlowTransfers,setShowFlowTransfers]=useState(true);
  const [mode,setMode]=useState<SurgicalInteraction['mode']>('navigate'),[preview,setPreview]=useState<SurgicalView|null>(null);
  const [previewOutcome,setPreviewOutcome]=useState<SurgicalOutcome|null>(null);
  const [pendingCommand,setPendingCommand]=useState<string|null>(null);
  const backendResult=useRef<SurgicalResponse|null>(null);
  const timelineStream=useRef(new SurgicalTimelineStream());
  const responseSource=useRef<ViewerVolume|null>(null);
  const acceptedResponseSource=useRef<ViewerVolume|null>(null);
  const displayAdmission=useRef(new SurgicalDisplayAdmission<SurgicalView>());
  const pendingPoseView=useRef<SurgicalView|null>(null);
  const fullCaseComputation=useRef(new FullCaseComputation());
  const [fullCaseVersion,setFullCaseVersion]=useState(0);
  const [playbackVersion,setPlaybackVersion]=useState(0);
  function pauseFullCase(){if(fullCaseComputation.current.scenario){fullCaseComputation.current.pause();setFullCaseVersion(n=>n+1);}}
  const generation=useRef(0),busy=useRef(false),current=useRef({source,api,result,slot,enabled,displayFailed});current.current={source,api,result,slot,enabled,displayFailed};
  const manualRequests=useRef(0);
  const [manualPending,setManualPending]=useState(0);
  const replayOwner=useRef<{source:ViewerVolume;generation:number;branchId:string;authority:string;prefix:boolean}|null>(null);
  const replayLane=useRef<SurgicalReplayPrefetch|null>(null);
  if(!replayLane.current)replayLane.current=new SurgicalReplayPrefetch(async selection=>{
    const c=current.current,owner=replayOwner.current;
    if(!owner||owner.source!==c.source||owner.generation!==generation.current||!c.enabled||!c.api?.surgicalRehearsal)throw Error('Recorded replay source is unavailable');
    if(!c.result||c.result.branchId!==owner.branchId||selection.branchId!==owner.branchId||replayAuthorityKey(c.result)!==owner.authority)throw Error('Recorded replay authority changed');
    const token=owner.generation,source=owner.source;
    const timelineToken=timelineStream.current.token(source.caseHash,selection.branchId);
    const next=await c.api.surgicalRehearsal({caseHash:source.caseHash,command:owner.prefix?'seekPrefixFrame':'seekFrame',...selection,
      ...(owner.prefix?{prefixToken:timelineToken,prefixFieldTokens:timelineStream.current.fieldTokens(source.caseHash,selection.branchId,timelineToken)}:{expectedStateId:backendResult.current?.view.stateId,timelineToken,viewFieldTokens:timelineStream.current.fieldTokens(source.caseHash,selection.branchId,timelineToken)})});
    if(token!==generation.current||!current.current.enabled||current.current.source!==source||replayOwner.current!==owner)throw Error('Recorded replay source changed');
    if(!next||next.caseHash!==source.caseHash||next.branchId!==selection.branchId)throw Error('Recorded replay response differs from its source');
    if(owner.prefix)checkedPrefixWireResponse(next);
    if(replayAuthorityKey(next)!==owner.authority)throw Error('Recorded replay response changed authority');
    checkedSurgicalPlaybackWire(next,selection,{caseHash:source.caseHash,branchId:owner.branchId,timelineToken,scene:c.result.view.scene});return next;
  });
  function stopPlayback(){replayOwner.current=null;replayLane.current!.stop();}
  const desired=useRef({result:desiredResult,preview:desiredPreview,previewOutcome:desiredPreviewOutcome});desired.current={result:desiredResult,preview:desiredPreview,previewOutcome:desiredPreviewOutcome};
  const requestQueue=useRef<SurgicalRequestQueue<{command:string;extra:Record<string,unknown>;caseHash:string;source:ViewerVolume;generation:number}>|null>(null);
  useEffect(()=>{stopPlayback();replayPendingLease.current=null;setReplayPending(false);timelineStream.current.clear();pauseFullCase();generation.current++;requestQueue.current?.invalidate();busy.current=false;backendResult.current=null;acceptedResponseSource.current=null;displayAdmission.current.invalidate();pendingPoseView.current=null;setResult(null);setDesiredResult(null);setDesiredPreview(null);setDesiredPreviewOutcome(null);setDisplayFailed(false);setPreview(null);setPreviewOutcome(null);setPath([]);setMode('navigate');setSlot('left');setPending(false);setError(null);},[source]);
  useEffect(()=>()=>{stopPlayback();fullCaseComputation.current.pause();generation.current++;requestQueue.current?.invalidate();},[]);
  async function execute(command:string,extra:Record<string,unknown>,isCurrent:()=>boolean):Promise<boolean>{
    const c=current.current;if(!c.enabled||!c.source||!c.api?.surgicalRehearsal||busy.current)return false;
    if(prefixCommand(command)&&(responseSource.current!==c.source||acceptedResponseSource.current!==c.source||c.result?.caseHash!==c.source.caseHash)){setError('Inspect this source checkpoint before reconstructing a read-only prefix.');return false;}
    const authorityIssue=surgicalAuthorityIssue(c.result,command,extra);if(authorityIssue){setError(authorityIssue);return false;}
    if(c.result?.playback?.readOnly&&!surgicalCursorCommand(command)){setError('Return to a recorded action boundary before applying or branching instrument actions.');return false;}
    if(c.displayFailed&&['step','preview','plan','fullCase'].includes(command)){setError('The accepted anatomy could not be displayed. Seek to a recorded state or reopen the case before applying another instrument action.');return false;}
    const prefixRestart=command==='seekPrefix'&&extra.position===0;
    const token=generation.current;busy.current=true;setPending(true);setPendingCommand(command);setError(null);
    try{
      if(prefixRestart)timelineStream.current.clear();
      const branchId=String(extra.branchId??c.result?.branchId??''),timelineToken=frameCommand(command)?timelineStream.current.token(c.source.caseHash,branchId):undefined;
      const requestExtra=prefixCommand(command)?{branchId,position:extra.position,...(prefixRestart&&Object.hasOwn(extra,'cacheOnly')?{cacheOnly:extra.cacheOnly as boolean}:{}),...(command==='seekPrefixFrame'?{frameIndex:extra.frameIndex,prefixToken:timelineToken,prefixFieldTokens:timelineStream.current.fieldTokens(c.source.caseHash,branchId,timelineToken)}:{})}:extra;
      let next=await c.api.surgicalRehearsal({caseHash:c.source.caseHash,command,...(!prefixCommand(command)?{expectedStateId:backendResult.current?.view.stateId}:{}),...requestExtra,
        ...(command==='seekFrame'?{timelineToken,viewFieldTokens:timelineStream.current.fieldTokens(c.source.caseHash,branchId,timelineToken)}:{})});
      if(token!==generation.current||!current.current.enabled||current.current.source!==c.source)return false;
      if(next){if(prefixCommand(command)){
        (command==='seekPrefixFrame'?checkedPrefixWireResponse:checkedPrefixResponse)(next);
        if(next.caseHash!==c.source.caseHash||next.branchId!==branchId||(reconstructedPrefix(c.result)&&replayAuthorityKey(next,!prefixRestart)!==replayAuthorityKey(c.result!,!prefixRestart)))throw Error('Reconstructed prefix response changed source or authority');
        if(prefixRestart&&((reconstructedPrefix(next)?.preparedCache?.mode==='only')!==(extra.cacheOnly===true)))throw Error('Reconstructed prefix response changed prepared-only mode');
      }next=timelineStream.current.hydrate(next);if(prefixCommand(command))checkedPrefixResponse(next);}
      // A superseded seek still moved the backend cursor. Preserve its state
      // identity for the next ordered request while withholding obsolete UI.
      if(frameCommand(command)&&next){checkedSurgicalPlayback(next,{...extra,branchId});setMode('navigate');setPath([]);}
      if(updatesEditableSurgicalCursor(command)&&!next?.verification?.readOnly)backendResult.current=next;
      if(!isCurrent())return false;
      setDisplayFailed(false);
      if(command==='preview'){const response=next as unknown as (SurgicalOutcome&{view:SurgicalView})|null;setDesiredPreview(response?.view??null);setDesiredPreviewOutcome(response?{status:response.status,reason:response.reason,completedFraction:response.completedFraction,events:response.events}:null);return !!response;}
      setDesiredPreview(null);setDesiredPreviewOutcome(null);pendingPoseView.current=next?.view??null;responseSource.current=c.source;setDesiredResult(next);
      return !!next;
    }catch(e){if(token===generation.current){timelineStream.current.clear();setError(String(e));}return false;}finally{if(token===generation.current){busy.current=false;setPending(false);setPendingCommand(null);}}
  }
  if(!requestQueue.current)requestQueue.current=new SurgicalRequestQueue(async(item,isCurrent)=>{
    if(item.generation!==generation.current||current.current.source!==item.source)return false;
    return execute(item.command,item.extra,isCurrent);
  },item=>['seek','seekFrame','seekPrefix','seekPrefixFrame'].includes(item.command));
  async function run(command:string,extra:Record<string,unknown>={}):Promise<boolean>{
    if(command!=='fullCase')pauseFullCase();
    const c=current.current;if(!c.enabled||!c.source||!c.api?.surgicalRehearsal)return false;
    const token=generation.current;manualRequests.current++;setManualPending(n=>n+1);setPlaybackVersion(n=>n+1);
    replayOwner.current=null;const barrier=replayLane.current!.barrier();
    try{
      if(barrier)await barrier;
      if(token!==generation.current||current.current.source!==c.source||!current.current.enabled)return false;
      return await requestQueue.current!.request({command,extra:structuredClone(extra),caseHash:c.source.caseHash,source:c.source,generation:token});
    }catch(cause){if(current.current.source===c.source)setError(String(cause));return false;}
    finally{manualRequests.current--;setManualPending(n=>n-1);}
  }
  function startPlayback(){
    const c=current.current;
    if(!c.enabled||!c.source||responseSource.current!==c.source||!c.result||busy.current||manualRequests.current||c.displayFailed||(c.result.verification?.readOnly&&!reconstructedPrefix(c.result))||desired.current.preview||desired.current.result!==c.result)return false;
    if(replayLane.current!.active)return replayOwner.current?.source===c.source&&replayOwner.current.generation===generation.current&&replayOwner.current.branchId===c.result.branchId&&replayOwner.current.authority===replayAuthorityKey(c.result);
    pauseFullCase();replayOwner.current={source:c.source,generation:generation.current,branchId:c.result.branchId,authority:replayAuthorityKey(c.result),prefix:!!reconstructedPrefix(c.result)};replayLane.current!.start();return true;
  }
  async function runPlayback(selection:SurgicalReplaySelection){
    const c=current.current,token=generation.current;
    if(!replayLane.current!.active||!c.enabled||!c.source||busy.current||manualRequests.current||c.displayFailed||(c.result?.verification?.readOnly&&!reconstructedPrefix(c.result))||desired.current.preview||desired.current.result!==c.result)return false;
    const expected=c.result?nextSurgicalPlaybackFrame(c.result,c.result.branches.find(branch=>branch.id===c.result!.branchId)?.steps??0):null;
    if(!expected||selection.branchId!==expected.branchId||selection.position!==expected.position||selection.frameIndex!==expected.frameIndex){stopPlayback();setError('Automatic replay must admit its exact next observation.');return false;}
    const lane=replayLane.current!,lease=lane.generation;replayPendingLease.current=lease;
    setReplayPending(true);setError(null);
    try{
      const receipt=await lane.take(selection);
      if(!receipt||!lane.isCurrent(lease)||token!==generation.current||current.current.source!==c.source||!current.current.enabled)return false;
      const next=timelineStream.current.hydrate(receipt);if(reconstructedPrefix(c.result))checkedPrefixResponse(next);checkedSurgicalPlayback(next,selection);
      // Only a demanded receipt enters React/model preparation. The prefetched
      // successor remains raw wire data until this exact view is acknowledged.
      setDisplayFailed(false);setMode('navigate');setPath([]);setDesiredPreview(null);setDesiredPreviewOutcome(null);
      pendingPoseView.current=next.view;responseSource.current=c.source;setDesiredResult(next);
      const steps=next.branches.find(branch=>branch.id===next.branchId)?.steps??0,successor=nextSurgicalPlaybackFrame(next,steps);
      if(successor)replayLane.current!.prefetch(successor);
      return true;
    }catch(cause){
      if(lane.isCurrent(lease)){stopPlayback();
        if(token===generation.current&&current.current.source===c.source){timelineStream.current.clear();setError(String(cause));}}
      return false;
    }finally{if(token===generation.current&&replayPendingLease.current===lease){replayPendingLease.current=null;setReplayPending(false);}}
  }
  useEffect(()=>{if(enabled&&source&&api?.surgicalRehearsal)void run(initialArchivePreview?'inspectArchive':'inspect');else {stopPlayback();replayPendingLease.current=null;setReplayPending(false);generation.current++;requestQueue.current?.invalidate();busy.current=false;displayAdmission.current.invalidate();setResult(null);setPending(false);setDesiredPreview(null);setDesiredPreviewOutcome(null);setPreview(null);setPreviewOutcome(null);}},[source,enabled,api?.surgicalRehearsal,initialArchivePreview]);
  const view=enabled&&acceptedResponseSource.current===source&&result?.caseHash===source?.caseHash?result?.view??null:null;
  const desiredView=enabled&&responseSource.current===source&&desiredResult&&desiredResult.caseHash===source?.caseHash?desiredResult.view:null;
  const activePreview=desiredView?desiredPreview:null,displayedPreview=view?preview:null;
  const requestedView=activePreview??desiredView,displayedView=displayedPreview??view;
  const displayPending=!displayFailed&&!!requestedView&&requestedView!==displayedView;
  const pending=requestPending||replayRequestPending||displayPending||manualPending>0;
  useEffect(()=>{
    const computation=fullCaseComputation.current,before=computation.scenario;
    const slice=computation.claim(result,pending,enabled&&!displayFailed&&!result?.playback?.readOnly&&!result?.verification?.readOnly);
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
    const next=desired.current.result!;acceptedResponseSource.current=displaySourceObject;setResult(next);setPreview(null);setPreviewOutcome(null);
    if(pendingPoseView.current===reported){
      pendingPoseView.current=null;setPath([]);
      const t=next.view.slots[current.current.slot];if(t){setRawPose({position:[...t.position],quaternion:[...t.quaternion]});setParameters({...t.parameters});}
    }
    if(!next.playback&&next.outcome?.reason)setError(`${next.outcome.status}: ${next.outcome.reason}`);
  }
  function onDisplayError(reported:SurgicalView|null,message:string){if(currentDisplay(reported)&&displayAdmission.current.isCurrent(displayTicket,reported)){stopPlayback();current.current.displayFailed=true;setDisplayFailed(true);setError(message);}}
  function clearPreview(){setDesiredPreview(null);setDesiredPreviewOutcome(null);if(!current.current.displayFailed)setError(null);}
  function setPose(p:SurgicalPose){setRawPose(p);setPath([]);clearPreview();}
  function selectSlot(s:string){setSlot(s);clearPreview();setPath([]);const t=result?.view.slots[s];if(t){setRawPose({position:[...t.position],quaternion:[...t.quaternion]});setParameters({...t.parameters});}}
  function onStroke(point:Point3,phase:'start'|'move'|'end'){
    if(phase==='start')setPath([{position:point,quaternion:pose.quaternion}]);
    else setPath(p=>p.length<512?[...p,{position:point,quaternion:pose.quaternion}]:p);
    setRawPose({...pose,position:point});clearPreview();
  }
  function action(operation:string,params:Record<string,number>=parameters,duration=1):SurgicalAction{return {slot,operation,parameters:params,duration,...(['apply','move','hold','grasp','place'].includes(operation)?{path:path.length?path:[pose]}:{})};}
  return {result,view,volume,previewVolume,pending,pendingCommand:requestPending?pendingCommand:replayRequestPending?(replayOwner.current?.prefix?'seekPrefixFrame':'seekFrame'):pendingCommand,simulationPending:requestPending||replayRequestPending,displayPending,error,slot,pose,path,mode,preview:displayedPreview,previewOutcome,parameters,setPose,setMode,selectSlot,run,action,
    playbackVersion,startPlayback,runPlayback,stopPlayback,showFlowTransfers,setShowFlowTransfers,
    pausePlayback:()=>{stopPlayback();setPlaybackVersion(n=>n+1);},
    computingFullCase:fullCaseComputation.current.scenario,pauseFullCase,
    startFullCase:(scenario:'primary'|'vascular')=>{if(current.current.enabled&&!busy.current&&!current.current.result?.playback?.readOnly&&!current.current.result?.verification?.readOnly){fullCaseComputation.current.start(scenario);setFullCaseVersion(n=>n+1);}},
    setParameters:(p:Record<string,number>)=>{setParameters(p);clearPreview();},
    clearPreview,
    create:(mapping?:SurgicalRegion[])=>{const issue=mapping?validateMapping(mapping):null;if(issue){setError(issue);return Promise.resolve(false);}return run('create',mapping?{mapping}:{});},
    cancel:()=>{clearPreview();setPath([]);const t=result?.view.slots[slot];if(t){setRawPose({position:t.position,quaternion:t.quaternion});setParameters({...t.parameters});}},
    interaction:{view:desiredView,preview:activePreview,recordedFrame:!activePreview&&desiredResult?.playback?{branchId:desiredResult.playback.branchId,position:desiredResult.playback.position,frameIndex:desiredResult.playback.frameIndex}:undefined,flowDisplay:!activePreview&&showFlowTransfers?desiredResult?.playback?.flowDisplay:undefined,outcome:activePreview?desiredPreviewOutcome:desiredResult?.outcome??null,readOnly:!!desiredResult?.playback||!!desiredResult?.verification?.readOnly,effectiveActivation:desiredResult?.playback?.effectiveActivation,pose,path,parameters,selectedSlot:slot,mode:desiredResult?.playback||desiredResult?.verification?.readOnly?'navigate':mode,onPose:setPose,onStroke,onDisplayReady,onDisplayError} as SurgicalInteraction};
}
export type SurgicalController=ReturnType<typeof useSurgical>;
