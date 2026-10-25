import type {SurgicalResponse,SurgicalPrefixVerification} from './surgical-types';
export const prefixCommand=(command:string)=>command==='seekPrefix'||command==='seekPrefixFrame';
export const frameCommand=(command:string)=>command==='seekFrame'||command==='seekPrefixFrame';
export function reconstructedPrefix(response:SurgicalResponse|null|undefined):SurgicalPrefixVerification|null {
  return response?.verification?.status==='reconstructed-prefix'?response.verification:null;
}
export function surgicalReplayRoutes(response:SurgicalResponse|null|undefined){
  const prefix=!!reconstructedPrefix(response),imported=!!response?.verification?.readOnly;
  return {prefix,imported,canReplay:!imported||prefix,boundary:prefix?'seekPrefix':'seek',frame:prefix?'seekPrefixFrame':'seekFrame',branch:imported?'inspectArchive':'selectBranch'};
}
/** Explicit authority domains: a reconstructed prefix grants inspection only. */
export function surgicalAuthorityIssue(response:SurgicalResponse|null,command:string,extra:Record<string,unknown>):string|null {
  if(Object.hasOwn(extra,'cacheOnly')&&(typeof extra.cacheOnly!=='boolean'||command!=='seekPrefix'||extra.position!==0))return 'Prepared-only mode can be selected only when starting read-only replay at the beginning.';
  const prefix=reconstructedPrefix(response),verification=response?.verification;
  if(verification?.readOnly){
    if(command==='inspectArchive'||command==='inspect')return null;
    const branch=extra.branchId??response!.branchId;
    if(branch!==response!.branchId)return 'Open the branch as an imported checkpoint before reconstructing its prefix.';
    if(!prefix){
      if(command==='seekPrefix'&&extra.position===0)return null;
      return 'Start read-only reconstruction from the archive beginning or verify the archive before replay and planning.';
    }
    if(prefixCommand(command)){
      if(!Number.isInteger(extra.position)||(extra.position as number)<0||(extra.position as number)>prefix.totalActions||
        (command==='seekPrefixFrame'&&((extra.position as number)>=prefix.totalActions||!Number.isInteger(extra.frameIndex)||(extra.frameIndex as number)<0)))return 'Choose a valid recorded prefix position.';
      return null;
    }
    return 'Reconstructed prefixes are read only. Verify the full archive before planning or instrument actions.';
  }
  return prefixCommand(command)?'Open an imported checkpoint before starting read-only prefix reconstruction.':null;
}
function checkedPrefix(response:SurgicalResponse,wire:boolean){
  const p=reconstructedPrefix(response),need=(condition:unknown)=>{if(!condition)throw Error('Reconstructed prefix authority or progress is unavailable.');};
  need(p?.readOnly===true);if(!p)throw Error('Reconstructed prefix authority is unavailable.');
  need(Number.isInteger(p.verifiedThrough)&&Number.isInteger(p.totalActions)&&p.verifiedThrough>=0&&p.verifiedThrough<=p.totalActions);
  need(response.branches.find(branch=>branch.id===response.branchId)?.steps===p.totalActions);
  need(Number.isInteger(response.position)&&response.position>=0&&response.position<=p.verifiedThrough);
  if(wire)need(response.timelineOmitted===undefined||typeof response.timelineOmitted==='boolean');
  if(wire&&response.timelineOmitted===true)need(typeof response.timelineToken==='string'&&!!response.timelineToken&&!Object.hasOwn(response,'timeline'));
  else need(Array.isArray(response.timeline)&&response.timeline.length===p.verifiedThrough);
  need(typeof p.archiveEndpointVerified==='boolean'&&typeof p.originSnapshotHash==='string'&&!!p.originSnapshotHash&&typeof p.actionLogHash==='string'&&!!p.actionLogHash);
  need(typeof p.implementation==='string'||(!!p.implementation&&typeof p.implementation==='object'&&!Array.isArray(p.implementation)));
  if(p.preparedCache!==undefined){
    need(!!p.preparedCache&&typeof p.preparedCache==='object'&&!Array.isArray(p.preparedCache));
    need(['prepare','only','disabled'].includes(p.preparedCache.mode));
    for(const count of [p.preparedCache.loadedActions,p.preparedCache.publishedActions,p.preparedCache.misses])need(Number.isSafeInteger(count)&&count>=0);
    need(p.preparedCache.lastSkip===null||typeof p.preparedCache.lastSkip==='string');
  }
  if(response.prefixOrigin!==undefined){
    const origin=response.prefixOrigin,keys=['schema','kind','activationId','caseHash','branchId','sourceArchiveSha256','sceneId'];
    need(!!origin&&typeof origin==='object'&&!Array.isArray(origin)&&Object.keys(origin).length===keys.length&&keys.every(key=>Object.hasOwn(origin,key)));
    need(origin.schema==='resectionlab.prefix-origin.v1'&&origin.kind==='fresh-reconstruction'&&
      typeof origin.activationId==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(origin.activationId)&&
      typeof origin.sourceArchiveSha256==='string'&&/^[0-9a-f]{64}$/.test(origin.sourceArchiveSha256)&&
      typeof origin.sceneId==='string'&&/^sha256:[0-9a-f]{64}$/.test(origin.sceneId)&&
      typeof origin.caseHash==='string'&&origin.caseHash===response.caseHash&&origin.caseHash.length>0&&origin.caseHash.length<=1024&&
      typeof origin.branchId==='string'&&origin.branchId===response.branchId&&origin.branchId.length>0&&origin.branchId.length<=128);
    need(response.readOnly===true&&response.outcome===null&&response.fullCase==null&&response.budgetStopped===false&&p.failure==null&&
      p.preparedCache?.mode==='only'&&p.preparedCache.loadedActions===p.verifiedThrough&&p.preparedCache.publishedActions===0&&
      p.preparedCache.misses===0&&p.preparedCache.lastSkip===null);
    if(!wire||Object.hasOwn(response.view,'scene'))need(origin.sceneId===response.view.scene?.sceneId);
  }
  for(const requested of [p.requestedPosition,response.requestedPosition])need(requested===undefined||(Number.isInteger(requested)&&requested>=0&&requested<=p.totalActions));
  for(const stopped of [p.budgetStopped,response.budgetStopped])need(stopped===undefined||typeof stopped==='boolean');
  need(p.requestedPosition===undefined||response.requestedPosition===undefined||p.requestedPosition===response.requestedPosition);
  need(p.budgetStopped===undefined||response.budgetStopped===undefined||p.budgetStopped===response.budgetStopped);return p;
}
/** A compact receipt still carries complete prefix authority and progress. */
export function checkedPrefixWireResponse(response:SurgicalResponse){return checkedPrefix(response,true);}
/** Demand hydration must restore the complete computed prefix before display. */
export function checkedPrefixResponse(response:SurgicalResponse){return checkedPrefix(response,false);}
/** The explicit start chooses preparation origin, while cache counters stay real. */
export function checkedPrefixStartMode(response:SurgicalResponse,cacheOnly:boolean){
  const p=checkedPrefixResponse(response),only=p.preparedCache?.mode==='only',fresh=!!response.prefixOrigin;
  if(cacheOnly?(!only||fresh):(only!==fresh))throw Error('Reconstructed prefix response changed prepared-only mode');
  return p;
}
export function replayAuthorityKey(response:SurgicalResponse,includePreparedMode=true){
  const p=reconstructedPrefix(response);
  const o=response.prefixOrigin,origin=o?[o.schema,o.kind,o.activationId,o.caseHash,o.branchId,o.sourceArchiveSha256,o.sceneId]:null;
  return p?JSON.stringify(['prefix',p.originSnapshotHash,p.actionLogHash,p.implementation,p.protectionContractHash,p.prefixImplementation,...(includePreparedMode?[p.preparedCache?.mode??'disabled',origin]:[])]):response.verification?.readOnly?'unverified-import':'canonical';
}
