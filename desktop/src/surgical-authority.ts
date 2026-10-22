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
  for(const requested of [p.requestedPosition,response.requestedPosition])need(requested===undefined||(Number.isInteger(requested)&&requested>=0&&requested<=p.totalActions));
  for(const stopped of [p.budgetStopped,response.budgetStopped])need(stopped===undefined||typeof stopped==='boolean');
  need(p.requestedPosition===undefined||response.requestedPosition===undefined||p.requestedPosition===response.requestedPosition);
  need(p.budgetStopped===undefined||response.budgetStopped===undefined||p.budgetStopped===response.budgetStopped);return p;
}
/** A compact receipt still carries complete prefix authority and progress. */
export function checkedPrefixWireResponse(response:SurgicalResponse){return checkedPrefix(response,true);}
/** Demand hydration must restore the complete computed prefix before display. */
export function checkedPrefixResponse(response:SurgicalResponse){return checkedPrefix(response,false);}
export function replayAuthorityKey(response:SurgicalResponse){
  const p=reconstructedPrefix(response);
  return p?JSON.stringify(['prefix',p.originSnapshotHash,p.actionLogHash,p.implementation,p.protectionContractHash,p.prefixImplementation]):response.verification?.readOnly?'unverified-import':'canonical';
}
