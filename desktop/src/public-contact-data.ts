import {hydrateNativeEpisodeReplay} from './native-episode-replay.ts';
import {checkedPublicContactAuthority,checkedPublicContactAssets,contactNeed,contactSame,CONTACT_COSTS} from './public-contact-authority.ts';
import {arrayDigest,sha256Bytes} from './source-integrity.ts';
import {validateCaseDescriptor} from './case-data.ts';
import type {ViewerPublicGoal} from './viewer/publicGoal';
import type {PublicContactResult,PublicContactEpisode} from './public-contact-types.ts';
import type {ResectionApi} from './types';
const close=(a:unknown,b:number)=>typeof a==='number'&&Number.isFinite(a)&&Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(a),Math.abs(b));
export interface ContactPrefix {retained:boolean;contactedAndRetained:boolean;removedVolumeMm3:number}
export async function hydratePublicContactEpisode(raw:PublicContactResult,api:ResectionApi){
 const input=structuredClone(raw);contactNeed(input&&Object.keys(input).sort().join(',')==='case,episode,episodeCanonicalJson','unexpected response fields.');
 checkedPublicContactAssets(input);
 const e=input.episode,expected=checkedPublicContactAuthority(e),binding=e.taskContract.detachedObservationBinding;
 const digest=async(text:string)=>'sha256:'+await sha256Bytes(new TextEncoder().encode(text));
 contactNeed(await digest(expected.objective)===e.publicGoal.objectiveHash&&await digest(expected.declaration)===binding.declaration_hash,'public objective/declaration digest mismatch.');
 const g=e.publicGoal.nativeIndex,goalRas=e.affine.slice(0,3).map(row=>row.slice(0,3).reduce((s,n,i)=>s+n*g[i],row[3]));
 contactNeed(contactSame(e.publicGoal.rasMm,goalRas),'goal RAS marker differs from native source cell.');
 const grid=new Uint8Array(13*13*12),index=(g[0]*13+g[1])*12+g[2];grid[index]=1;
 contactNeed(await arrayDigest(grid,[13,13,12],'|b1')===e.publicGoal.goalGridHash,'one-hot goal differs from declared crop.');
 const geometry=await hydrateNativeEpisodeReplay(input,api,2);
 const {voxelVolumeMm3}=validateCaseDescriptor(input.case);
 const prefixes:ContactPrefix[]=[];let touched=false;
 for(const[i,f]of e.replayFrames.entries()){
  if(f.probeContactIndicesNative.some(cell=>contactSame(cell,g)))touched=true;
  const frame=geometry.frames[i],retained=frame.removedMask[index]===0;
  prefixes.push({retained,contactedAndRetained:retained&&touched,removedVolumeMm3:frame.removedTargetVolumeMm3+frame.removedNormalVolumeMm3});
 }
 let before=0,contacted=false,retained=true,removed=0,reward=0,path=0,cost=0,priorTool:string|undefined;
 for(const [actionIndex,h] of e.history.entries()){
  const cellKey=(cell:number[])=>cell.join(',');
  const committedProbe=[...new Set(e.replayFrames.filter(f=>f.actionIndex===actionIndex&&f.phase==='insertion').flatMap(f=>f.probeContactIndicesNative.map(cellKey)))].sort();
  const macroProbe=h.probe_contact_indices_native.map(cellKey).sort();
  contactNeed(contactSame(macroProbe,committedProbe),'macro probe contact differs from retained committed insertion contact.');
  const stop=h.interaction_mode==='stop',volume=h.removed_indices_native.length*voxelVolumeMm3;
  if(h.removed_indices_native.some(cell=>contactSame(cell,g)))retained=false;
  if(h.interaction_mode==='probe'&&h.probe_contact_indices_native.some(cell=>contactSame(cell,g)))contacted=true;
  const after=Number(retained&&contacted),distance=stop?0:Math.hypot(...h.tip_mm!.map((n,i)=>n-h.entry_mm![i]));
  const change=Number(!stop&&priorTool!==undefined&&priorTool!==h.tool_id),charge=CONTACT_COSTS.normal_per_mm3*volume+CONTACT_COSTS.action_cost*Number(!stop)+2*CONTACT_COSTS.motion_per_mm*distance+CONTACT_COSTS.tool_change_cost*change;
  contactNeed(h.goal_potential_before===before&&h.goal_potential_after===after&&h.public_objective_hash===e.publicGoal.objectiveHash&&h.clinical_deficit_probability===null&&h.outcome_scope==='public_geometric_retained_surface_contact'&&close(h.removal_cost_volume_mm3,volume)&&close(h.insertion_distance_mm,distance)&&close(h.complete_tool_path_length_mm,2*distance)&&h.tool_change_count===change&&close(h.effort_and_removal_cost,charge)&&close(h.reward,after-before-charge),'goal potential, effort or reward differs from executed geometry.');
  removed+=volume;reward+=h.reward;path+=2*distance;cost+=charge;before=after;if(!stop)priorTool=h.tool_id;
 }
 const final=prefixes.at(-1)!;
 contactNeed(retained===final.retained&&Boolean(before)===final.contactedAndRetained&&e.metrics.goal_retained===final.retained&&e.metrics.goal_contacted_and_retained===final.contactedAndRetained&&close(e.metrics.removed_volume_mm3,removed)&&close(e.metrics.total_reward,reward)&&close(final.removedVolumeMm3,removed),'final contact/removal metrics differ from replay.');
 return {...geometry,episode:e,prefixes,outcome:{modeledReturn:reward,totalRemovedMm3:removed,completePathMm:path,totalCost:cost,retained:final.retained,contactedAndRetained:final.contactedAndRetained,termination:e.history.at(-1)!.interaction_mode==='stop'?'STOP' as const:'horizon' as const}};
}
export type PublicContactView=Awaited<ReturnType<typeof hydratePublicContactEpisode>>;
export function requirePersistableContact(view:PublicContactView|null){if(view)throw Error('This public-contact episode is not saved in workspaces yet. Open or execute another case before saving.');}
export function contactGoalMarker(view:PublicContactView,step:number):ViewerPublicGoal{
 const goal=view.episode.publicGoal,prefix=view.prefixes[step];
 contactNeed(prefix,'invalid goal replay frame.');
 return {scope:'declared-public-geometric-goal',caseHash:view.source.caseHash,episodeId:view.episode.episodeId,objectiveHash:goal.objectiveHash,
  nativeIndex:[...goal.nativeIndex],rasMm:[...goal.rasMm],state:!prefix.retained?'removed':prefix.contactedAndRetained?'retained-contacted':'retained-uncontacted'};
}
