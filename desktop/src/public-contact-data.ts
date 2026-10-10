import {hydrateNativeEpisodeReplay} from './native-episode-replay.ts';
import {checkedPublicContactAuthority,checkedPublicContactAssets,contactNeed,contactSame} from './public-contact-authority.ts';
import {arrayDigest,sha256Bytes} from './source-integrity.ts';
import {validateCaseDescriptor} from './case-data.ts';
import {checkedNativeContactOutcome} from './native-contact-outcome.ts';
import type {ContactPrefix} from './native-contact-outcome.ts';
import type {ViewerPublicGoal} from './viewer/publicGoal.ts';
import type {PublicContactResult} from './public-contact-types.ts';
import type {ResectionApi,CasePayload,Vec3} from './types.ts';
export type {ContactPrefix} from './native-contact-outcome.ts';
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
 return {...geometry,episode:e,...checkedNativeContactOutcome(e,geometry.frames,voxelVolumeMm3)};
}
export type PublicContactView=Awaited<ReturnType<typeof hydratePublicContactEpisode>>;
export function requirePersistableContact(view:unknown|null){if(view)throw Error('This public-contact episode is not saved in workspaces yet. Open or execute another case before saving.');}
interface GoalMarkerView {
 source:CasePayload;
 episode:{episodeId:string;publicGoal:{nativeIndex:Vec3;rasMm:Vec3;objectiveHash:string}};
 prefixes:ContactPrefix[];
}
export function contactGoalMarker(view:GoalMarkerView,step:number):ViewerPublicGoal{
 const goal=view.episode.publicGoal,prefix=view.prefixes[step];
 contactNeed(prefix,'invalid goal replay frame.');
 return {scope:'declared-public-geometric-goal',caseHash:view.source.caseHash,episodeId:view.episode.episodeId,objectiveHash:goal.objectiveHash,
  nativeIndex:[...goal.nativeIndex],rasMm:[...goal.rasMm],state:!prefix.retained?'removed':prefix.contactedAndRetained?'retained-contacted':'retained-uncontacted'};
}
