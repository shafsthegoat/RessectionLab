import {hydrateDevelopmentEpisode} from './episode-data.ts';
import type {EpisodeView} from './episode-data.ts';
import type {ResectionApi} from './types';
import type {EpisodeComparisonResult} from './episode-comparison-types';
const HASH=/^sha256:[a-f0-9]{64}$/;
const stable=(v:unknown):unknown=>Array.isArray(v)?v.map(stable):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b)).map(([k,x])=>[k,stable(x)])):v;
const same=(a:unknown,b:unknown)=>JSON.stringify(stable(a))===JSON.stringify(stable(b));
function need(v:unknown,m:string):asserts v{if(!v)throw Error(`Matched comparison withheld: ${m}`)}
export interface EpisodeOutcome {modeledReturn:number;targetRemoved:number;normalRemoved:number;actions:number;termination:'STOP'|'horizon'}
export interface EpisodeComparisonView {actor:EpisodeView;companion:EpisodeView;actorOutcome:EpisodeOutcome;searchOutcome:EpisodeOutcome;binding:EpisodeComparisonResult}
export function checkedEpisodeOutcome(view:EpisodeView):EpisodeOutcome{
 const e=view.episode,m=e.metrics,last=view.frames.at(-1)!;
 const values=[m.total_reward,m.target_removed_mm3,m.normal_removed_mm3,...e.history.map(h=>h.reward)];
 need(values.every(n=>typeof n==='number'&&Number.isFinite(n)),'nonfinite modeled outcome.');
 const close=(a:number,b:number)=>Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(a),Math.abs(b));
 need(close(Number(m.total_reward),e.history.reduce((sum,h)=>sum+h.reward,0))&&close(Number(m.target_removed_mm3),last.removedTargetVolumeMm3)&&close(Number(m.normal_removed_mm3),last.removedNormalVolumeMm3),'outcome differs from checked history/removal masks.');
 return {modeledReturn:Number(m.total_reward),targetRemoved:last.removedTargetVolumeMm3,normalRemoved:last.removedNormalVolumeMm3,actions:e.history.length,termination:e.history.at(-1)?.interaction_mode==='stop'?'STOP':'horizon'};
}
export async function loadEpisodeComparison(api:ResectionApi,actor:EpisodeView,isCurrent:()=>boolean):Promise<EpisodeComparisonView>{
 need(actor.origin==='live'&&actor.authorship?.status==='verified_live_backend_run'&&actor.episode.selector==='RL256_ASPIRATION_TRANSFER','a retained live actor pair is required.');
 need(api.inspectDevelopmentEpisodeComparison,'this engine has no matched comparison operation.');
 const current=()=>{if(!isCurrent())throw new DOMException('A newer episode replaced this inspection.','AbortError')};
 const expected=structuredClone(actor.episode),source=structuredClone(actor.source);current();
 const binding=structuredClone(await api.inspectDevelopmentEpisodeComparison({caseHash:source.caseHash,episodeId:expected.episodeId}));current();
 need(same(actor.episode,expected)&&same(actor.source,source),'actor mutated during inspection.');
 need(binding&&Object.keys(binding).sort().join(',')===['caseHash','actorEpisodeId','actorStrategySeal','pairSeal','projectionHash','initialProjectedObservationHash','companion'].sort().join(',')&&binding.companion&&Object.keys(binding.companion).sort().join(',')==='episode,episodeCanonicalJson','unexpected comparison fields.');
 const e=binding.companion?.episode,p=e?.planning,a=expected.planning;
 const pending:unknown[]=[e];while(pending.length){const value=pending.pop();if(value&&typeof value==='object'){const row=value as Record<string,unknown>;need(!Object.hasOwn(row,'assetId')&&!(typeof row.path==='string'&&'dtype' in row&&'byteLength' in row),'comparison cannot introduce transfer assets.');pending.push(...Object.values(row));}}
 const trace=(v:unknown)=>Array.isArray(v)?v[0] as Record<string,unknown>:null;
 need(binding.caseHash===source.caseHash&&binding.actorEpisodeId===expected.episodeId&&binding.actorStrategySeal===a.strategySeal&&
  binding.pairSeal===a.pairSeal&&binding.projectionHash===a.projectionHash&&[binding.actorStrategySeal,binding.pairSeal,binding.projectionHash,binding.initialProjectedObservationHash].every(h=>HASH.test(h)), 'response differs from the current actor/pair.');
 need(e?.selector==='RL256_ASPIRATION_MATCHED_SEARCH'&&e.caseHash===expected.caseHash&&e.sourceHash===expected.sourceHash&&e.decisionModelHash===expected.decisionModelHash&&e.initialStateId===expected.initialStateId&&
  same(e.sourceBinding,expected.sourceBinding)&&same(e.tools,expected.tools)&&same(e.access,expected.access)&&
  p.pairSeal===binding.pairSeal&&p.projectionHash===binding.projectionHash&&p.matchedActorEpisodeId===expected.episodeId&&p.matchedActorStrategySeal===a.strategySeal&&
  trace(p.projectionTrace)?.projected_observation_hash===binding.initialProjectedObservationHash&&trace(a.projectionTrace)?.projected_observation_hash===binding.initialProjectedObservationHash&&
  trace(p.projectionTrace)?.full_observation_hash===trace(a.projectionTrace)?.full_observation_hash,'source, full tool registry or projected initial state differs.');
 const companion=await hydrateDevelopmentEpisode({case:source,...binding.companion},api,'comparison');current();
 need(same(actor.episode,expected)&&same(actor.source,source),'actor mutated during replay hydration.');
 return {actor,companion,binding,actorOutcome:checkedEpisodeOutcome(actor),searchOutcome:checkedEpisodeOutcome(companion)};
}
export function requirePersistableEpisode(view:EpisodeView|null){if(view?.origin==='comparison')throw Error('Return to the actor replay to save. This matched comparison is temporary.');}
