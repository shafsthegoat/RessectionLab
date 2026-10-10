import type {PublicContactEpisode} from './public-contact-types.ts';
/** Only existing generated structural/support/fixture-mask transfers are public.
 * Metadata, reference evidence and the episode may not introduce transfer slots. */
export function checkedPublicContactAssets(result:unknown){
 const allowed=new Set(['case.mri','case.brainMask','case.compartments.0.array','case.compartments.0.sourceArray']);
 const pending:Array<[unknown,string]>=[[result,'']];
 while(pending.length){const[value,path]=pending.pop()!;if(value&&typeof value==='object'){
  const row=value as Record<string,unknown>;
  if('assetId' in row||(typeof row.path==='string'&&'dtype' in row&&'byteLength' in row))contactNeed(allowed.has(path),'transfer asset outside public source slots.');
  for(const[key,item]of Object.entries(row))pending.push([item,path?`${path}.${key}`:key]);
 }}
}
export const CONTACT_SCHEMA='resectionlab.shared-native-development-episode.v2';
export const CONTACT_FIXTURE='generated-public-surface-contact-v1';
export const CONTACT_COSTS={action_cost:.03,graph_edge_cost:0,language_per_mm3:0,motion_per_mm:.001,motor_per_mm3:0,normal_per_mm3:.2,target_per_mm3:0,tool_change_cost:.03};
const HASH=/^sha256:[a-f0-9]{64}$/;
export const contactStable=(v:unknown):unknown=>Array.isArray(v)?v.map(contactStable):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,x])=>[k,contactStable(x)])):v;
export const contactSame=(a:unknown,b:unknown)=>JSON.stringify(contactStable(a))===JSON.stringify(contactStable(b));
export function contactNeed(v:unknown,m:string):asserts v{if(!v)throw Error(`Public contact withheld: ${m}`)}
/** This fixed metadata schema contains explicit float fields. Preserve Python's
 * .0 when recomputing those public objective/declaration hashes after JSON parse. */
export function metadataJson(v:unknown,key=''):string{
 if(typeof v==='number')return Number.isInteger(v)&&(key in CONTACT_COSTS||['completion_value','seconds'].includes(key))?`${v}.0`:JSON.stringify(v);
 if(Array.isArray(v))return '['+v.map(x=>metadataJson(x)).join(',')+']';
 if(v&&typeof v==='object')return '{'+Object.entries(v).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,x])=>JSON.stringify(k)+':'+metadataJson(x,k)).join(',')+'}';
 return JSON.stringify(v);
}
export function checkedPublicContactAuthority(e:PublicContactEpisode){
 contactNeed(e?.schema===CONTACT_SCHEMA&&e.taskKind==='public_retained_surface_contact'&&e.fixture===CONTACT_FIXTURE&&['scripted','SEARCH'].includes(e.selector),'unsupported task/schema/selector.');
 const g=e.publicGoal,t=e.taskContract,p=e.planning,b=t?.detachedObservationBinding;
 const index=g?.goalId==='near'?[6,6,3]:g?.goalId==='costly'?[6,6,7]:null;
 contactNeed(index&&g.frame==='RAS+'&&g.physicalUnits==='mm'&&contactSame(g.nativeIndex,index)&&g.rasMm?.length===3&&g.rasMm.every(Number.isFinite)&&[g.goalGridHash,g.objectiveHash].every(h=>HASH.test(h)),'goal identity or native coordinates changed.');
 const objective={version:'public-retained-surface-contact-objective-v1',source_hash:e.sourceHash,native_index:index,completion_value:1,costs:CONTACT_COSTS,meaning:'committed_probe_contact_AND_currently_retained_cell',clinical_or_sensor_claim:false};
 const budget={max_calls:256,beam_width:96,seconds:4};
 const declaration={fixture:CONTACT_FIXTURE,selector:e.selector,goal_id:g.goalId,objective,observation_version:'public-goal-sequential-spatial-observation-v2',max_steps:2,search_budget:budget};
 contactNeed(t.objectiveVersion===objective.version&&t.observationVersion===declaration.observation_version&&t.contextVersion==='generated-public-contact-context-v2'&&t.maxSteps===2&&t.learnedPolicySupported===false&&t.trainingAdmission===false&&t.nominalTargetRole==='permitted generated fixture background; unused by contact reward'&&contactSame(t.objective,objective)&&contactSame(t.declaration,declaration),'objective, costs, horizon or declaration changed.');
 contactNeed(b&&b.version===t.contextVersion&&b.source_hash===e.sourceHash&&b.decision_model_hash===e.decisionModelHash&&b.objective_hash===g.objectiveHash&&b.goal_grid_hash===g.goalGridHash&&b.crop_affine_hash===e.sourceBinding?.affine_hash&&contactSame(b.crop_origin_native,[0,0,0])&&typeof b.declaration_hash==='string'&&HASH.test(b.declaration_hash),'detached goal/crop context changed.');
 contactNeed(p&&p.sealedBeforeExecution===true&&p.referenceScoringPerformed===false&&p.sealedBeforeReferenceScoring===undefined&&p.learnedPolicyExecuted===false&&p.actor_forward_calls===0&&p.optimizer_updates===0&&p.policyIdentity===undefined&&p.objectiveSource==='public_goal_and_shared_observed_native_state_only'&&contactSame(p.searchBudget,budget),'execution/public-only provenance changed.');
 contactNeed(Number.isSafeInteger(p.model_transition_calls)&&Number(p.model_transition_calls)>=1&&Number(p.model_transition_calls)<=256&&p.selector===(e.selector==='scripted'?'scripted_public_contact_demonstration':'existing_observed_beam_search'),'invalid bounded search/script accounting.');
 const plan=p.strategy as Record<string,unknown>;
 contactNeed(Array.isArray(e.history)&&e.history.length>=1&&e.history.length<=2&&HASH.test(String(p.strategySeal))&&plan&&plan.max_steps===2&&plan.source_hash===e.sourceHash&&plan.decision_model_hash===e.decisionModelHash&&plan.observation_contract===t.observationVersion&&contactSame(plan.actions,e.history.map(h=>h.action_id))&&contactSame(plan.history,e.history),'complete sealed strategy differs.');
 contactNeed(e.metrics?.task_version===t.objectiveVersion&&e.metrics.clinical_validation===false&&e.metrics.source_hash===e.sourceHash&&e.metrics.decision_model_hash===e.decisionModelHash&&e.metrics.steps===e.history.length&&e.metrics.terminated===true&&e.metrics.planning_estimator_only===false&&contactSame(e.metrics.objective,objective)&&contactSame(e.metrics.history,e.history)&&!('target_removed_mm3' in e.metrics)&&!('normal_removed_mm3' in e.metrics),'public metrics do not belong to this task.');
 return {objective:metadataJson(objective),declaration:metadataJson(declaration)};
}
