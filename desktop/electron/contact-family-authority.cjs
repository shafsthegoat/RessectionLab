'use strict';
const shared=(()=>{

/** Only existing generated structural/support/fixture-mask transfers are public.
 * Metadata, reference evidence and the episode may not introduce transfer slots. */
function checkedPublicContactAssets(result        ){
 const allowed=new Set(['case.mri','case.brainMask','case.compartments.0.array','case.compartments.0.sourceArray']);
 const pending                        =[[result,'']];
 while(pending.length){const[value,path]=pending.pop() ;if(value&&typeof value==='object'){
  const row=value                          ;
  if('assetId' in row||(typeof row.path==='string'&&'dtype' in row&&'byteLength' in row))contactNeed(allowed.has(path),'transfer asset outside public source slots.');
  for(const[key,item]of Object.entries(row))pending.push([item,path?`${path}.${key}`:key]);
 }}
}
const CONTACT_SCHEMA='resectionlab.shared-native-development-episode.v2';
const CONTACT_FIXTURE='generated-public-surface-contact-v1';
const CONTACT_COSTS={action_cost:.03,graph_edge_cost:0,language_per_mm3:0,motion_per_mm:.001,motor_per_mm3:0,normal_per_mm3:.2,target_per_mm3:0,tool_change_cost:.03};
const HASH=/^sha256:[a-f0-9]{64}$/;
const contactStable=(v        )        =>Array.isArray(v)?v.map(contactStable):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,x])=>[k,contactStable(x)])):v;
const contactSame=(a        ,b        )=>JSON.stringify(contactStable(a))===JSON.stringify(contactStable(b));
function contactNeed(v        ,m       )          {if(!v)throw Error(`Public contact withheld: ${m}`)}
/** This fixed metadata schema contains explicit float fields. Preserve Python's
 * .0 when recomputing those public objective/declaration hashes after JSON parse. */
function metadataJson(v        ,key='')       {
 if(typeof v==='number')return Number.isInteger(v)&&(key in CONTACT_COSTS||['completion_value','seconds'].includes(key))?`${v}.0`:JSON.stringify(v);
 if(Array.isArray(v))return '['+v.map(x=>metadataJson(x)).join(',')+']';
 if(v&&typeof v==='object')return '{'+Object.entries(v).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,x])=>JSON.stringify(k)+':'+metadataJson(x,k)).join(',')+'}';
 return JSON.stringify(v);
}
function checkedPublicContactAuthority(e                     ){
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
 const plan=p.strategy                          ;
 contactNeed(Array.isArray(e.history)&&e.history.length>=1&&e.history.length<=2&&HASH.test(String(p.strategySeal))&&plan&&plan.max_steps===2&&plan.source_hash===e.sourceHash&&plan.decision_model_hash===e.decisionModelHash&&plan.observation_contract===t.observationVersion&&contactSame(plan.actions,e.history.map(h=>h.action_id))&&contactSame(plan.history,e.history),'complete sealed strategy differs.');
 contactNeed(e.metrics?.task_version===t.objectiveVersion&&e.metrics.clinical_validation===false&&e.metrics.source_hash===e.sourceHash&&e.metrics.decision_model_hash===e.decisionModelHash&&e.metrics.steps===e.history.length&&e.metrics.terminated===true&&e.metrics.planning_estimator_only===false&&contactSame(e.metrics.objective,objective)&&contactSame(e.metrics.history,e.history)&&!('target_removed_mm3' in e.metrics)&&!('normal_removed_mm3' in e.metrics),'public metrics do not belong to this task.');
 return {objective:metadataJson(objective),declaration:metadataJson(declaration)};
}

return {contactNeed,contactSame,metadataJson,CONTACT_COSTS,checkedPublicContactAssets};})();
const request=(()=>{


function checkedContactFamilyRequest(value         )                       {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Choose a generated layout, goal and method.');
  }
  const input = value                           ;
  if (Object.keys(input).sort().join(',') !== 'fixture,goalId,layoutId,selector' ||
      input.fixture !== 'generated-public-contact-family-v2' ||
      typeof input.layoutId !== 'string' || !/^pcf-(?:0[0-9]|1[0-9]|2[0-3])$/.test(input.layoutId) ||
      typeof input.goalId !== 'string' || !['surface', 'deep'].includes(input.goalId) ||
      typeof input.selector !== 'string' || !['STOP', 'SEARCH', 'IL', 'RL', 'IL_TRAIN_REFIT'].includes(input.selector)) {
    throw new Error('Only the fixed generated family and named methods are supported.');
  }
  // Canonical role and released method availability belong to the owned backend.
  // A syntactically valid ID does not authorize held-out execution or a checkpoint.
  return {fixture: input.fixture, layoutId: input.layoutId,
    goalId: input.goalId                                  ,
    selector: input.selector                                    };
}

return {checkedContactFamilyRequest};})();
const availability=(()=>{const {contactNeed,contactSame}=shared;const {checkedContactFamilyRequest}=request;

const HASH = /^sha256:[a-f0-9]{64}$/;
const digest = (value         ) => typeof value === 'string' && HASH.test(value);
const keys = (value        ) => Object.keys(value).sort().join(',');

function checkedFamilyAvailability(raw                           )                            {
  const value = structuredClone(raw);
  contactNeed(value && keys(value) === 'experimentHash,familyHash,fixture,layouts,methods,releaseHash,version' &&
    ['generated-public-contact-learning-availability-v1', 'generated-public-contact-learning-availability-v2'].includes(value.version) &&
    value.fixture === 'generated-public-contact-family-v2' && digest(value.familyHash) &&
    (value.experimentHash === null || digest(value.experimentHash)) &&
    (value.releaseHash === null || digest(value.releaseHash)) &&
    (value.releaseHash === null) === (value.experimentHash === null),
    'family availability identity changed.');
  contactNeed(Array.isArray(value.layouts) && value.layouts.length === 24 &&
    new Set(value.layouts.map(row => row.layoutId)).size === 24, 'incomplete family layout inventory.');
  const counts = {TRAIN: 0, SELECT: 0, MEASUREMENT_EVAL: 0};
  for (const row of value.layouts) {
    contactNeed(keys(row) === 'goals,interactive,layoutId,role' &&
      typeof row.layoutId === 'string' && /^pcf-(?:0[0-9]|1[0-9]|2[0-3])$/.test(row.layoutId) &&
      Object.hasOwn(counts, row.role) && contactSame(row.goals, ['surface', 'deep']) &&
      row.interactive === (row.role === 'TRAIN' || row.role === 'SELECT'), 'layout role or interactive admission changed.');
    counts[row.role]++;
  }
  contactNeed(contactSame(counts, {TRAIN: 12, SELECT: 4, MEASUREMENT_EVAL: 8}), 'layout role denominator changed.');
  contactNeed(value.methods && keys(value.methods) === (value.version === 'generated-public-contact-learning-availability-v2' ? 'IL,IL_TRAIN_REFIT,RL,SEARCH,STOP' : 'IL,RL,SEARCH,STOP'), 'missing fixed method slot.');
  for (const method of ['STOP', 'SEARCH', 'IL', 'RL']         ) {
    const row = value.methods[method];
    contactNeed(row && keys(row) === 'available,reason' && typeof row.available === 'boolean' &&
      (row.reason === null || typeof row.reason === 'string' && row.reason.length > 0 && row.reason.length <= 1000) &&
      (row.available ? row.reason === null : row.reason !== null), 'method availability is ambiguous.');
  }
  contactNeed(value.methods.IL.available === value.methods.RL.available &&
    (value.releaseHash !== null || !value.methods.IL.available), 'unreleased learned method cannot be enabled.');
  if (value.version === 'generated-public-contact-learning-availability-v2') {
    const row = value.methods.IL_TRAIN_REFIT;
    contactNeed(row && keys(row) === 'allowedRoles,available,checkpointFileSha256,evidence,experimentHash,knownTRAINOutcome,parameterHash,reason,releaseHash,trainingBudget' &&
      typeof row.available === 'boolean' && contactSame(row.allowedRoles, ['TRAIN']) &&
      (row.reason === null || typeof row.reason === 'string' && row.reason.length > 0 && row.reason.length <= 1000) &&
      (row.available ? row.reason === null : row.reason !== null), 'TRAIN refit availability is ambiguous.');
    if (row.releaseHash === null) {
      contactNeed(!row.available && row.experimentHash === null && row.parameterHash === null && row.checkpointFileSha256 === null &&
        row.evidence === null && row.trainingBudget === null && row.knownTRAINOutcome === null, 'unpublished refit carries artifact identity.');
    } else {
      contactNeed(row.releaseHash === 'sha256:68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887' &&
        row.experimentHash === 'sha256:fd211e00dbe6dd1c9ed50a58d6c3b9f9f8896a0acd3a7815b2c52e4a8dc1014f' &&
        row.parameterHash === 'sha256:0bdd6713358937ac2c665ff700210b24eb6a88433f6f410fc87bb62f23f7fe20' &&
        row.checkpointFileSha256 === '5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7' &&
        row.evidence && keys(row.evidence) === 'fitResultSha256,independentAuditSha256,rolloutResultSha256' &&
        Object.values(row.evidence).every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)) &&
        contactSame(row.trainingBudget, {updates: 32, statesPerUpdate: 40, lossForwards: 1280, fixedReadoutForwards: 80}),
        'TRAIN refit artifact or extra training budget changed.');
      const outcome = row.knownTRAINOutcome;
      contactNeed(outcome && keys(outcome) === 'STOPOnly,goalContacts,meanReturn,savedSEARCHContacts,scope,tasks' &&
        outcome.tasks === 24 && outcome.goalContacts === 6 && outcome.savedSEARCHContacts === 16 && outcome.STOPOnly === 18 &&
        typeof outcome.meanReturn === 'number' && Math.abs(outcome.meanReturn - 0.05633333333333332) < 1e-12 &&
        outcome.scope === 'generated_TRAIN_native_results_no_heldout_claim', 'fixed TRAIN refit negative outcomes changed.');
    }
  }
  return value;
}

function requireInteractiveFamilyRequest(raw         , catalog                           ) {
  const request = checkedContactFamilyRequest(raw);
  const value = checkedFamilyAvailability(catalog);
  const layout = value.layouts.find(row => row.layoutId === request.layoutId);
  contactNeed(layout?.interactive === true && layout.role !== 'MEASUREMENT_EVAL', 'held-out layouts are unavailable for interactive execution.');
  if (request.selector === 'IL_TRAIN_REFIT') contactNeed(layout.role === 'TRAIN', 'Full-teacher imitation is available only on TRAIN layouts.');
  const method = value.methods[request.selector];
  contactNeed(method?.available, method?.reason ?? 'method unavailable.');
  return {request, layout, catalog: value};
}

return {checkedFamilyAvailability,requireInteractiveFamilyRequest};})();
const authority=(()=>{const {contactNeed,contactSame,metadataJson,CONTACT_COSTS}=shared;const {requireInteractiveFamilyRequest}=availability;

const FAMILY_SCHEMA = 'resectionlab.shared-native-contact-learning-episode.v3';
const FAMILY_SHAPE = [15, 15, 14]         ;
const HASH = /^sha256:[a-f0-9]{64}$/;
const digest = (value         ) => typeof value === 'string' && HASH.test(value);
const uint = (value         , maximum        ) => Number.isSafeInteger(value) && Number(value) >= 0 && Number(value) <= maximum;
/** Native strokes are straight, fully recorded physical paths. Cell indices
 * remain discrete; entry/endpoints are never rounded to a lattice. */
function checkedFamilyTrajectory(e                                                 ) {
  const vector = (v         )                => Array.isArray(v) && v.length === 3 && v.every(n => typeof n === 'number' && Number.isFinite(n));
  const close = (a        , b        ) => Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(a), Math.abs(b));
  const sameVector = (a          , b          ) => a.every((n, i) => close(n, b[i]));
  for (const action of e.history) {
    if (action.interaction_mode === 'stop') continue;
    const tool = e.tools.find(t => t.tool_id === action.tool_id);
    contactNeed(tool && [tool.tip_radius_mm, tool.shaft_radius_mm, tool.tip_length_mm, tool.working_length_mm].every(n => typeof n === 'number' && Number.isFinite(n) && n > 0) &&
      tool.working_length_mm >= tool.tip_length_mm && vector(action.entry_mm) && vector(action.tip_mm) && vector(action.axis_unit), 'family physical tool or endpoints changed.');
    const entry = action.entry_mm, tip = action.tip_mm, axis = action.axis_unit;
    const length = Math.hypot(...tip.map((n, i) => n - entry[i]));
    contactNeed(length > 1e-9 && sameVector(axis, tip.map((n, i) => (n - entry[i]) / length)) && action.microsteps.length > 0,
      'family macro endpoint and physical axis disagree.');
    let previous = entry, previousDistance = 0;
    for (const micro of action.microsteps) {
      contactNeed(vector(micro.tip_start_mm) && vector(micro.tip_end_mm) && vector(micro.active_stroke_start_mm) && vector(micro.active_stroke_end_mm) &&
        sameVector(micro.tip_start_mm, previous) && sameVector(micro.active_stroke_end_mm, micro.tip_end_mm) &&
        sameVector(micro.active_stroke_start_mm, micro.tip_start_mm.map((n, i) => n - tool.tip_length_mm * axis[i])) &&
        micro.active_radius_mm === tool.tip_radius_mm, 'family microstep continuity or active tool geometry changed.');
      const distance = micro.tip_end_mm.reduce((sum, n, i) => sum + (n - entry[i]) * axis[i], 0);
      contactNeed(distance >= previousDistance - 1e-8 && distance <= length + 1e-8 &&
        sameVector(micro.tip_end_mm, entry.map((n, i) => n + distance * axis[i])), 'family microstep leaves the recorded straight path.');
      previous = micro.tip_end_mm; previousDistance = distance;
    }
    contactNeed(sameVector(previous, tip), 'family microsteps do not reach the physical endpoint.');
  }
}
function checkedFamilyAuthority(
  episode                      , request                      , catalog                           ,
) {
  const admitted = requireInteractiveFamilyRequest(request, catalog);
  const e = episode, goal = e.publicGoal, task = e.taskContract, planning = e.planning;
  contactNeed(e.schema === FAMILY_SCHEMA && e.fixture === request.fixture &&
    e.taskKind === 'generated_family_public_retained_surface_contact' &&
    e.selector === (request.selector === 'IL_TRAIN_REFIT' ? 'IL' : request.selector) && e.layoutId === request.layoutId &&
    e.splitRole === admitted.layout.role && e.familyHash === admitted.catalog.familyHash &&
    contactSame(e.shape, FAMILY_SHAPE), 'family task/layout/role differs from requested generated world.');
  contactNeed(goal && goal.goalId === request.goalId && goal.frame === 'RAS+' && goal.physicalUnits === 'mm' &&
    goal.nativeIndex.length === 3 && goal.nativeIndex.every((n, i) => Number.isSafeInteger(n) && n >= 0 && n < e.shape[i]) &&
    goal.rasMm.length === 3 && goal.rasMm.every(Number.isFinite) && digest(goal.goalGridHash) && digest(goal.objectiveHash),
    'invalid family public goal.');
  const objective = {version: 'public-retained-surface-contact-objective-v1', source_hash: e.sourceHash,
    native_index: goal.nativeIndex, completion_value: 1, costs: CONTACT_COSTS,
    meaning: 'committed_probe_contact_AND_currently_retained_cell', clinical_or_sensor_claim: false};
  const declaration = task.declaration, binding = task.detachedObservationBinding;
  const experimentHash = request.selector === 'IL_TRAIN_REFIT' ? catalog.methods.IL_TRAIN_REFIT?.experimentHash : catalog.experimentHash ?? planning.experimentHash;
  contactNeed(digest(experimentHash), 'family execution has no frozen experiment identity.');
  contactNeed(task.maxSteps === 2 && task.proposalMode === 'fixed_lattice_access_centerline_v1' &&
    task.sourceCandidateVersion === task.proposalMode && task.objectiveVersion === objective.version &&
    task.observationVersion === 'public-goal-sequential-spatial-observation-v2' &&
    task.contextVersion === 'generated-public-contact-context-v2' &&
    task.learnedPolicySupported === true && task.trainingAdmission === false &&
    task.nominalTargetRole === 'zero compatibility field unused by public contact objective' &&
    contactSame(task.objective, objective), 'family public objective or task versions changed.');
  contactNeed(Object.keys(declaration).sort().join(',') === 'decision_model_hash,experiment_hash,family_hash,goal_id,layout_id,objective_hash,proposal_mode,recipe_hash,role,scope,source_candidate_version,source_hash,training_admission,version' &&
    declaration.version === request.fixture && declaration.proposal_mode === task.proposalMode &&
    declaration.source_candidate_version === task.sourceCandidateVersion && declaration.layout_id === request.layoutId &&
    declaration.goal_id === request.goalId && declaration.role === e.splitRole &&
    declaration.family_hash === e.familyHash && declaration.experiment_hash === experimentHash &&
    declaration.source_hash === e.sourceHash && declaration.objective_hash === goal.objectiveHash &&
    declaration.decision_model_hash === e.decisionModelHash && digest(declaration.recipe_hash) &&
    declaration.training_admission === false && declaration.scope === 'generated_forward_context_only',
    'family recipe/context or interactive role changed.');
  contactNeed(binding.version === task.contextVersion && binding.source_hash === e.sourceHash &&
    binding.decision_model_hash === e.decisionModelHash && binding.objective_hash === goal.objectiveHash &&
    binding.goal_grid_hash === goal.goalGridHash && digest(binding.crop_affine_hash) &&
    contactSame(binding.crop_origin_native, [0, 0, 0]) && digest(binding.declaration_hash), 'family detached observation binding changed.');
  const learned = e.selector === 'IL' || e.selector === 'RL';
  contactNeed(planning.sealedBeforeExecution === true && planning.referenceScoringPerformed === false &&
    planning.sealedBeforeReferenceScoring === undefined && planning.learnedPolicyExecuted === learned &&
    planning.training_or_checkpoint_lineage_verified === learned && planning.optimizer_updates === 0 &&
    planning.experimentHash === experimentHash &&
    planning.objectiveSource === 'same_public_goal_observed_state_and_geometry' &&
    uint(planning.actor_forward_calls, 2) && uint(planning.model_transition_calls, 256), 'family inference provenance changed.');
  const plan = planning.strategy                           ;
  contactNeed(Array.isArray(e.history) && e.history.length >= 1 && e.history.length <= 2 && digest(planning.strategySeal) &&
    plan && plan.source_hash === e.sourceHash && plan.decision_model_hash === e.decisionModelHash &&
    plan.max_steps === 2 && plan.observation_contract === task.observationVersion &&
    contactSame(plan.actions, e.history.map(row => row.action_id)) && contactSame(plan.history, e.history),
    'family sealed strategy differs from physical history.');
  contactNeed(e.metrics.source_hash === e.sourceHash && e.metrics.decision_model_hash === e.decisionModelHash &&
    e.metrics.task_version === objective.version && e.metrics.clinical_validation === false &&
    e.metrics.steps === e.history.length && e.metrics.terminated === true && e.metrics.planning_estimator_only === false &&
    contactSame(e.metrics.objective, objective) && contactSame(e.metrics.history, e.history) &&
    !('target_removed_mm3' in e.metrics) && !('normal_removed_mm3' in e.metrics), 'family metrics differ from the public task.');
  if (learned) {
    const author = e.learnedAuthorship, lineage = task.checkpoint;
    contactNeed(author && author.version === 'public-contact-learned-authorship-v1' && author.method === e.selector &&
      typeof author.checkpointFileSha256 === 'string' && /^[a-f0-9]{64}$/.test(author.checkpointFileSha256) && author.checkpointVersion === 'public-goal-mode-spatial-checkpoint-v1' &&
      [author.architectureHash, author.parameterHash, author.trainingLineageHash].every(digest) &&
      author.architectureHash === planning.architecture_hash && author.parameterHash === planning.parameter_hash &&
      author.experimentHash === experimentHash && author.familyHash === e.familyHash &&
      author.completedUpdates === 32 && author.checkpointKind === 'final' && author.inferenceOptimizerUpdates === 0 &&
      author.verificationScope === 'bounded_checkpoint_bytes_and_declared_lineage_owned_native_replay_not_signed_training_proof' &&
      lineage && lineage.kind === 'final' && lineage.method === e.selector && lineage.optimizer_updates === 32 &&
      lineage.training_status === 'completed_fixed_endpoint' && lineage.real_patient_count === 0 &&
      lineage.experiment_hash === author.experimentHash && lineage.family_hash === author.familyHash &&
      lineage.parameter_hash === author.parameterHash && Array.isArray(lineage.training_bindings) && lineage.training_bindings.length > 0 &&
      planning.selector === 'GOAL_MODE_POLICY_V1' && planning.actor_forward_calls === e.history.length &&
      planning.model_transition_calls === e.history.length && planning.context_declaration_hash === binding.declaration_hash &&
      Array.isArray(planning.observation_ids) && planning.observation_ids.length === e.history.length &&
      planning.observation_ids.every(digest), 'learned method lacks matching final artifact or inference provenance.');
  } else {
    contactNeed(e.learnedAuthorship === null && task.checkpoint === null && planning.actor_forward_calls === 0 &&
      planning.parameter_hash === undefined, 'SEARCH/STOP cannot carry learned authorship.');
    if (e.selector === 'STOP') contactNeed(e.history.length === 1 && e.history[0].action_id === 'STOP' &&
      planning.selector === 'immediate_STOP' && planning.model_transition_calls === 0, 'immediate STOP was replaced.');
    else {
      contactNeed(planning.time_cap_reached === false, 'SEARCH reached its time limit; no completed result can be displayed.');
      contactNeed(planning.max_calls === 256 && planning.beam_width === 96 && planning.guidance === 'none' &&
      planning.transition_mode === 'lazy_planning' &&
      typeof planning.call_cap_reached === 'boolean' && uint(planning.beam_pruned_prefixes, 256),
      'SEARCH differs from its bounded public protocol.');
    }
  }
  checkedFamilyTrajectory(e);
  return {objectiveJson: metadataJson(objective), declarationJson: metadataJson(declaration)};
}

return {checkedFamilyAuthority};})();
const {contactNeed}=shared;
function checkedFamilyExecution(result                     , catalog                           , request) {
  const e = result.episode, p = result.executionProvenance, author = e.learnedAuthorship;
  const refit = request.selector === 'IL_TRAIN_REFIT';
  if (refit) {
    const method = catalog.methods.IL_TRAIN_REFIT;
    contactNeed(Object.keys(result).sort().join(',') === 'case,episode,episodeCanonicalJson,executionProvenance,policyVariant' &&
      result.policyVariant === 'IL_TRAIN_REFIT' && e.selector === 'IL' && e.splitRole === 'TRAIN' && method?.available &&
      p?.version === 'generated-contact-train-refit-execution-v1', 'refit response is not the requested distinct TRAIN method.');
    const expectedKeys = ['version', 'variant', 'algorithm', 'layoutId', 'goalId', 'splitRole', 'experimentHash', 'familyHash',
      'releaseManifestSha256', 'fitResultSha256', 'rolloutResultSha256', 'independentAuditSha256', 'checkpointFileSha256',
      'architectureHash', 'parameterHash', 'trainingLineageHash', 'completedUpdates', 'statesPerUpdate',
      'inferenceOptimizerUpdates', 'ownedResultSha256', 'ownedSupervisionSha256'];
    contactNeed(author && Object.keys(p).sort().join(',') === expectedKeys.sort().join(',') && p.variant === 'IL_TRAIN_REFIT' &&
      p.algorithm === 'IL' && p.layoutId === e.layoutId && p.goalId === e.publicGoal.goalId && p.splitRole === 'TRAIN' &&
      p.familyHash === e.familyHash && p.experimentHash === method.experimentHash && p.experimentHash === author.experimentHash &&
      p.architectureHash === author.architectureHash && p.parameterHash === method.parameterHash && p.parameterHash === author.parameterHash &&
      p.trainingLineageHash === author.trainingLineageHash && p.checkpointFileSha256 === method.checkpointFileSha256 &&
      p.checkpointFileSha256 === author.checkpointFileSha256 && p.completedUpdates === 32 && p.statesPerUpdate === 40 &&
      p.inferenceOptimizerUpdates === 0 && method.releaseHash === 'sha256:' + p.releaseManifestSha256 &&
      p.fitResultSha256 === method.evidence?.fitResultSha256 && p.rolloutResultSha256 === method.evidence?.rolloutResultSha256 &&
      p.independentAuditSha256 === method.evidence?.independentAuditSha256 &&
      [p.releaseManifestSha256, p.fitResultSha256, p.rolloutResultSha256, p.independentAuditSha256, p.checkpointFileSha256,
        p.ownedResultSha256, p.ownedSupervisionSha256].every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)),
      'live TRAIN refit differs from its separate audited publication.');
    return;
  }
  contactNeed(result.policyVariant === undefined, 'original method cannot carry refit identity.');
  const learned = e.selector === 'IL' || e.selector === 'RL';
  contactNeed(Object.keys(result).sort().join(',') === (learned ?
    'case,episode,episodeCanonicalJson,executionProvenance' : 'case,episode,episodeCanonicalJson'), 'unexpected family response fields.');
  if (!learned) { contactNeed(p === undefined && author === null, 'nonlearned method claims live actor provenance.'); return; }
  contactNeed(p?.version === 'generated-contact-family-execution-v1', 'original method requires its paired release.');
  const expectedKeys = ['version', 'selector', 'layoutId', 'goalId', 'splitRole', 'experimentHash', 'familyHash',
    'releaseManifestSha256', 'pilotResultSha256', 'finalFreezeSha256', 'checkpointFileSha256', 'architectureHash',
    'parameterHash', 'trainingLineageHash', 'completedUpdates', 'inferenceOptimizerUpdates', 'ownedResultSha256', 'ownedSupervisionSha256'];
  contactNeed(p && author && Object.keys(p).sort().join(',') === expectedKeys.sort().join(',') &&
    p.version === 'generated-contact-family-execution-v1' && p.selector === e.selector &&
    p.layoutId === e.layoutId && p.goalId === e.publicGoal.goalId && p.splitRole === e.splitRole &&
    p.familyHash === e.familyHash && p.experimentHash === catalog.experimentHash &&
    p.experimentHash === author.experimentHash && p.architectureHash === author.architectureHash &&
    p.parameterHash === author.parameterHash && p.trainingLineageHash === author.trainingLineageHash &&
    p.checkpointFileSha256 === author.checkpointFileSha256 &&
    p.completedUpdates === 32 && p.inferenceOptimizerUpdates === 0 &&
    catalog.releaseHash === 'sha256:' + p.releaseManifestSha256 &&
    [p.releaseManifestSha256, p.pilotResultSha256, p.finalFreezeSha256, p.checkpointFileSha256,
      p.ownedResultSha256, p.ownedSupervisionSha256].every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)),
    'live learned response differs from released artifacts or owned worker.');
}


module.exports={...shared,...request,...availability,...authority,checkedFamilyExecution};
