import type {ContactFamilyAvailability, ContactFamilyEpisode, ContactFamilyRequest} from './contact-family-types.ts';
import {contactNeed, contactSame, metadataJson, CONTACT_COSTS} from './public-contact-authority.ts';
import {requireInteractiveFamilyRequest} from './contact-family-availability.ts';
export const FAMILY_SCHEMA = 'resectionlab.shared-native-contact-learning-episode.v3';
export const FAMILY_SHAPE = [15, 15, 14] as const;
const HASH = /^sha256:[a-f0-9]{64}$/;
const digest = (value: unknown) => typeof value === 'string' && HASH.test(value);
const uint = (value: unknown, maximum: number) => Number.isSafeInteger(value) && Number(value) >= 0 && Number(value) <= maximum;
/** Native strokes are straight, fully recorded physical paths. Cell indices
 * remain discrete; entry/endpoints are never rounded to a lattice. */
export function checkedFamilyTrajectory(e: Pick<ContactFamilyEpisode, 'tools' | 'history'>) {
  const vector = (v: unknown): v is number[] => Array.isArray(v) && v.length === 3 && v.every(n => typeof n === 'number' && Number.isFinite(n));
  const close = (a: number, b: number) => Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(a), Math.abs(b));
  const sameVector = (a: number[], b: number[]) => a.every((n, i) => close(n, b[i]));
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
export function checkedFamilyAuthority(
  episode: ContactFamilyEpisode, request: ContactFamilyRequest, catalog: ContactFamilyAvailability,
) {
  const admitted = requireInteractiveFamilyRequest(request, catalog);
  const e = episode, goal = e.publicGoal, task = e.taskContract, planning = e.planning;
  contactNeed(e.schema === FAMILY_SCHEMA && e.fixture === request.fixture &&
    e.taskKind === 'generated_family_public_retained_surface_contact' &&
    e.selector === request.selector && e.layoutId === request.layoutId &&
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
  const experimentHash = catalog.experimentHash ?? planning.experimentHash;
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
  const plan = planning.strategy as Record<string, unknown>;
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
      author.experimentHash === catalog.experimentHash && author.familyHash === e.familyHash &&
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
