import type {CasePayload, Vec3} from './types';
import type {NativeEpisodeGeometry} from './native-episode-replay';

export type ContactFamilyMethod = 'STOP' | 'SEARCH' | 'IL' | 'RL';
export type ContactFamilyGoal = 'surface' | 'deep';
export type InteractiveContactRole = 'TRAIN' | 'SELECT';
export interface ContactFamilyAvailability {
  version: 'generated-public-contact-learning-availability-v1';
  fixture: 'generated-public-contact-family-v2';
  familyHash: string;
  experimentHash: string | null;
  releaseHash: string | null;
  layouts: Array<{layoutId: string; role: InteractiveContactRole | 'MEASUREMENT_EVAL';
    goals: ['surface', 'deep']; interactive: boolean}>;
  methods: Record<ContactFamilyMethod, {available: boolean; reason: string | null}>;
}
export interface ContactLearnedAuthorship {
  version: 'public-contact-learned-authorship-v1';
  method: 'IL' | 'RL';
  checkpointFileSha256: string;
  checkpointVersion: 'public-goal-mode-spatial-checkpoint-v1';
  architectureHash: string;
  parameterHash: string;
  experimentHash: string;
  familyHash: string;
  trainingLineageHash: string;
  completedUpdates: 32;
  checkpointKind: 'final';
  inferenceOptimizerUpdates: 0;
  verificationScope: 'bounded_checkpoint_bytes_and_declared_lineage_owned_native_replay_not_signed_training_proof';
}
export interface ContactInitialObservation {
  version: 'public-contact-initial-observation-binding-v1';
  observationHash: string;
  sourceHash: string;
  decisionModelHash: string;
  declarationHash: string;
  objectiveHash: string;
  goalGridHash: string;
  cropOriginNative: Vec3;
  cropShape: Vec3;
  cropAffine: number[][];
  cropAffineHash: string;
  frame: 'RAS+';
  physicalUnits: 'mm';
  channelNames: string[];
  channelAvailable: boolean[];
  channelValueHashes: string[];
  channelCoverageHashes: string[];
  scope: 'raw_public_crop_values_before_policy_normalization_not_full_native_coverage';
  extraChannelCoverage: 'inherits_required_structural_intensity_coverage';
}
export interface ContactFamilyRequest {
  fixture: 'generated-public-contact-family-v2';
  layoutId: string;
  goalId: ContactFamilyGoal;
  selector: ContactFamilyMethod;
}

/** Distinct family contract: never accepted through the fixed-fixture v2 API. */
export interface ContactFamilyEpisode extends Omit<NativeEpisodeGeometry, 'schema' | 'history'> {
  schema: 'resectionlab.shared-native-contact-learning-episode.v3';
  fixture: ContactFamilyRequest['fixture'];
  taskKind: 'generated_family_public_retained_surface_contact';
  selector: ContactFamilyMethod;
  familyHash: string;
  layoutId: string;
  splitRole: InteractiveContactRole;
  publicGoal: {
    goalId: ContactFamilyGoal;
    nativeIndex: Vec3;
    rasMm: Vec3;
    frame: 'RAS+';
    physicalUnits: 'mm';
    goalGridHash: string;
    objectiveHash: string;
    meaning: string;
  };
  taskContract: {
    objectiveVersion: 'public-retained-surface-contact-objective-v1';
    observationVersion: 'public-goal-sequential-spatial-observation-v2';
    contextVersion: 'generated-public-contact-context-v2';
    maxSteps: 2;
    proposalMode: 'fixed_lattice_access_centerline_v1';
    sourceCandidateVersion: 'fixed_lattice_access_centerline_v1';
    objective: Record<string, unknown>;
    declaration: Record<string, unknown>;
    detachedObservationBinding: Record<string, unknown>;
    nominalTargetRole: 'zero compatibility field unused by public contact objective';
    learnedPolicySupported: true;
    trainingAdmission: false;
    checkpoint: Record<string, unknown> | null;
  };
  planning: Record<string, unknown>;
  metrics: Record<string, unknown>;
  history: Array<NativeEpisodeGeometry['history'][number] & {
    reward: number;
    goal_potential_before: number;
    goal_potential_after: number;
    public_objective_hash: string;
    removal_cost_volume_mm3: number;
    insertion_distance_mm: number;
    complete_tool_path_length_mm: number;
    tool_change_count: number;
    effort_and_removal_cost: number;
    outcome_scope: string;
    clinical_deficit_probability: null;
  }>;
  learnedAuthorship: ContactLearnedAuthorship | null;
  initialObservationBinding: ContactInitialObservation;
}
export interface ContactFamilyResult {
  case: CasePayload;
  episode: ContactFamilyEpisode;
  episodeCanonicalJson: string;
  executionProvenance?: ContactFamilyExecution;
}
export interface ContactFamilyExecution {
  version: 'generated-contact-family-execution-v1';
  selector: 'IL' | 'RL';
  layoutId: string;
  goalId: ContactFamilyGoal;
  splitRole: InteractiveContactRole;
  experimentHash: string;
  familyHash: string;
  releaseManifestSha256: string;
  pilotResultSha256: string;
  finalFreezeSha256: string;
  checkpointFileSha256: string;
  architectureHash: string;
  parameterHash: string;
  trainingLineageHash: string;
  completedUpdates: 32;
  inferenceOptimizerUpdates: 0;
  ownedResultSha256: string;
  ownedSupervisionSha256: string;
}
