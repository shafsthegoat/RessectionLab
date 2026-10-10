import type {NativeEpisodeGeometry} from './native-episode-replay';
import type {CasePayload} from './types';
export type ContactSelector='scripted'|'SEARCH';
export type ContactGoalId='near'|'costly';
export interface PublicContactRequest {fixture:'generated-public-surface-contact-v1';selector:ContactSelector;goalId:ContactGoalId}
export interface PublicContactEpisode extends Omit<NativeEpisodeGeometry,'schema'|'history'> {
 schema:'resectionlab.shared-native-development-episode.v2';selector:ContactSelector;fixture:PublicContactRequest['fixture'];taskKind:'public_retained_surface_contact';
 publicGoal:{goalId:ContactGoalId;nativeIndex:[number,number,number];rasMm:[number,number,number];frame:'RAS+';physicalUnits:'mm';goalGridHash:string;objectiveHash:string;meaning:string};
 taskContract:{objectiveVersion:string;observationVersion:string;contextVersion:string;maxSteps:2;objective:Record<string,unknown>;declaration:Record<string,unknown>;detachedObservationBinding:Record<string,unknown>;nominalTargetRole:string;learnedPolicySupported:false;trainingAdmission:false};
 planning:Record<string,unknown>;metrics:Record<string,unknown>;
 history:Array<NativeEpisodeGeometry['history'][number]&{reward:number;goal_potential_before:number;goal_potential_after:number;public_objective_hash:string;removal_cost_volume_mm3:number;insertion_distance_mm:number;complete_tool_path_length_mm:number;tool_change_count:number;effort_and_removal_cost:number;outcome_scope:string;clinical_deficit_probability:null}>;
}
export interface PublicContactResult {case:CasePayload;episode:PublicContactEpisode;episodeCanonicalJson:string}
