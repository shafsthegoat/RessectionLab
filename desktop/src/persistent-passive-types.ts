import type {CasePayload} from './types';

export type PassiveActionId = 'PASSIVE_MOVE'|'ACQUIRE_GENERATED_STATUS'|'PASSIVE_WITHDRAW'|'PASSIVE_WITHDRAW_STATIONARY'|'STOP';
export type PassivePhase = 'ready'|'held'|'inspected'|'withdrawn'|'both_withdrawn'|'stopped';
export type PassiveStatus = 'not_acquired'|'verified'|'unresolved';
export type PhysicalPose = {tip_mm:[number,number,number];axis_unit:[number,number,number]};
export interface PassiveToolState {
  tool_id:string|null;pose:PhysicalPose|null;activation:'off'|'passive_only';
  retained_at_depth:boolean;pose_status:'held_in_declared_void'|'access_pose_recorded'|'not_present_in_this_protocol';
  physical_absence_claim:false;
}
export interface PassiveOperativeState {
  phase:PassivePhase;elapsed_generated_seconds:number;
  moving_tool:PassiveToolState;other_tool:PassiveToolState;
  inspection_status:PassiveStatus;
}
export interface PassiveHistoryRow {
  action_id:PassiveActionId;interaction_mode:'passive_motion'|'generated_information'|'stop';
  tool_id:string|null;pose_before:PhysicalPose;pose_after:PhysicalPose;
  duration_generated_seconds:number;elapsed_generated_seconds:number;
  time_source:'fixed_generated_schedule_not_measured_surgical_time';
  source_state_hash:string;result_state_hash:string;operative_before:string;operative_after:string;
  geometry:Record<string,unknown>|null;
  independent_tissue_geometry:{feasible:true;checker_version:string;unknowns:string[];stationary_tool_pair_independently_checked:false}|null;
  information_event:{version:'generated-registration-status-v1';status:'verified'|'unresolved';input_fingerprint:string;meaning:string}|null;
  stop_reason:string|null;removed_indices_native:[];contact_indices_native:[];reward:0;
  clinical_deficit_probability:null;physiological_response:null;
}
export interface PassiveProtocol {
  version:'generated-persistent-passive-v1';moving_tool_id:string;
  start_pose:PhysicalPose;hold_pose:PhysicalPose;
  stationary_tool_id:string|null;stationary_pose:PhysicalPose|null;stationary_exit_pose:PhysicalPose|null;
  generated_seconds:number[];time_meaning:string;movement_scope:string;
}
export interface PersistentPassiveEpisode {
  schema:'resectionlab.generated-persistent-passive-episode.v1';
  sourceHash:string;decisionModelHash:string;protocol:PassiveProtocol;
  actions:PassiveActionId[];history:PassiveHistoryRow[];
  physicalHistoryCanonicalJson:string;physicalHistoryHash:string;
  frames:Array<{frameIndex:number;operative:PassiveOperativeState}>;
  frameMeaning:string;finalStateHash:string;scope:string;episodeId:string;
}
export interface PassiveToolGeometry {
  tool_id:string;tip_radius_mm:number;shaft_radius_mm:number;
  working_length_mm:number;max_access_angle_deg:number;tip_length_mm:number;
  parameter_source:string;
}
/** A generated live result. Saved-workspace admission requires separate fresh replay. */
export interface PersistentPassiveResult {
  case:CasePayload;episode:PersistentPassiveEpisode;episodeCanonicalJson:string;
  sourceBinding:{caseHash:string;sourceHash:string;decisionModelHash:string;
    structuralIntensityHash:string;sourceFrameHash:string;supportHash:string;nominalTargetMaskHash:string;affineHash:string;
    tools:[PassiveToolGeometry,PassiveToolGeometry];bindingHash:string};
}
