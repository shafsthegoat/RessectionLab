import type {DevelopmentEpisode} from './episode-types';
export interface VascularContactCount {
  touched_reference_cells:number;positive_reference_cells:number;unknown_reference_cells:number;
  outside_reference_fov:boolean;annotated_positive_encounter:boolean|null;annotation_coverage_complete_for_sweep:boolean;
  positive_cell_volume_upper_bound_mm3:number;unknown_in_grid_cell_volume_mm3:number;
  biological_vessel_free:null;clinical_injury_probability:null;
}
export interface EpisodeVascularEvaluation {
  schema:'generated-shared-vascular-encounter-v1';status:'evaluated_generated_vascular_reference';evaluationId:string;
  episodeId:string;caseHash:string;sourceHash:string;decisionModelHash:string;strategySeal:string;
  physicalHistoryHash:string;physicalHistoryCanonicalJson:string;actionIds:string[];referenceBindingHash:string;
  perAction:Array<{actionIndex:number;actionId:string;interactionMode:'stop'|'aspirate'|'probe';sweepCount:number;
    shaft:VascularContactCount|null;tip:VascularContactCount|null;wholeTool:VascularContactCount|null}>;
  shaft:VascularContactCount;tip:VascularContactCount;wholeTool:VascularContactCount;
  removedOverlap:{status:'not_evaluated_by_contact_kernel';outcomes:null};clinicalInjuryProbability:null;
  scope:'generated_geometry_annotation_contact_only';patientAdmission:false;
}
export interface EpisodeVascularRequest {caseHash:string;episodeId:string}
export interface EpisodeVascularResult {caseHash:string;evaluation:EpisodeVascularEvaluation;evaluationCanonicalJson:string}
export type EpisodeVascularApi={evaluateDevelopmentEpisodeVascular?(request:EpisodeVascularRequest):Promise<EpisodeVascularResult>};
export type EvaluatedEpisode = {episode:DevelopmentEpisode;evaluation:EpisodeVascularEvaluation};
