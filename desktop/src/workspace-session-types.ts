import type {CasePayload,Vec3} from './types';
import type {DisplaySeriesPayload} from './workspace-imaging-types';
import type {DevelopmentEpisode} from './episode-types';
export interface ImageViewState {cursor:Vec3|null;visibleLayers:Record<string,boolean>}
export interface ImagingSessionState {selectedSeriesId:string|null;states:Record<string,ImageViewState>}
export interface EpisodeReplaySelection {episodeId:string;frameIndex:number;visible:boolean}
export interface WorkspaceEvidenceRow {
  seriesId:string;sourceCaseHash:string;sourceFrameHash:string;modality:string;
  sourceKind:'primary_source'|'source_image'|'source_annotation'|'estimated_annotation';
  displayAvailable:true;planningInput:boolean;
  readiness:'primary_case_contract'|'display_only_registration_unreviewed';reasons:string[];
}
/** This envelope is produced only after the backend validates persisted sources/history. */
export interface WorkspaceSessionPayload {
  schema:'integrated-workspace-session-v1';sessionHash:string;referenceCaseHash:string;
  displaySeries:DisplaySeriesPayload[];imagingState:ImagingSessionState;
  episodeReplay: {episode:DevelopmentEpisode;episodeCanonicalJson:string;frameIndex:number;visible:boolean;
    validation:'authoritative_generated_replay';accountingQualification:'historical_timings_not_remeasured'}|null;
  evidenceInventory:WorkspaceEvidenceRow[];
}
export type WorkspaceCasePayload = CasePayload & {workspaceSession?:WorkspaceSessionPayload};
