import type {Affine, Point3, Shape3} from './viewer/coordinates';

export interface RehearsalTool {
  toolId: string;
  mode: 'aspiration' | 'field_management';
  tipRasMm: Point3;
  axis: Point3;
  active: boolean;
  workingLengthMm: number;
  tipLengthMm: number;
  shaftRadiusMm: number;
  tipRadiusMm: number;
}
export interface RehearsalMetrics {
  targetRemovedMm3: number;
  residualTargetMm3: number;
  nonTargetRemovedMm3: number;
  fieldBurden: number;
  vascularEvents: null | number;
  motorRisk: null;
  languageRisk: null;
  [key: string]: unknown;
}
export interface RehearsalFrame {
  frameIndex: number;
  actionIndex: number;
  kind: string;
  elapsedSeconds: number;
  tools: RehearsalTool[];
  removedIndicesNative: number[];
  contactIndicesNative: number[];
  fieldContactIndicesNative: number[];
  stateBefore: string;
  stateAfter: string;
  metrics: RehearsalMetrics;
  stopReason?: string | null;
}
export interface RehearsalSourceBinding {
  caseHash: string;
  shape: Shape3;
  affine: Affine;
  frame: string;
  [key: string]: unknown;
}
export interface RehearsalEpisode {
  schema?: string;
  episodeId?: string;
  sourceBinding: RehearsalSourceBinding;
  initialStateId: string;
  finalStateId: string;
  replayFrames: RehearsalFrame[];
  history: Array<Record<string, unknown>>;
  tools: RehearsalTool[];
  assumedExposureIndicesNative: number[];
  finalRemovedIndicesNative: number[];
  metrics?: RehearsalMetrics;
  stopReason?: string | null;
  [key: string]: unknown;
}
export type RehearsalAction = Record<string, unknown>;
export interface RehearsalSuggestedAction {id: string; label: string; action: RehearsalAction;}
export interface RehearsalAlternative {id: string; label: string; episode: RehearsalEpisode;}
export interface RehearsalComparison {snapshotId: string; alternatives: RehearsalAlternative[];}
export interface RehearsalViewPreferences {
  branchId: string;
  frameIndex: number;
  selectedTool: string;
  visible: boolean;
  episodeInitialStateId: string;
}
export interface RehearsalResponse {
  caseHash: string;
  episode: RehearsalEpisode;
  stateId: string;
  suggestedActions: RehearsalSuggestedAction[];
  comparison?: RehearsalComparison | null;
  savedView?: RehearsalViewPreferences;
  accepted?: boolean;
  reason?: string;
}
export interface RehearsalApi {
  createRehearsal(input: {caseHash: string}): Promise<RehearsalResponse>;
  inspectRehearsal(input: {caseHash: string}): Promise<RehearsalResponse | null>;
  stepRehearsal(input: {caseHash: string; expectedStateId: string; action: RehearsalAction}): Promise<RehearsalResponse>;
  compareRehearsal(input: {caseHash: string; expectedStateId: string}): Promise<RehearsalResponse>;
}
