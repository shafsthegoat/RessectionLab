// Generated development result from execute_development_episode(selector, cancelled).
// The sidecar supplies `case` through its existing _install_case / CasePayload.
export type EpisodeSelector = 'scripted' | 'SEARCH' | 'RL256_ASPIRATION_TRANSFER';
export type EpisodeOutputSelector = EpisodeSelector | 'RL256_ASPIRATION_MATCHED_SEARCH';
export type EpisodeOrigin = 'live' | 'reopened' | 'comparison';
/** Response-only qualification; never inferred from the imported episode's claims. */
export interface EpisodeAuthorship {
  status: 'verified_live_backend_run' | 'unverified_imported';
  checkpointSha256: string;
  projectionHash: string;
}
export type XYZ = [number, number, number];
export type NativeCell = [number, number, number];
export interface DevelopmentEpisode {
  schema: 'resectionlab.shared-native-development-episode.v1';
  episodeId: string;
  caseHash: string;
  sourceHash: string;
  decisionModelHash: string;
  selector: EpisodeOutputSelector;
  evidenceKind: 'generated_software_fixture';
  fidelity: 'native_grid_connected_exposed_tip_aspiration_and_nonremoving_geometric_probe';
  backendStatus: 'generated_executed';
  patientAdmission: false;
  clinicalValidation: false;
  frame: 'RAS+';
  physicalUnits: 'mm';
  shape: XYZ;
  affine: number[][];
  sourceBinding: {display_case_hash: string; native_source_hash: string;
    structural_intensity_hash: string; affine_hash: string; source_frame_hash: string; support_hash: string;
    nominal_target_mask_hash: string; private_reference_published: false};
  tools: Array<{tool_id: string; tip_radius_mm: number; shaft_radius_mm: number;
    working_length_mm: number; max_access_angle_deg: number; tip_length_mm: number;
    interactionMode: 'aspirate' | 'probe'}>;
  access: {center_mm: XYZ; normal_inward: XYZ; radius_mm: number; window_id: string};
  history: Array<{action_id: string; interaction_mode: 'stop' | 'aspirate' | 'probe';
    source_state_hash: string; result_state_hash: string;
    tool_id?: string; entry_mm?: XYZ; tip_mm?: XYZ; axis_unit?: XYZ;
    removed_indices_native: NativeCell[]; contact_indices_native: NativeCell[];
    probe_contact_indices_native: NativeCell[];
    microsteps: Array<{tip_start_mm: XYZ; tip_end_mm: XYZ;
      active_stroke_start_mm: XYZ; active_stroke_end_mm: XYZ; active_radius_mm: number;
      removed_indices_native: NativeCell[]; contact_indices_native: NativeCell[]}>;
    reward: number; target_removed_mm3: number; normal_removed_mm3: number}>;
  replayFrames: Array<{frameIndex: number; actionIndex: number;
    phase: 'initial' | 'insertion' | 'withdrawal' | 'stop';
    toolId: string | null; mode: 'stop' | 'aspirate' | 'probe' | null;
    tipRasMm: XYZ | null; axis: XYZ | null;
    removedIndicesNative: NativeCell[]; contactIndicesNative: NativeCell[];
    probeContactIndicesNative: NativeCell[];
    stateBefore: string; stateAfter: string;
    cavityHash: string; remainingHash: string; contactHash: string; probeContactHash: string}>;
  initialStateId: string;
  finalStateId: string;
  nativeEngineFinalStateId: string;
  finalRemovedIndicesNative: NativeCell[];
  geometryAudit: {feasible: boolean; complete_tool_checked: boolean;
    frontier_checked: boolean; checker_version: string; failures: string[];
    source_case_hash: string; [key: string]: unknown};
  metrics: Record<string, unknown>;
  planning: Record<string, unknown>;
  sequentialEffect: Record<string, unknown>;
  attemptDiagnostics: Array<{status: 'rejected'; interactionMode: 'probe'; reason: string;
    toolId: string; tipRasMm: XYZ; stateBefore: string; stateAfter: string;
    removedIndicesNative: NativeCell[]}>;
  hashEncoding: string;
  replayStateInterpretation: string;
  unsupported: string[];
  interpretation: string;
}
// Request: executeDevelopmentEpisode({fixture:'generated-sequential-v1',selector:EpisodeSelector})
// Response: DevelopmentEpisodeResult; learned authorship is response-only.
// Only recorded frames are displayed; no invented timing, force, sensor or interpolation.
// Tissue is case.brainMask, target is case.compartments; exact cell deltas are native-grid indices.
// All *IndicesNative frame fields are per-frame deltas, not cumulative masks.
// Only insertion applies microstep removal/contact deltas; withdrawal/STOP are empty.
// Probe delta = microstep contact cells that remain occupied, only on probe insertions.
// Fixed fixture bounds: shape=[13,13,12], history<=6, replayFrames<=512, tools=2.

export interface DevelopmentEpisodeRequest {fixture:"generated-sequential-v1";selector:EpisodeSelector;}
export interface DevelopmentEpisodeResult {episodeAuthorship?:EpisodeAuthorship;episodeCanonicalJson:string;case:import("./types").CasePayload;episode:DevelopmentEpisode;}
