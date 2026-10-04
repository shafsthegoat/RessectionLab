export type Vec3 = [number, number, number];
export type Mat4 = number[][];

export interface ArrayDescriptor {
  assetId: string;
  dtype: string;
  shape: number[];
  byteOrder: "little";
  order: "C";
  sha256: string;
  byteLength: number;
}

export interface StructuralEvidence {
  evidenceId: string;
  kind: string;
  provenance: string;
  reviewStatus: string;
  reviewRequired: boolean;
  corticalAccessPermitted: false;
  sourceHash: string;
  sourceFrameHash: string;
  sourceFileHash: string | null;
  maskHash: string;
  modelHash: string | null;
  runHash: string;
  evidenceHash: string;
  method: string;
  array: ArrayDescriptor;
  metadata: Record<string, unknown>;
  review: unknown;
}

export interface StructuralProposalView {
  caseHash: string;
  evidenceId: string;
  mask: Uint8Array;
  shape: Vec3;
  affine: Mat4;
  frame: "RAS+";
  reviewStatus: string;
  provenance: "estimated";
  scope: "display-only-estimate";
  label: string;
  annotationOutsideVoxelCount: number;
  outsideAnnotationPointMm: Vec3 | null;
  voxelBounds: { min: Vec3; max: Vec3 };
  estimatedVoxelCount: number;
}

export type PriorComponent =
  | "motor"
  | "phonology"
  | "semantics"
  | "speech_articulation";
export type PriorMapKind = "functional_concordance" | "structural_mask";
export interface PriorProposal {
  proposalId: string;
  mapId: string;
  component: PriorComponent;
  mapKind: PriorMapKind;
  title: string;
  provenance: "prior";
  reviewStatus: "alignment_review_required";
  viewOnly: true;
  planningEligible: false;
  patientSpecificFunction: false;
  clinicalDeficitProbability: null;
  clinicalRiskReason: string;
  frame: "RAS+";
  spatialUnits: "mm";
  valueUnits: "unitless";
  affine: Mat4;
  shape: Vec3;
  data: ArrayDescriptor;
  samplingCoverage: ArrayDescriptor;
  samplingCoverageFraction: number;
  coverageMeaning: "atlas field of view, not patient functional coverage";
  interpolation: "linear" | "nearest_neighbor";
  evidenceHash: string;
  registrationHash: string;
  sourceImageHash: string;
  sourceFrameHash: string;
  sourcePlanningHash: string;
  registrationCaseHash: string;
  source: {
    source_id: string;
    uri: string;
    sha256: string;
    license?: string;
    native_frame: string;
    provenance: "prior";
  };
  metadata: Record<string, unknown>;
  provenanceRecord: Record<string, unknown>;
}
export interface PriorLayerView {
  caseHash: string;
  proposalId: string;
  mapId: string;
  title: string;
  component: PriorComponent;
  mapKind: PriorMapKind;
  values: Float32Array;
  coverage: Uint8Array;
  shape: Vec3;
  affine: Mat4;
  frame: "RAS+";
  scope: "view-only-population-prior";
  provenance: "prior";
  reviewStatus: "alignment_review_required";
  planningEligible: false;
  patientSpecificFunction: false;
  valueUnits: "unitless";
  spatialUnits: "mm";
  proposal: PriorProposal;
}

export interface CasePayload {
  caseId: string;
  caseHash: string;
  planningHash?: string;
  frame: string;
  affine: Mat4;
  shape: Vec3;
  spacingMm: Vec3;
  mri: ArrayDescriptor;
  brainMask?: ArrayDescriptor | null;
  brainSupport?: {
    usableForResearchSimulation: boolean;
    reviewStatus: string;
    corticalAccessPermitted: false;
    reason?: string;
  };
  structuralEvidence?: StructuralEvidence[];
  priorProposals?: PriorProposal[];
  compartments: {
    name: string;
    volumeMm3: number;
    array: ArrayDescriptor;
    sourceArray?: ArrayDescriptor;
  }[];
  unknowns: string[];
  metadata: Record<string, unknown>;
  sourceRefs?: {
    source_id: string;
    uri: string;
    sha256?: string;
    license?: string;
    native_frame?: string;
    provenance?: string;
  }[];
  context?: unknown;
  artifacts?: Record<string, unknown>;
  intensityRange?: [number, number];
  targetCentroidMm?: Vec3;
}

export interface ViewerCase {
  caseId: string;
  caseHash: string;
  frame: string;
  affine: Mat4;
  shape: Vec3;
  spacingMm?: Vec3;
  intensityRange?: [number, number];
  mri: Float32Array;
  compartments: {
    name: string;
    volumeMm3: number;
    mask: Uint8Array;
    color: string;
    mesh?: {
      positions: Float32Array;
      indices: Uint32Array;
      frame: "voxel" | "RAS+";
    };
  }[];
}

export interface RouteCandidate {
  route_id: string;
  planning_model_hash?: string;
  case_hash: string;
  entry_mm: Vec3;
  target_mm: Vec3;
  tool_id: string;
  target_compartment: string;
  tool: {
    tool_id: string;
    tip_radius_mm: number;
    shaft_radius_mm: number;
    working_length_mm: number;
    tip_length_mm: number;
    max_access_angle_deg: number;
  };
  window: {
    center_mm: Vec3;
    normal_inward: Vec3;
    radius_mm: number;
    window_id: string;
  };
  geometry: {
    feasible: boolean;
    clearance_mm: number | null;
    failures: { reason: string; detail: string; position_mm: Vec3 }[];
  };
  route_length_mm: number;
  accessible_target_volume_mm3: number;
  accessible_target_volume_mm3_by_compartment: Record<string, number>;
  normal_tissue_exposure_mm3: number | null;
  structure_contact_volume_mm3: Record<string, number | null>;
  category: "pareto" | "dominated" | "rejected";
  assessment: string;
  unknowns: string[];
  assumptions: string[];
  simulated_removed_target_volume_mm3: null;
  clinical_deficit_probability: null;
}

export interface RefinementReadiness {
  caseHash: string;
  routeId: string;
  status: "ready" | "no_actionable_moves";
  legalNonStopActions: number;
  reasons: string[];
  decision_model_hash: string;
  route_binding: Record<string, unknown>;
  optimizationChoiceScope: "STOP_or_declared_native_stroke";
}

export interface SearchResult {
  combined_models?: boolean;
  planning_model_hash?: string | null;
  planning_model_hashes?: string[];
  search_models?: Record<string, unknown>[];
  candidates: RouteCandidate[];
  elapsed_seconds: number;
  cancelled: boolean;
  assumptions: string[];
  case_hash: string;
}

export interface BridgeEvent {
  id: string;
  op?: string;
  event:
    | "started"
    | "progress"
    | "result"
    | "error"
    | "cancelled"
    | "engineStopped"
    | "menuAction";
  action?: "openCase" | "saveCase";
  progress?: { fraction: number; message: string };
  error?: { code: string; message: string };
  result?: unknown;
}

export interface TrainingStats {
  status?: string;
  readiness?: RefinementReadiness;
  gradient_steps?: number;
  optimization_environment_steps?: number;
  selection_environment_steps?: number;
  elapsed_seconds?: number;
  initial_selection_return?: number | null;
  selected_selection_return?: number | null;
  actor_parameters_changed?: boolean;
  role?: string;
  final_evaluation?: boolean;
  replay_status?: string;
  selection_history?: { gradient_steps: number; mean_return: number }[];
  replay?: {
    stepCount: number;
    metrics: Record<string, unknown>;
    native_certificate: Record<string, unknown>;
  } | null;
}
export interface TrainingRun {
  runId: string;
  caseHash: string;
  status: string;
  createdAt?: number;
  hasCheckpoint?: boolean;
  hasAcceptedReplay?: boolean;
  config: {
    budgetSeconds: number;
    seed: number;
    routeId?: string;
    optimizationChoiceScope?: string;
  };
  training?: TrainingStats;
}
export interface ReplayResult {
  caseHash: string;
  runId: string;
  step: number;
  stepCount: number;
  role: "selection";
  finalEvaluation: false;
  frame: "RAS+";
  affine: number[][];
  removedMask: ArrayDescriptor;
  clinicalDeficitProbability: null;
  simulatedRemovedTargetVolumeMm3: number;
  simulatedRemovedNormalVolumeMm3: number;
  modeledResidualTargetVolumeMm3: number;
  training: TrainingStats;
}
export interface CertifiedReplay {
  result: ReplayResult;
  mask: Uint8Array;
}
export interface TrainingApi {
  trainPatient(args: {
    caseHash: string;
    budgetSeconds?: number;
    seed?: number;
    routeId?: string;
    resumeRunId?: string;
  }): Promise<TrainingRun>;
  listRuns(args: {
    caseHash: string;
  }): Promise<{ caseHash: string; runs: TrainingRun[] }>;
  replayTraining(args: {
    caseHash: string;
    runId: string;
    step?: number;
  }): Promise<ReplayResult>;
  exportCandidate(args: {
    caseHash: string;
    runId: string;
  }): Promise<{ exported?: boolean } | null>;
}

export interface ResectionApi extends TrainingApi {
  readOnly?: boolean;
  ping(): Promise<unknown>;
  startupCase(): Promise<CasePayload | null>;
  createSyntheticCase(): Promise<CasePayload>;
  openCase(): Promise<CasePayload | null>;
  importNifti(): Promise<CasePayload | null>;
  importStructuralEvidence?(args: {
    caseHash: string;
    variant: "main" | "nocsf";
  }): Promise<CasePayload | null>;
  saveCase(args: {
    caseHash: string;
    workspace?: Record<string, unknown>;
  }): Promise<{ saved?: boolean } | null>;
  generateRoutes(args?: Record<string, unknown>): Promise<SearchResult>;
  generateNativeRoutes?(args: { caseHash: string }): Promise<SearchResult>;
  inspectRefinement?(args: {
    caseHash: string;
    routeId: string;
  }): Promise<RefinementReadiness>;
  inspectEvidence(args?: Record<string, unknown>): Promise<unknown>;
  cancel(requestId: string): Promise<unknown>;
  readAsset(assetId: string): Promise<Uint8Array>;
  onEvent(callback: (event: BridgeEvent) => void): () => void;
}

declare global {
  interface Window {
    resectionApi?: ResectionApi;
  }
}
