import type { Affine, Point3, Shape3 } from "./coordinates";

export interface AlignmentQuality {status:string;warnings:string[];notes?:string[];metrics:Record<string,number|null>;selected?:string}
export interface VesselDisplayFilter {scope:string;hiddenVoxels?:number;reason?:string;excludedVolumeMm3?:number;excludedFragments?:number;outsideVoxels?:number;boneConflictVoxels?:number;uncertainVoxels?:number}
export interface ViewerVolume {
  alignmentQuality?:Record<string,AlignmentQuality>;
  /** Native display identity; includes revision/preview generation, never just object identity. */
  displayIdentity?: string;
  /** Prepared off-thread for native sources. Legacy callers retain synchronous preparation. */
  packedLabels?: {data:Uint8Array;codes:Array<[number,number]>};
  caseId: string;
  caseHash: string;
  planningHash?: string;
  frame: string;
  affine: Affine;
  shape: Shape3;
  mri: Float32Array;
  /** Shared categorical volumes in the displayed grid, sampled with nearest cells. */
  labelSets?: Array<{id:string;data:Uint16Array}>;
  tractography?: Array<{id:string;label:string;points:Float32Array;offsets:Uint32Array}>;
  compartments: Array<{
    name: string;
    kind?: string;
    sourceIdentity?: string;
    qa?: AnatomyQuality;
    surfaceDiagnostics?: import('./surfacePolicy').SurfaceDiagnostics;
    mask: Uint8Array;
    labelSet?: string;
    labelValue?: number;
    color: string;
    volumeMm3?: number;
    displayFilter?: VesselDisplayFilter;
    /** Cached indexed display geometry. Masks remain the quantitative source. */
    surface?: {positions: Float32Array; source: Float32Array; normals: Float32Array; sourceNormals: Float32Array; indices: Uint32Array};
  }>;
}

export interface AnatomyQuality {
  sourceIdentity:string; voxelCount:number; componentCount:number;
  largestComponentFraction:number|null; boundaryTouchingVoxels:number;
  probabilityMismatchVoxels:number|null; reviewStatus:string;
  issues:string[]; accuracy:'unmeasured';
}

export interface ViewerRoute {
  route_id: string;
  comparisonSlot?: "A" | "B";
  entry_mm: number[];
  target_mm: number[];
  category?: string;
  tool: {
    working_length_mm: number;
    tip_length_mm: number;
    shaft_radius_mm: number;
    tip_radius_mm: number;
  };
  window?: { center_mm: number[]; normal_inward: number[]; radius_mm: number };
  geometry?: { failures?: Array<{ position_mm: number[]; reason: string }> };
}

export interface ViewerWorkspaceProps {
  surgicalInteraction?: import("../surgical-types").SurgicalInteraction;
  onSliceStroke?: (point: Point3 | null, phase: 'start' | 'move' | 'end') => void;
  workspaceSettings?: import('./workspaceSettings').ViewerWorkspaceSettings;
  onWorkspaceSettingsChange?: (settings: import('./workspaceSettings').ViewerWorkspaceSettings) => void;
  /** Acquired modality label; synthetic signals retain their separate label. */
  sourceLabel?: string;
  registeredFusion?: import('./registeredFusion').ViewerRegisteredFusion | null;
  /** Separate, unreviewed scan diagnostic; never becomes a CaseData compartment. */
  diagnosticLayer?: import("./diagnosticLayer").LoadedDiagnosticLayer | null;
  /** SHA-256 of the selected, displayed atlas-grid T1c image bytes. */
  diagnosticSourceSha256?: string | null;
  publicGoal?: import('./publicGoal').ViewerPublicGoal | null;
  /** Analytic software-fixture intensities; never an acquired MRI. */
  generatedSignal?: boolean;
  segmentationOnly?: boolean;
  caseData: ViewerVolume | null;
  visibleLayers: Record<string, boolean>;
  overlayOpacity: number;
  cursor: Point3 | null;
  onCursorChange: (point: Point3) => void;
  routes: ViewerRoute[];
  cameraMode: "anatomy" | "instruments";
  replay?: ViewerReplay | null;
  /** Separate source-bound patient research state; never legacy replay acceptance. */
  rehearsalState?: ViewerRehearsalState | null;
  /** Checked generated passive motion; separate from removal replay. */
  passiveReplay?: ViewerPassiveReplay | null;
  structuralProposal?: ViewerStructuralProposal | null;
  priorLayer?: ViewerPriorLayer | null;
  inspectionTool?: ViewerInspectionTool | null;
  onClearInspection?: () => void;
}

/** Host-validated complete inspector report; never a route or replay payload. */
export interface ViewerInspectionTool {
  scope: "unexecuted-native-axis-inspection";
  caseHash: string;
  planningHash: string;
  bindingHash: string;
  inspectionHash: string;
  actionId: string;
  report: unknown;
}

export interface ViewerPriorLayer {
  caseHash: string;
  proposalId: string;
  mapId: string;
  title: string;
  component: string;
  mapKind: "functional_concordance" | "structural_mask";
  values: Float32Array;
  coverage: Uint8Array;
  shape: Shape3;
  affine: Affine;
  frame: "RAS+";
  scope: "view-only-population-prior";
  provenance: "prior";
  reviewStatus: "alignment_review_required";
  planningEligible: false;
  patientSpecificFunction: false;
  valueUnits: "unitless";
  spatialUnits: "mm";
}

export interface ViewerStructuralProposal {
  caseHash: string;
  evidenceId: string;
  mask: Uint8Array;
  shape: Shape3;
  affine: Affine;
  frame: "RAS+";
  reviewStatus: string;
  provenance: "estimated";
  label: string;
  scope: "display-only-estimate";
}

export interface ViewerReplay {
  recordedEpisodeBounds?: import("./recordedTool").RecordedEpisodeBounds | null;
  recordedTool?: import("./recordedTool").RecordedToolPose | null;
  removedMask: Uint8Array;
  step: number;
  stepCount: number;
  scope: "native-source-grid";
  caseHash: string;
  shape: Shape3;
  affine: Affine;
  independentlyAccepted: boolean;
  removedTargetVolumeMm3: number;
  removedNormalVolumeMm3: number;
  residualTargetVolumeMm3: number;
}

/** A checked display frame, hydrated from a fresh-replayed generated episode.
 * Neither the generated clock nor these flags establish surgical validity. */
export interface ViewerPassiveReplay {
  scope: "generated-persistent-passive-episode";
  caseHash: string;
  episodeId: string;
  frameIndex: number;
  stepCount: number;
  shape: Shape3;
  affine: Affine;
  phase: "ready" | "held" | "inspected" | "withdrawn" | "both_withdrawn" | "stopped";
  elapsedGeneratedSeconds: number;
  inspectionStatus: "not_acquired" | "verified" | "unresolved";
  event: null | { kind: "generated_inspection" | "stop"; status?: "verified" | "unresolved"; reason?: string };
  recordedTools: import("./recordedTool").PassiveRecordedToolPose[];
  /** Locally derived from all accepted frame poses, never trusted from the backend. */
  recordedEpisodeBounds: import("./recordedTool").RecordedEpisodeBounds;
  /** Primary geometry checker covers pair clearance; independent checker covers tissue only. */
  primaryPairClearanceChecked: true;
  independentTissueMotionChecked: true;
}

/** Patient-source research episode. Its scope never asserts clinical validation or
 * acceptance by the generated-fixture legacy replay checker. */
export interface ViewerRehearsalState {
  scope: 'source-bound-post-exposure-research';
  caseHash: string;
  episodeId: string;
  stateId: string;
  frameIndex: number;
  stepCount: number;
  shape: Shape3;
  affine: Affine;
  frame: 'RAS+';
  elapsedSeconds: number;
  kind: string;
  tools: import('../rehearsal-types').RehearsalTool[];
  removedMask: Uint8Array;
  assumedExposureMask: Uint8Array;
  fieldContactMask: Uint8Array;
  contactMask: Uint8Array;
  metrics: import('../rehearsal-types').RehearsalMetrics;
  recordedEpisodeBounds: import('./recordedTool').RecordedEpisodeBounds;
}
