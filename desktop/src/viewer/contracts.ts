import type { Affine, Bounds3, Point3, Shape3 } from "./coordinates";

export interface AlignmentQuality {status:string;warnings:string[];notes?:string[];metrics:Record<string,number|null>;selected?:string}
export interface TissueDisplaySupport {
  version:string; status:'supported'|'unfiltered'|'unavailable'; reason?:string;
  sourceSha256:string; rawDataSha256:string; sampledDataSha256:string;
  sourceVoxels?:number; retainedVoxels?:number; hiddenVoxels?:number;
}
export interface VesselDisplayFilter {scope:string;hiddenVoxels?:number;reason?:string;excludedVolumeMm3?:number;excludedFragments?:number;outsideVoxels?:number;boneConflictVoxels?:number;uncertainVoxels?:number}
export interface PatientContextSource {
  seriesId:string; modality:string; sourceSha256:string; alignment:string; reason?:string;
  inPatientBounds:boolean; voxelToPatientRasMm?:Affine; nativeShape?:Shape3;
}
export interface PatientContext {
  sources:PatientContextSource[];
  boundsRasMm?:Bounds3;
}
/** Immutable scalar samples and physical display frame of a surface's own acquisition. */
export interface SurfaceAppearanceSource {
  seriesId:string; modality:string; sourceSha256:string; signalSha256:string;
  intensityUnits?:'HU'|'source';
  softTissueWindow?:[number,number];
  nativeWallCrop?:true;
  affine:Affine; shape:Shape3; data:Float32Array; nativeAffine?:Affine; nativeShape?:Shape3;
}
export interface ViewerVolume {
  patientContext?:PatientContext;
  headSupport?:{data:Uint8Array;sha256:string};
  /** Display-only exclusion of a detached CT sheet on the exact native grid, in C order. */
  ctExterior?:{offset:Point3;shape:Shape3;data:Uint8Array;sha256:string};
  sourceFrameIssue?:string|null;
  displayPreview?:{factor:number;meaning:string}|null;
  alignmentQuality?:Record<string,AlignmentQuality>;
  /** Native display identity; includes revision/preview generation, never just object identity. */
  displayIdentity?: string;
  /** Modality of the selected acquisition in the native workspace. */
  modality?: string;
  /** Identity of the acquisition supplying the scalar grid. */
  sourceSeriesId?: string;
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
  tractography?: Array<{id:string;label:string;sourceSeries?:string;alignment?:string;points:Float32Array;offsets:Uint32Array}>;
  compartments: Array<{
    name: string;
    /** Source-facing anatomical name; name remains the stable layer identifier. */
    label?: string;
    kind?: string;
    sourceIdentity?: string;
    appearanceSource?:SurfaceAppearanceSource;
    /** Display-only cut support from accepted, locally stationary source cells. */
    cavity?:import('./cavityGeometry').AcceptedCavity;
    displayKind?: string;
    sourceSeries?:string;
    alignment?:string;
    qa?: AnatomyQuality;
    surfaceDiagnostics?: import('./surfacePolicy').SurfaceDiagnostics;
    mask: Uint8Array;
    /** Static remainder when an accepted body has its own continuous surface. */
    generatedFinish?:boolean;
    surfaceMask?: Uint8Array;
    labelSet?: string;
    labelValue?: number;
    color: string;
    volumeMm3?: number;
    displayFilter?: VesselDisplayFilter;
    sourceSupport?: TissueDisplaySupport;
    /** Cached indexed display geometry. Masks remain the quantitative source. */
    surface?: {positions: Float32Array; source: Float32Array; normals: Float32Array; sourceNormals: Float32Array; indices: Uint32Array};
    /** For a score-derived mesh on the exact engine grid, one native source cell per vertex. */
    vesselSourceCells?: Uint32Array;
    vesselGeometryBinding?:string;
    /** Optional source-score display covector and weight (xyzw) per original wall vertex. */
    vesselScoreLighting?:Float32Array;
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
  comparisonSlot?: "A" | "B" | "C";
  /** Research planning geometry only; never a physical execution admission. */
  scope?: "geometric-approach";
  label?: string;
  selected?: boolean;
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

export interface ViewerVesselTrace {
  layerId:string;
  points:Point3[];
  /** Display a physical neighborhood of the verified source centerline. */
  isolationRadiusMm?:number;
}

export interface ViewerWorkspaceProps {
  approachOverview?:boolean;
  /** Inspect the modeled grid committed with the displayed instrument state. */
  engineInspection?:boolean;
  readCamera?: import('react').Ref<()=>import('./workspaceSettings').ViewerCameraState|null>;
  surgicalInteraction?: import("../surgical-types").SurgicalInteraction;
  onSliceStroke?: (point: Point3 | null, phase: 'start' | 'move' | 'end') => void;
  workspaceSettings?: import('./workspaceSettings').ViewerWorkspaceSettings;
  onWorkspaceSettingsChange?: (settings: import('./workspaceSettings').ViewerWorkspaceSettings) => void;
  /** Acquired modality label; synthetic signals retain their separate label. */
  sourceLabel?: string;
  /** The acquired scan is interactive while its exact anatomy is prepared. */
  sourcePreview?: boolean;
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
  /** Explicit framing command; mode changes alone preserve a restored camera. */
  cameraFitRequest?:number;
  /** A compartment name or supplied tract id. Display selection never changes assessment. */
  selectedStructure?: string | null;
  vesselTrace?: ViewerVesselTrace | null;
  /** A double-click on a registered native label selects its source evidence. */
  onPickStructure?: (name: string, point:Point3) => void;
  ctaCoverage?: {mode:'combined'|'focused'|null;onChange:(combined:boolean)=>void};
  /** Explicit focus command; selecting a row alone does not move a restored camera. */
  structureFocusRequest?: number;
  /** An assessed source-RAS concern location, when focusing a local intersection. */
  structureFocusPoint?: Point3 | null;
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
