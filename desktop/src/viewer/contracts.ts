import type { Affine, Point3, Shape3 } from "./coordinates";

export interface ViewerVolume {
  caseId: string;
  caseHash: string;
  planningHash?: string;
  frame: string;
  affine: Affine;
  shape: Shape3;
  mri: Float32Array;
  compartments: Array<{
    name: string;
    mask: Uint8Array;
    color: string;
    volumeMm3?: number;
  }>;
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
  /** Analytic software-fixture intensities; never an acquired MRI. */
  generatedSignal?: boolean;
  caseData: ViewerVolume | null;
  visibleLayers: Record<string, boolean>;
  overlayOpacity: number;
  cursor: Point3 | null;
  onCursorChange: (point: Point3) => void;
  routes: ViewerRoute[];
  cameraMode: "anatomy" | "instruments";
  replay?: ViewerReplay | null;
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
