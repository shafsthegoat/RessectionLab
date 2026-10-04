import type { Affine, Point3, Shape3 } from "./coordinates";

export interface ViewerVolume {
  caseId: string;
  caseHash: string;
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
  caseData: ViewerVolume | null;
  visibleLayers: Record<string, boolean>;
  overlayOpacity: number;
  cursor: Point3 | null;
  onCursorChange: (point: Point3) => void;
  routes: ViewerRoute[];
  cameraMode: "anatomy" | "instruments";
}
