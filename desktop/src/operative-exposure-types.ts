import type {Point3} from './viewer/coordinates';

/** Accepted research exposure state. Source IDs index scene.shape in C order;
 * positions, vector axes and fixture locations use the scene's patient RAS. */
export type ExposureBodyName = 'scalp' | 'bone' | 'dura';
export interface OperativeExposureBody {
  cells: number[];
  centersMm: Point3[];
  status: string;
  heldBy: 'left' | 'right' | null;
  angleRad: number;
  hingeOriginMm: Point3;
  hingeAxis: Point3;
  cutCells: number[];
  seamEdges: Array<[number, number]>;
  cutCount: number;
  edgeCount: number;
  gripMm: Point3;
  anchorsMm: Point3[];
  closedAnchors: number[];
}
export interface OperativeExposureObject {
  id: string;
  kind: 'raney-clip' | 'retaining-stay' | 'cottonoid' | 'hemostat' | 'bone-anchor' | 'suture';
  positionMm: Point3;
  status: 'placed' | 'carried' | 'retrieved' | 'intended-retained';
  radiusMm: number;
  sourceCell?: number;
  normal?: Point3;
  endpointsMm?: [Point3, Point3];
  sourceContactMm?: Point3;
  clipSegments?: Array<{aMm: Point3; bMm: Point3; radiusMm: number}>;
  body: ExposureBodyName | 'field';
}
export interface OperativeExposure {
  schema: string;
  exteriorNormalRas?: Point3;
  boneRest?: {pads: Array<{aMm: Point3; bMm: Point3; radiusMm: number}>; engaged: boolean};
  bodies: Record<ExposureBodyName, OperativeExposureBody>;
  objects: OperativeExposureObject[];
  accessOpen: boolean;
  complete: boolean;
}
