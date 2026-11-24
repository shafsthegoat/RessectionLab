import {engineImplicitSurface} from './engineImplicitSurface.ts';
import type {EngineSurfaceReference} from './engineImplicitSurface';
import {maskSurface} from './surface.ts';
import {indexSurface,fairIndexedSurface,vesselLimits} from './surfaceSmoothing.ts';
import {indexedSurfaceNormals} from './surfaceNormals.ts';
import {surfaceKind} from './surfacePolicy.ts';
import type {Affine,Bounds3,Shape3} from './coordinates';
import type {SurfaceDiagnostics} from './surfacePolicy';
import type {SurfaceIntersectionAudit} from './surfaceSafety';
import type {SurfacePolynomialFilter} from './incrementalSurfaceFilter';
import type {OperativeVesselSurface} from './operativeVesselSurface';
export interface PreparedSurface {positions:Float32Array;normals:Float32Array;smoothPositions?:Float32Array;smoothNormals?:Float32Array;smoothIndices?:Uint32Array;indices:Uint32Array;diagnostics?:SurfaceDiagnostics;smoothingPrepared?:boolean;representation?:'stable-source-ids'|'operative-vessel';activeBounds?:Bounds3}
export function prepareOperativeVessel({positions,normals,indices,mode}:OperativeVesselSurface):PreparedSurface {
  const raw=mode==='source-cell-boundary';
  return {positions,normals,indices,representation:'operative-vessel',diagnostics:{
    policy:raw?'engine-source-vessel-caps-v1':'engine-native-probability-v1',
    kind:'vessel',triangles:indices.length/3,maxDisplacementMm:0,restoredComponents:0,
    ...(raw?{intersectionCheck:'source-restored',surfaceField:'categorical',fallbackReason:'Accepted vessel contact uses source cell faces after this edit.'}:
      {intersectionCheck:'passed',surfaceField:'native-probability'}),
  }};
}
export function prepareSurface(mask:Uint8Array,shape:Shape3,affine:Affine,kind?:string,smooth=true,reference?:EngineSurfaceReference|null):PreparedSurface {
  const started=performance.now(),result=prepareSurfaceSoup(maskSurface(mask,shape),mask,shape,affine,kind,smooth,reference);
  if(result.diagnostics)result.diagnostics.preparationMs=performance.now()-started;return result;
}
export function prepareSurfaceSoup(soup:Float32Array,mask:Uint8Array,shape:Shape3,affine:Affine,kind?:string,smooth=true,reference?:EngineSurfaceReference|null):PreparedSurface {
  const started=performance.now(),indexed=indexSurface(soup,shape),result=prepareIndexedSurface(indexed.positions,indexed.indices,mask,shape,affine,kind,smooth,undefined,undefined,undefined,reference);
  if(result.diagnostics)result.diagnostics.preparationMs=performance.now()-started;return result;
}
/** Already canonical indexed source buffers; conditioning is identical to the
 * full extractor, including shared seam normals and physical vessel limits. */
export function prepareIndexedSurface(positions:Float32Array,indices:Uint32Array,mask:Uint8Array,shape:Shape3,affine:Affine,kind?:string,smooth=true,preparedNormals?:Float32Array,audit?:SurfaceIntersectionAudit,filter?:SurfacePolynomialFilter,reference?:EngineSurfaceReference|null):PreparedSurface {
  const started=performance.now();
  if(preparedNormals&&preparedNormals.length!==positions.length)throw Error('Cached source normals differ from canonical vertices');
  const normals=preparedNormals??indexedSurfaceNormals(positions,indices,affine);
  if(!smooth)return {positions,indices,normals,smoothingPrepared:false};
  if(reference!==undefined){
    const display=engineImplicitSurface(positions,indices,normals,mask,shape,affine,kind,reference);
    return {positions,indices,normals,smoothPositions:display.positions,smoothNormals:display.smoothNormals,smoothIndices:display.smoothIndices,diagnostics:display.diagnostics,smoothingPrepared:true};
  }
  const limits=surfaceKind(kind)==='vessel'?vesselLimits(positions,affine,mask,shape):undefined;
  const display=fairIndexedSurface(positions,indices,affine,kind,limits,audit,filter);
  display.diagnostics.preparationMs=performance.now()-started;
  return {positions,indices,normals,smoothPositions:display.positions,smoothNormals:indexedSurfaceNormals(display.positions,indices,affine),diagnostics:display.diagnostics,smoothingPrepared:true};
}
