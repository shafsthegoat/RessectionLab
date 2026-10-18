import {maskSurface} from './surface.ts';
import {indexSurface,fairIndexedSurface,vesselLimits} from './surfaceSmoothing.ts';
import {indexedSurfaceNormals} from './surfaceNormals.ts';
import {surfaceKind} from './surfacePolicy.ts';
import type {Affine,Shape3} from './coordinates';
import type {SurfaceDiagnostics} from './surfacePolicy';
export interface PreparedSurface {positions:Float32Array;normals:Float32Array;smoothPositions?:Float32Array;smoothNormals?:Float32Array;indices:Uint32Array;diagnostics?:SurfaceDiagnostics}
export function prepareSurface(mask:Uint8Array,shape:Shape3,affine:Affine,kind?:string,smooth=true):PreparedSurface {
  const started=performance.now(),indexed=indexSurface(maskSurface(mask,shape)),positions=indexed.positions,indices=indexed.indices;
  const normals=indexedSurfaceNormals(positions,indices,affine);
  if(!smooth)return {positions,indices,normals};
  const limits=surfaceKind(kind)==='vessel'?vesselLimits(positions,affine,mask,shape):undefined;
  const display=fairIndexedSurface(positions,indices,affine,kind,limits);
  display.diagnostics.preparationMs=performance.now()-started;
  return {positions,indices,normals,smoothPositions:display.positions,smoothNormals:indexedSurfaceNormals(display.positions,indices,affine),diagnostics:display.diagnostics};
}
