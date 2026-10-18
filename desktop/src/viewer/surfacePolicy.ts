// Generated policy copy is checked against the Python canonical resource by the cross-stack suite.
import policy from './surface_policy.json' with {type:'json'};
export const SURFACE_POLICY = policy;
export type SurfaceKind = 'context' | 'vessel' | 'tumor';
/** Only declared anatomy/material is used. Names and colors are not evidence. */
export function surfaceKind(kind?: string): SurfaceKind {
  return kind === 'tumor' ? 'tumor' : ['vessel','vessels','artery','vein','sinus'].includes(kind ?? '') ? 'vessel' : 'context';
}
export interface SurfaceDiagnostics {
  policy: string;
  kind: SurfaceKind;
  maxDisplacementMm: number;
  restoredComponents: number;
  backedOffVertices?:number;
  fallbackReason?:string;
  intersectionCheck: 'passed' | 'source-restored';
  preparationMs?: number;
}
/** Hann-windowed Chebyshev low-pass coefficients, with exact DC preservation.
 * The graph operator is the neighbour mean (spectrum [-1,1]); frequencies near
 * +1 describe smooth modes. This formula is shared with surface_fairing.py.
 */
export function sincCoefficients(kind: SurfaceKind): Float64Array {
  const {iterations:n,passBand}=policy.profiles[kind], theta=Math.acos(1-passBand);
  const c=Float64Array.from({length:n+1},(_,j)=>j===0?theta/Math.PI:
    2*Math.sin(j*theta)/(Math.PI*j)*(.5+.5*Math.cos(Math.PI*j/(n+1))));
  const dc=c.reduce((a,b)=>a+b,0);
  for(let j=0;j<c.length;j++)c[j]/=dc;
  return c;
}
