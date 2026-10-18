import type {PreparedSurface} from './prepareSurface';
import type {Affine} from './coordinates';
import {SURFACE_POLICY,surfaceKind} from './surfacePolicy.ts';
// Weak source-mask keys do not keep patient volumes alive after replacement.
// Source and modeled masks are immutable; their owners replace them on edits.
const cache=new WeakMap<Uint8Array,Map<string,PreparedSurface>>();
const signature=(affine:Affine,kind?:string,shape?:number[])=>JSON.stringify([SURFACE_POLICY.version,affine,surfaceKind(kind),shape]);
export function cachedSurface(mask:Uint8Array,affine:Affine,kind?:string,shape?:number[]){return cache.get(mask)?.get(signature(affine,kind,shape));}
export function retainSurface(mask:Uint8Array,affine:Affine,kind:string|undefined,surface:PreparedSurface,shape?:number[]){
  const entries=cache.get(mask)??new Map<string,PreparedSurface>();entries.set(signature(affine,kind,shape),surface);cache.set(mask,entries);
}
