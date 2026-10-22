import type {PreparedSurface} from './prepareSurface';
import type {Affine} from './coordinates';
import {SURFACE_POLICY,surfaceKind} from './surfacePolicy.ts';
// Weak source-mask keys do not keep patient volumes alive after replacement.
// Source and modeled masks are immutable; their owners replace them on edits.
const cache=new WeakMap<Uint8Array,Map<string,PreparedSurface>>();
const signature=(affine:Affine,kind:string|undefined,shape:number[]|undefined,smooth:boolean)=>JSON.stringify([SURFACE_POLICY.version,affine,surfaceKind(kind),shape,smooth]);
export function cachedSurface(mask:Uint8Array,affine:Affine,kind?:string,shape?:number[],smooth=true){
  const entries=cache.get(mask);return entries?.get(signature(affine,kind,shape,smooth))??(!smooth?entries?.get(signature(affine,kind,shape,true)):undefined);
}
export function retainSurface(mask:Uint8Array,affine:Affine,kind:string|undefined,surface:PreparedSurface,shape?:number[]){
  // Keep raw representation and active bounds with the immutable snapshot;
  // historical source IDs cannot be interpreted as compact active vertices.
  const entries=cache.get(mask)??new Map<string,PreparedSurface>();entries.set(signature(affine,kind,shape,surface.smoothingPrepared!==false),surface);cache.set(mask,entries);
}
