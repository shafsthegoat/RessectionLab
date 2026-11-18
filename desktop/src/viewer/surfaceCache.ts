import type {PreparedSurface} from './prepareSurface';
import type {Affine} from './coordinates';
import {SURFACE_POLICY,surfaceKind} from './surfacePolicy.ts';
// Weak source-mask keys do not keep patient volumes alive after replacement.
// Source and modeled masks are immutable; their owners replace them on edits.
type SurfaceCache=WeakMap<Uint8Array,Map<string,PreparedSurface>>;
const cache:SurfaceCache=new WeakMap();
const partitions=new WeakMap<object,SurfaceCache>();
function partitionCache(reference?:object):SurfaceCache{
  if(!reference)return cache;
  let result=partitions.get(reference);
  if(!result){result=new WeakMap();partitions.set(reference,result);}
  return result;
}
const signature=(affine:Affine,kind:string|undefined,shape:number[]|undefined,smooth:boolean)=>JSON.stringify([SURFACE_POLICY.version,affine,surfaceKind(kind),shape,smooth]);
export function cachedSurface(mask:Uint8Array,affine:Affine,kind?:string,shape?:number[],smooth=true,reference?:object){
  const entries=partitionCache(reference).get(mask);return entries?.get(signature(affine,kind,shape,smooth))??(!smooth?entries?.get(signature(affine,kind,shape,true)):undefined);
}
export function retainSurface(mask:Uint8Array,affine:Affine,kind:string|undefined,surface:PreparedSurface,shape?:number[],reference?:object){
  // Keep raw representation and active bounds with the immutable snapshot;
  // historical source IDs cannot be interpreted as compact active vertices.
  const selected=partitionCache(reference),entries=selected.get(mask)??new Map<string,PreparedSurface>();entries.set(signature(affine,kind,shape,surface.smoothingPrepared!==false),surface);selected.set(mask,entries);
}
