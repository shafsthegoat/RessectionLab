import type {PreparedDisplaySurface} from './displaySurfaceNormals';
import type {Affine} from './coordinates';
import {SURFACE_POLICY,surfaceKind} from './surfacePolicy.ts';
// Weak source-mask keys do not keep patient volumes alive after replacement.
// Source and modeled masks are immutable; their owners replace them on edits.
type SurfaceCache=WeakMap<Uint8Array,Map<string,PreparedDisplaySurface>>;
const cache:SurfaceCache=new WeakMap();
const partitions=new WeakMap<object,SurfaceCache>();
function partitionCache(reference?:object):SurfaceCache{
  if(!reference)return cache;
  let result=partitions.get(reference);
  if(!result){result=new WeakMap();partitions.set(reference,result);}
  return result;
}
const signature=(affine:Affine,kind:string|undefined,shape:number[]|undefined,smooth:boolean,displayNormals:boolean)=>JSON.stringify([SURFACE_POLICY.version,affine,surfaceKind(kind),shape,smooth,displayNormals]);
export function cachedSurface(mask:Uint8Array,affine:Affine,kind?:string,shape?:number[],smooth=true,reference?:object,displayNormals=false){
  const entries=partitionCache(reference).get(mask);if(!entries)return;
  const find=(smoothing:boolean)=>(displayNormals?entries.get(signature(affine,kind,shape,smoothing,true)):undefined)??entries.get(signature(affine,kind,shape,smoothing,false));
  return find(smooth)??(!smooth?find(true):undefined);
}
export function retainSurface(mask:Uint8Array,affine:Affine,kind:string|undefined,surface:PreparedDisplaySurface,shape?:number[],reference?:object){
  // Keep raw representation and active bounds with the immutable snapshot;
  // historical source IDs cannot be interpreted as compact active vertices.
  const selected=partitionCache(reference),entries=selected.get(mask)??new Map<string,PreparedDisplaySurface>();
  entries.set(signature(affine,kind,shape,surface.smoothingPrepared!==false,surface.displayNormalsPrepared===true),surface);selected.set(mask,entries);
}
