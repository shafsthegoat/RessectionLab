import {ChunkedMaskSurface} from './chunkedSurface.ts';
import {prepareSurfaceSoup,prepareIndexedSurface} from './prepareSurface.ts';
import {SurfaceIndexBudgetExceeded} from './sourceVertexIndex.ts';
import {maskSurface} from './surface.ts';
import type {Affine,Shape3} from './coordinates';
export interface IncrementalSurfaceRequest {jobId:number;layerId:string;previousIdentity:string;identity:string;shape:Shape3;affine:Affine;kind?:string;mask?:Uint8Array;indices?:Uint32Array;values?:Uint8Array;retainChunks?:boolean;smooth?:boolean}
/** One worker owns these cloned source cells. Cache budget covers masks and
 * chunk buffers and vertex registry; returned indexed/fair buffers are
 * transferred, never retained. Budget exhaustion falls back to exact soups. */
export class IncrementalSurfaceEngine {
  private readonly cache=new Map<string,{identity:string;signature:string;surface:ChunkedMaskSurface}>();
  private readonly maxBytes:number;
  constructor(maxBytes=128*1024*1024){this.maxBytes=maxBytes;}
  prepare(request:IncrementalSurfaceRequest){
    const started=performance.now(),signature=JSON.stringify([request.shape,request.affine,request.kind]),old=this.cache.get(request.layerId);
    // Initial mapping without tissue effects needs a display, not a hot edit
    // cache. Use the exact full extractor and release its cloned mask/soup
    // after transfer; the first real cut admits ordinary resident chunks.
    if(request.retainChunks===false){
      if(!request.mask||request.mask.some(n=>n>1))throw Error('Missing immutable source-mask admission for surface preparation');
      const soup=maskSurface(request.mask,request.shape),extractionMs=performance.now()-started,fairStarted=performance.now();
      const result=prepareSurfaceSoup(soup,request.mask,request.shape,request.affine,request.kind,request.smooth!==false);
      this.cache.delete(request.layerId);
      return {...result,jobId:request.jobId,layerId:request.layerId,identity:request.identity,
        resident:[...this.cache].map(([layerId,state])=>({layerId,identity:state.identity})),
        work:{incremental:false,fullReference:true,remeshedChunks:0,totalChunks:request.shape.reduce((n,s)=>n*Math.ceil((s+1)/32),1),
          retainedBytes:[...this.cache.values()].reduce((sum,c)=>sum+c.surface.retainedBytes,0),extractionMs,
          smoothingPrepared:result.smoothingPrepared!==false,globalFairingMs:performance.now()-fairStarted,preparationMs:performance.now()-started}};
    }
    const incremental=!!old&&old.identity===request.previousIdentity&&old.signature===signature&&!request.mask;
    let surface:ChunkedMaskSurface;
    if(incremental){
      const {indices,values}=request;if(!indices||!values||indices.length!==values.length||values.some(n=>n>1)||indices.some(i=>i>=old!.surface.mask.length))throw Error('Invalid incremental source cells');
      surface=old!.surface;for(let i=0;i<indices.length;i++)surface.mask[indices[i]]=values[i];
      try{surface.update(indices);}catch(error){
        if(!(error instanceof SurfaceIndexBudgetExceeded))throw error;
        this.cache.delete(request.layerId);surface=new ChunkedMaskSurface(surface.mask,request.shape);
      }
    }else {
      if(!request.mask||request.mask.some(n=>n>1))throw Error('Missing immutable source-mask admission for surface preparation');
      try{surface=new ChunkedMaskSurface(request.mask,request.shape,32,this.maxBytes);}catch(error){
        if(!(error instanceof SurfaceIndexBudgetExceeded))throw error;
        surface=new ChunkedMaskSurface(request.mask,request.shape);
      }
    }
    const extractionMs=performance.now()-started,assemblyStarted=performance.now();
    const indexed=surface.indexedChunks?surface.indexed(request.smooth===false&&request.kind==='bone'?request.affine:undefined):undefined,soup=indexed?undefined:surface.positions(),assemblyMs=performance.now()-assemblyStarted,fairStarted=performance.now();
    const result=indexed?prepareIndexedSurface(indexed.positions,indexed.indices,surface.mask,request.shape,request.affine,request.kind,request.smooth!==false,indexed.normals):prepareSurfaceSoup(soup!,surface.mask,request.shape,request.affine,request.kind,request.smooth!==false);
    this.cache.delete(request.layerId);
    if(surface.retainedBytes<=this.maxBytes)this.cache.set(request.layerId,{identity:request.identity,signature,surface});
    let retainedBytes=[...this.cache.values()].reduce((sum,c)=>sum+c.surface.retainedBytes,0);
    while(retainedBytes>this.maxBytes&&this.cache.size){const key=this.cache.keys().next().value!;retainedBytes-=this.cache.get(key)!.surface.retainedBytes;this.cache.delete(key);}
    return {...result,jobId:request.jobId,layerId:request.layerId,identity:request.identity,
      resident:[...this.cache].map(([layerId,state])=>({layerId,identity:state.identity})),
      work:{incremental,...surface.normalWork,indexedChunks:surface.indexedChunks,indexedCorners:surface.lastIndexedCorners,remeshedChunks:surface.lastRemeshedChunks,totalChunks:surface.totalChunks,retainedBytes,extractionMs,assemblyMs,
        smoothingPrepared:result.smoothingPrepared!==false,globalFairingMs:performance.now()-fairStarted,preparationMs:performance.now()-started}};
  }
}
