import {ChunkedMaskSurface} from './chunkedSurface.ts';
import {prepareSurfaceSoup,prepareIndexedSurface} from './prepareSurface.ts';
import {SurfaceIndexBudgetExceeded} from './sourceVertexIndex.ts';
import {maskSurface} from './surface.ts';
import {IncrementalSurfaceIntersectionAudit} from './surfaceSafety.ts';
import type {Affine,Shape3} from './coordinates';
export interface IncrementalSurfaceRequest {jobId:number;layerId:string;previousIdentity:string;identity:string;shape:Shape3;affine:Affine;kind?:string;mask?:Uint8Array;indices?:Uint32Array;values?:Uint8Array;retainChunks?:boolean;smooth?:boolean}
// Small lesions remesh most of their source chunks on each cut. The measured
// certificate overhead exceeded any savings there; retain the full audit.
const MIN_CERTIFICATE_FACES=100000;
/** One worker owns these cloned source cells. Cache budget covers masks and
 * chunk buffers and vertex registry; returned indexed/fair buffers are
 * transferred, never retained. Budget exhaustion falls back to exact soups. */
export class IncrementalSurfaceEngine {
  private readonly cache=new Map<string,{identity:string;signature:string;surface:ChunkedMaskSurface;audit?:IncrementalSurfaceIntersectionAudit}>();
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
          retainedBytes:[...this.cache.values()].reduce((sum,c)=>sum+c.surface.retainedBytes+(c.audit?.retainedBytes??0),0),extractionMs,
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
      try{surface=new ChunkedMaskSurface(request.mask,request.shape,32,this.maxBytes,request.kind==='bone');}catch(error){
        if(!(error instanceof SurfaceIndexBudgetExceeded))throw error;
        surface=new ChunkedMaskSurface(request.mask,request.shape);
      }
    }
    const extractionMs=performance.now()-started,assemblyStarted=performance.now();
    const stable=incremental&&request.smooth===false&&request.kind==='bone'?surface.stableIndexed(request.affine):undefined;
    const indexed=stable??(surface.indexedChunks?surface.indexed(request.smooth===false&&request.kind==='bone'?request.affine:undefined,request.smooth!==false):undefined),soup=indexed?undefined:surface.positions(),assemblyMs=performance.now()-assemblyStarted,fairStarted=performance.now();
    const layout=indexed&&indexed.indices.length/3>=MIN_CERTIFICATE_FACES&&'auditLayout'in indexed?indexed.auditLayout:undefined;
    const audit=layout?(incremental?old?.audit:undefined)??new IncrementalSurfaceIntersectionAudit():undefined;
    const observer=audit&&layout?audit.begin(layout,indexed!.indices,Math.max(0,this.maxBytes-surface.retainedBytes)):undefined;
    const result=stable?{...stable,smoothingPrepared:false}:indexed?prepareIndexedSurface(indexed.positions,indexed.indices,surface.mask,request.shape,request.affine,request.kind,request.smooth!==false,indexed.normals,observer):prepareSurfaceSoup(soup!,surface.mask,request.shape,request.affine,request.kind,request.smooth!==false);
    this.cache.delete(request.layerId);
    if(surface.retainedBytes+(audit?.retainedBytes??0)<=this.maxBytes)this.cache.set(request.layerId,{identity:request.identity,signature,surface,audit});
    let retainedBytes=[...this.cache.values()].reduce((sum,c)=>sum+c.surface.retainedBytes+(c.audit?.retainedBytes??0),0);
    while(retainedBytes>this.maxBytes&&this.cache.size){const key=this.cache.keys().next().value!,entry=this.cache.get(key)!;retainedBytes-=entry.surface.retainedBytes+(entry.audit?.retainedBytes??0);this.cache.delete(key);}
    return {...result,jobId:request.jobId,layerId:request.layerId,identity:request.identity,
      resident:[...this.cache].map(([layerId,state])=>({layerId,identity:state.identity})),
      work:{incremental,stableSourceIds:!!stable,...surface.normalWork,indexedChunks:surface.indexedChunks,indexedCorners:surface.lastIndexedCorners,remeshedChunks:surface.lastRemeshedChunks,totalChunks:surface.totalChunks,retainedBytes,extractionMs,assemblyMs,
        intersectionAudit:audit?.statistics,smoothingPrepared:result.smoothingPrepared!==false,globalFairingMs:performance.now()-fairStarted,preparationMs:performance.now()-started}};
  }
}
