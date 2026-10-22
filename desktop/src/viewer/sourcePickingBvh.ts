import * as THREE from 'three';
import {acceleratedRaycast,MeshBVH} from 'three-mesh-bvh';
import type {SourceBvhInput,SourceBvhResult} from './sourcePickingBvhBuild';

type Geometry=THREE.BufferGeometry;
type Signature={position:THREE.BufferAttribute;array:Float32Array;version:number;index:THREE.BufferAttribute|null;indexArray:Uint16Array|Uint32Array|null;indexVersion:number;layout:string};
type Entry={id:number;geometry:Geometry;signature:Signature;tree:MeshBVH|null;bytes:number;status:'queued'|'building'|'ready'|'evicted'|'fallback';dispose:()=>void};
type Binding={source:Geometry;original:THREE.Mesh['raycast'];bound:THREE.Mesh['raycast'];deferred:boolean;dispose?:()=>void};
interface Options {
  maxRetainedBytes?:number;maxPending?:number;maxInputBytes?:number;maxGeometries?:number;
  onEvent?:(detail:Record<string,unknown>)=>void;
  createWorker?:()=>Worker;
}
const ordinaryRaycast=THREE.Mesh.prototype.raycast;

/** One source-owned visual picking worker. This never checks surgical contact.
 * Source attributes are immutable between ordinary Three.js version updates.
 * Pending requests retain geometry references; only the active job copies data.
 */
export class SourcePickingBvh {
  private readonly entries=new Map<Geometry,Entry>();
  private readonly ready=new Map<Geometry,Entry>();
  private readonly bindings=new Map<THREE.Mesh,Binding>();
  private queue:Entry[]=[];
  private active:Entry|null=null;
  private worker:Worker|null=null;
  private serial=0;
  private disposed=false;
  private retained=0;
  private readonly maxRetainedBytes:number;
  private readonly maxPending:number;
  private readonly maxInputBytes:number;
  private readonly maxGeometries:number;
  private readonly options:Options;
  private readonly counters={buildsRequested:0,buildsCompleted:0,workerStarts:0,workerFailures:0,copyBytes:0,transferredInputBytes:0,transferredResultBytes:0,evictions:0,fallbacks:0,staleResults:0,acceleratedRaycasts:0,ordinaryRaycasts:0};
  constructor(options:Options={}){
    this.options=options;
    this.maxRetainedBytes=options.maxRetainedBytes??128*1024**2;
    this.maxPending=options.maxPending??128;
    this.maxInputBytes=options.maxInputBytes??64*1024**2;
    this.maxGeometries=options.maxGeometries??128;
    if([this.maxRetainedBytes,this.maxPending,this.maxInputBytes,this.maxGeometries].some(v=>!Number.isSafeInteger(v)||v<1))throw Error('Positive integer BVH budgets required');
  }
  private signature(geometry:Geometry):Signature|null {
    const position=geometry.getAttribute('position'),index=geometry.getIndex();
    if(!(position instanceof THREE.BufferAttribute)||!(position.array instanceof Float32Array)||position.itemSize!==3||position.normalized)return null;
    if(index&&(!(index.array instanceof Uint16Array||index.array instanceof Uint32Array)||index.itemSize!==1||index.normalized))return null;
    return {position,array:position.array,version:position.version,index,indexArray:index?.array as Uint16Array|Uint32Array|null??null,
      indexVersion:index?.version??0,layout:JSON.stringify([geometry.groups,geometry.drawRange])};
  }
  private current(entry:Entry):boolean {
    const now=this.signature(entry.geometry),old=entry.signature;
    return !!now&&now.position===old.position&&now.array===old.array&&now.version===old.version&&now.index===old.index&&now.indexArray===old.indexArray&&now.indexVersion===old.indexVersion&&now.layout===old.layout;
  }
  prepare(geometry:Geometry):boolean {
    if(this.disposed)return false;
    const previous=this.entries.get(geometry);
    if(previous){
      if(this.current(previous)){
        // Only an LRU eviction is eligible for re-admission. Failed builds and
        // over-budget results stay ordinary instead of retrying on every pick.
        if(previous.status==='evicted'&&!geometry.boundsTree&&this.queue.length+(this.active?1:0)<this.maxPending){
          previous.id=++this.serial;previous.status='queued';this.queue.push(previous);this.counters.buildsRequested++;this.pump();
        }
        return previous.status==='ready';
      }
      this.drop(previous);
    }
    const signature=this.signature(geometry);
    if(!signature||geometry.boundsTree||this.queue.length+(this.active?1:0)>=this.maxPending||this.entries.size>=this.maxPending+this.maxGeometries){this.counters.fallbacks++;return false;}
    const inputBytes=signature.array.byteLength+(signature.indexArray?.byteLength??0);
    if(inputBytes>this.maxInputBytes){this.counters.fallbacks++;return false;}
    const entry:Entry={id:++this.serial,geometry,signature,tree:null,bytes:0,status:'queued',dispose:()=>this.forget(geometry)};
    geometry.addEventListener('dispose',entry.dispose);this.entries.set(geometry,entry);this.queue.push(entry);this.counters.buildsRequested++;
    this.pump();return false;
  }
  /** Transient read-only frames bind exact ordinary picking immediately. Their
   * acceleration is demanded only by a real pick, avoiding a full tree/copy
   * for a surface that the next observed microstep will replace. */
  install(mesh:THREE.Mesh,sourceGeometry:Geometry,eager=true):void {
    if(this.disposed)return;
    const previous=this.bindings.get(mesh);
    if(previous?.source===sourceGeometry){if(eager){previous.deferred=false;this.prepare(sourceGeometry);}return;}
    if(previous)this.forget(mesh);
    const owner=this,original=mesh.raycast;
    const bound:THREE.Mesh['raycast']=function(this:THREE.Mesh,raycaster,hits){
      const entry=owner.entries.get(sourceGeometry);
      if(this.geometry===sourceGeometry&&entry?.tree&&sourceGeometry.boundsTree===entry.tree&&owner.current(entry)){
        owner.ready.delete(sourceGeometry);owner.ready.set(sourceGeometry,entry);owner.counters.acceleratedRaycasts++;
        // All hits are required: a clipped nearer hit must allow a deeper one.
        const first=raycaster.firstHitOnly,near=raycaster.near,far=raycaster.far,start=hits.length;
        // The library scales local near/far by decomposed axis lengths, which
        // is insufficient for sheared source affines. Collect all local hits,
        // then apply Three.js's original world-distance interval exactly.
        try{raycaster.firstHitOnly=false;raycaster.near=0;raycaster.far=Infinity;acceleratedRaycast.call(this,raycaster,hits);}
        finally{raycaster.near=near;raycaster.far=far;if(first===undefined)delete raycaster.firstHitOnly;else raycaster.firstHitOnly=first;}
        let write=start;for(let read=start;read<hits.length;read++)if(!(hits[read].distance<near||hits[read].distance>far))hits[write++]=hits[read];hits.length=write;
      }else{
        // This call keeps the ordinary comparator. Repeated picks while the
        // replacement is queued/building cannot enqueue a duplicate job.
        if(this.geometry===sourceGeometry&&(binding.deferred||entry?.status==='evicted')){binding.deferred=false;owner.prepare(sourceGeometry);}
        owner.counters.ordinaryRaycasts++;ordinaryRaycast.call(this,raycaster,hits);
      }
    };
    const binding:Binding={source:sourceGeometry,original,bound,deferred:!eager};
    // Deferred geometries have no Entry/Entry.dispose listener until a pick.
    // Still release their mesh binding when a never-picked frame is retired.
    if(!eager){binding.dispose=()=>this.forget(mesh);sourceGeometry.addEventListener('dispose',binding.dispose);}
    mesh.raycast=bound;this.bindings.set(mesh,binding);if(eager)this.prepare(sourceGeometry);
  }
  private pump():void {
    if(this.disposed||this.active)return;
    const entry=this.queue.shift();if(!entry)return;
    if(this.entries.get(entry.geometry)!==entry||!this.current(entry)){this.drop(entry);this.pump();return;}
    try{
      this.active=entry;entry.status='building';
      if(!this.worker){
        this.worker=this.options.createWorker?.()??new Worker(new URL('./sourcePickingBvh.worker.ts',import.meta.url),{type:'module'});
        this.counters.workerStarts++;
        this.worker.onmessage=event=>this.receive(event.data);
        this.worker.onerror=()=>this.failure('source BVH worker failed');
      }
      const positions=new Float32Array(entry.signature.array),index=entry.signature.indexArray?.slice()??null;
      const input:SourceBvhInput={id:entry.id,positions,index,groups:entry.geometry.groups.map(g=>({...g})),drawRange:{...entry.geometry.drawRange}};
      const transfer:Transferable[]=[positions.buffer];if(index)transfer.push(index.buffer as ArrayBuffer);
      const bytes=positions.byteLength+(index?.byteLength??0);this.counters.copyBytes+=bytes;this.counters.transferredInputBytes+=bytes;
      this.worker.postMessage(input,transfer);
    }catch(cause){this.failure(cause instanceof Error?cause.message:String(cause));}
  }
  private receive(result:SourceBvhResult&{error?:string}):void {
    const entry=this.active;if(!entry||result.id!==entry.id){this.counters.staleResults++;return;}
    this.active=null;
    if(this.disposed)return;
    if(result.error){entry.status='fallback';this.counters.fallbacks++;this.emit({kind:'source-bvh-error',error:result.error});this.pump();this.emit({kind:'source-bvh-complete',id:result.id,status:'fallback'});return;}
    const bytes=result.serialized.roots.reduce((n,b)=>n+b.byteLength,0)+(result.serialized.indirectBuffer?.byteLength??0);
    this.counters.transferredResultBytes+=bytes;
    if(this.entries.get(entry.geometry)!==entry||!this.current(entry)||entry.geometry.boundsTree||bytes>this.maxRetainedBytes){
      entry.status='fallback';this.counters.fallbacks++;this.counters.staleResults++;this.pump();this.emit({kind:'source-bvh-complete',id:result.id,status:'fallback'});return;
    }
    try{
      while(this.ready.size&&(this.retained+bytes>this.maxRetainedBytes||this.ready.size>=this.maxGeometries)){
        const oldest=this.ready.values().next().value as Entry;this.releaseTree(oldest);oldest.status='evicted';this.counters.evictions++;
      }
      entry.tree=MeshBVH.deserialize(result.serialized,entry.geometry,{setIndex:false});entry.geometry.boundsTree=entry.tree;
      entry.bytes=bytes;entry.status='ready';this.retained+=bytes;this.ready.set(entry.geometry,entry);this.counters.buildsCompleted++;
      this.emit({kind:'source-bvh-ready',id:entry.id,inputBytes:entry.signature.array.byteLength+(entry.signature.indexArray?.byteLength??0),rootBytes:bytes,buildMs:result.buildMs});
    }catch(cause){entry.status='fallback';this.counters.fallbacks++;this.emit({kind:'source-bvh-error',error:String(cause)});}
    this.pump();this.emit({kind:'source-bvh-complete',id:result.id,status:entry.status});
  }
  private failure(message:string):void {
    this.counters.workerFailures++;if(this.active){this.active.status='fallback';this.counters.fallbacks++;}this.active=null;
    this.worker?.terminate();this.worker=null;this.emit({kind:'source-bvh-error',error:message});this.pump();
  }
  private emit(detail:Record<string,unknown>){this.options.onEvent?.(detail);}
  private releaseTree(entry:Entry){
    if(entry.tree&&entry.geometry.boundsTree===entry.tree)delete entry.geometry.boundsTree;
    this.retained-=entry.bytes;entry.bytes=0;entry.tree=null;this.ready.delete(entry.geometry);
  }
  private drop(entry:Entry){
    this.releaseTree(entry);entry.geometry.removeEventListener('dispose',entry.dispose);
    if(this.entries.get(entry.geometry)===entry)this.entries.delete(entry.geometry);
    this.queue=this.queue.filter(item=>item!==entry);
  }
  forget(target:THREE.Mesh|Geometry):void {
    if(target instanceof THREE.Mesh){const binding=this.bindings.get(target);if(binding){if(target.raycast===binding.bound)target.raycast=binding.original;if(binding.dispose)binding.source.removeEventListener('dispose',binding.dispose);this.bindings.delete(target);}return;}
    for(const [mesh,binding] of this.bindings)if(binding.source===target)this.forget(mesh);
    const entry=this.entries.get(target);if(entry)this.drop(entry);
  }
  stats(){return {...this.counters,workerLive:this.worker?1:0,activeJobs:this.active?1:0,
    activeBuildInputBytes:this.active?this.active.signature.array.byteLength+(this.active.signature.indexArray?.byteLength??0):0,
    pendingGeometryRefs:this.queue.length,retainedGeometries:this.ready.size,retainedRootBytes:this.retained,
    maxRetainedBytes:this.maxRetainedBytes,maxInputBytes:this.maxInputBytes,maxPending:this.maxPending,
    memoryMeaning:'Retained serialized roots plus indirect triangle buffers; source geometry stays with renderer; one disposable input copy is bounded separately'};}
  dispose():void {
    if(this.disposed)return;this.disposed=true;this.worker?.terminate();this.worker=null;this.active=null;
    for(const mesh of [...this.bindings.keys()])this.forget(mesh);
    for(const entry of [...this.entries.values()])this.drop(entry);
    this.queue=[];
  }
}
