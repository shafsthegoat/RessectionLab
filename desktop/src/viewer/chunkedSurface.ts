import {maskSurfaceRegion} from './surface.ts';
import {SourceVertexIndex,SurfaceIndexBudgetExceeded} from './sourceVertexIndex.ts';
import type {Point3,Shape3} from './coordinates';
interface Chunk {low:Point3;high:Point3;data:Float32Array|Uint32Array;rowOffsets:Uint32Array}
/** Worker-owned source chunks. Row stitching preserves complete reference
 * x/y/z face order. Optional stable IDs avoid rehashing unchanged corners. */
export class ChunkedMaskSurface {
  private readonly chunks=new Map<string,Chunk>();
  private readonly vertices?:SourceVertexIndex;
  private chunkBytes=0;
  private readonly indexBudget?:number;
  readonly chunkSize:number;
  readonly shape:Shape3;
  readonly mask:Uint8Array;
  lastRemeshedChunks=0;
  lastIndexedCorners=0;
  constructor(mask:Uint8Array,shape:Shape3,chunkSize=32,indexBudget?:number){
    if(mask.length!==shape.reduce((a,b)=>a*b,1)||!Number.isInteger(chunkSize)||chunkSize<2)throw Error('Invalid chunked surface source');
    this.mask=mask;this.shape=shape;this.chunkSize=chunkSize;this.indexBudget=indexBudget;
    if(indexBudget!==undefined){
      const admit=(bytes:number)=>{if(mask.byteLength+this.chunkBytes+bytes>indexBudget)throw new SurfaceIndexBudgetExceeded('Indexed source chunk budget exceeded');};
      admit(0);this.vertices=new SourceVertexIndex(shape,admit);
    }
    const low=[...shape],high=[-1,-1,-1],yz=shape[1]*shape[2];
    for(let i=0;i<mask.length;i++)if(mask[i]){const p=[Math.floor(i/yz),Math.floor(i/shape[2])%shape[1],i%shape[2]];for(let a=0;a<3;a++){low[a]=Math.min(low[a],p[a]);high[a]=Math.max(high[a],p[a]);}}
    if(high[0]<0)return;
    const min=low.map(n=>Math.floor(n/chunkSize)),max=high.map(n=>Math.floor((n+1)/chunkSize));
    for(let x=min[0];x<=max[0];x++)for(let y=min[1];y<=max[1];y++)for(let z=min[2];z<=max[2];z++)this.extract([x,y,z]);
  }
  get indexedChunks(){return !!this.vertices;}
  get retainedBytes(){return this.mask.byteLength+this.chunkBytes+(this.vertices?.retainedBytes??0);}
  get totalChunks(){return this.shape.reduce((n,s)=>n*Math.ceil((s+1)/this.chunkSize),1);}
  private extract(coordinate:Point3){
    const low=coordinate.map(n=>n*this.chunkSize-1) as Point3,high=coordinate.map((n,a)=>Math.min(this.shape[a]-1,(n+1)*this.chunkSize-2)) as Point3;
    const key=coordinate.join(','),result=maskSurfaceRegion(this.mask,this.shape,low,high);this.lastRemeshedChunks++;
    const old=this.chunks.get(key);if(old)this.chunkBytes-=old.data.byteLength+old.rowOffsets.byteLength;
    if(!result.positions.length){this.chunks.delete(key);return;}
    let data:Float32Array|Uint32Array=result.positions;
    if(this.vertices){
      this.chunkBytes+=result.positions.length/3*4+result.rowOffsets.byteLength;
      // Reserve the new row storage before the registry grows. A failed
      // admission discards this entire optimization, never a partial mesh.
      if(this.retainedBytes>this.indexBudget!)throw new SurfaceIndexBudgetExceeded('Indexed source chunk budget exceeded');
      data=new Uint32Array(result.positions.length/3);this.lastIndexedCorners+=data.length;
      for(let i=0;i<data.length;i++)data[i]=this.vertices.intern(result.positions[3*i],result.positions[3*i+1],result.positions[3*i+2]);
      for(let i=0;i<result.rowOffsets.length;i++)result.rowOffsets[i]/=3;
    }else this.chunkBytes+=data.byteLength+result.rowOffsets.byteLength;
    this.chunks.set(key,{low,high,data,rowOffsets:result.rowOffsets});
  }
  update(indices:Iterable<number>){
    const dirty=new Set<string>();this.lastRemeshedChunks=0;this.lastIndexedCorners=0;
    for(const index of indices){
      if(!Number.isInteger(index)||index<0||index>=this.mask.length)throw Error('Invalid changed source cell');
      const p=[Math.floor(index/(this.shape[1]*this.shape[2])),Math.floor(index/this.shape[2])%this.shape[1],index%this.shape[2]];
      for(let dx=0;dx<2;dx++)for(let dy=0;dy<2;dy++)for(let dz=0;dz<2;dz++)dirty.add(p.map((v,a)=>Math.floor((v+[dx,dy,dz][a])/this.chunkSize)).join(','));
    }
    for(const key of dirty)this.extract(key.split(',').map(Number) as Point3);
  }
  private rows(){
    const rows:Array<{x:number;y:number;z:number;data:Float32Array|Uint32Array}>=[];let length=0;
    for(const chunk of this.chunks.values()){
      let row=0;for(let x=chunk.low[0];x<=chunk.high[0];x++)for(let y=chunk.low[1];y<=chunk.high[1];y++){
        const begin=chunk.rowOffsets[row],end=chunk.rowOffsets[++row];if(begin===end)continue;
        const data=chunk.data.subarray(begin,end);rows.push({x,y,z:chunk.low[2],data});length+=data.length;
      }
    }
    rows.sort((a,b)=>a.x-b.x||a.y-b.y||a.z-b.z);return {rows,length};
  }
  /** Fresh transferable buffers, with IDs assigned in original first-appearance
   * order. Stale registry vertices never enter the output or affect normals. */
  indexed(){
    if(!this.vertices)throw Error('Indexed source chunks were not admitted');
    const {rows,length}=this.rows(),indices=new Uint32Array(length),positions=new Float32Array(this.vertices.vertexCount*3),remap=new Uint32Array(this.vertices.vertexCount);
    let offset=0,count=0;
    for(const row of rows)for(const id of row.data){
      let mapped=remap[id];if(!mapped){mapped=++count;remap[id]=mapped;this.vertices.copy(id,positions,(mapped-1)*3);}
      indices[offset++]=mapped-1;
    }
    return {positions:count===this.vertices.vertexCount?positions:positions.slice(0,count*3),indices};
  }
  positions():Float32Array {
    if(this.vertices){const {positions,indices}=this.indexed(),soup=new Float32Array(indices.length*3);for(let i=0;i<indices.length;i++)for(let a=0;a<3;a++)soup[3*i+a]=positions[3*indices[i]+a];return soup;}
    const {rows,length}=this.rows(),result=new Float32Array(length);let offset=0;
    for(const row of rows){result.set(row.data,offset);offset+=row.data.length;}return result;
  }
}
