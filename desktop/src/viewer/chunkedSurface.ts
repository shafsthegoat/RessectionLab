import {maskSurfaceRegion} from './surface.ts';
import type {Point3,Shape3} from './coordinates';
interface Chunk {low:Point3;high:Point3;positions:Float32Array;rowOffsets:Uint32Array}
/** Worker-owned source triangle soups. Row stitching preserves the complete
 * reference x/y/z face order, so global indexing/normals/fairing stay exact. */
export class ChunkedMaskSurface {
  private readonly chunks=new Map<string,Chunk>();
  readonly chunkSize:number;
  readonly shape:Shape3;
  readonly mask:Uint8Array;
  lastRemeshedChunks=0;
  constructor(mask:Uint8Array,shape:Shape3,chunkSize=32){
    if(mask.length!==shape.reduce((a,b)=>a*b,1)||!Number.isInteger(chunkSize)||chunkSize<2)throw Error('Invalid chunked surface source');
    this.mask=mask;this.shape=shape;this.chunkSize=chunkSize;
    const low=[...shape],high=[-1,-1,-1],yz=shape[1]*shape[2];
    for(let i=0;i<mask.length;i++)if(mask[i]){const p=[Math.floor(i/yz),Math.floor(i/shape[2])%shape[1],i%shape[2]];for(let a=0;a<3;a++){low[a]=Math.min(low[a],p[a]);high[a]=Math.max(high[a],p[a]);}}
    if(high[0]<0)return;
    const min=low.map(n=>Math.floor(n/chunkSize)),max=high.map(n=>Math.floor((n+1)/chunkSize));
    for(let x=min[0];x<=max[0];x++)for(let y=min[1];y<=max[1];y++)for(let z=min[2];z<=max[2];z++)this.extract([x,y,z]);
  }
  get retainedBytes(){let bytes=this.mask.byteLength;for(const c of this.chunks.values())bytes+=c.positions.byteLength+c.rowOffsets.byteLength;return bytes;}
  get totalChunks(){return this.shape.reduce((n,s)=>n*Math.ceil((s+1)/this.chunkSize),1);}
  private extract(coordinate:Point3){
    const low=coordinate.map(n=>n*this.chunkSize-1) as Point3,high=coordinate.map((n,a)=>Math.min(this.shape[a]-1,(n+1)*this.chunkSize-2)) as Point3;
    const key=coordinate.join(','),result=maskSurfaceRegion(this.mask,this.shape,low,high);this.lastRemeshedChunks++;
    if(result.positions.length)this.chunks.set(key,{low,high,...result});else this.chunks.delete(key);
  }
  update(indices:Iterable<number>){
    const dirty=new Set<string>();this.lastRemeshedChunks=0;
    for(const index of indices){
      if(!Number.isInteger(index)||index<0||index>=this.mask.length)throw Error('Invalid changed source cell');
      const p=[Math.floor(index/(this.shape[1]*this.shape[2])),Math.floor(index/this.shape[2])%this.shape[1],index%this.shape[2]];
      for(let dx=0;dx<2;dx++)for(let dy=0;dy<2;dy++)for(let dz=0;dz<2;dz++)dirty.add(p.map((v,a)=>Math.floor((v+[dx,dy,dz][a])/this.chunkSize)).join(','));
    }
    for(const key of dirty)this.extract(key.split(',').map(Number) as Point3);
  }
  positions():Float32Array {
    const rows:Array<{x:number;y:number;z:number;data:Float32Array}>=[];let length=0;
    for(const chunk of this.chunks.values()){
      let row=0;for(let x=chunk.low[0];x<=chunk.high[0];x++)for(let y=chunk.low[1];y<=chunk.high[1];y++){
        const begin=chunk.rowOffsets[row],end=chunk.rowOffsets[++row];if(begin===end)continue;
        const data=chunk.positions.subarray(begin,end);rows.push({x,y,z:chunk.low[2],data});length+=data.length;
      }
    }
    rows.sort((a,b)=>a.x-b.x||a.y-b.y||a.z-b.z);const result=new Float32Array(length);let offset=0;
    for(const row of rows){result.set(row.data,offset);offset+=row.data.length;}return result;
  }
}
