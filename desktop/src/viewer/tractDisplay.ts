import {transformPoint} from './coordinates.ts';
import type {Affine, Point3} from './coordinates.ts';
import type {ViewerVolume} from './contracts.ts';

export type ViewerTract = NonNullable<ViewerVolume['tractography']>[number];
export const TRACT_COLORS = ['#f8b76b','#bb9cf3','#90d48b','#f38fac','#83abf4','#f1da73','#e6a9e1','#dd9f73','#82d5bd','#68d8ed','#b6cd76','#b9b9f7'] as const;
export function tractColor(index:number):string {return TRACT_COLORS[index%TRACT_COLORS.length];}
export interface TractPreview {positions:Float32Array;sourceSegments:number;displaySegments:number;streamlineStride:number}
const previews = new WeakMap<ViewerTract,Map<string,TractPreview>>();

/** Complete streamlines are selected uniformly for a reversible display preview.
 * The authoritative points/offsets are never changed or used as reduced evidence.
 * Native bridge points are displayed-grid voxel coordinates, not RAS coordinates. */
export function tractPreview(tract:ViewerTract,affine:Affine,maxSegments=70000):TractPreview {
  const signature=JSON.stringify([affine,maxSegments]);
  let cached=previews.get(tract);if(!cached){cached=new Map();previews.set(tract,cached);}
  const previous=cached.get(signature);if(previous)return previous;
  let sourceSegments=0;
  for(let line=0;line<tract.offsets.length-1;line++)sourceSegments+=Math.max(0,tract.offsets[line+1]-tract.offsets[line]-1);
  const streamlineStride=Math.max(1,Math.ceil(sourceSegments/maxSegments));
  let displaySegments=0;
  for(let line=0;line<tract.offsets.length-1;line+=streamlineStride)displaySegments+=Math.max(0,tract.offsets[line+1]-tract.offsets[line]-1);
  const positions=new Float32Array(displaySegments*6);let offset=0;
  for(let line=0;line<tract.offsets.length-1;line+=streamlineStride){
    for(let i=tract.offsets[line];i<tract.offsets[line+1]-1;i++){
      for(const index of [i,i+1]){
        const p=transformPoint(affine,[tract.points[index*3],tract.points[index*3+1],tract.points[index*3+2]]);
        positions.set(p,offset);offset+=3;
      }
    }
  }
  const preview={positions,sourceSegments,displaySegments,streamlineStride};cached.set(signature,preview);return preview;
}

/** Exact line/slab clipping. No projection of remote fibers onto the MRI plane. */
export function clipSegmentToSlab(a:Point3,b:Point3,axis:0|1|2,position:number,halfWidth:number):[Point3,Point3]|null {
  const delta=b[axis]-a[axis];let low=0,high=1;
  if(Math.abs(delta)<1e-12){if(Math.abs(a[axis]-position)>halfWidth)return null;}
  else {const t0=(position-halfWidth-a[axis])/delta,t1=(position+halfWidth-a[axis])/delta;low=Math.max(0,Math.min(t0,t1));high=Math.min(1,Math.max(t0,t1));if(low>high)return null;}
  return [low,high].map(t=>a.map((n,i)=>n+t*(b[i]-n)) as Point3) as [Point3,Point3];
}

/** A small source-plane spatial index over display segments, shared across MPRs. */
export class TractSliceIndex {
  private readonly bins=[new Map<number,number[]>(),new Map<number,number[]>(),new Map<number,number[]>()];
  readonly preview:TractPreview;
  private readonly binMm:number;
  constructor(preview:TractPreview,binMm=3){
    this.preview=preview;this.binMm=binMm;
    const p=preview.positions;
    for(let i=0;i<p.length;i+=6)for(let axis=0;axis<3;axis++){
      const lo=Math.floor(Math.min(p[i+axis],p[i+3+axis])/binMm),hi=Math.floor(Math.max(p[i+axis],p[i+3+axis])/binMm);
      for(let bin=lo;bin<=hi;bin++){const list=this.bins[axis].get(bin);if(list)list.push(i);else this.bins[axis].set(bin,[i]);}
    }
  }
  segments(axis:0|1|2,position:number,halfWidth=.65,maxSegments=5000):Array<[Point3,Point3]>{
    const candidates=new Set<number>();
    for(let bin=Math.floor((position-halfWidth)/this.binMm);bin<=Math.floor((position+halfWidth)/this.binMm);bin++)for(const i of this.bins[axis].get(bin)??[])candidates.add(i);
    const result:Array<[Point3,Point3]>=[],p=this.preview.positions;
    const stride=Math.max(1,Math.ceil(candidates.size/maxSegments));let count=0;
    for(const i of candidates){if(count++%stride)continue;const clipped=clipSegmentToSlab([p[i],p[i+1],p[i+2]],[p[i+3],p[i+4],p[i+5]],axis,position,halfWidth);if(clipped)result.push(clipped);}
    return result;
  }
}
