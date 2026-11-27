/** Nearest tract and vessel to the working instrument tip, from the layer
 * masks the renderer already holds. An estimate: masks are label estimates on
 * the source grid, distances use voxel centres, and under a brain-shift
 * scenario the tip is carried back into the source frame by the displayed field. */
import type {Affine, Point3, Shape3} from './coordinates';
import {inverseAffine} from './coordinates.ts';
import type {TipProximity} from '../operative-scenarios';

export interface ProximityLayer {name:string;mask:Uint8Array;group:'tract'|'vessel'}
interface SlotLike {instrument:string;position:Point3;inserted?:boolean}

const label=(name:string)=>name.replace(/\s*\((tract|angiography)\)\s*$/i,'').trim();

/** Distances (mm) from `point` to the nearest nonzero cell of each layer within `searchMm`. */
export function nearestLayers(point:Point3,layers:readonly ProximityLayer[],shape:Shape3,affine:Affine,searchMm:number,removed?:ReadonlySet<number>) {
  const inverse=inverseAffine(affine),[nx,ny,nz]=shape;
  const ijk=[0,1,2].map(a=>inverse[a][0]*point[0]+inverse[a][1]*point[1]+inverse[a][2]*point[2]+inverse[a][3]);
  const spacing=[0,1,2].map(c=>Math.hypot(affine[0][c],affine[1][c],affine[2][c]));
  const reach=spacing.map(s=>Math.ceil(searchMm/s));
  const low=ijk.map((v,a)=>Math.max(0,Math.floor(v)-reach[a])),high=ijk.map((v,a)=>Math.min(shape[a]-1,Math.ceil(v)+reach[a]));
  const result:Array<{name:string;group:ProximityLayer['group'];distanceMm:number}>=[];
  for(const layer of layers){
    if(layer.mask.length!==nx*ny*nz)continue;
    let best=Infinity;
    for(let x=low[0];x<=high[0];x++)for(let y=low[1];y<=high[1];y++){
      const row=(x*ny+y)*nz;
      for(let z=low[2];z<=high[2];z++){
        const cell=row+z;
        if(!layer.mask[cell]||removed?.has(cell))continue;
        const d=[x-ijk[0],y-ijk[1],z-ijk[2]];
        const r=[0,1,2].map(a=>affine[a][0]*d[0]+affine[a][1]*d[1]+affine[a][2]*d[2]);
        const distance=Math.hypot(r[0],r[1],r[2]);
        if(distance<best)best=distance;
      }
    }
    if(best<=searchMm)result.push({name:label(layer.name),group:layer.group,distanceMm:best});
  }
  return result.sort((a,b)=>a.distanceMm-b.distanceMm);
}

/** The working tip: the selected hand when inserted, else the first inserted hand. */
export function workingTip(slots:Record<string,SlotLike>,selected?:string):{slot:string;instrument:string;position:Point3}|null {
  const ordered=[...(selected&&slots[selected]?[selected]:[]),...Object.keys(slots).filter(k=>k!==selected)];
  for(const key of ordered){const slot=slots[key];if(slot?.inserted&&slot.position?.every(Number.isFinite))return {slot:key,instrument:slot.instrument,position:slot.position};}
  return null;
}

export function tipProximity(slots:Record<string,SlotLike>,selected:string|undefined,layers:readonly ProximityLayer[],shape:Shape3,affine:Affine,
  options:{searchMm?:number;removed?:ReadonlySet<number>;toSource?:(p:Point3)=>Point3}={}):TipProximity|null {
  const tip=workingTip(slots,selected);
  if(!tip||!layers.length)return null;
  const searchMm=options.searchMm??20,point=options.toSource?options.toSource(tip.position):tip.position;
  const found=nearestLayers(point,layers,shape,affine,searchMm,options.removed);
  const pick=(group:ProximityLayer['group'])=>{const row=found.find(r=>r.group===group);return row?{name:row.name,distanceMm:row.distanceMm}:null;};
  return {tip:tip.position,slot:tip.slot,instrument:tip.instrument,tract:pick('tract'),vessel:pick('vessel'),searchMm};
}
