import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import type {Point3, Shape3} from './coordinates';
import {cIndex,transformPoint} from './coordinates.ts';
import {modeledAnatomy} from './modeledAnatomy.ts';

interface SourceTargets {owned:Uint32Array;overlap:Uint32Array}
interface ResultMembership {
  cavity:Uint32Array;residual:Uint32Array;targetNames:Set<string>;
  overlapRemaining:number;overlapOriginal:number;cellMm3:number;
}
const targetIndex=new WeakMap<Uint16Array,SourceTargets>();
const memberships=new WeakMap<ViewerVolume,ResultMembership>();

/** Native indices of the same categorical owners used to extract the accepted
 * meshes. Unowned overlapping annotations still belong to the separate ledger. */
export function resultMembership(volume:ViewerVolume,view:SurgicalView) {
  const cached=memberships.get(volume);if(cached)return cached;
  const modeled=modeledAnatomy(volume);
  const reference=modeled?.surfaceReference;
  if(!modeled||!reference)throw Error('Result inspection requires a source-bound engine grid.');
  const ids=new Set(view.scene.regions.filter(r=>r.target).map(r=>r.id));
  const targetNames=new Set(view.scene.regions.filter(r=>r.target).map(r=>r.name));
  const targets=volume.compartments.filter(l=>targetNames.has(l.name));
  const ownedNames=new Set(view.scene.regions.map(r=>r.name));
  const owners=volume.compartments.filter(l=>ownedNames.has(l.name));
  let initial=targetIndex.get(reference.owner);
  if(!initial){
    const sourceLayers=modeled.source.compartments.filter(l=>targetNames.has(l.name)).map(layer=>({
      mask:layer.mask,labelValue:layer.labelValue,
      labels:modeled.source.labelSets?.find(set=>set.id===layer.labelSet)?.data
    }));
    const owned:number[]=[],overlap:number[]=[];
    for(let i=0;i<reference.owner.length;i++){
      if(ids.has(reference.owner[i]))owned.push(i);
      else if(sourceLayers.some(l=>l.labels?l.labels[i]===l.labelValue:!!l.mask[i]))overlap.push(i);
    }
    initial={owned:Uint32Array.from(owned),overlap:Uint32Array.from(overlap)};
    targetIndex.set(reference.owner,initial);
  }
  const residual=Array.from(initial.owned).filter(i=>targets.some(l=>l.mask[i]));
  for(const [,dest] of view.moved)
    if(!ids.has(reference.owner[dest])&&targets.some(l=>l.mask[dest]))residual.push(dest);
  const cavity=view.removed.filter(i=>!owners.some(l=>l.mask[i]));
  const sorted=(indices:number[])=>Uint32Array.from(new Set(indices)).sort();
  const removed=new Set(view.removed);
  const [[a,b,c],[d,e,f],[g,h,i]]=view.scene.affine;
  const cellMm3=Math.abs(a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g));
  const result={cavity:sorted(cavity),residual:sorted(residual),targetNames,cellMm3,
    overlapOriginal:initial.overlap.length,overlapRemaining:initial.overlap.reduce((n,i)=>n+Number(!removed.has(i)),0)};
  memberships.set(volume,result);return result;
}

export function resultMaskHash(indices:Uint32Array) {
  let hash=2166136261;
  for(const i of indices)hash=Math.imul(hash^i,16777619);
  return (hash>>>0).toString(16).padStart(8,'0');
}

/** Cropping is only an extraction optimization. The translated affine places
 * the cavity at exactly the same patient coordinates as the retained tissue. */
export function resultMaskCrop(indices:Uint32Array,view:SurgicalView) {
  if(!indices.length)return null;
  const {shape,affine}=view.scene;
  const low:Point3=[Infinity,Infinity,Infinity],high:Point3=[-1,-1,-1];
  const point=(i:number):Point3=>[Math.floor(i/(shape[1]*shape[2])),Math.floor(i/shape[2])%shape[1],i%shape[2]];
  for(const index of indices){
    const p=point(index);
    for(let a=0;a<3;a++){low[a]=Math.min(low[a],p[a]);high[a]=Math.max(high[a],p[a]);}
  }
  const size=high.map((n,a)=>n-low[a]+1) as Shape3;
  const mask=new Uint8Array(size[0]*size[1]*size[2]);
  for(const index of indices){
    const p=point(index);
    mask[cIndex(size,[p[0]-low[0],p[1]-low[1],p[2]-low[2]])]=1;
  }
  const origin=transformPoint(affine,low);
  const localAffine=affine.map((row,i)=>row.map((n,j)=>i<3&&j===3?origin[i]:n));
  return {mask,size,affine:localAffine};
}
