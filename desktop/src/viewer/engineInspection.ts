import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import {cIndex, inverseAffine, transformPoint} from './coordinates.ts';
import type {Point3, Shape3} from './coordinates';
import {modeledAnatomy} from './modeledAnatomy.ts';

type Axis=0|1|2;
const axes:Axis[]=[0,1,2];
const clamp=(n:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,n));
const voxelAt=(i:number,shape:Shape3):Point3=>[Math.floor(i/(shape[1]*shape[2])), Math.floor(i/shape[2])%shape[1],i%shape[2]];
type Bounds={low:Point3;high:Point3};
const emptyBounds=():Bounds=>({low:[Infinity,Infinity,Infinity],high:[-Infinity,-Infinity,-Infinity]});
function include(bounds:Bounds,p:Point3) {
  for(const a of axes){bounds.low[a]=Math.min(bounds.low[a],p[a]);bounds.high[a]=Math.max(bounds.high[a],p[a]);}
}
function planePeak(counts:Uint32Array,center:number) {
  let best=clamp(Math.round(center),0,counts.length-1);
  for(let i=0;i<counts.length;i++)if(counts[i]>counts[best]||
    (counts[i]===counts[best]&&Math.abs(i-center)<Math.abs(best-center)))best=i;
  return best;
}

/** Built once per immutable native ownership field, including overlap priority. */
export function inspectionSource(owner:Uint16Array,view:SurgicalView) {
  const targetIds=new Set(view.scene.regions.filter(r=>r.target).map(r=>r.id));
  const bounds=emptyBounds();
  const counts=view.scene.shape.map(n=>new Uint32Array(n));
  for(let i=0;i<owner.length;i++)if(targetIds.has(owner[i])){
    const p=voxelAt(i,view.scene.shape);include(bounds,p);
    for(const a of axes)counts[a][p[a]]++;
  }
  return {owner,targetIds,bounds,counts};
}
type InspectionSource=ReturnType<typeof inspectionSource>;
export const sectionClasses={tissue:0,target:1,cavity:2,vessel:3,protected:4} as const;
export function inspectionSection(volume:ViewerVolume,view:SurgicalView,base:InspectionSource,axis:Axis,fixed:number|null) {
  const {shape,affine}=view.scene,spacing=view.scene.resolutionMm;
  const [u,v]=axes.filter(a=>a!==axis);
  const geometry=modeledAnatomy(volume);
  if(geometry?.surfaceReference?.owner!==base.owner)throw Error('Operated section lost its source ownership binding');
  const layers=view.scene.regions.map(r=>{
    const layer=volume.compartments.find(c=>c.name===r.name);
    if(!layer||layer.mask.length!==base.owner.length)throw Error('Operated section is missing the accepted region mask');
    const declaredProtected=('protected' in r&&r.protected===true)||('critical' in r&&r.critical===true);
    const kind=r.target?sectionClasses.target:/^(vessel|vessels|artery|vein|sinus)$/.test(r.material)?sectionClasses.vessel:
      declaredProtected?sectionClasses.protected:sectionClasses.tissue;
    return {mask:layer.mask,kind};
  });
  const observedTargets=view.removed.filter(i=>base.targetIds.has(base.owner[i]));
  const work=observedTargets.length?observedTargets:view.removed;
  const workBounds=emptyBounds(),workCounts=new Uint32Array(shape[axis]);
  for(const i of work){const p=voxelAt(i,shape);include(workBounds,p);workCounts[p[axis]]++;}
  const targetBounds=Number.isFinite(base.bounds.low[0])?base.bounds:null;
  const bounds=targetBounds??(work.length?workBounds:{low:[0,0,0],high:shape.map(n=>n-1)});
  const targetCenter=axes.map(a=>(bounds.low[a]+bounds.high[a])/2);
  const center=work.length?axes.map(a=>(workBounds.low[a]+workBounds.high[a])/2):targetCenter;
  const depth=fixed===null?planePeak(work.length?workCounts:base.counts[axis],center[axis]):clamp(fixed,0,shape[axis]-1);
  // Preserve target framing through treatment. Larger lesions get a bounded detail
  // around the effect so a small cavity remains legible inside their source extent.
  const spanMm=clamp(Math.max(...[u,v].map(a=>(bounds.high[a]-bounds.low[a]+1)*spacing[a]))*2.5,16,42);
  const frameCenter=axes.map(a=>Math.abs(center[a]-targetCenter[a])*spacing[a]<spanMm*.32?targetCenter[a]:center[a]);
  const dims=[u,v].map(a=>clamp(Math.round(spanMm/spacing[a]),1,shape[a]));
  const origin=[u,v].map((a,j)=>clamp(Math.round(frameCenter[a]-(dims[j]-1)/2),0,shape[a]-dims[j]));
  const [width,height]=dims,removed=new Set(view.removed);
  const membership=new Uint8Array(width*height),signal=new Float32Array(width*height);
  const point:Point3=[0,0,0];point[axis]=depth;
  let targetPixels=0,cavityPixels=0;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    point[u]=origin[0]+x;point[v]=origin[1]+height-1-y;
    const i=cIndex(shape,point),pixel=y*width+x;
    const layer=layers.find(l=>l.mask[i]);
    membership[pixel]=layer?.kind??(removed.has(i)?sectionClasses.cavity:sectionClasses.tissue);
    signal[pixel]=volume.mri[i];
    if(membership[pixel]===sectionClasses.target)targetPixels++;
    if(membership[pixel]===sectionClasses.cavity)cavityPixels++;
  }
  const inverse=inverseAffine(affine);
  const tips=Object.entries(view.slots).filter(([,slot])=>slot.inserted).map(([hand,slot])=>{
    const voxel=transformPoint(inverse,slot.position);
    return {hand,instrument:slot.instrument,x:voxel[u]-origin[0]+.5,y:origin[1]+height-.5-voxel[v],inPlane:Math.abs(voxel[axis]-depth)<=.5};
  });
  const direction=(a:Axis,sign:number)=>{
    const vector=affine.slice(0,3).map(row=>row[a]*sign);
    const dominant=axes.reduce((best,i)=>Math.abs(vector[i])>Math.abs(vector[best])?i:best,0 as Axis);
    return (vector[dominant]>=0?['R','A','S']:['L','P','I'])[dominant];
  };
  return {axis,u,v,origin,depth,width,height,widthMm:width*spacing[u],heightMm:height*spacing[v],
    labels:[direction(u,-1),direction(u,1),direction(v,1),direction(v,-1)],
    signal,membership,targetPixels,cavityPixels,tips,
    hasVessels:layers.some(l=>l.kind===sectionClasses.vessel),hasProtected:layers.some(l=>l.kind===sectionClasses.protected)};
}
export type InspectionSection=ReturnType<typeof inspectionSection>;
