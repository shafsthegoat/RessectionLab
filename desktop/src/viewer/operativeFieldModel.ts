import type { OperativeField, OperativePhase } from '../operative-field';
import { aspirationSeedIndex, eligibleTumor } from './operativeSeed.ts';
export { hasAspirationSeed } from './operativeSeed.ts';
import { emptyVoxelBounds, extendVoxelBounds } from './operativeSurface.ts';

type XYZ = [number, number, number];
export const layerNames = ['bone','boneFlap','dura','duraFlap','brain','tumor','vessel'] as const;
export type FieldLayer = typeof layerNames[number];
const TAU = Math.PI * 2;
export const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
export const smooth = (n: number) => { const t = clamp01(n); return t*t*(3-2*t); };
const phases: OperativePhase[] = ['overview','bone','dura','brain','tumor'];
export function phaseAmount(phase: OperativePhase, progress: number, stage: OperativePhase) {
  return phase === stage ? clamp01(progress) : Number(phases.indexOf(phase) > phases.indexOf(stage));
}
function pointAt(field: OperativeField, index: number): XYZ {
  const [,ny,nz] = field.shape, s = field.spacingMm;
  return [field.origin[0]+Math.floor(index/(ny*nz))*s, field.origin[1]+(Math.floor(index/nz)%ny)*s, field.origin[2]+(index%nz)*s];
}
export function operativeDimensions(field: OperativeField) {
  const s = field.spacingMm, route = field.routeRadiusMm;
  const opening = Math.min(field.radiusMm*.72, Math.max(4*s, route*1.7, field.radiusMm*.36));
  return { opening, duralOpening: opening*.81, tipRadius: Math.max(s*.85, Math.min(2.2, route*.42)), kerf: Math.max(.7*s, opening*.065) };
}

/** Plan a continuous illustrative sweep through a local connected tumor label.
 * Adjacent centres share a voxel face. Protected cells neither admit the tip
 * nor receive a removal time. A brush affects only eligible source labels. */
function aspiration(field: OperativeField, tipRadius: number) {
  const {shape: [nx,ny,nz], voxels, spacingMm: s} = field, yz = ny*nz;
  const firstHit = new Uint32Array(voxels.length), path: number[] = [];
  const allowed = (i:number) => eligibleTumor(voxels[i]);
  const seed=aspirationSeedIndex(field);
  if(seed===null)return {path,firstHit,seed:null};
  const center = pointAt(field,seed), reach = Math.max(tipRadius*2.2, Math.min(field.radiusMm*.44, field.routeRadiusMm*1.6));
  const depth = Math.max(3*s, reach*1.8);
  const neighbors = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,-1],[0,0,1]];
  const radiusSquared=(tipRadius/s)**2;
  const record = (i: number) => {
    path.push(i);
    const z0=i%nz, y0=Math.floor(i/nz)%ny, x0=Math.floor(i/yz);
    // Flood within this tip footprint. A nearby island cannot be reached
    // through empty tissue or a vessel, even if it lies inside the tip radius.
    const queue=[i], seen=new Set(queue);
    for(let cursor=0;cursor<queue.length;cursor++) {
      const j=queue[cursor], x=Math.floor(j/yz), y=Math.floor(j/nz)%ny, z=j%nz;
      if(!firstHit[j]) firstHit[j]=path.length;
      for(const [dx,dy,dz] of neighbors) {
        const xx=x+dx, yy=y+dy, zz=z+dz;
        if(xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
        if((xx-x0)**2+(yy-y0)**2+(zz-z0)**2>radiusSquared) continue;
        const id=xx*yz+yy*nz+zz;
        if(allowed(id) && !seen.has(id)){seen.add(id);queue.push(id);}
      }
    }
  };
  // Advance by face-adjacent eligible cells toward successive spiral guide
  // points, holding or gliding along an obstructing labelled boundary.
  const passes=Math.max(2,Math.min(6,Math.ceil(depth/(tipRadius*2.4))));
  const samples=passes*Math.ceil(14*reach/s);
  let current=seed; record(seed);
  for(let sample=1;sample<=samples;sample++) {
    const t=sample/samples, phase=t*passes;
    const radius=reach*.84*Math.abs(Math.sin(Math.PI*phase));
    const angle=phase*TAU*1.22;
    const gx=center[0]+radius*Math.cos(angle), gy=center[1]+radius*Math.sin(angle), gz=center[2]-depth*t;
    const pos=pointAt(field,current), x=Math.floor(current/yz), y=Math.floor(current/nz)%ny, z=current%nz;
    let closest=(pos[0]-gx)**2+(pos[1]-gy)**2+(pos[2]-gz)**2;
    let next=current;
    for(const [dx,dy,dz] of neighbors) {
      const xx=x+dx, yy=y+dy, zz=z+dz;
      if(xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
      const j=xx*yz+yy*nz+zz;
      const d=(pos[0]+dx*s-gx)**2+(pos[1]+dy*s-gy)**2+(pos[2]+dz*s-gz)**2;
      if(d<closest-1e-8 && allowed(j)){closest=d; next=j;}
    }
    if(next!==current){current=next; record(current);}
  }
  return {path, firstHit, seed:center};
}

export function buildOperativeModel(field: OperativeField) {
  const {origin, spacingMm:s, stations} = field, [nx,ny,nz] = field.shape;
  const {opening,duralOpening,tipRadius,kerf}=operativeDimensions(field);
  const layers = Object.fromEntries(layerNames.map(name=>[name,{mask:new Uint8Array(field.voxels.length), bounds:emptyVoxelBounds()}])) as Record<FieldLayer,{mask:Uint8Array;bounds:ReturnType<typeof emptyVoxelBounds>}>;
  const source = (name: FieldLayer, i:number, x:number,y:number,z:number) => {
    layers[name].mask[i]=1; extendVoxelBounds(layers[name].bounds,x,y,z);
  };
  const cutIndices = {bone:[] as number[], dura:[] as number[]};
  const cutAt = new Uint8Array(field.voxels.length), duralCutAt = new Uint8Array(field.voxels.length);
  const flapIndices = {bone:[] as number[], dura:[] as number[]};
  const swept = aspiration(field, tipRadius), affected: number[]=[];
  const maxOffset = Math.max(8,opening*.9,4*s);
  for(let x=0,i=0;x<nx;x++) for(let y=0;y<ny;y++) {
    const px=origin[0]+x*s, py=origin[1]+y*s, r=Math.hypot(px,py);
    const angle=(Math.atan2(py,px)+TAU)%TAU;
    for(let z=0;z<nz;z++,i++) {
      const bits=field.voxels[i]; if(!bits) continue;
      const pz=origin[2]+z*s;
      for (const kind of ['bone','dura'] as const) {
        if (!(bits & (kind==='bone'?1:2))) continue;
        source(kind,i,x,y,z);
        const station=stations[kind], radius=kind==='bone'?opening:duralOpening;
        if(station===null || Math.abs(pz-station)>maxOffset) continue;
        const hinge=kind==='dura' && px>radius*.74;
        if(!hinge && Math.abs(r-radius)<=kerf) {
          const fraction=kind==='bone'?angle/TAU:clamp01((angle-.70)/(TAU-1.40));
          (kind==='bone'?cutAt:duralCutAt)[i]=1+Math.round(fraction*127); cutIndices[kind].push(i);
        } else if(!hinge && r<radius) {
          flapIndices[kind].push(i); source(kind==='bone'?'boneFlap':'duraFlap',i,x,y,z);
        }
      }
      if(bits&16) source('vessel',i,x,y,z);
      if(eligibleTumor(bits)) source('tumor',i,x,y,z);
      if((bits&4) && !(bits&(1|2|8|16))) source('brain',i,x,y,z);
      if(swept.firstHit[i]) affected.push(i);
    }
  }
  function openingState(kind: 'bone'|'dura', amount:number) {
    const cutEnd=kind==='bone'?.68:.50;
    return {cut:Math.round(clamp01(amount/cutEnd)*128),lift:smooth((amount-cutEnd)/(1-cutEnd))};
  }
  function applyOpening(kind:'bone'|'dura',cut:number,lifted:boolean) {
    const target=layers[kind].mask, timing=kind==='bone'?cutAt:duralCutAt;
    for(const i of cutIndices[kind]) target[i]=Number(timing[i]>cut);
    for(const i of flapIndices[kind]) target[i]=Number(!lifted);
  }
  function applyAspiration(step:number) {
    let removed=0;
    for(const i of affected){
      const intact=swept.firstHit[i]>step;
      layers.tumor.mask[i]=Number(intact);
      if(!intact)removed++;
    }
    return removed;
  }
  function surfaceHeight(kind:'bone'|'dura',px:number,py:number): number|null {
    const station=stations[kind]; if(station===null) return null;
    const x0=Math.round((px-origin[0])/s), y0=Math.round((py-origin[1])/s);
    let nearest=Infinity, height:number|null=null;
    for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++) {
      const x=x0+dx,y=y0+dy; if(x<0||x>=nx||y<0||y>=ny) continue;
      for(let z=nz-1;z>=0;z--) {
        const pz=origin[2]+z*s;
        if(Math.abs(pz-station)>maxOffset || !(field.voxels[(x*ny+y)*nz+z]&(kind==='bone'?1:2))) continue;
        const d=dx*dx+dy*dy;
        if(d<nearest) {nearest=d; height=pz+s*.45;}
        break;
      }
    }
    return height;
  }
  const tipAt=(step:number)=>swept.path.length?pointAt(field,swept.path[Math.max(0,Math.min(swept.path.length-1,step-1))]):null;
  const sectionY = (swept.seed?.[1]??0) - tipRadius*.45;
  return {field,layers,opening,duralOpening,tipRadius,openingState,applyOpening,applyAspiration,surfaceHeight,tipAt,stepCount:swept.path.length,
    firstSectionRow:Math.max(0,Math.ceil((sectionY-origin[1])/s))};
}
export type OperativeModel = ReturnType<typeof buildOperativeModel>;
