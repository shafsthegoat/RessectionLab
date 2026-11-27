import type {Affine, Point3, Shape3} from './coordinates';
import {sampleTrilinear} from './coordinates.ts';
import {operativeColors} from './operativeMaterial.ts';

/** A display section through the current operative state: each pixel takes
 * the layer that owns its nearest engine cell and the case scan intensity at
 * its physical position. Nothing is classified from the image; unlabelled
 * space inside the head stays unlabelled (CSF in a host head is not segmented). */
export interface SectionLayerInput {name:string;kind?:string;displayKind?:string;color:string;mask:Uint8Array}
export interface SectionPlane {origin:Point3;across:Point3;down:Point3;width:number;height:number;stepMm:number}
export interface SectionSample {width:number;height:number;stepMm:number;label:Uint8Array;signal:Float32Array;inside:Uint8Array}

/** Illustrative colour of each host display kind on a cut face. */
const KIND_COLOURS:Record<string,string>={scalp:'#c4917f',galea:'#d6bba3',tissue:'#a8685a',extracranial:'#a8685a',
  skull:'#e1cfa6',diploe:'#b48f6c',dura:'#a9868a',grayMatter:operativeColors.grayMatter,whiteMatter:'#e2cfbb',tract:'#e2cfbb',
  tumor:'#9a5f7c',artery:'#b0453c',vein:'#5f5486',vessel:'#b0453c'};
const KIND_NAMES:Record<string,string>={scalp:'Scalp',galea:'Galea / subcutaneous',tissue:'Muscle and fat',extracranial:'Muscle and fat',
  skull:'Bone (cortical)',diploe:'Bone (diploe)',dura:'Dura (assumed)',grayMatter:'Cortex',whiteMatter:'White matter',tract:'White matter tract',
  tumor:'Tumour',artery:'Artery',vein:'Vein'};
export const UNLABELLED_INSIDE='Unlabelled space';

export function sectionColour(layer:SectionLayerInput){
  const kind=layer.displayKind??layer.kind??'';
  return KIND_COLOURS[kind]??layer.color;
}
export function sectionName(layer:SectionLayerInput){
  const kind=layer.displayKind??layer.kind??'';
  if(kind==='tumor')return layer.name;
  return KIND_NAMES[kind]??layer.name;
}

/** Sample a plane at `stepMm` pixels. `rasToVoxel` maps RAS millimetres to source voxel indices. */
export function sampleSection(layers:readonly SectionLayerInput[],mri:Float32Array,shape:Shape3,rasToVoxel:Affine,plane:SectionPlane):SectionSample {
  const {width,height,origin,across,down,stepMm}=plane,[nx,ny,nz]=shape;
  const label=new Uint8Array(width*height),signal=new Float32Array(width*height).fill(NaN),inside=new Uint8Array(width*height);
  const voxel:Point3=[0,0,0],weights=new Float64Array(layers.length+1);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const u=(x+.5)*stepMm,v=(y+.5)*stepMm;
    const p=[0,1,2].map(a=>origin[a]+across[a]*u+down[a]*v);
    for(let a=0;a<3;a++)voxel[a]=rasToVoxel[a][0]*p[0]+rasToVoxel[a][1]*p[1]+rasToVoxel[a][2]*p[2]+rasToVoxel[a][3];
    const i=y*width+x;
    if(voxel[0]<-.5||voxel[1]<-.5||voxel[2]<-.5||voxel[0]>nx-.5||voxel[1]>ny-.5||voxel[2]>nz-.5)continue;
    inside[i]=1;
    // Each layer's trilinear occupancy from its eight surrounding cells; the
    // pixel takes the layer covering it most, so boundaries follow the same
    // half-cell contour as the display surfaces instead of whole-cell blocks.
    const x0=Math.floor(voxel[0]),y0=Math.floor(voxel[1]),z0=Math.floor(voxel[2]);
    const fx=voxel[0]-x0,fy=voxel[1]-y0,fz=voxel[2]-z0;
    weights.fill(0);
    for(let corner=0;corner<8;corner++){
      const cx=x0+(corner&1),cy=y0+((corner>>1)&1),cz=z0+(corner>>2);
      if(cx<0||cy<0||cz<0||cx>=nx||cy>=ny||cz>=nz)continue;
      const w=(corner&1?fx:1-fx)*((corner>>1)&1?fy:1-fy)*(corner>>2?fz:1-fz),cell=(cx*ny+cy)*nz+cz;
      let owned=0;
      for(let l=0;l<layers.length;l++)if(layers[l].mask[cell]){owned=l+1;break;}
      weights[owned]+=w;
    }
    let best=0;for(let l=1;l<weights.length;l++)if(weights[l]>weights[best])best=l;
    label[i]=best;
    signal[i]=sampleTrilinear(mri,shape,voxel);
  }
  return {width,height,stepMm,label,signal,inside};
}

/** 2nd to 98th percentile of the scan over labelled pixels. */
export function sectionWindow(sample:SectionSample):[number,number]{
  const values:number[]=[];
  for(let i=0;i<sample.label.length;i++)if(sample.label[i]&&Number.isFinite(sample.signal[i]))values.push(sample.signal[i]);
  values.sort((a,b)=>a-b);
  const low=values[Math.floor(values.length*.02)]??0,high=values[Math.floor(values.length*.98)]??1;
  return [low,high>low?high:low+1];
}

const rgb=(hex:string)=>{const n=parseInt(hex.replace('#','').slice(0,6),16);return [(n>>16)&255,(n>>8)&255,n&255];};
/** RGBA pixels: layer colour shaded by the scan; unlabelled space inside the
 * head is grey scan with a cool tint; outside the grid or above the first
 * tissue (`exterior`) is transparent. */
export function paintSection(sample:SectionSample,layers:readonly SectionLayerInput[],window:[number,number],exterior?:Uint8Array){
  const out=new Uint8ClampedArray(sample.label.length*4),colours=layers.map(l=>rgb(sectionColour(l)));
  const [low,high]=window;
  for(let i=0;i<sample.label.length;i++){
    if(!sample.inside[i]||exterior?.[i])continue;
    const t=Math.max(0,Math.min(1,(sample.signal[i]-low)/(high-low)))||0,l=sample.label[i];
    if(l){
      const c=colours[l-1],shade=.5+.7*t;
      out.set([c[0]*shade,c[1]*shade,c[2]*shade,255],4*i);
    }else{
      // Unlabelled space keeps its scan intensity with a cool tint (CSF in a host head is not segmented).
      const g=45+130*t;
      out.set([g*.82,g*.93,g*1.05+10,255],4*i);
    }
  }
  return out;
}

/** Pixels above the first tissue in each column (the image's down axis): outside the head. */
export function exteriorAbove(sample:SectionSample){
  const out=new Uint8Array(sample.label.length);
  for(let x=0;x<sample.width;x++)for(let y=0;y<sample.height;y++){
    const i=y*sample.width+x;
    if(sample.label[i])break;
    out[i]=1;
  }
  return out;
}

export interface DepthRun {name:string;colour:string;fromMm:number;toMm:number}
/** Layers met along one column, from its first tissue, merging runs of one name. */
export function depthRuns(sample:SectionSample,layers:readonly SectionLayerInput[],column:number,minimumMm=.75):{surfaceMm:number|null;runs:DepthRun[]}{
  let surface:number|null=null;const runs:DepthRun[]=[];
  for(let y=0;y<sample.height;y++){
    const i=y*sample.width+column;
    if(!sample.inside[i])continue;
    const l=sample.label[i];
    if(surface===null){if(!l)continue;surface=y;}
    const layer=l?layers[l-1]:undefined,name=layer?sectionName(layer):UNLABELLED_INSIDE;
    const depth=(y-surface)*sample.stepMm,last=runs.at(-1);
    if(last&&last.name===name)last.toMm=depth+sample.stepMm;
    else runs.push({name,colour:layer?sectionColour(layer):'#4a5866',fromMm:depth,toMm:depth+sample.stepMm});
  }
  // Fold slivers into their neighbours so the key reads as a sequence of layers.
  const merged:DepthRun[]=[];
  for(const run of runs){
    const last=merged.at(-1);
    if(last&&run.toMm-run.fromMm<minimumMm){last.toMm=run.toMm;continue;}
    if(last&&last.name===run.name){last.toMm=run.toMm;continue;}
    merged.push({...run});
  }
  return {surfaceMm:surface===null?null:surface*sample.stepMm,runs:merged};
}
