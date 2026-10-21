import type {ViewerVolume} from './viewer/contracts';
import type {SurgicalRegion,SurgicalView} from './surgical-types';
import {rasAffine} from './viewer/coordinates.ts';
import {surfaceKind} from './viewer/surfacePolicy.ts';
import {registerModeledAnatomy} from './viewer/modeledAnatomy.ts';

type Layer=ViewerVolume['compartments'][number];
type Base={owner:Uint16Array;regions:SurgicalRegion[];originals:Array<{layer:Layer|undefined;labels:Uint8Array|Uint16Array|undefined}>;priority:Map<number,number>};
type State={volume:ViewerVolume;overrides:Map<number,number>;differences:number[]};
type Cache={mappingKey:string;base:Base;current:State;frames:Map<string,State>};
const caches=new WeakMap<ViewerVolume,Cache>();
export interface ModeledVolumeWork {fullBuild:boolean;sourceCellsScanned:number;copiedMaskBytes:number;patchedCells:number;baseOwnerBytes:number}
const work=new WeakMap<ViewerVolume,ModeledVolumeWork>();
export const modeledVolumeWork=(volume:ViewerVolume)=>work.get(volume);
export function modeledVolumeCacheStatistics(source:ViewerVolume){
  const cache=caches.get(source);return {frames:cache?.frames.size??0,maxFrames:3,baseOwnerBytes:cache?.base.owner.byteLength??0};
}
function geometryIdentity(key:string){let a=2166136261,b=5381;for(let i=0;i<key.length;i++){a=Math.imul(a^key.charCodeAt(i),16777619);b=Math.imul(b,33)^key.charCodeAt(i);}return `${key.length}:${a>>>0}:${b>>>0}`;}
function validateGrid(source:ViewerVolume,view:SurgicalView){
  if(view.scene.shape.some((n,i)=>n!==source.shape[i]))throw Error('Simulation/source grid mismatch');
  const sourceRas=rasAffine(source.affine,source.frame),sceneRas=rasAffine(view.scene.affine,'RAS+');
  if(sourceRas.some((row,i)=>row.some((value,j)=>Math.abs(value-sceneRas[i][j])>1e-6)))throw Error('Simulation/source affine mismatch');
}
function compartment(region:SurgicalRegion,original:Base['originals'][number],mask:Uint8Array,difference:number):Layer {
  const {layer,labels}=original,unchanged=!!layer&&difference===0,sameKind=surfaceKind(layer?.kind)===surfaceKind(region.material);
  return {...(unchanged?layer:{}),name:region.name,color:region.color,kind:region.material,
    mask:unchanged&&layer&&!labels?layer.mask:mask,labelSet:undefined,labelValue:undefined,
    surface:unchanged&&sameKind?layer?.surface:undefined,surfaceDiagnostics:unchanged&&sameKind?layer?.surfaceDiagnostics:undefined,
    sourceIdentity:unchanged?layer?.sourceIdentity:undefined,qa:unchanged?layer?.qa:undefined};
}
function displayVolume(source:ViewerVolume,compartments:Layer[],key:string):ViewerVolume {
  return {...source,displayIdentity:`surgical:${source.caseHash}:${geometryIdentity(key)}`,packedLabels:undefined,labelSets:undefined,compartments};
}
function stateKey(view:SurgicalView,coupled:boolean){return JSON.stringify([view.scene.regions,view.removed,view.moved,...(coupled?['coupled-v2']:[])]);}
/** Complete retained reference: independently scan ownership and every mask.
 * Initial installation also captures this pass's private source owner array. */
export function referenceModeledVolume(source:ViewerVolume,view:SurgicalView,capture?:(base:Base,differences:number[])=>void):ViewerVolume {
  validateGrid(source,view);
  const count=source.shape.reduce((a,b)=>a*b,1),regions=view.scene.regions,coupled=view.physics?.profile?.schema==='resectionlab.tissue-physics.v2';
  const owner=new Uint16Array(count),originals=regions.map(region=>{const layer=source.compartments.find(l=>l.name===region.name);return {layer,labels:source.labelSets?.find(s=>s.id===layer?.labelSet)?.data};});
  regions.forEach((region,r)=>{const {layer,labels}=originals[r];if(layer)for(let i=0;i<count;i++)if(labels?labels[i]===layer.labelValue:layer.mask[i])owner[i]=region.id;});
  const original=owner.slice(),valid=(i:number)=>Number.isInteger(i)&&i>=0&&i<count;
  for(const i of view.removed){if(!valid(i))throw Error('Invalid removed cell');owner[i]=0;}
  for(const [a,b]of view.moved){if(!valid(a)||!valid(b))throw Error('Invalid displacement');owner[a]=0;}
  const priority=new Map(regions.map((r,i)=>[r.id,i]));
  for(const [a,b]of view.moved)if(!coupled||(priority.get(original[a])??-1)>=(priority.get(owner[b])??-1))owner[b]=original[a];
  const differences:number[]=[];
  const compartments=regions.map((region,r)=>{
    const {layer,labels}=originals[r],mask=new Uint8Array(count);let difference=0;
    for(let i=0;i<count;i++){mask[i]=Number(owner[i]===region.id);if(layer&&mask[i]!==Number(labels?labels[i]===layer.labelValue:!!layer.mask[i]))difference++;}
    differences.push(difference);return compartment(region,originals[r],mask,difference);
  });
  for(const layer of source.compartments)if(!regions.some(r=>r.name===layer.name)){
    const labels=source.labelSets?.find(s=>s.id===layer.labelSet)?.data;
    compartments.push({...layer,labelSet:undefined,labelValue:undefined,mask:labels?Uint8Array.from(labels,n=>Number(n===layer.labelValue)):layer.mask});
  }
  capture?.({owner:original,regions:structuredClone(regions),originals,priority},differences);
  return displayVolume(source,compartments,stateKey(view,coupled));
}
function overrides(base:Base,view:SurgicalView,coupled:boolean):Map<number,number>{
  const values=new Map<number,number>(),valid=(i:number)=>Number.isInteger(i)&&i>=0&&i<base.owner.length;
  for(const i of view.removed){if(!valid(i))throw Error('Invalid removed cell');values.set(i,0);}
  for(const [a,b]of view.moved){if(!valid(a)||!valid(b))throw Error('Invalid displacement');values.set(a,0);}
  for(const [a,b]of view.moved){const sourceOwner=base.owner[a],targetOwner=values.get(b)??base.owner[b];
    if(!coupled||(base.priority.get(sourceOwner)??-1)>=(base.priority.get(targetOwner)??-1))values.set(b,sourceOwner);}
  for(const [i,value]of values)if(value===base.owner[i])values.delete(i);
  return values;
}
/** Source ownership is built once per mapping. Follow-up states inspect only
 * the sparse old/new effect union; only changed layer masks receive a private
 * full-length copy. No writable mask or override map is shared across frames.
 * One mapping/source plus three demanded frames bounds retained ownership. */
export function modeledVolume(source:ViewerVolume,view:SurgicalView):ViewerVolume {
  validateGrid(source,view);
  const coupled=view.physics?.profile?.schema==='resectionlab.tissue-physics.v2',key=stateKey(view,coupled),mappingKey=JSON.stringify([view.scene.regions,coupled]);
  let cache=caches.get(source);
  if(cache?.mappingKey===mappingKey){const existing=cache.frames.get(key);if(existing){
    cache.current=existing;cache.frames.delete(key);cache.frames.set(key,existing);
    work.set(existing.volume,{fullBuild:false,sourceCellsScanned:0,copiedMaskBytes:0,patchedCells:0,baseOwnerBytes:cache.base.owner.byteLength});
    return existing.volume;
  }}
  else cache=undefined;
  let state:State,measurement:ModeledVolumeWork;
  if(!cache){
    let base!:Base,differences!:number[];
    const volume=referenceModeledVolume(source,view,(value,diff)=>{base=value;differences=diff;});
    state={volume,overrides:overrides(base,view,coupled),differences};
    cache={mappingKey,base,current:state,frames:new Map()};caches.set(source,cache);
    measurement={fullBuild:true,sourceCellsScanned:base.owner.length*(base.regions.length+base.originals.filter(original=>original.layer).length),copiedMaskBytes:base.owner.length*view.scene.regions.length,patchedCells:0,baseOwnerBytes:base.owner.byteLength};
  }else{
    const base=cache.base,previous=cache.current,next=overrides(base,view,coupled),changed=new Set([...previous.overrides.keys(),...next.keys()]);
    const patches=new Map<number,Array<[number,number]>>(),differences=[...previous.differences];let changedCells=0;
    for(const index of changed){const before=previous.overrides.get(index)??base.owner[index],after=next.get(index)??base.owner[index];if(before===after)continue;changedCells++;
      base.regions.forEach((region,r)=>{if(region.id!==before&&region.id!==after)return;const old=Number(before===region.id),value=Number(after===region.id),{layer,labels}=base.originals[r];
        if(layer){const sourceValue=Number(labels?labels[index]===layer.labelValue:!!layer.mask[index]);differences[r]+=Number(value!==sourceValue)-Number(old!==sourceValue);}
        const cells=patches.get(r)??[];cells.push([index,value]);patches.set(r,cells);
      });
    }
    let copiedMaskBytes=0;
    const compartments=previous.volume.compartments.map((layer,r)=>{
      const cells=patches.get(r);if(!cells)return layer;
      const original=base.originals[r],restoreSource=!!original.layer&&!original.labels&&differences[r]===0;
      const mask=restoreSource?original.layer!.mask:layer.mask.slice();if(!restoreSource){copiedMaskBytes+=mask.byteLength;for(const [index,value]of cells)mask[index]=value;}
      return compartment(base.regions[r],original,mask,differences[r]);
    });
    const volume=displayVolume(source,compartments,key);state={volume,overrides:next,differences};cache.current=state;
    measurement={fullBuild:false,sourceCellsScanned:0,copiedMaskBytes,patchedCells:changedCells,baseOwnerBytes:base.owner.byteLength};
  }
  registerModeledAnatomy(source,state.volume,mappingKey,[...view.removed,...view.moved.flat()]);
  cache.frames.set(key,state);if(cache.frames.size>3)cache.frames.delete(cache.frames.keys().next().value!);
  work.set(state.volume,measurement);return state.volume;
}
