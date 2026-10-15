import type {CasePayload} from './types';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import {validCameraState} from './viewer/workspaceSettings.ts';
import {displayBrainMask} from './case-data.ts';

export function checkedViewerSettings(source:CasePayload,input:unknown):ViewerWorkspaceSettings|undefined {
  if(input===undefined)return undefined;
  const fail=()=>{throw new Error('Workspace reopen withheld: invalid anatomy view settings.');};
  if(!input || typeof input!=='object' || Array.isArray(input))return fail();
  const value=input as ViewerWorkspaceSettings;
  const labels=[...source.compartments.map(layer=>layer.name),...(displayBrainMask(source)?['brain']:[])];
  if(Object.keys(value).some(key=>!['skullMode','volumeRendering','layerOpacity','layerColor','camera','clipping','fusion','layout','activePlane','showPlane','contrast','clipBox','obliqueClip'].includes(key)))return fail();
  if(value.skullMode!==undefined&&!['exterior','transparent','cutaway'].includes(value.skullMode))return fail();
  if(value.volumeRendering!==undefined){
    const volume=value.volumeRendering as Record<string,unknown>;
    if(!volume||typeof volume!=='object'||Array.isArray(volume)||Object.keys(volume).some(k=>!['enabled','preset','density','window','points','shadows','isolateLabels'].includes(k))||
      ('enabled'in volume&&typeof volume.enabled!=='boolean')||
      ('preset'in volume&&!['auto','ct-bone','ct-tissue','mri'].includes(volume.preset as string))||
      ('density'in volume&&(typeof volume.density!=='number'||!Number.isFinite(volume.density)||volume.density<0||volume.density>2)))return fail();
  }
  const volume=value.volumeRendering;
  if(volume){
    if(['shadows','isolateLabels'].some(k=>k in volume&&typeof volume[k as 'shadows']!=='boolean'))return fail();
    if(volume.window&&(!Array.isArray(volume.window)||volume.window.length!==2||!volume.window.every(Number.isFinite)||volume.window[1]<=volume.window[0]))return fail();
    if(volume.points&&(!Array.isArray(volume.points)||volume.points.length<2||volume.points.length>32||volume.points.some((p,i)=>!Array.isArray(p)||p.length!==5||p.some(v=>!Number.isFinite(v)||v<0||v>1)||(i>0&&p[0]<=volume.points![i-1][0]))))return fail();
  }
  const vector=(v:unknown):v is number[]=>Array.isArray(v)&&v.length===3&&v.every(n=>Number.isFinite(n)&&Math.abs(n)<1e7);
  if(value.clipBox&&(!vector(value.clipBox.min)||!vector(value.clipBox.max)||typeof value.clipBox.enabled!=='boolean'||value.clipBox.min.some((v,i)=>v>=value.clipBox!.max[i])))return fail();
  if(value.obliqueClip&&(!vector(value.obliqueClip.normal)||Math.hypot(...value.obliqueClip.normal)<1e-8||!Number.isFinite(value.obliqueClip.offset)||typeof value.obliqueClip.enabled!=='boolean'))return fail();
  if(value.layout!==undefined&&!['mri-review','3d-focus'].includes(value.layout))return fail();
  if(value.activePlane!==undefined&&!['axial','coronal','sagittal'].includes(value.activePlane))return fail();
  if(value.showPlane!==undefined&&typeof value.showPlane!=='boolean')return fail();
  if(value.contrast!==undefined&&(!Number.isFinite(value.contrast)||value.contrast<.4||value.contrast>2))return fail();
  for(const [key,check] of [['layerOpacity',(v:unknown)=>typeof v==='number'&&Number.isFinite(v)&&v>=0&&v<=1],
    ['layerColor',(v:unknown)=>typeof v==='string'&&/^#[a-f0-9]{6}$/i.test(v)]] as const){
    const map=value[key];
    if(map && (typeof map!=='object'||Array.isArray(map)||Object.entries(map).some(([k,v])=>!labels.includes(k)||!check(v))))return fail();
  }
  if(value.camera && !validCameraState(value.camera))return fail();
  const clip=value.clipping;
  if(clip && (!['x','y','z'].includes(clip.axis)||!Number.isFinite(clip.value)||Math.abs(clip.value)>1e6||typeof clip.enabled!=='boolean'||
    (clip.flipped!==undefined&&typeof clip.flipped!=='boolean')))return fail();
  const fusion=value.fusion;
  if(fusion && (!(fusion.seriesId===null || typeof fusion.seriesId==='string')||!Number.isFinite(fusion.opacity)||fusion.opacity<0||fusion.opacity>1))return fail();
  return structuredClone(value);
}
