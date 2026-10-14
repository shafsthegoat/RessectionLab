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
  if(Object.keys(value).some(key=>!['layerOpacity','layerColor','camera','clipping','fusion','layout','activePlane','showPlane','contrast'].includes(key)))return fail();
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
