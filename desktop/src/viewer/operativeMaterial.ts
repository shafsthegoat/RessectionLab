import * as THREE from 'three';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
// Tissue colour is illustrative. Linked slice overlays retain the original label palette.
export const operativeColors={brain:'#c68788',grayMatter:'#b38b8b',whiteMatter:'#dbc1b1',bone:'#d5bd91',
  dura:'#c8aaa0',tumor:'#aa7188',csf:'#a3b9bc',ventricles:'#82a8b3',artery:'#b85f58',vein:'#7e7489',sinus:'#746a85'} as const;

export function operativeSurfaceStyle(kind:string|undefined,settings:ViewerWorkspaceSettings){
  if(settings.operativeView!=='surgeon'||settings.microscope?.tissueStyle==='labels')return null;
  switch(kind){
    case 'scalp':return {color:'#bf8d7e',roughness:.64};
    case 'galea':return {color:'#d4b8a2',roughness:.67};
    case 'diploe':return {color:'#ae8b71',roughness:.88};
    case 'brain':case 'soft_tissue':case 'tissue':case 'head-envelope':return {color:operativeColors.brain,roughness:.53};
    case 'gray-matter':case 'grayMatter':return {color:operativeColors.grayMatter,roughness:.53};
    case 'white-matter':case 'whiteMatter':return {color:operativeColors.whiteMatter,roughness:.64};
    case 'tumor':return {color:operativeColors.tumor,roughness:.58};
    case 'vessel':case 'vessels':return {color:'#ac6579',roughness:.67};
    case 'artery':case 'vein':case 'sinus':return {color:operativeColors[kind],roughness:.46};
    case 'bone':case 'skull':return {color:operativeColors.bone,roughness:.86};
    case 'membrane':case 'dura':return {color:operativeColors.dura,roughness:.55};
    case 'csf':case 'ventricles':return {color:operativeColors[kind],roughness:.44};
    default:return null;
  }
}

export function applyOperativeSurface(material:THREE.MeshStandardMaterial,kind:string|undefined,settings:ViewerWorkspaceSettings){
  const active=material.userData.tissueFinish;
  if(active)active.value=settings.microscope?.tissueStyle!=='labels'&&(settings.operativeView==='surgeon'||settings.layerOpacity&&Object.values(settings.layerOpacity).every(o=>o===1));
  const style=operativeSurfaceStyle(kind,settings);
  if(style){material.color.set(style.color);material.roughness=style.roughness;material.metalness=0;}
  else if(material instanceof THREE.MeshPhysicalMaterial)material.roughness=.43;
  else material.roughness=kind==='tumor'?.48:['vessel','vessels','artery','vein','sinus'].includes(kind??'')?.32:.52;
}
