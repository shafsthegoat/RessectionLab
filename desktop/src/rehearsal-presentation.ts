import type {SurgicalRegion} from './surgical-types';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
const palette:Record<string,string>={soft_tissue:'#b9c5c9',tumor:'#9863db',bone:'#e5d6b8',vessel:'#e54491',membrane:'#82c8d8',unknown:'#92a7b8'};
const opacity:Record<string,number>={soft_tissue:.12,tumor:1,bone:.08,vessel:1,membrane:.25,unknown:.3};
/** Display-only preset. Uses declared materials; never guesses anatomy from names. */
export function rehearsalPresentation(regions:SurgicalRegion[],settings:ViewerWorkspaceSettings={}):ViewerWorkspaceSettings {
  return {...settings,layout:'3d-focus',showPlane:false,volumeRendering:{...settings.volumeRendering,enabled:false},
    layerColor:{...settings.layerColor,...Object.fromEntries(regions.map(r=>[r.name,palette[r.material]??r.color]))},
    layerOpacity:{...settings.layerOpacity,...Object.fromEntries(regions.map(r=>[r.name,opacity[r.material]??.3]))}};
}
