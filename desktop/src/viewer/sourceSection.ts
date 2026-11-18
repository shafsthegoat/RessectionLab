import type { Bounds3, Point3, SlicePlane } from './coordinates.ts';
import { clippingForBounds } from './workspaceSettings.ts';
import type { ViewerWorkspaceSettings } from './workspaceSettings.ts';

type SectionKey = 'activePlane'|'showPlane'|'clipping';
export type SourceSectionSnapshot = Pick<ViewerWorkspaceSettings, SectionKey>;
const PLANES = {x:'sagittal',y:'coronal',z:'axial'} as const;

export function sourceSectionSnapshot(settings: ViewerWorkspaceSettings): SourceSectionSnapshot {
  const snapshot: SourceSectionSnapshot = {};
  for (const key of ['activePlane','showPlane','clipping'] as const)
    if (Object.hasOwn(settings,key)) Object.assign(snapshot,{[key]:key==='clipping'&&settings.clipping?{...settings.clipping}:settings[key]});
  return snapshot;
}

/** Source-plane sampling stays in the renderer's original affine/window. This
 * command only places the existing source plane and primary cut at RAS mm. */
export function sourceSectionAtCursor(settings: ViewerWorkspaceSettings, cursor: Point3, bounds: Bounds3, cameraPosition?: Point3): ViewerWorkspaceSettings {
  const axis = settings.clipping?.axis??'z';
  const index = {x:0,y:1,z:2}[axis];
  // Retain the far half so that the image section faces the current camera.
  const flipped = cameraPosition ? cameraPosition[index]>cursor[index] : settings.clipping?.flipped===true;
  return followSourceSection({...settings,clipping:{...settings.clipping,axis,value:cursor[index],enabled:true,flipped}},cursor,bounds);
}

export function followSourceSection(settings: ViewerWorkspaceSettings, cursor: Point3, bounds: Bounds3): ViewerWorkspaceSettings {
  const clipping = clippingForBounds(settings.clipping,bounds);
  clipping.enabled = true;
  const index = {x:0,y:1,z:2}[clipping.axis];
  clipping.value = Math.max(bounds[0][index],Math.min(bounds[1][index],cursor[index]));
  const activePlane: SlicePlane = PLANES[clipping.axis];
  if (settings.showPlane===true&&settings.activePlane===activePlane&&settings.clipping?.enabled===true&&settings.clipping.value===clipping.value) return settings;
  return {...settings,activePlane,showPlane:true,clipping};
}

export function restoreSourceSection(settings: ViewerWorkspaceSettings, snapshot: SourceSectionSnapshot): ViewerWorkspaceSettings {
  const {activePlane:_activePlane,showPlane:_showPlane,clipping:_clipping,...unchanged} = settings;
  return {...unchanged,...sourceSectionSnapshot(snapshot)};
}
