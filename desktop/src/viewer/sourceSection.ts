import type { Bounds3, Point3, SlicePlane } from './coordinates.ts';
import { rasAffine, volumeBounds } from './coordinates.ts';
import { clippingForBounds } from './workspaceSettings.ts';
import type { ViewerWorkspaceSettings } from './workspaceSettings.ts';
import type { ViewerVolume } from './contracts';
import { isCtAngiography } from './volumeTransfer.ts';
import type { Layer } from '../native-anatomy-data';

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

/** Restrict only the depth of the 3D presentation in patient-frame millimeters. */
export function sourceDepthSlabAtCursor(cursor: Point3, bounds: Bounds3, plane: SlicePlane, thicknessMm: number): NonNullable<ViewerWorkspaceSettings['clipBox']> {
  const axis = {sagittal:0,coronal:1,axial:2}[plane];
  const min: Point3 = [...bounds[0]], max: Point3 = [...bounds[1]];
  min[axis] = cursor[axis] - thicknessMm/2;
  max[axis] = cursor[axis] + thicknessMm/2;
  return {enabled:true,min,max};
}

/** A native CTA can open above a compact, fully loaded arterial field. Bound
 * both the source and conditioned meshes so the opening plane clears every
 * visible arterial vertex. The CTA acquisition and linked cursor stay intact. */
export function ctaArterialCutaway(volume:ViewerVolume,seriesId:string,visible:Readonly<Record<string,boolean>>,
  settings:ViewerWorkspaceSettings,declarations:ReadonlyArray<Pick<Layer,'id'|'role'|'provenance'>>=[],
  coverage?:{combinedId:string;focusIds:ReadonlyArray<string>}):ViewerWorkspaceSettings|undefined {
  if(!volume.displayIdentity||!isCtAngiography(volume.modality??'')||
    !['segmentation','vascular'].includes(settings.presentation??'')||settings.operativeView||
    settings.volumeRendering?.enabled===false||settings.volumeRendering?.density===0||settings.volumeRendering?.isolateLabels)return;
  const layers=volume.compartments.filter(layer=>visible[layer.name]&&settings.layerOpacity?.[layer.name]!==0);
  const arterial=layers.filter(layer=>layer.kind==='artery');
  const bone=layers.filter(layer=>['bone','skull'].includes(layer.kind??'')&&layer.sourceSeries===seriesId);
  if(!arterial.length||arterial.some(layer=>layer.sourceSeries!==seriesId||!layer.surface?.indices.length)||
    !bone.some(layer=>layer.surface?.indices.length))return;
  const useCoverage=coverage&&arterial.length===1&&arterial[0].name===coverage.combinedId?coverage:undefined;
  const focused=useCoverage?volume.compartments.filter(layer=>useCoverage.focusIds.includes(layer.name)&&
    layer.kind==='artery'&&layer.sourceSeries===seriesId&&layer.surface?.indices.length):arterial;
  if(useCoverage&&focused.length!==useCoverage.focusIds.length)return;
  const affine=rasAffine(volume.affine,volume.frame);
  const measure=(members:typeof layers,superior=Infinity):Bounds3|undefined=>{
    const lower:Point3=[Infinity,Infinity,Infinity],upper:Point3=[-Infinity,-Infinity,-Infinity];
    for(const {surface} of members){
      if(!surface?.indices.length)continue;
      const arrays=surface.source===surface.positions?[surface.source]:[surface.source,surface.positions];
      for(const points of arrays)for(let i=0;i<points.length;i+=3){
        const x=points[i],y=points[i+1],z=points[i+2];
        const worldZ=affine[2][0]*x+affine[2][1]*y+affine[2][2]*z+affine[2][3];
        if(worldZ>superior)continue;
        for(let axis=0;axis<3;axis++){
          const a=affine[axis],value=axis===2?worldZ:a[0]*x+a[1]*y+a[2]*z+a[3];
          lower[axis]=Math.min(lower[axis],value);
          upper[axis]=Math.max(upper[axis],value);
        }
      }
    }
    return lower.every(Number.isFinite)&&upper.every(Number.isFinite)?[lower,upper]:undefined;
  };
  const vessels=measure(focused);
  if(!vessels||vessels[1][2]-vessels[0][2]>75)return;
  const source=volumeBounds(volume.shape,affine,true);
  const plane=Math.ceil(vessels[1][2]+8);
  if(source[1][2]-plane<12||plane-source[0][2]<25)return;
  const coverageSurface=useCoverage?arterial[0].surface:undefined;
  if(coverageSurface){
    // The declared focus can coexist with a wide field when the opening still
    // displays the large majority of its source and conditioned contour.
    for(const points of [coverageSurface.source,coverageSurface.positions]){
      let retained=0;
      for(let i=0;i<points.length;i+=3){
        if(affine[2][0]*points[i]+affine[2][1]*points[i+1]+affine[2][2]*points[i+2]+affine[2][3]<=plane)retained++;
      }
      if(retained<points.length/3*.9)return;
    }
  }
  const byId=new Map(declarations.map(layer=>[layer.id,layer]));
  const lesions=layers.filter(layer=>{
    const declared=byId.get(layer.name);
    return (declared?.provenance as {kind?:string}|null)?.kind!=='simulated_observation'&&
      (declared?.role==='target'||['tumor','enhancing-tumor','tumor-core','tumor-whole','edema',
        'nonenhancing-tumor','lesion-context'].includes(layer.kind??''));
  });
  if(lesions.some(layer=>!layer.surface?.indices.length)||(measure(lesions)?.[1][2]??-Infinity)>plane)return;
  const skull=measure(bone,plane);
  if(!skull)return;
  const target:Point3=[(skull[0][0]+skull[1][0])/2,(skull[0][1]+skull[1][1])/2,plane-12];
  const halfWidth=Math.max(target[0]-skull[0][0],skull[1][0]-target[0]);
  const halfHeight=Math.max(target[1]-skull[0][1],skull[1][1]-target[1]);
  const distance=Math.max(100,Math.max(halfWidth/.86,halfHeight)/Math.tan(17*Math.PI/180)*1.08+12);
  return {...settings,showPlane:false,skullMode:'cutaway',clipping:{axis:'z',enabled:true,flipped:true,value:plane},
    camera:{target,position:[target[0]+.10*distance,target[1]+.20*distance,target[2]+distance],up:[0,1,0]}};
}
