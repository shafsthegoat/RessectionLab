import type {ViewerVolume} from './contracts';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
import {ctaVesselColor} from './anatomySurfaceMaterial.ts';
import {surfaceKind} from './surfacePolicy.ts';
import {layerAppearance} from './workspaceSettings.ts';

type SliceLayer = Pick<ViewerVolume['compartments'][number],'name'|'kind'|'displayKind'>;

/**
 * Declared lesion and planning-target kinds. This mirrors the engine's lesion
 * candidates (`anatomy_planning.LESION_CANDIDATE_KINDS`) plus the source tumour
 * compartments the desktop already treats as lesion anatomy. Names and colours
 * are not evidence; only the declared kind is used.
 */
export const SLICE_TARGET_KINDS: ReadonlySet<string> = new Set([
  'tumor','lesion','tumor-target','lesion-target','synthetic-target',
  'enhancing-lesion','nonenhancing-lesion','nonenhancing-abnormality','edema','edema-like-signal',
  'enhancing-tumor','tumor-core','tumor-whole','nonenhancing-tumor','lesion-context',
]);

export type SliceStructureScope = NonNullable<ViewerWorkspaceSettings['sliceStructures']>;

/** Saved views without this choice use the restrained target-and-selection default. */
export function sliceStructureScope(settings:ViewerWorkspaceSettings|undefined):SliceStructureScope {
  return settings?.sliceStructures==='all'?'all':'focus';
}

/** Saved views without a style use outlines; filled labels stay an explicit choice. */
export function sliceOverlayStyle(settings:ViewerWorkspaceSettings|undefined):'contour'|'fill' {
  return settings?.sliceOverlay==='fill'?'fill':'contour';
}

/**
 * A layer is study target anatomy when it declares a lesion kind. Layers that
 * declare no kind at all (historical engine compartments, explicit review
 * highlights) are the supplied annotation itself and keep their outline.
 */
export function isSliceTargetLayer(layer:SliceLayer):boolean {
  const kinds=[layer.kind,layer.displayKind].filter((kind):kind is string=>typeof kind==='string'&&kind.length>0);
  return kinds.length===0||kinds.some(kind=>SLICE_TARGET_KINDS.has(kind));
}

/**
 * Per-layer slice focus: target/lesion layers and the selected structure.
 * Visibility is applied separately so a hidden layer never reappears here.
 */
export function sliceFocusFlags(layers:readonly SliceLayer[],selected:string|null|undefined):boolean[] {
  return layers.map(layer=>layer.name===selected||isSliceTargetLayer(layer));
}

/** Does this layer draw on the linked slices (before visibility and opacity)? */
export function drawsOnSlices(layer:SliceLayer,selected:string|null|undefined,settings:ViewerWorkspaceSettings|undefined):boolean {
  return sliceStructureScope(settings)==='all'||layer.name===selected||isSliceTargetLayer(layer);
}

/** Pack up to 128 layer flags into the four 32-bit words of an ivec4 uniform. */
export function sliceLayerBits(flags:readonly boolean[],out:Int32Array=new Int32Array(4)):Int32Array {
  out.fill(0);
  flags.forEach((on,index)=>{if(on&&index<128)out[index>>5]|=1<<(index&31);});
  return out;
}

/**
 * The colour a layer's slice outline uses: the same colour as its 3D surface,
 * including user colour overrides and the CTA vessel-type palette.
 */
export function sliceOutlineColor(layer:SliceLayer&{color:string},settings:ViewerWorkspaceSettings,
  volume:{displayIdentity?:string;modality?:string},operativeVessel=false):string {
  const color=layerAppearance(layer.name,layer.color,settings,layer.displayKind??layer.kind).color;
  if(surfaceKind(layer.kind)!=='vessel'||!volume.displayIdentity||operativeVessel)return color;
  return ctaVesselColor(layer.kind,settings,volume.modality)??color;
}

export const SLICE_OUTLINE_CSS_PX=1.5;
/** Outline width in CSS pixels, rounded to whole device pixels so the stroke stays crisp. */
export function sliceOutlineWidth(pixelRatio:number):number {
  const ratio=Number.isFinite(pixelRatio)&&pixelRatio>0?pixelRatio:1;
  return Math.max(1,Math.round(SLICE_OUTLINE_CSS_PX*ratio))/ratio;
}
